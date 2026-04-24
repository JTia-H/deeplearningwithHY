from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch import nn
from torch.optim import AdamW

from llps_diffusion.config import TrainConfig, load_config
from llps_diffusion.data.pairs import ProteinPair, iter_triplets, load_pairs_csv
from llps_diffusion.losses.infonce import infonce_loss
from llps_diffusion.models.cross_attention import PairCrossAttentionScorer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate_pairs(
    model: PairCrossAttentionScorer,
    pairs: list[ProteinPair],
    temperature: float,
    loss_type: str,
    bce_criterion: nn.BCEWithLogitsLoss,
) -> tuple[float, float, float]:
    model.eval()
    losses: list[float] = []
    probs: list[float] = []
    labels: list[int] = []
    with torch.no_grad():
        for pair in pairs:
            score = model.score(pair.seq_a, pair.seq_b)
            prob = float(torch.sigmoid(score).item())
            if not np.isfinite(prob):
                prob = 0.5
            probs.append(prob)
            labels.append(pair.label)
            if loss_type == "bce":
                target = torch.tensor([float(pair.label)], dtype=torch.float32)
                loss = bce_criterion(score.reshape(1), target)
            else:
                if pair.label == 1:
                    pos_score = score
                    neg_scores = torch.stack([torch.tensor(0.0, dtype=torch.float32)])
                else:
                    pos_score = torch.tensor(0.0, dtype=torch.float32)
                    neg_scores = torch.stack([score])
                loss = infonce_loss(
                    pos_score=pos_score, neg_scores=neg_scores, temperature=temperature
                )
            loss_value = float(loss.item())
            if not np.isfinite(loss_value):
                loss_value = 1e6
            losses.append(loss_value)
    if not losses:
        raise RuntimeError("Validation split is empty.")
    acc = float(accuracy_score(labels, [1 if p >= 0.5 else 0 for p in probs]))
    valid = np.isfinite(np.array(probs, dtype=np.float64))
    valid_labels = [label for label, ok in zip(labels, valid, strict=False) if ok]
    valid_probs = [prob for prob, ok in zip(probs, valid, strict=False) if ok]
    if len(set(valid_labels)) < 2 or len(valid_probs) == 0:
        auc = float("nan")
    else:
        auc = float(roc_auc_score(valid_labels, valid_probs))
    return float(np.mean(losses)), acc, auc


def compute_epoch_lr(
    epoch_idx: int, total_epochs: int, base_lr: float, min_lr: float, warmup_epochs: int
) -> float:
    if total_epochs <= 0:
        return base_lr
    if epoch_idx < warmup_epochs:
        # Linear warmup from 10% base_lr to base_lr.
        warmup_ratio = (epoch_idx + 1) / max(warmup_epochs, 1)
        return base_lr * (0.1 + 0.9 * warmup_ratio)
    # Cosine decay after warmup.
    decay_steps = max(total_epochs - warmup_epochs, 1)
    progress = (epoch_idx - warmup_epochs) / decay_steps
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


def build_balanced_bce_epoch_pairs(pairs: list[ProteinPair], seed: int) -> list[ProteinPair]:
    positives = [p for p in pairs if p.label == 1]
    negatives = [p for p in pairs if p.label == 0]
    if not positives or not negatives:
        return pairs
    rng = random.Random(seed)
    target_n = max(len(positives), len(negatives))
    if len(positives) < target_n:
        positives = positives + rng.choices(positives, k=target_n - len(positives))
    if len(negatives) < target_n:
        negatives = negatives + rng.choices(negatives, k=target_n - len(negatives))
    merged = positives + negatives
    rng.shuffle(merged)
    return merged


def train_bce_one_epoch(
    model: PairCrossAttentionScorer,
    pairs: list[ProteinPair],
    optimizer: AdamW,
    bce_criterion: nn.BCEWithLogitsLoss,
    grad_clip_norm: float,
    seed: int,
    balance_sampling: bool,
) -> None:
    epoch_pairs = pairs
    if balance_sampling:
        epoch_pairs = build_balanced_bce_epoch_pairs(epoch_pairs, seed=seed)
    else:
        epoch_pairs = epoch_pairs.copy()
        random.Random(seed).shuffle(epoch_pairs)
    model.train()
    for pair in epoch_pairs:
        optimizer.zero_grad()
        score = model.score(pair.seq_a, pair.seq_b)
        score = torch.nan_to_num(score, nan=0.0, posinf=20.0, neginf=-20.0)
        target = torch.tensor([float(pair.label)], dtype=torch.float32)
        loss = bce_criterion(score.reshape(1), target)
        if not torch.isfinite(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()


def auto_select_pos_weight(
    cfg: TrainConfig,
    train_pairs: list[ProteinPair],
    val_pairs: list[ProteinPair],
    input_dim: int,
) -> float:
    best_weight = cfg.bce_pos_weight
    best_auc = float("-inf")
    best_val_loss = float("inf")
    selection_rows: list[dict[str, float]] = []
    for idx, candidate in enumerate(cfg.bce_pos_weight_grid):
        set_seed(cfg.seed + idx * 100)
        model = PairCrossAttentionScorer(input_dim=input_dim, hidden_dim=cfg.hidden_dim)
        bce_criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([candidate], dtype=torch.float32)
        )
        optimizer = AdamW(model.parameters(), lr=cfg.learning_rate)
        for epoch in range(max(cfg.bce_pos_weight_tune_epochs, 1)):
            train_bce_one_epoch(
                model=model,
                pairs=train_pairs,
                optimizer=optimizer,
                bce_criterion=bce_criterion,
                grad_clip_norm=cfg.grad_clip_norm,
                seed=cfg.seed + idx * 1000 + epoch,
                balance_sampling=cfg.bce_balance_sampling,
            )
        val_loss, _, val_auc = evaluate_pairs(
            model=model,
            pairs=val_pairs,
            temperature=cfg.temperature,
            loss_type="bce",
            bce_criterion=bce_criterion,
        )
        score_auc = val_auc if np.isfinite(val_auc) else float("-inf")
        selection_rows.append(
            {
                "pos_weight": float(candidate),
                "val_auc": float(val_auc) if np.isfinite(val_auc) else float("nan"),
                "val_loss": float(val_loss),
            }
        )
        if score_auc > best_auc or (
            np.isclose(score_auc, best_auc) and np.isfinite(val_loss) and val_loss < best_val_loss
        ):
            best_auc = score_auc
            best_val_loss = val_loss
            best_weight = float(candidate)
    report_path = Path("models/checkpoints/pos_weight_selection.csv")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["pos_weight", "val_auc", "val_loss"])
        writer.writeheader()
        for row in selection_rows:
            writer.writerow(row)
    print(f"Auto-selected bce_pos_weight={best_weight:.4f} (report: {report_path})")
    return float(best_weight)


def train(config_path: str) -> Path:
    cfg = load_config(config_path)
    set_seed(cfg.seed)
    input_dim = 21
    train_file = Path("data/processed/splits/train.csv")
    fallback_file = Path("data/processed/protein_pairs_selected.csv")
    val_file = Path("data/processed/splits/val.csv")

    train_pairs: list[ProteinPair] | None
    if train_file.exists():
        train_pairs = load_pairs_csv(train_file)
    elif fallback_file.exists():
        train_pairs = load_pairs_csv(fallback_file)
    else:
        train_pairs = None
    val_pairs = load_pairs_csv(val_file) if val_file.exists() else []
    selected_pos_weight = cfg.bce_pos_weight
    if (
        cfg.loss_type == "bce"
        and cfg.bce_auto_pos_weight
        and isinstance(train_pairs, list)
        and len(train_pairs) > 0
        and len(val_pairs) > 0
    ):
        selected_pos_weight = auto_select_pos_weight(
            cfg=cfg,
            train_pairs=train_pairs,
            val_pairs=val_pairs,
            input_dim=input_dim,
        )
    model = PairCrossAttentionScorer(input_dim=input_dim, hidden_dim=cfg.hidden_dim)
    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate)
    bce_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([selected_pos_weight], dtype=torch.float32)
    )
    out = Path("models/checkpoints/pair_cross_attention.pt")
    best_out = Path("models/checkpoints/pair_cross_attention_best.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    log_path = Path(cfg.log_csv_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_auc = float("-inf")
    no_improve_epochs = 0

    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "val_loss",
                "val_acc",
                "val_auc",
                "lr",
                "grad_norm_mean",
                "grad_norm_max",
                "skipped",
            ],
        )
        writer.writeheader()

        model.train()
        for epoch in range(cfg.epochs):
            epoch_lr = compute_epoch_lr(
                epoch_idx=epoch,
                total_epochs=cfg.epochs,
                base_lr=cfg.learning_rate,
                min_lr=cfg.min_learning_rate,
                warmup_epochs=cfg.warmup_epochs,
            )
            for group in optimizer.param_groups:
                group["lr"] = epoch_lr
            epoch_loss = 0.0
            batch_count = 0
            skipped = 0
            grad_norm_values: list[float] = []
            current_lr = float(epoch_lr)
            if train_pairs is None:
                raise RuntimeError(
                    "Training pairs not found. Please run latest data pipeline first: "
                    "make prepare-multi-source && make assemble-training-pairs && make split-pairs"
                )
            train_pairs_epoch = train_pairs.copy()
            random.Random(cfg.seed + epoch).shuffle(train_pairs_epoch)
            if cfg.loss_type == "bce":
                if not isinstance(train_pairs_epoch, list):
                    raise RuntimeError("BCE training requires explicit pair rows.")
                if cfg.bce_balance_sampling:
                    train_pairs_epoch = build_balanced_bce_epoch_pairs(
                        train_pairs_epoch, seed=cfg.seed + epoch
                    )
                for pair in train_pairs_epoch:
                    optimizer.zero_grad()
                    score = model.score(pair.seq_a, pair.seq_b)
                    score = torch.nan_to_num(score, nan=0.0, posinf=20.0, neginf=-20.0)
                    target = torch.tensor([float(pair.label)], dtype=torch.float32)
                    loss = bce_criterion(score.reshape(1), target)
                    if not torch.isfinite(loss):
                        skipped += 1
                        continue
                    loss.backward()
                    grad_norm = float(
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), max_norm=cfg.grad_clip_norm
                        )
                    )
                    if np.isfinite(grad_norm):
                        grad_norm_values.append(grad_norm)
                    optimizer.step()
                    epoch_loss += float(loss.detach().item())
                    batch_count += 1
            else:
                for seq_a, pos_b, neg_bs in iter_triplets(
                    num_negatives=cfg.num_negatives,
                    pairs=train_pairs_epoch,
                    seed=cfg.seed + epoch,
                ):
                    optimizer.zero_grad()
                    pos_score = model.score(seq_a, pos_b)
                    neg_scores = torch.stack([model.score(seq_a, neg_b) for neg_b in neg_bs], dim=0)
                    pos_score = torch.nan_to_num(pos_score, nan=0.0, posinf=20.0, neginf=-20.0)
                    neg_scores = torch.nan_to_num(neg_scores, nan=0.0, posinf=20.0, neginf=-20.0)
                    loss = infonce_loss(
                        pos_score=pos_score, neg_scores=neg_scores, temperature=cfg.temperature
                    )
                    if not torch.isfinite(loss):
                        skipped += 1
                        continue
                    loss.backward()
                    grad_norm = float(
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), max_norm=cfg.grad_clip_norm
                        )
                    )
                    if np.isfinite(grad_norm):
                        grad_norm_values.append(grad_norm)
                    optimizer.step()
                    epoch_loss += float(loss.detach().item())
                    batch_count += 1
            if batch_count == 0:
                print(f"Epoch {epoch + 1}/{cfg.epochs} skipped: no finite training loss.")
                writer.writerow(
                    {
                        "epoch": epoch + 1,
                        "train_loss": "",
                        "val_loss": "",
                        "val_acc": "",
                        "val_auc": "",
                        "lr": f"{current_lr:.8f}",
                        "grad_norm_mean": "",
                        "grad_norm_max": "",
                        "skipped": skipped,
                    }
                )
                continue

            train_loss = epoch_loss / batch_count
            grad_norm_mean = float(np.mean(grad_norm_values)) if grad_norm_values else float("nan")
            grad_norm_max = float(np.max(grad_norm_values)) if grad_norm_values else float("nan")
            val_loss = float("nan")
            val_acc = float("nan")
            val_auc = float("nan")
            if val_pairs:
                val_loss, val_acc, val_auc = evaluate_pairs(
                    model=model,
                    pairs=val_pairs,
                    temperature=cfg.temperature,
                    loss_type=cfg.loss_type,
                    bce_criterion=bce_criterion,
                )
                print(
                    f"Epoch {epoch + 1}/{cfg.epochs} "
                    f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                    f"val_acc={val_acc:.4f} val_auc={val_auc:.4f} "
                    f"grad_norm_mean={grad_norm_mean:.4f} grad_norm_max={grad_norm_max:.4f} "
                    f"skipped={skipped}"
                )
                if np.isfinite(val_auc):
                    if val_auc > best_val_auc + cfg.early_stopping_min_delta:
                        best_val_auc = val_auc
                        no_improve_epochs = 0
                        torch.save(model.state_dict(), best_out)
                    else:
                        no_improve_epochs += 1
            else:
                print(
                    f"Epoch {epoch + 1}/{cfg.epochs} train_loss={train_loss:.4f} skipped={skipped}"
                )

            writer.writerow(
                {
                    "epoch": epoch + 1,
                    "train_loss": f"{train_loss:.8f}",
                    "val_loss": "" if not np.isfinite(val_loss) else f"{val_loss:.8f}",
                    "val_acc": "" if not np.isfinite(val_acc) else f"{val_acc:.8f}",
                    "val_auc": "" if not np.isfinite(val_auc) else f"{val_auc:.8f}",
                    "lr": f"{current_lr:.8f}",
                    "grad_norm_mean": ""
                    if not np.isfinite(grad_norm_mean)
                    else f"{grad_norm_mean:.8f}",
                    "grad_norm_max": ""
                    if not np.isfinite(grad_norm_max)
                    else f"{grad_norm_max:.8f}",
                    "skipped": skipped,
                }
            )
            model.train()

            if val_pairs and no_improve_epochs >= cfg.early_stopping_patience:
                print(
                    f"Early stopping at epoch {epoch + 1}: "
                    f"val_auc did not improve for {cfg.early_stopping_patience} epochs."
                )
                break

    torch.save(model.state_dict(), out)
    if not best_out.exists():
        torch.save(model.state_dict(), best_out)
    print(f"Training log saved to: {log_path}")
    print(f"Best checkpoint saved to: {best_out}")
    print(f"Training bce_pos_weight used: {selected_pos_weight:.4f}")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LLPS diffusion baseline model.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ckpt = train(args.config)
    print(f"Checkpoint saved to: {ckpt}")
