from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from llps_diffusion.config import load_config
from llps_diffusion.data.diffusion_dataset import (
    ConditionalDiffusionDataset,
    load_diffusion_examples,
)
from llps_diffusion.data.tokenization import SequenceTokenizer, TokenizerConfig
from llps_diffusion.experiments.reporting import generate_experiment_report
from llps_diffusion.models.conditional_diffusion import ConditionalDiffusionModel


def resolve_device(device: str = "auto") -> torch.device:
    normalized = device.lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but not available. Use --device auto or --device cpu."
            )
        return torch.device("cuda")
    if normalized == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device: {device}. Expected one of: auto, cuda, cpu.")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate_diffusion(
    model: ConditionalDiffusionModel,
    data_loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch in data_loader:
            cond_tokens = batch["cond_tokens"].to(device)
            target_tokens = batch["target_tokens"].to(device)
            loss = model.diffusion_loss(cond_tokens=cond_tokens, target_tokens=target_tokens)
            if torch.isfinite(loss):
                losses.append(float(loss.item()))
    if not losses:
        return float("nan")
    return float(np.mean(losses))


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


def train(config_path: str, device: str = "auto") -> Path:
    cfg = load_config(config_path)
    set_seed(cfg.seed)
    resolved_device = resolve_device(device)
    print(f"Using device: {resolved_device}")
    train_file = Path("data/processed/splits/train.csv")
    fallback_file = Path("data/processed/protein_pairs_selected.csv")
    val_file = Path("data/processed/splits/val.csv")
    train_csv = train_file if train_file.exists() else fallback_file
    if not train_csv.exists():
        raise RuntimeError(
            "Training pairs not found. Please run latest data pipeline first: "
            "make prepare-multi-source && make assemble-training-pairs && make split-pairs"
        )
    tokenizer = SequenceTokenizer(config=TokenizerConfig(max_length=cfg.max_seq_len))
    train_examples = load_diffusion_examples(train_csv, positives_only=True)
    val_examples = (
        load_diffusion_examples(val_file, positives_only=True) if val_file.exists() else []
    )
    if not train_examples:
        raise RuntimeError("No positive training examples found for diffusion training.")
    train_dataset = ConditionalDiffusionDataset(train_examples, tokenizer=tokenizer)
    val_dataset = ConditionalDiffusionDataset(val_examples, tokenizer=tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)

    model = ConditionalDiffusionModel(
        vocab_size=tokenizer.vocab_size,
        embed_dim=cfg.hidden_dim,
        hidden_dim=cfg.hidden_dim,
        max_seq_len=cfg.max_seq_len,
        num_diffusion_steps=cfg.diffusion_steps,
        beta_start=cfg.beta_start,
        beta_end=cfg.beta_end,
        pad_id=tokenizer.pad_id,
    ).to(resolved_device)
    model.sync_schedule_device()
    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    out = Path("models/checkpoints/conditional_diffusion.pt")
    best_out = Path("models/checkpoints/conditional_diffusion_best.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    log_path = Path(cfg.log_csv_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    no_improve_epochs = 0

    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "val_loss",
                "lr",
                "grad_norm_mean",
                "grad_norm_max",
                "skipped",
            ],
        )
        writer.writeheader()

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
            model.train()
            for batch in train_loader:
                cond_tokens = batch["cond_tokens"].to(resolved_device)
                target_tokens = batch["target_tokens"].to(resolved_device)
                optimizer.zero_grad()
                loss = model.diffusion_loss(cond_tokens=cond_tokens, target_tokens=target_tokens)
                if not torch.isfinite(loss):
                    skipped += 1
                    continue
                loss.backward()
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip_norm)
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
            if len(val_dataset) > 0:
                val_loss = evaluate_diffusion(
                    model=model, data_loader=val_loader, device=resolved_device
                )
                print(
                    f"Epoch {epoch + 1}/{cfg.epochs} "
                    f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                    f"grad_norm_mean={grad_norm_mean:.4f} grad_norm_max={grad_norm_max:.4f} "
                    f"skipped={skipped}"
                )
                if np.isfinite(val_loss):
                    if val_loss < best_val_loss - cfg.early_stopping_min_delta:
                        best_val_loss = val_loss
                        no_improve_epochs = 0
                        torch.save(
                            {
                                "state_dict": model.state_dict(),
                                "model_version": "token_diffusion_v2",
                                "config_path": str(config_path),
                            },
                            best_out,
                        )
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
            if len(val_dataset) > 0 and no_improve_epochs >= cfg.early_stopping_patience:
                print(
                    f"Early stopping at epoch {epoch + 1}: "
                    f"val_loss did not improve for {cfg.early_stopping_patience} epochs."
                )
                break

    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_version": "token_diffusion_v2",
            "config_path": str(config_path),
        },
        out,
    )
    if not best_out.exists():
        torch.save(
            {
                "state_dict": model.state_dict(),
                "model_version": "token_diffusion_v2",
                "config_path": str(config_path),
            },
            best_out,
        )
    print(f"Training log saved to: {log_path}")
    print(f"Best checkpoint saved to: {best_out}")
    report_path = generate_experiment_report(
        config_path=config_path,
        device=str(resolved_device),
        checkpoints_dir=out.parent,
    )
    print(f"Auto-generated experiment report: {report_path}")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LLPS diffusion baseline model.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Execution device. auto prefers GPU and falls back to CPU.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ckpt = train(args.config, device=args.device)
    print(f"Checkpoint saved to: {ckpt}")
