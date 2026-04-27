from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

from llps_diffusion.models.cross_attention import PairCrossAttentionScorer


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


def collect_probs_labels(
    val_csv: str | Path, checkpoint: str | Path, device: str = "auto"
) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(val_csv)
    required = {"seq_a", "seq_b", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in val csv: {sorted(missing)}")

    resolved_device = resolve_device(device)
    model = PairCrossAttentionScorer(input_dim=21, hidden_dim=64).to(resolved_device)
    state = torch.load(checkpoint, map_location=resolved_device)
    model.load_state_dict(state)
    model.eval()

    probs: list[float] = []
    labels: list[int] = []
    with torch.no_grad():
        for _, row in df.iterrows():
            score = model.score(str(row["seq_a"]), str(row["seq_b"]))
            prob = float(torch.sigmoid(score).item())
            if not np.isfinite(prob):
                prob = 0.5
            probs.append(prob)
            labels.append(int(row["label"]))
    return np.array(probs, dtype=np.float64), np.array(labels, dtype=np.int64)


def sweep_thresholds(
    val_csv: str | Path = "data/processed/splits/val.csv",
    checkpoint: str | Path = "models/checkpoints/pair_cross_attention_best.pt",
    output_json: str | Path = "models/checkpoints/best_threshold.json",
    step: float = 0.01,
    device: str = "auto",
) -> dict[str, float]:
    if step <= 0 or step >= 1:
        raise ValueError("step must be in (0, 1).")
    y_prob, y_true = collect_probs_labels(val_csv=val_csv, checkpoint=checkpoint, device=device)
    thresholds = np.arange(step, 1.0, step)

    best = {"threshold": 0.5, "f1": -1.0, "accuracy": 0.0}
    for t in thresholds:
        y_pred = (y_prob >= t).astype(np.int64)
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        acc = float(accuracy_score(y_true, y_pred))
        if f1 > best["f1"]:
            best = {"threshold": float(t), "f1": f1, "accuracy": acc}

    out = Path(output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(best, indent=2), encoding="utf-8")
    print(json.dumps(best, indent=2))
    print(f"Saved best threshold to: {out}")
    return best


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep threshold on validation set and save best F1 threshold."
    )
    parser.add_argument("--val-csv", type=str, default="data/processed/splits/val.csv")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/checkpoints/pair_cross_attention_best.pt",
    )
    parser.add_argument("--output-json", type=str, default="models/checkpoints/best_threshold.json")
    parser.add_argument("--step", type=float, default=0.01)
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
    sweep_thresholds(
        val_csv=args.val_csv,
        checkpoint=args.checkpoint,
        output_json=args.output_json,
        step=args.step,
        device=args.device,
    )
