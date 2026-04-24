from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_training_log(
    log_csv: str | Path = "models/checkpoints/train_log.csv",
    output_png: str | Path = "models/checkpoints/train_curves.png",
    eval_comparison_json: str | Path = "models/checkpoints/final_eval_comparison.json",
) -> Path:
    df = pd.read_csv(log_csv)
    if "epoch" not in df.columns or "train_loss" not in df.columns:
        raise ValueError("train_log.csv must contain at least epoch and train_loss columns.")

    epochs = df["epoch"].to_numpy()
    train_loss = pd.to_numeric(df["train_loss"], errors="coerce")
    val_loss = pd.to_numeric(df["val_loss"], errors="coerce") if "val_loss" in df else None
    val_auc = pd.to_numeric(df["val_auc"], errors="coerce") if "val_auc" in df else None
    lr = pd.to_numeric(df["lr"], errors="coerce") if "lr" in df else None

    fig, axes = plt.subplots(4, 1, figsize=(10, 15), sharex=True)

    axes[0].plot(epochs, train_loss, label="train_loss", color="#1f77b4")
    axes[0].set_ylabel("Train Loss")
    axes[0].set_title("Training Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    if val_loss is not None and val_loss.notna().any():
        axes[1].plot(epochs, val_loss, label="val_loss", color="#ff7f0e")
    axes[1].set_ylabel("Val Loss")
    axes[1].set_title("Validation Loss")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    if val_auc is not None and val_auc.notna().any():
        axes[2].plot(epochs, val_auc, label="val_auc", color="#2ca02c")
        valid_auc = val_auc.to_numpy(dtype=float)
        best_idx = int(np.nanargmax(valid_auc))
        best_epoch = int(epochs[best_idx])
        best_auc = float(valid_auc[best_idx])
        axes[2].scatter([best_epoch], [best_auc], color="red", s=40, label="best_val_auc")
        axes[2].axvline(best_epoch, color="red", linestyle="--", alpha=0.35)
        axes[2].text(
            best_epoch,
            best_auc,
            f" best@{best_epoch} ({best_auc:.4f})",
            color="red",
            va="bottom",
            fontsize=9,
        )
    axes[2].set_ylabel("AUC")
    axes[2].set_title("Validation AUC")
    axes[2].grid(alpha=0.3)
    axes[2].legend()

    if lr is not None and lr.notna().any():
        axes[3].plot(epochs, lr, label="lr", color="#9467bd")
    axes[3].set_xlabel("Epoch")
    axes[3].set_ylabel("Learning Rate")
    axes[3].set_title("Learning Rate Schedule")
    axes[3].grid(alpha=0.3)
    axes[3].legend()

    best_method = None
    cmp_path = Path(eval_comparison_json)
    if cmp_path.exists():
        try:
            payload = json.loads(cmp_path.read_text(encoding="utf-8"))
            best_method = payload.get("summary", {}).get("best_pr_auc")
        except json.JSONDecodeError:
            best_method = None
    if best_method:
        fig.suptitle(f"Training Curves (Best Calibration: {best_method})", fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    out = Path(output_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training curves from CSV log.")
    parser.add_argument("--log-csv", type=str, default="models/checkpoints/train_log.csv")
    parser.add_argument("--output-png", type=str, default="models/checkpoints/train_curves.png")
    parser.add_argument(
        "--eval-comparison-json", type=str, default="models/checkpoints/final_eval_comparison.json"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out_file = plot_training_log(
        log_csv=args.log_csv,
        output_png=args.output_png,
        eval_comparison_json=args.eval_comparison_json,
    )
    print(f"Saved training plot to: {out_file}")
