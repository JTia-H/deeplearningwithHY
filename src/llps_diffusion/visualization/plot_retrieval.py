from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _extract_k_series(metrics: dict[str, float], prefix: str) -> tuple[list[int], list[float]]:
    pairs: list[tuple[int, float]] = []
    for key, value in metrics.items():
        if not key.startswith(prefix):
            continue
        suffix = key.split("@", maxsplit=1)[-1]
        if not suffix.isdigit():
            continue
        pairs.append((int(suffix), float(value)))
    pairs.sort(key=lambda x: x[0])
    ks = [k for k, _ in pairs]
    vals = [v for _, v in pairs]
    return ks, vals


def plot_retrieval_metrics(
    retrieval_json: str | Path = "models/checkpoints/retrieval_metrics.json",
    output_png: str | Path = "models/checkpoints/retrieval_curves.png",
) -> Path:
    payload = json.loads(Path(retrieval_json).read_text(encoding="utf-8"))
    metrics = payload.get("metrics", {})
    if not isinstance(metrics, dict):
        raise ValueError("Invalid retrieval json: missing metrics object.")

    recall_k, recall_vals = _extract_k_series(metrics, "recall@")
    ndcg_k, ndcg_vals = _extract_k_series(metrics, "ndcg@")
    mrr = float(metrics.get("mrr", 0.0))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    if recall_k:
        axes[0].plot(recall_k, recall_vals, marker="o", label="Recall@K")
    if ndcg_k:
        axes[0].plot(ndcg_k, ndcg_vals, marker="s", label="NDCG@K")
    axes[0].set_xlabel("K")
    axes[0].set_ylabel("Score")
    axes[0].set_title("Retrieval Curves")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    metric_names = ["MRR", "Recall@1", "Recall@5", "Recall@10", "NDCG@10"]
    metric_values = [
        mrr,
        float(metrics.get("recall@1", 0.0)),
        float(metrics.get("recall@5", 0.0)),
        float(metrics.get("recall@10", 0.0)),
        float(metrics.get("ndcg@10", 0.0)),
    ]
    axes[1].bar(np.arange(len(metric_names)), metric_values)
    axes[1].set_xticks(np.arange(len(metric_names)))
    axes[1].set_xticklabels(metric_names, rotation=20, ha="right")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_ylabel("Score")
    axes[1].set_title("Retrieval Summary")
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = Path(output_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot retrieval metrics curves.")
    parser.add_argument(
        "--retrieval-json", type=str, default="models/checkpoints/retrieval_metrics.json"
    )
    parser.add_argument("--output-png", type=str, default="models/checkpoints/retrieval_curves.png")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out = plot_retrieval_metrics(retrieval_json=args.retrieval_json, output_png=args.output_png)
    print(f"Saved retrieval plot to: {out}")
