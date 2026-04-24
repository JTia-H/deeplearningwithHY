from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score

from llps_diffusion.models.cross_attention import PairCrossAttentionScorer


def evaluate_test_set(
    test_csv: str | Path,
    checkpoint: str | Path,
    output_json: str | Path = "models/checkpoints/test_metrics.json",
    threshold: float | None = None,
    threshold_json: str | Path = "models/checkpoints/best_threshold.json",
    calibration_method: str = "none",
    val_csv_for_calibration: str | Path = "data/processed/splits/val.csv",
) -> dict[str, float | str]:
    resolved_threshold = 0.5
    threshold_source = "default_0.5"
    if threshold is not None:
        resolved_threshold = float(threshold)
        threshold_source = "cli"
    else:
        threshold_path = Path(threshold_json)
        if threshold_path.exists():
            payload = json.loads(threshold_path.read_text(encoding="utf-8"))
            if "threshold" in payload:
                resolved_threshold = float(payload["threshold"])
                threshold_source = str(threshold_path)

    df = pd.read_csv(test_csv)
    required = {"seq_a", "seq_b", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in test csv: {sorted(missing)}")

    model = PairCrossAttentionScorer(input_dim=21, hidden_dim=64)
    ckpt = Path(checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    state = torch.load(ckpt, map_location="cpu")
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

    y_true = np.array(labels, dtype=np.int64)
    y_prob = np.array(probs, dtype=np.float64)
    calibration_source = "none"
    if calibration_method != "none":
        val_df = pd.read_csv(val_csv_for_calibration)
        required_val = {"seq_a", "seq_b", "label"}
        missing_val = required_val - set(val_df.columns)
        if missing_val:
            raise ValueError(f"Missing required columns in val csv: {sorted(missing_val)}")
        val_probs: list[float] = []
        val_labels: list[int] = []
        with torch.no_grad():
            for _, row in val_df.iterrows():
                score = model.score(str(row["seq_a"]), str(row["seq_b"]))
                prob = float(torch.sigmoid(score).item())
                if not np.isfinite(prob):
                    prob = 0.5
                val_probs.append(prob)
                val_labels.append(int(row["label"]))
        x_val = np.array(val_probs, dtype=np.float64)
        y_val = np.array(val_labels, dtype=np.int64)
        if calibration_method == "platt":
            clf = LogisticRegression(max_iter=1000)
            clf.fit(x_val.reshape(-1, 1), y_val)
            y_prob = clf.predict_proba(y_prob.reshape(-1, 1))[:, 1]
            calibration_source = "platt@val"
        elif calibration_method == "isotonic":
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(x_val, y_val)
            y_prob = iso.predict(y_prob)
            calibration_source = "isotonic@val"
        else:
            raise ValueError("calibration_method must be one of: none, platt, isotonic")
        y_prob = np.clip(y_prob, 0.0, 1.0)
    y_pred = (y_prob >= resolved_threshold).astype(np.int64)

    metrics: dict[str, float | str] = {
        "n_samples": float(len(y_true)),
        "threshold": float(resolved_threshold),
        "threshold_source": threshold_source,
        "calibration": calibration_source,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if len(np.unique(y_true)) > 1:
        metrics["auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
    else:
        metrics["auc"] = float("nan")
        metrics["pr_auc"] = float("nan")

    out = Path(output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved test metrics to: {out}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate best checkpoint on test split.")
    parser.add_argument("--test-csv", type=str, default="data/processed/splits/test.csv")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/checkpoints/pair_cross_attention_best.pt",
    )
    parser.add_argument("--output-json", type=str, default="models/checkpoints/test_metrics.json")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument(
        "--threshold-json", type=str, default="models/checkpoints/best_threshold.json"
    )
    parser.add_argument(
        "--calibration-method", type=str, default="none", choices=["none", "platt", "isotonic"]
    )
    parser.add_argument(
        "--val-csv-for-calibration", type=str, default="data/processed/splits/val.csv"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_test_set(
        test_csv=args.test_csv,
        checkpoint=args.checkpoint,
        output_json=args.output_json,
        threshold=args.threshold,
        threshold_json=args.threshold_json,
        calibration_method=args.calibration_method,
        val_csv_for_calibration=args.val_csv_for_calibration,
    )
