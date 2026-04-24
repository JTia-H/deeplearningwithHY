from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score

from llps_diffusion.models.cross_attention import PairCrossAttentionScorer


def collect_probs_labels(
    csv_path: str | Path, checkpoint: str | Path
) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    required = {"seq_a", "seq_b", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in csv: {sorted(missing)}")

    model = PairCrossAttentionScorer(input_dim=21, hidden_dim=64)
    state = torch.load(checkpoint, map_location="cpu")
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


def fit_calibrator(
    val_probs: np.ndarray, val_labels: np.ndarray, method: str
) -> tuple[Callable[[np.ndarray], np.ndarray], dict[str, str | float]]:
    if method == "platt":
        clf = LogisticRegression(max_iter=1000)
        clf.fit(val_probs.reshape(-1, 1), val_labels)

        def calibrate_fn(x: np.ndarray) -> np.ndarray:
            return clf.predict_proba(x.reshape(-1, 1))[:, 1]

        return calibrate_fn, {"method": "platt"}
    if method == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(val_probs, val_labels)

        def calibrate_fn(x: np.ndarray) -> np.ndarray:
            return iso.predict(x)

        return calibrate_fn, {"method": "isotonic"}
    raise ValueError(f"Unknown calibration method: {method}")


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(np.int64)
    out: dict[str, float] = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if len(np.unique(y_true)) > 1:
        out["auc"] = float(roc_auc_score(y_true, y_prob))
        out["pr_auc"] = float(average_precision_score(y_true, y_prob))
    else:
        out["auc"] = float("nan")
        out["pr_auc"] = float("nan")
    return out


def calibrate_and_evaluate(
    val_csv: str | Path = "data/processed/splits/val.csv",
    test_csv: str | Path = "data/processed/splits/test.csv",
    checkpoint: str | Path = "models/checkpoints/pair_cross_attention_best.pt",
    threshold_json: str | Path = "models/checkpoints/best_threshold.json",
    output_json: str | Path = "models/checkpoints/calibration_report.json",
    method: str = "platt",
) -> dict[str, object]:
    threshold = 0.5
    th_path = Path(threshold_json)
    if th_path.exists():
        payload = json.loads(th_path.read_text(encoding="utf-8"))
        if "threshold" in payload:
            threshold = float(payload["threshold"])

    val_probs, val_labels = collect_probs_labels(val_csv, checkpoint)
    test_probs, test_labels = collect_probs_labels(test_csv, checkpoint)
    calibrate_fn, meta = fit_calibrator(val_probs, val_labels, method=method)
    test_probs_cal = np.clip(calibrate_fn(test_probs), 0.0, 1.0)

    before = compute_metrics(test_labels, test_probs, threshold=threshold)
    after = compute_metrics(test_labels, test_probs_cal, threshold=threshold)

    report: dict[str, object] = {
        "calibration_method": meta["method"],
        "threshold_used": threshold,
        "n_val": int(len(val_labels)),
        "n_test": int(len(test_labels)),
        "before_calibration": before,
        "after_calibration": after,
    }
    out = Path(output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved calibration report to: {out}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate probabilities on val split and compare test metrics."
    )
    parser.add_argument("--val-csv", type=str, default="data/processed/splits/val.csv")
    parser.add_argument("--test-csv", type=str, default="data/processed/splits/test.csv")
    parser.add_argument(
        "--checkpoint", type=str, default="models/checkpoints/pair_cross_attention_best.pt"
    )
    parser.add_argument(
        "--threshold-json", type=str, default="models/checkpoints/best_threshold.json"
    )
    parser.add_argument(
        "--output-json", type=str, default="models/checkpoints/calibration_report.json"
    )
    parser.add_argument("--method", type=str, default="platt", choices=["platt", "isotonic"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    calibrate_and_evaluate(
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        checkpoint=args.checkpoint,
        threshold_json=args.threshold_json,
        output_json=args.output_json,
        method=args.method,
    )
