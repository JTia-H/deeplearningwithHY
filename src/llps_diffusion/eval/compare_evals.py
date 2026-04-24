from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast


def load_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing eval json: {p}")
    return cast(dict[str, Any], json.loads(p.read_text(encoding="utf-8")))


def compare_eval_reports(
    none_json: str | Path = "models/checkpoints/test_metrics.json",
    platt_json: str | Path = "models/checkpoints/test_metrics_platt.json",
    isotonic_json: str | Path = "models/checkpoints/test_metrics_isotonic.json",
    output_json: str | Path = "models/checkpoints/final_eval_comparison.json",
) -> Path:
    none = load_json(none_json)
    platt = load_json(platt_json)
    isotonic = load_json(isotonic_json)

    def pack(name: str, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "name": name,
            "calibration": payload.get("calibration", "unknown"),
            "threshold": payload.get("threshold"),
            "accuracy": payload.get("accuracy"),
            "f1": payload.get("f1"),
            "auc": payload.get("auc"),
            "pr_auc": payload.get("pr_auc"),
        }

    rows = [
        pack("none", none),
        pack("platt", platt),
        pack("isotonic", isotonic),
    ]

    best_f1 = max(rows, key=lambda r: float(r["f1"]))
    best_pr_auc = max(rows, key=lambda r: float(r["pr_auc"]))
    report = {
        "summary": {
            "best_f1": best_f1["name"],
            "best_pr_auc": best_pr_auc["name"],
        },
        "rows": rows,
    }

    out = Path(output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved eval comparison to: {out}")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare none/platt/isotonic eval results.")
    parser.add_argument("--none-json", type=str, default="models/checkpoints/test_metrics.json")
    parser.add_argument(
        "--platt-json", type=str, default="models/checkpoints/test_metrics_platt.json"
    )
    parser.add_argument(
        "--isotonic-json", type=str, default="models/checkpoints/test_metrics_isotonic.json"
    )
    parser.add_argument(
        "--output-json", type=str, default="models/checkpoints/final_eval_comparison.json"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    compare_eval_reports(
        none_json=args.none_json,
        platt_json=args.platt_json,
        isotonic_json=args.isotonic_json,
        output_json=args.output_json,
    )
