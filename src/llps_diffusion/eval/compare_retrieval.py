from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast


def _load_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing retrieval json: {p}")
    return cast(dict[str, Any], json.loads(p.read_text(encoding="utf-8")))


def compare_retrieval_reports(
    baseline_json: str | Path,
    candidate_json: str | Path,
    output_json: str | Path = "models/checkpoints/retrieval_comparison.json",
    baseline_name: str = "baseline",
    candidate_name: str = "candidate",
) -> Path:
    baseline = _load_json(baseline_json)
    candidate = _load_json(candidate_json)
    base_metrics = baseline.get("metrics", {})
    cand_metrics = candidate.get("metrics", {})

    metric_keys = [
        "mrr",
        "recall@1",
        "recall@5",
        "recall@10",
        "recall@20",
        "ndcg@1",
        "ndcg@5",
        "ndcg@10",
        "ndcg@20",
    ]
    rows: list[dict[str, float | str]] = []
    for key in metric_keys:
        base_val = float(base_metrics.get(key, 0.0))
        cand_val = float(cand_metrics.get(key, 0.0))
        rows.append(
            {
                "metric": key,
                "baseline": base_val,
                "candidate": cand_val,
                "delta": cand_val - base_val,
            }
        )

    wins = sum(1 for row in rows if float(row["delta"]) > 0)
    losses = sum(1 for row in rows if float(row["delta"]) < 0)
    ties = len(rows) - wins - losses
    report = {
        "baseline_name": baseline_name,
        "candidate_name": candidate_name,
        "summary": {"wins": wins, "losses": losses, "ties": ties},
        "rows": rows,
    }
    out = Path(output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved retrieval comparison to: {out}")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two retrieval evaluation reports.")
    parser.add_argument("--baseline-json", type=str, required=True)
    parser.add_argument("--candidate-json", type=str, required=True)
    parser.add_argument(
        "--output-json", type=str, default="models/checkpoints/retrieval_comparison.json"
    )
    parser.add_argument("--baseline-name", type=str, default="baseline")
    parser.add_argument("--candidate-name", type=str, default="candidate")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    compare_retrieval_reports(
        baseline_json=args.baseline_json,
        candidate_json=args.candidate_json,
        output_json=args.output_json,
        baseline_name=args.baseline_name,
        candidate_name=args.candidate_name,
    )
