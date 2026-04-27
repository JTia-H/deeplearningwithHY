from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_retrieval_eval_csv(
    input_csv: str | Path = "data/processed/splits/test.csv",
    output_csv: str | Path = "data/processed/retrieval_eval.csv",
    report_path: str | Path = "data/processed/retrieval_eval_report.txt",
    max_candidates_per_anchor: int = 0,
    seed: int = 42,
) -> tuple[Path, Path]:
    df = pd.read_csv(input_csv)
    required = {"id_a", "seq_a", "id_b", "seq_b", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in input csv: {sorted(missing)}")

    out_df = df[["id_a", "seq_a", "id_b", "seq_b", "label"]].copy()
    out_df = out_df.drop_duplicates(subset=["id_a", "id_b", "seq_a", "seq_b", "label"]).copy()

    if max_candidates_per_anchor > 0:
        sampled_parts: list[pd.DataFrame] = []
        for _, group in out_df.groupby("id_a", sort=False):
            if len(group) <= max_candidates_per_anchor:
                sampled_parts.append(group.copy())
            else:
                sampled_parts.append(group.sample(n=max_candidates_per_anchor, random_state=seed))
        out_df = pd.concat(sampled_parts, ignore_index=True)

    out_df = out_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    out_csv = Path(output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    n_anchors = int(out_df["id_a"].nunique())
    avg_candidates = float(len(out_df) / n_anchors) if n_anchors > 0 else 0.0
    n_positives = int((out_df["label"] == 1).sum())
    n_negatives = int((out_df["label"] == 0).sum())
    per_anchor_pos = out_df.groupby("id_a")["label"].sum()
    anchors_with_positive = int((per_anchor_pos > 0).sum())

    lines = [
        "Build Retrieval Eval Report",
        f"input_csv: {input_csv}",
        f"output_csv: {out_csv}",
        f"n_rows: {len(out_df)}",
        f"n_anchors: {n_anchors}",
        f"avg_candidates_per_anchor: {avg_candidates:.3f}",
        f"n_positives: {n_positives}",
        f"n_negatives: {n_negatives}",
        f"anchors_with_positive: {anchors_with_positive}",
        f"max_candidates_per_anchor: {max_candidates_per_anchor}",
    ]
    out_report = Path(report_path)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return out_csv, out_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build retrieval evaluation CSV for p(B|A, y=1) metrics."
    )
    parser.add_argument("--input-csv", type=str, default="data/processed/splits/test.csv")
    parser.add_argument("--output-csv", type=str, default="data/processed/retrieval_eval.csv")
    parser.add_argument(
        "--report-path", type=str, default="data/processed/retrieval_eval_report.txt"
    )
    parser.add_argument("--max-candidates-per-anchor", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_retrieval_eval_csv(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        report_path=args.report_path,
        max_candidates_per_anchor=args.max_candidates_per_anchor,
        seed=args.seed,
    )
