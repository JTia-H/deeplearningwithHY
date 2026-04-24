from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def run_qc(input_csv: str | Path, report_path: str | Path | None = None) -> dict[str, float]:
    df = pd.read_csv(input_csv)
    required = {"id_a", "id_b", "seq_a", "seq_b", "label", "source"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    n_total = len(df)
    n_dup = int(df.duplicated(subset=["id_a", "id_b", "seq_a", "seq_b", "label"]).sum())
    n_empty_seq = int(((df["seq_a"].str.len() == 0) | (df["seq_b"].str.len() == 0)).sum())
    pos = int((df["label"] == 1).sum())
    neg = int((df["label"] == 0).sum())
    pos_ratio = float(pos / n_total) if n_total else 0.0
    neg_ratio = float(neg / n_total) if n_total else 0.0

    seq_a_len = df["seq_a"].str.len()
    seq_b_len = df["seq_b"].str.len()
    stats = {
        "total_pairs": float(n_total),
        "duplicate_pairs": float(n_dup),
        "empty_sequence_rows": float(n_empty_seq),
        "positive_count": float(pos),
        "negative_count": float(neg),
        "positive_ratio": pos_ratio,
        "negative_ratio": neg_ratio,
        "seq_a_len_mean": float(seq_a_len.mean()) if n_total else 0.0,
        "seq_b_len_mean": float(seq_b_len.mean()) if n_total else 0.0,
    }

    lines = [
        f"Total pairs: {n_total}",
        f"Duplicates: {n_dup}",
        f"Empty-sequence rows: {n_empty_seq}",
        f"Positive: {pos} ({pos_ratio:.3f})",
        f"Negative: {neg} ({neg_ratio:.3f})",
        f"Mean seq_a length: {stats['seq_a_len_mean']:.2f}",
        f"Mean seq_b length: {stats['seq_b_len_mean']:.2f}",
    ]
    report = "\n".join(lines)
    print(report)

    if report_path is not None:
        out = Path(report_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report + "\n", encoding="utf-8")

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run quality checks for generated protein pair CSV."
    )
    parser.add_argument("--input", type=str, default="data/processed/protein_pairs.csv")
    parser.add_argument("--report", type=str, default="data/processed/pairs_qc_report.txt")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_qc(input_csv=args.input, report_path=args.report)
