from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def merge_positive_sources(
    base_pairs_csv: str | Path = "data/processed/protein_pairs.csv",
    strict_candidates_csv: str | Path = "data/processed/strict_positive_candidates.csv",
    llpsdb_csv: str | Path = "data/processed/llpsdb_positive_pairs.csv",
    output_csv: str | Path = "data/processed/protein_pairs_multi_source.csv",
    report_path: str | Path = "data/processed/multi_source_merge_report.txt",
) -> tuple[Path, Path]:
    base = pd.read_csv(base_pairs_csv)
    required = {"id_a", "id_b", "seq_a", "seq_b", "label", "source"}
    missing = required - set(base.columns)
    if missing:
        raise ValueError(f"Missing columns in base pairs: {sorted(missing)}")

    frames = [base]
    merge_counts: list[tuple[str, int]] = [("base_pairs", len(base))]
    for name, path in [
        ("strict_candidates", Path(strict_candidates_csv)),
        ("llpsdb_pairs", Path(llpsdb_csv)),
    ]:
        if path.exists():
            df = pd.read_csv(path)
            miss = required - set(df.columns)
            if miss:
                raise ValueError(f"Missing columns in {name}: {sorted(miss)}")
            frames.append(df)
            merge_counts.append((name, len(df)))

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["id_a", "id_b", "seq_a", "seq_b", "label"])

    out_csv = Path(output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)

    out_report = Path(report_path)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    lines = ["Multi-source Positive Merge Report"]
    for name, n in merge_counts:
        lines.append(f"{name}: {n}")
    lines.extend(
        [
            f"merged_total: {len(merged)}",
            f"merged_positive: {int((merged['label'] == 1).sum())}",
            f"merged_negative: {int((merged['label'] == 0).sum())}",
            f"output_csv: {out_csv}",
        ]
    )
    out_report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return out_csv, out_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge PhasePro/strict/LLPSDB positive sources.")
    parser.add_argument("--base-pairs", type=str, default="data/processed/protein_pairs.csv")
    parser.add_argument(
        "--strict-candidates", type=str, default="data/processed/strict_positive_candidates.csv"
    )
    parser.add_argument("--llpsdb", type=str, default="data/processed/llpsdb_positive_pairs.csv")
    parser.add_argument(
        "--output", type=str, default="data/processed/protein_pairs_multi_source.csv"
    )
    parser.add_argument(
        "--report", type=str, default="data/processed/multi_source_merge_report.txt"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    merge_positive_sources(
        base_pairs_csv=args.base_pairs,
        strict_candidates_csv=args.strict_candidates,
        llpsdb_csv=args.llpsdb,
        output_csv=args.output,
        report_path=args.report,
    )
