from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ALLOWED_MODES = {"strict_only", "strict_plus_supported", "all"}


def select_positives_by_mode(positives_df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode not in ALLOWED_MODES:
        raise ValueError(f"Unknown mode: {mode}. Allowed: {sorted(ALLOWED_MODES)}")
    if mode == "strict_only":
        return positives_df[positives_df["evidence_tier"] == "tier1_strict"].copy()
    if mode == "strict_plus_supported":
        return positives_df[
            positives_df["evidence_tier"].isin(["tier1_strict", "tier2_supported"])
        ].copy()
    return positives_df.copy()


def assemble_training_pairs(
    all_pairs_csv: str | Path = "data/processed/protein_pairs_multi_source.csv",
    positives_with_tiers_csv: str | Path = "data/processed/positives_with_tiers.csv",
    mode: str = "strict_only",
    output_csv: str | Path = "data/processed/protein_pairs_selected.csv",
    report_path: str | Path = "data/processed/assemble_training_pairs_report.txt",
) -> tuple[Path, Path]:
    all_pairs_path = Path(all_pairs_csv)
    if not all_pairs_path.exists():
        fallback = Path("data/processed/protein_pairs.csv")
        if fallback.exists():
            all_pairs_path = fallback
        else:
            raise FileNotFoundError(f"Neither {all_pairs_path} nor fallback {fallback} exists.")
    all_df = pd.read_csv(all_pairs_path)
    pos_df = pd.read_csv(positives_with_tiers_csv)
    required_pos_cols = {
        "id_a",
        "id_b",
        "seq_a",
        "seq_b",
        "label",
        "source",
        "evidence_tier",
        "pair_id",
    }
    missing = required_pos_cols - set(pos_df.columns)
    if missing:
        raise ValueError(f"Missing required columns in positives_with_tiers.csv: {sorted(missing)}")

    negatives = all_df[all_df["label"] == 0].copy()
    selected_pos = select_positives_by_mode(pos_df, mode=mode)
    selected_pos = selected_pos[["id_a", "id_b", "seq_a", "seq_b", "label", "source"]].copy()

    selected = pd.concat([selected_pos, negatives], ignore_index=True)
    selected = selected.drop_duplicates(subset=["id_a", "id_b", "seq_a", "seq_b", "label"]).copy()
    selected = selected.sample(frac=1.0, random_state=42).reset_index(drop=True)

    out_csv = Path(output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(out_csv, index=False)

    out_report = Path(report_path)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "Assemble Training Pairs Report",
        f"mode: {mode}",
        f"all_pairs_source: {all_pairs_path}",
        f"selected_positives: {len(selected_pos)}",
        f"negatives: {len(negatives)}",
        f"total_selected_pairs: {len(selected)}",
        f"output_csv: {out_csv}",
    ]
    out_report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return out_csv, out_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble training pairs by positive-source mode.")
    parser.add_argument(
        "--all-pairs", type=str, default="data/processed/protein_pairs_multi_source.csv"
    )
    parser.add_argument(
        "--positives-with-tiers", type=str, default="data/processed/positives_with_tiers.csv"
    )
    parser.add_argument("--mode", type=str, default="strict_only", choices=sorted(ALLOWED_MODES))
    parser.add_argument("--output", type=str, default="data/processed/protein_pairs_selected.csv")
    parser.add_argument(
        "--report", type=str, default="data/processed/assemble_training_pairs_report.txt"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    assemble_training_pairs(
        all_pairs_csv=args.all_pairs,
        positives_with_tiers_csv=args.positives_with_tiers,
        mode=args.mode,
        output_csv=args.output,
        report_path=args.report,
    )
