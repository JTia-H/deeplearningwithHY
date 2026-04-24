from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def assign_evidence_tier(source: str) -> str:
    if source == "phasepro_positive":
        return "tier1_strict"
    if source == "phasepro_string_strict_candidate":
        return "tier1_strict"
    if source == "phasepro_string_proxy_positive":
        return "tier2_supported"
    if source == "phasepro_cohort_proxy_positive":
        return "tier3_proxy"
    return "tier3_proxy"


def canonical_pair_id(id_a: str, id_b: str) -> str:
    left, right = sorted((id_a, id_b))
    return f"{left}__{right}"


def curate_positives(
    input_csv: str | Path = "data/processed/protein_pairs_multi_source.csv",
    strict_candidates_csv: str | Path | None = "data/processed/strict_positive_candidates.csv",
    output_all: str | Path = "data/processed/positives_with_tiers.csv",
    output_strict: str | Path = "data/processed/positives_strict.csv",
    report_path: str | Path = "data/processed/curation_report.txt",
    min_len: int = 30,
    max_len: int = 4000,
) -> tuple[Path, Path, Path]:
    input_path = Path(input_csv)
    if not input_path.exists():
        fallback = Path("data/processed/protein_pairs.csv")
        if fallback.exists():
            input_path = fallback
        else:
            raise FileNotFoundError(f"Neither {input_path} nor fallback {fallback} exists.")
    df = pd.read_csv(input_path)
    n_strict_extra = 0
    if strict_candidates_csv is not None:
        strict_path = Path(strict_candidates_csv)
        if strict_path.exists():
            strict_df = pd.read_csv(strict_path)
            required = {"id_a", "id_b", "seq_a", "seq_b", "label", "source"}
            miss_strict = required - set(strict_df.columns)
            if miss_strict:
                raise ValueError(
                    f"Missing required columns in strict candidates: {sorted(miss_strict)}"
                )
            n_strict_extra = len(strict_df)
            df = pd.concat([df, strict_df], ignore_index=True)
    required = {"id_a", "id_b", "seq_a", "seq_b", "label", "source"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Keep only positives.
    pos = df[df["label"] == 1].copy()
    n_pos_raw = len(pos)

    # Basic sequence sanity filters.
    pos["len_a"] = pos["seq_a"].str.len()
    pos["len_b"] = pos["seq_b"].str.len()
    pos = pos[
        pos["len_a"].between(min_len, max_len)
        & pos["len_b"].between(min_len, max_len)
        & pos["seq_a"].notna()
        & pos["seq_b"].notna()
    ].copy()
    n_pos_after_length = len(pos)

    # Remove symmetric duplicates (A-B == B-A).
    pos["pair_id"] = [
        canonical_pair_id(a, b) for a, b in zip(pos["id_a"], pos["id_b"], strict=False)
    ]
    pos = pos.drop_duplicates(subset=["pair_id"]).copy()
    n_pos_after_dedup = len(pos)

    # Evidence tiers.
    pos["evidence_tier"] = pos["source"].map(assign_evidence_tier)
    tier_counts = pos["evidence_tier"].value_counts(dropna=False).to_dict()

    # Strict set: only tier1 strict positives.
    strict = pos[pos["evidence_tier"] == "tier1_strict"].copy()

    out_all = Path(output_all)
    out_strict = Path(output_strict)
    out_report = Path(report_path)
    out_all.parent.mkdir(parents=True, exist_ok=True)
    out_strict.parent.mkdir(parents=True, exist_ok=True)
    out_report.parent.mkdir(parents=True, exist_ok=True)

    cols = ["id_a", "id_b", "seq_a", "seq_b", "label", "source", "evidence_tier", "pair_id"]
    pos[cols].to_csv(out_all, index=False)
    strict[cols].to_csv(out_strict, index=False)

    report_lines = [
        "Positive Curation Report",
        f"Input file: {input_path}",
        f"Merged strict candidates: {n_strict_extra}",
        f"Raw positive rows: {n_pos_raw}",
        f"After length/sanity filter: {n_pos_after_length}",
        f"After symmetric de-duplication: {n_pos_after_dedup}",
        f"Strict positives (tier1_strict): {len(strict)}",
        "Tier counts:",
    ]
    for key in ("tier1_strict", "tier2_supported", "tier3_proxy"):
        report_lines.append(f"- {key}: {int(tier_counts.get(key, 0))}")
    out_report.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"Saved curated positives: {out_all}")
    print(f"Saved strict positives: {out_strict}")
    print(f"Saved report: {out_report}")
    return out_all, out_strict, out_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Curate high-purity positive LLPS pairs.")
    parser.add_argument(
        "--input", type=str, default="data/processed/protein_pairs_multi_source.csv"
    )
    parser.add_argument(
        "--strict-candidates", type=str, default="data/processed/strict_positive_candidates.csv"
    )
    parser.add_argument("--output-all", type=str, default="data/processed/positives_with_tiers.csv")
    parser.add_argument("--output-strict", type=str, default="data/processed/positives_strict.csv")
    parser.add_argument("--report", type=str, default="data/processed/curation_report.txt")
    parser.add_argument("--min-len", type=int, default=30)
    parser.add_argument("--max-len", type=int, default=4000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    curate_positives(
        input_csv=args.input,
        strict_candidates_csv=args.strict_candidates,
        output_all=args.output_all,
        output_strict=args.output_strict,
        report_path=args.report,
        min_len=args.min_len,
        max_len=args.max_len,
    )
