from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_candidate_b_csv(
    input_csv: str | Path = "data/processed/splits/test.csv",
    fallback_csv: str | Path = "data/processed/protein_pairs_selected.csv",
    output_csv: str | Path = "data/processed/candidate_b.csv",
) -> Path:
    source = Path(input_csv)
    if not source.exists():
        source = Path(fallback_csv)
    if not source.exists():
        raise FileNotFoundError(
            f"Neither input_csv ({input_csv}) nor fallback_csv ({fallback_csv}) exists."
        )

    df = pd.read_csv(source)
    required = {"id_b", "seq_b"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in source csv: {sorted(missing)}")

    out_df = df[["id_b", "seq_b"]].copy()
    out_df["id_b"] = out_df["id_b"].astype(str)
    out_df["seq_b"] = out_df["seq_b"].astype(str).str.strip()
    out_df = out_df[out_df["seq_b"] != ""].drop_duplicates(subset=["id_b", "seq_b"])
    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)
    print(f"Built candidate B pool: {out} (n={len(out_df)}, source={source})")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build candidate B pool CSV for diffusion retrieval prediction."
    )
    parser.add_argument("--input-csv", type=str, default="data/processed/splits/test.csv")
    parser.add_argument(
        "--fallback-csv", type=str, default="data/processed/protein_pairs_selected.csv"
    )
    parser.add_argument("--output-csv", type=str, default="data/processed/candidate_b.csv")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_candidate_b_csv(
        input_csv=args.input_csv,
        fallback_csv=args.fallback_csv,
        output_csv=args.output_csv,
    )
