from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REQUIRED_OUTPUT_COLUMNS = ["id_a", "id_b", "seq_a", "seq_b", "label", "source"]


def _pick_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return str(lower_map[cand.lower()])
    return None


def normalize_llpsdb_pairs(input_csv: str | Path, output_csv: str | Path) -> Path:
    df = pd.read_csv(input_csv)
    col_id_a = _pick_column(df, ["id_a", "uniprot_a", "protein_a", "protein1_id", "acc_a"])
    col_id_b = _pick_column(df, ["id_b", "uniprot_b", "protein_b", "protein2_id", "acc_b"])
    col_seq_a = _pick_column(df, ["seq_a", "sequence_a", "protein_a_seq", "seq1"])
    col_seq_b = _pick_column(df, ["seq_b", "sequence_b", "protein_b_seq", "seq2"])
    col_label = _pick_column(df, ["label", "llps_label", "is_positive", "target"])

    missing = [
        name
        for name, col in [
            ("id_a", col_id_a),
            ("id_b", col_id_b),
            ("seq_a", col_seq_a),
            ("seq_b", col_seq_b),
        ]
        if col is None
    ]
    if missing:
        raise ValueError(f"Cannot map required LLPSDB columns: {missing}")

    out = pd.DataFrame(
        {
            "id_a": df[col_id_a].astype(str),
            "id_b": df[col_id_b].astype(str),
            "seq_a": df[col_seq_a].astype(str),
            "seq_b": df[col_seq_b].astype(str),
            "label": 1
            if col_label is None
            else pd.to_numeric(df[col_label], errors="coerce").fillna(1).astype(int),
            "source": "llpsdb_curated_positive",
        }
    )

    out = out[(out["seq_a"].str.len() > 0) & (out["seq_b"].str.len() > 0)].copy()
    out = out[out["label"] == 1].copy()
    out = out.drop_duplicates(subset=["id_a", "id_b", "seq_a", "seq_b", "label"])

    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    out[REQUIRED_OUTPUT_COLUMNS].to_csv(output, index=False)
    print(f"Saved normalized LLPSDB positives to: {output} count={len(out)}")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize LLPSDB pairs into project schema.")
    parser.add_argument("--input", type=str, default="data/raw/llpsdb_pairs.csv")
    parser.add_argument("--output", type=str, default="data/processed/llpsdb_positive_pairs.csv")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    normalize_llpsdb_pairs(input_csv=args.input, output_csv=args.output)
