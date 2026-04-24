from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def split_by_anchor_protein(
    input_csv: str | Path,
    out_dir: str | Path = "data/processed/splits",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[Path, Path, Path]:
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1.0:
        raise ValueError("Invalid split ratios: require 0 < train, 0 <= val, and train+val < 1.")

    df = pd.read_csv(input_csv)
    if "id_a" not in df.columns:
        raise ValueError("Input CSV must contain id_a column for leakage-safe anchor split.")

    unique_ids = df["id_a"].dropna().unique()
    id_df = pd.DataFrame({"id_a": unique_ids})
    id_df = id_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    n = len(id_df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    if n_train == 0 or n_test == 0:
        raise ValueError("Split produced empty train/test. Increase dataset size or adjust ratios.")

    train_ids = set(id_df.iloc[:n_train]["id_a"])
    val_ids = set(id_df.iloc[n_train : n_train + n_val]["id_a"])
    test_ids = set(id_df.iloc[n_train + n_val :]["id_a"])

    train_df = df[df["id_a"].isin(train_ids)].copy()
    val_df = df[df["id_a"].isin(val_ids)].copy()
    test_df = df[df["id_a"].isin(test_ids)].copy()

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_path = out / "train.csv"
    val_path = out / "val.csv"
    test_path = out / "test.csv"
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(
        f"Saved split files to {out}: "
        f"train={len(train_df)} val={len(val_df)} test={len(test_df)} "
        f"(anchor proteins: train={len(train_ids)} val={len(val_ids)} test={len(test_ids)})"
    )
    return train_path, val_path, test_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leakage-safe split for protein-pair dataset.")
    parser.add_argument("--input", type=str, default="data/processed/protein_pairs.csv")
    parser.add_argument("--out-dir", type=str, default="data/processed/splits")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    split_by_anchor_protein(
        input_csv=args.input,
        out_dir=args.out_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
