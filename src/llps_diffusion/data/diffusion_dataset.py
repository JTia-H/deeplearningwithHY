from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from llps_diffusion.data.tokenization import SequenceTokenizer


@dataclass(frozen=True)
class DiffusionExample:
    id_a: str
    seq_a: str
    id_b: str
    seq_b: str
    label: int


class ConditionalDiffusionDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, examples: list[DiffusionExample], tokenizer: SequenceTokenizer) -> None:
        self.examples = examples
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ex = self.examples[idx]
        return {
            "cond_tokens": torch.tensor(self.tokenizer.encode(ex.seq_a), dtype=torch.long),
            "target_tokens": torch.tensor(self.tokenizer.encode(ex.seq_b), dtype=torch.long),
            "label": torch.tensor(ex.label, dtype=torch.long),
        }


def load_diffusion_examples(
    input_csv: str | Path, positives_only: bool = True
) -> list[DiffusionExample]:
    path = Path(input_csv)
    if not path.exists():
        return []
    df = pd.read_csv(path)
    required = {"id_a", "seq_a", "id_b", "seq_b", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in diffusion csv: {sorted(missing)}")
    if positives_only:
        df = df[df["label"] == 1].copy()
    out: list[DiffusionExample] = []
    for _, row in df.iterrows():
        out.append(
            DiffusionExample(
                id_a=str(row["id_a"]),
                seq_a=str(row["seq_a"]),
                id_b=str(row["id_b"]),
                seq_b=str(row["seq_b"]),
                label=int(row["label"]),
            )
        )
    return out
