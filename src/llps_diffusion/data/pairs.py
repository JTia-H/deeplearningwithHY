from __future__ import annotations

import csv
import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProteinPair:
    id_a: str
    id_b: str
    seq_a: str
    seq_b: str
    label: int
    source: str


def build_demo_pairs() -> list[ProteinPair]:
    # Placeholder examples aligned with the PDF classes:
    # positive, random negative, and hard negative.
    return [
        ProteinPair(
            id_a="P35637",
            id_b="Q9H0H5",
            seq_a="MSTNPKPQRKTKRNTNRRPQDVKFPGG",
            seq_b="GGGGSSSSQQQQNNNNKKKK",
            label=1,
            source="phasepro_positive",
        ),
        ProteinPair(
            id_a="P35637",
            id_b="P99999",
            seq_a="MSTNPKPQRKTKRNTNRRPQDVKFPGG",
            seq_b="ACDEFGHIKLMNPQRSTVWY",
            label=0,
            source="swissprot_random_negative",
        ),
        ProteinPair(
            id_a="P35637",
            id_b="Q99999",
            seq_a="MSTNPKPQRKTKRNTNRRPQDVKFPGG",
            seq_b="VVVVVVVVVVVVVVVVVVVV",
            label=0,
            source="string_hard_negative",
        ),
    ]


def save_pairs_csv(pairs: list[ProteinPair], output_path: str | Path) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id_a", "id_b", "seq_a", "seq_b", "label", "source"])
        writer.writeheader()
        for pair in pairs:
            writer.writerow(
                {
                    "id_a": pair.id_a,
                    "id_b": pair.id_b,
                    "seq_a": pair.seq_a,
                    "seq_b": pair.seq_b,
                    "label": pair.label,
                    "source": pair.source,
                }
            )
    return out


def load_pairs_csv(input_path: str | Path) -> list[ProteinPair]:
    path = Path(input_path)
    if not path.exists():
        return []
    pairs: list[ProteinPair] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs.append(
                ProteinPair(
                    id_a=row["id_a"],
                    id_b=row["id_b"],
                    seq_a=row["seq_a"],
                    seq_b=row["seq_b"],
                    label=int(row["label"]),
                    source=row["source"],
                )
            )
    return pairs


def iter_triplets(
    num_negatives: int = 2,
    pairs: list[ProteinPair] | None = None,
    seed: int = 42,
) -> Iterator[tuple[str, str, list[str]]]:
    current_pairs = pairs if pairs is not None else build_demo_pairs()
    positives = [pair for pair in current_pairs if pair.label == 1]
    negatives = [pair.seq_b for pair in current_pairs if pair.label == 0]
    if not positives or not negatives:
        return
    rng = random.Random(seed)
    for pair in positives:
        if len(negatives) <= num_negatives:
            neg_pool = negatives
        else:
            neg_pool = rng.sample(negatives, num_negatives)
        yield pair.seq_a, pair.seq_b, neg_pool
