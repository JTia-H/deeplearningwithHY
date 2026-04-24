from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def sequence_to_features(seq: str) -> list[float]:
    clean_seq = seq.strip().upper()
    length = max(len(clean_seq), 1)
    counts = [clean_seq.count(aa) / length for aa in AMINO_ACIDS]
    return counts + [float(len(clean_seq))]


@dataclass(frozen=True)
class LabeledSequence:
    sequence: str
    label: int


def iter_dummy_dataset() -> Iterator[LabeledSequence]:
    # Baseline placeholder data; replace with real dataset construction later.
    examples = [
        LabeledSequence("MSTNPKPQRKTKRNTNRRPQDVKFPGG", 1),
        LabeledSequence("ACDEFGHIKLMNPQRSTVWY", 0),
        LabeledSequence("GGGGGGGGGGGGGGGG", 1),
        LabeledSequence("VVVVVVVVVV", 0),
    ]
    yield from examples
