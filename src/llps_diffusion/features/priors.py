from __future__ import annotations


def estimate_idr_ratio(seq: str) -> float:
    # Placeholder heuristic: approximate IDR by low-complexity residue ratio.
    seq = seq.strip().upper()
    if not seq:
        return 0.0
    idr_like = set("GPQSNEDKR")
    count = sum(1 for aa in seq if aa in idr_like)
    return count / len(seq)


def estimate_prld_score(seq: str) -> float:
    # Placeholder heuristic: approximate PrLD by Q/N enrichment.
    seq = seq.strip().upper()
    if not seq:
        return 0.0
    count = sum(1 for aa in seq if aa in {"Q", "N"})
    return count / len(seq)
