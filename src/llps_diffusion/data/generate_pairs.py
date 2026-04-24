from __future__ import annotations

import argparse
import json
import random
import re
from typing import Any

import requests  # type: ignore[import-untyped]

from llps_diffusion.data.pairs import ProteinPair, save_pairs_csv

PHASEPRO_FULL_URL = "https://phasepro.elte.hu/download_full.json"
UNIPROT_ENTRY_URL = "https://rest.uniprot.org/uniprotkb/{accession}.json"
UNIPROT_SWISSPROT_QUERY_URL = "https://rest.uniprot.org/uniprotkb/search"
STRING_INTERACTION_URL = "https://string-db.org/api/json/interaction_partners"
UNIPROT_ID_REGEX = re.compile(r"\b(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]{5})\b")


def fetch_json(url: str, params: dict[str, Any] | None = None) -> Any:
    response = requests.get(url, params=params, timeout=90)
    response.raise_for_status()
    return response.json()


def extract_uniprot_ids(value: Any) -> list[str]:
    text = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
    return sorted(set(UNIPROT_ID_REGEX.findall(text)))


def fetch_uniprot_sequence(accession: str, cache: dict[str, str]) -> str:
    if accession in cache:
        return cache[accession]
    payload = fetch_json(UNIPROT_ENTRY_URL.format(accession=accession))
    seq = str(payload.get("sequence", {}).get("value", ""))
    if seq:
        cache[accession] = seq
    return seq


def load_phasepro_positives(seq_cache: dict[str, str], max_pairs: int) -> list[ProteinPair]:
    payload = fetch_json(PHASEPRO_FULL_URL)
    positives: list[ProteinPair] = []
    seen: set[tuple[str, str]] = set()

    if not isinstance(payload, dict):
        return positives

    for key, value in payload.items():
        id_a = key if UNIPROT_ID_REGEX.fullmatch(str(key)) else None
        if id_a is None:
            continue
        partners = extract_uniprot_ids(value.get("partners", {}))
        seq_a = value.get("sequence", "") or fetch_uniprot_sequence(id_a, seq_cache)
        if not seq_a:
            continue
        for id_b in partners:
            if id_b == id_a:
                continue
            seq_b = fetch_uniprot_sequence(id_b, seq_cache)
            if not seq_b:
                continue
            key_pair = tuple(sorted((id_a, id_b)))
            if key_pair in seen:
                continue
            seen.add(key_pair)
            positives.append(
                ProteinPair(
                    id_a=id_a,
                    id_b=id_b,
                    seq_a=seq_a,
                    seq_b=seq_b,
                    label=1,
                    source="phasepro_positive",
                )
            )
            if len(positives) >= max_pairs:
                return positives
    return positives


def load_phasepro_drivers(seq_cache: dict[str, str]) -> list[tuple[str, str]]:
    payload = fetch_json(PHASEPRO_FULL_URL)
    drivers: list[tuple[str, str]] = []
    if not isinstance(payload, dict):
        return drivers
    for key, value in payload.items():
        accession = key if UNIPROT_ID_REGEX.fullmatch(str(key)) else ""
        if not accession:
            continue
        seq = value.get("sequence", "") or fetch_uniprot_sequence(accession, seq_cache)
        if seq:
            drivers.append((accession, seq))
    return drivers


def load_swissprot_pool(seq_cache: dict[str, str], taxon: int, size: int) -> list[tuple[str, str]]:
    safe_size = min(size, 500)
    payload = fetch_json(
        UNIPROT_SWISSPROT_QUERY_URL,
        params={
            "query": f"reviewed:true AND organism_id:{taxon}",
            "format": "json",
            "fields": "accession,sequence",
            "size": safe_size,
        },
    )
    pool: list[tuple[str, str]] = []
    for item in payload.get("results", []):
        accession = item.get("primaryAccession", "")
        sequence = item.get("sequence", {}).get("value", "")
        if accession and sequence:
            seq_cache[accession] = sequence
            pool.append((accession, sequence))
    return pool


def fetch_string_interactors(accession: str, taxon: int, required_score: int = 150) -> set[str]:
    try:
        payload = fetch_json(
            STRING_INTERACTION_URL,
            params={"identifiers": accession, "species": taxon, "required_score": required_score},
        )
    except requests.HTTPError:
        return set()
    interactors: set[str] = set()
    if not isinstance(payload, list):
        return interactors
    for item in payload:
        for key in ("preferredName_A", "preferredName_B", "stringId_A", "stringId_B"):
            val = item.get(key, "")
            interactors.update(extract_uniprot_ids(val))
    return interactors


def build_phasepro_cohort_positives(
    drivers: list[tuple[str, str]], max_pairs: int, seed: int
) -> list[ProteinPair]:
    rng = random.Random(seed)
    if len(drivers) < 2:
        return []
    shuffled = drivers[:]
    rng.shuffle(shuffled)
    positives: list[ProteinPair] = []
    for i in range(0, len(shuffled) - 1, 2):
        id_a, seq_a = shuffled[i]
        id_b, seq_b = shuffled[i + 1]
        positives.append(
            ProteinPair(
                id_a=id_a,
                id_b=id_b,
                seq_a=seq_a,
                seq_b=seq_b,
                label=1,
                source="phasepro_cohort_proxy_positive",
            )
        )
        if len(positives) >= max_pairs:
            break
    return positives


def build_proxy_positives_from_string(
    drivers: list[tuple[str, str]],
    seq_cache: dict[str, str],
    taxon: int,
    max_pairs: int,
    required_score: int = 700,
) -> list[ProteinPair]:
    positives: list[ProteinPair] = []
    seen: set[tuple[str, str]] = set()
    for id_a, seq_a in drivers:
        partner_ids = fetch_string_interactors(
            accession=id_a, taxon=taxon, required_score=required_score
        )
        for id_b in partner_ids:
            if id_b == id_a:
                continue
            seq_b = fetch_uniprot_sequence(id_b, seq_cache)
            if not seq_b:
                continue
            pair_key = (id_a, id_b) if id_a < id_b else (id_b, id_a)
            if pair_key in seen:
                continue
            seen.add(pair_key)
            positives.append(
                ProteinPair(
                    id_a=id_a,
                    id_b=id_b,
                    seq_a=seq_a,
                    seq_b=seq_b,
                    label=1,
                    source="phasepro_string_proxy_positive",
                )
            )
            if len(positives) >= max_pairs:
                return positives
    return positives


def build_negative_pairs(
    positives: list[ProteinPair],
    swiss_pool: list[tuple[str, str]],
    taxon: int,
    seed: int,
) -> list[ProteinPair]:
    rng = random.Random(seed)
    all_positive_partners: dict[str, set[str]] = {}
    for p in positives:
        all_positive_partners.setdefault(p.id_a, set()).add(p.id_b)

    random_negatives: list[ProteinPair] = []
    hard_negatives: list[ProteinPair] = []

    for pos in positives:
        candidates = [(pid, seq) for pid, seq in swiss_pool if pid not in {pos.id_a, pos.id_b}]
        if not candidates:
            continue

        # Random negatives
        rand_id, rand_seq = rng.choice(candidates)
        random_negatives.append(
            ProteinPair(
                id_a=pos.id_a,
                id_b=rand_id,
                seq_a=pos.seq_a,
                seq_b=rand_seq,
                label=0,
                source="swissprot_random_negative",
            )
        )

        # Hard negatives: exclude strong STRING interactions and known positive partners
        strong_interactors = fetch_string_interactors(pos.id_a, taxon=taxon, required_score=150)
        blocked = strong_interactors | all_positive_partners.get(pos.id_a, set()) | {pos.id_a}
        hard_candidates = [(pid, seq) for pid, seq in candidates if pid not in blocked]
        if not hard_candidates:
            continue
        hard_id, hard_seq = rng.choice(hard_candidates)
        hard_negatives.append(
            ProteinPair(
                id_a=pos.id_a,
                id_b=hard_id,
                seq_a=pos.seq_a,
                seq_b=hard_seq,
                label=0,
                source="string_hard_negative",
            )
        )

    return random_negatives + hard_negatives


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate LLPS protein pairs from open databases.")
    parser.add_argument("--output", type=str, default="data/processed/protein_pairs.csv")
    parser.add_argument("--max-positives", type=int, default=200)
    parser.add_argument("--taxon", type=int, default=9606, help="NCBI taxon id, default human.")
    parser.add_argument(
        "--swissprot-pool-size",
        type=int,
        default=500,
        help="Swiss-Prot random pool size for negative sampling.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seq_cache: dict[str, str] = {}

    positives = load_phasepro_positives(seq_cache=seq_cache, max_pairs=args.max_positives)
    drivers = load_phasepro_drivers(seq_cache=seq_cache)
    if not positives:
        positives = build_proxy_positives_from_string(
            drivers=drivers,
            seq_cache=seq_cache,
            taxon=args.taxon,
            max_pairs=args.max_positives,
            required_score=700,
        )
    if not positives:
        positives = build_phasepro_cohort_positives(
            drivers=drivers, max_pairs=args.max_positives, seed=args.seed
        )
    if not positives:
        raise RuntimeError(
            "No positive pairs could be built from PhasePro/STRING. "
            "Please check network, species(taxon), or API limits."
        )

    swiss_pool = load_swissprot_pool(
        seq_cache=seq_cache, taxon=args.taxon, size=args.swissprot_pool_size
    )
    if not swiss_pool:
        raise RuntimeError("Swiss-Prot pool is empty. Please check UniProt API availability.")

    negatives = build_negative_pairs(
        positives=positives,
        swiss_pool=swiss_pool,
        taxon=args.taxon,
        seed=args.seed,
    )

    all_pairs = positives + negatives
    out_path = save_pairs_csv(all_pairs, args.output)
    print(
        "Saved pairs to:"
        f" {out_path} | positives={len(positives)} "
        f"negatives={len(negatives)} total={len(all_pairs)}"
    )


if __name__ == "__main__":
    main()
