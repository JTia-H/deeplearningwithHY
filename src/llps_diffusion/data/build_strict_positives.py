from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import requests  # type: ignore[import-untyped]

from llps_diffusion.data.pairs import ProteinPair, save_pairs_csv

PHASEPRO_FULL_URL = "https://phasepro.elte.hu/download_full.json"
UNIPROT_ENTRY_URL = "https://rest.uniprot.org/uniprotkb/{accession}.json"
UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
STRING_INTERACTION_URL = "https://string-db.org/api/json/interaction_partners"
GENE_SYMBOL_SAFE = re.compile(r"^[A-Za-z0-9_.-]+$")


def fetch_json(url: str, params: dict[str, Any] | None = None) -> Any:
    response = requests.get(url, params=params, timeout=90)
    response.raise_for_status()
    return response.json()


def fetch_uniprot_entry(accession: str, cache: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if accession in cache:
        return cache[accession]
    payload = fetch_json(UNIPROT_ENTRY_URL.format(accession=accession))
    if not isinstance(payload, dict):
        return {}
    cache[accession] = payload
    return payload


def get_taxon_id(entry: dict[str, Any]) -> int | None:
    try:
        return int(entry.get("organism", {}).get("taxonId"))
    except (TypeError, ValueError):
        return None


def get_sequence(entry: dict[str, Any]) -> str:
    return str(entry.get("sequence", {}).get("value", ""))


def map_gene_to_uniprot_accession(
    gene: str,
    taxon_id: int,
    map_cache: dict[tuple[str, int], tuple[str, str] | None],
) -> tuple[str, str] | None:
    if not GENE_SYMBOL_SAFE.fullmatch(gene):
        map_cache[(gene, taxon_id)] = None
        return None
    key = (gene, taxon_id)
    if key in map_cache:
        return map_cache[key]
    try:
        payload = fetch_json(
            UNIPROT_SEARCH_URL,
            params={
                "query": f"reviewed:true AND gene_exact:{gene} AND organism_id:{taxon_id}",
                "format": "json",
                "fields": "accession,sequence",
                "size": 1,
            },
        )
    except requests.HTTPError:
        map_cache[key] = None
        return None
    results = payload.get("results", []) if isinstance(payload, dict) else []
    if not results:
        map_cache[key] = None
        return None
    accession = str(results[0].get("primaryAccession", ""))
    sequence = str(results[0].get("sequence", {}).get("value", ""))
    if not accession or not sequence:
        map_cache[key] = None
        return None
    map_cache[key] = (accession, sequence)
    return map_cache[key]


def build_strict_positive_candidates(
    required_score: int,
    max_pairs: int,
    max_partners_per_anchor: int,
) -> list[ProteinPair]:
    phasepro = fetch_json(PHASEPRO_FULL_URL)
    if not isinstance(phasepro, dict):
        return []

    entry_cache: dict[str, dict[str, Any]] = {}
    gene_map_cache: dict[tuple[str, int], tuple[str, str] | None] = {}
    positives: list[ProteinPair] = []
    seen_pairs: set[tuple[str, str]] = set()

    for accession in phasepro:
        if len(positives) >= max_pairs:
            break
        entry_a = fetch_uniprot_entry(accession, entry_cache)
        if not entry_a:
            continue
        seq_a = get_sequence(entry_a)
        taxon_a = get_taxon_id(entry_a)
        if not seq_a or taxon_a is None:
            continue

        try:
            interactions = fetch_json(
                STRING_INTERACTION_URL,
                params={"identifiers": accession, "required_score": required_score},
            )
        except requests.HTTPError:
            continue
        if not isinstance(interactions, list):
            continue

        added_for_anchor = 0
        for item in interactions:
            if len(positives) >= max_pairs:
                break
            if added_for_anchor >= max_partners_per_anchor:
                break
            taxon = item.get("ncbiTaxonId")
            if taxon is None or int(taxon) != taxon_a:
                continue

            # We queried with accession as A, so partner symbol is usually preferredName_B.
            partner_gene = str(item.get("preferredName_B", "")).strip()
            if not partner_gene:
                continue
            mapped = map_gene_to_uniprot_accession(partner_gene, taxon_a, gene_map_cache)
            if mapped is None:
                continue
            accession_b, seq_b = mapped
            if accession_b == accession:
                continue
            key = tuple(sorted((accession, accession_b)))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            positives.append(
                ProteinPair(
                    id_a=accession,
                    id_b=accession_b,
                    seq_a=seq_a,
                    seq_b=seq_b,
                    label=1,
                    source="phasepro_string_strict_candidate",
                )
            )
            added_for_anchor += 1
    return positives


def write_report(
    report_path: str | Path,
    strict_count: int,
    required_score: int,
    max_pairs: int,
    max_partners_per_anchor: int,
) -> Path:
    out = Path(report_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "Strict Positive Builder Report",
        f"required_score: {required_score}",
        f"max_pairs: {max_pairs}",
        f"max_partners_per_anchor: {max_partners_per_anchor}",
        f"strict_candidate_count: {strict_count}",
        "source_label: phasepro_string_strict_candidate",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build strict positive candidates from PhasePro + STRING + UniProt mapping."
    )
    parser.add_argument(
        "--output", type=str, default="data/processed/strict_positive_candidates.csv"
    )
    parser.add_argument("--report", type=str, default="data/processed/strict_builder_report.txt")
    parser.add_argument("--required-score", type=int, default=900)
    parser.add_argument("--max-pairs", type=int, default=300)
    parser.add_argument("--max-partners-per-anchor", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    positives = build_strict_positive_candidates(
        required_score=args.required_score,
        max_pairs=args.max_pairs,
        max_partners_per_anchor=args.max_partners_per_anchor,
    )
    out_csv = save_pairs_csv(positives, args.output)
    out_report = write_report(
        report_path=args.report,
        strict_count=len(positives),
        required_score=args.required_score,
        max_pairs=args.max_pairs,
        max_partners_per_anchor=args.max_partners_per_anchor,
    )
    print(f"Saved strict candidates: {out_csv} count={len(positives)}")
    print(f"Saved strict builder report: {out_report}")


if __name__ == "__main__":
    main()
