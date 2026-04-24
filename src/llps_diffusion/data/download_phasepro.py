from __future__ import annotations

import json
from pathlib import Path

import requests  # type: ignore[import-untyped]

PHASEPRO_URL = "https://phasepro.elte.hu/download_full.json"


def download_phasepro(output_path: str | Path = "data/raw/phasepro_full.json") -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(PHASEPRO_URL, timeout=60)
    response.raise_for_status()
    data = response.json()
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    path = download_phasepro()
    print(f"Saved PhasePro data to: {path}")
