from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from llps_diffusion.features.priors import estimate_idr_ratio, estimate_prld_score
from llps_diffusion.models.cross_attention import PairCrossAttentionScorer
from llps_diffusion.scoring.pspi import compute_cfg_gap, compute_crl


def predict_pair(
    seq_a: str, seq_b: str, checkpoint: str = "models/checkpoints/pair_cross_attention.pt"
) -> dict[str, float]:
    model = PairCrossAttentionScorer(input_dim=21, hidden_dim=64)
    ckpt = Path(checkpoint)
    if ckpt.exists():
        state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state)

    model.eval()
    with torch.no_grad():
        score_ab = model.score(seq_a, seq_b).item()
        score_ba = model.score(seq_b, seq_a).item()
        crl = compute_crl(score_ab, score_ba)

        # Placeholder CFG-Gap: unconditional logits approximated by zeros.
        cond_logits = np.array([score_ab, score_ba], dtype=np.float32)
        uncond_logits = np.zeros_like(cond_logits)
        cfg_gap = compute_cfg_gap(cond_logits=cond_logits, uncond_logits=uncond_logits)

        idr_mean = (estimate_idr_ratio(seq_a) + estimate_idr_ratio(seq_b)) / 2.0
        prld_mean = (estimate_prld_score(seq_a) + estimate_prld_score(seq_b)) / 2.0

        # Placeholder PSPI when no trained fusion model is available.
        proxy_logit = crl + cfg_gap + idr_mean + prld_mean
        pspi = torch.sigmoid(torch.tensor(proxy_logit)).item()
    return {
        "crl": float(crl),
        "cfg_gap": float(cfg_gap),
        "idr_mean": float(idr_mean),
        "prld_mean": float(prld_mean),
        "pspi": float(pspi),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict LLPS score for a protein pair.")
    parser.add_argument("--seq-a", type=str, required=True, help="Protein sequence A.")
    parser.add_argument("--seq-b", type=str, required=True, help="Protein sequence B.")
    parser.add_argument(
        "--checkpoint", type=str, default="models/checkpoints/pair_cross_attention.pt"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out = predict_pair(seq_a=args.seq_a, seq_b=args.seq_b, checkpoint=args.checkpoint)
    print(f"CRL={out['crl']:.4f} CFG-Gap={out['cfg_gap']:.4f} PSPI={out['pspi']:.4f}")
