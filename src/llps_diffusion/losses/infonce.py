from __future__ import annotations

import torch


def infonce_loss(
    pos_score: torch.Tensor, neg_scores: torch.Tensor, temperature: float
) -> torch.Tensor:
    """
    InfoNCE form used in the technical document:
    - numerator: exp(pos / tau)
    - denominator: positive term + K negative terms
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    pos = pos_score.reshape(1) / temperature
    neg = neg_scores.reshape(-1) / temperature
    logits = torch.cat([pos, neg], dim=0)
    # Stable InfoNCE: -log( exp(pos) / sum(exp(all)) ) = -pos + logsumexp(all)
    loss = -pos[0] + torch.logsumexp(logits, dim=0)
    return loss
