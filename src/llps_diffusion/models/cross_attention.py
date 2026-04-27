from __future__ import annotations

import torch
from torch import nn

from llps_diffusion.data.datasets import sequence_to_features


class PairCrossAttentionScorer(nn.Module):
    """
    Placeholder two-tower + cross-attention scorer.
    This module aligns with the PDF architecture and can be replaced later
    by EvoDiff/DPLM + ESM2-based integration.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.encoder_a = nn.Linear(input_dim, hidden_dim)
        self.encoder_b = nn.Linear(input_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )
        self.proj = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))

    def encode_seq(self, seq: str) -> torch.Tensor:
        device = next(self.parameters()).device
        x = torch.tensor(sequence_to_features(seq), dtype=torch.float32, device=device).unsqueeze(0)
        return x

    def score(self, seq_a: str, seq_b: str) -> torch.Tensor:
        feat_a = self.encode_seq(seq_a)
        feat_b = self.encode_seq(seq_b)
        q = self.encoder_a(feat_a).unsqueeze(1)
        kv = self.encoder_b(feat_b).unsqueeze(1)
        fused, _ = self.cross_attn(q, kv, kv)
        logits = self.proj(fused.squeeze(1))
        return torch.as_tensor(logits.squeeze(-1))
