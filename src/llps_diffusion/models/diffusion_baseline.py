from __future__ import annotations

import torch
from torch import nn


class DiffusionClassifierBaseline(nn.Module):  # type: ignore[misc]
    """
    Lightweight baseline model.
    - Input: amino-acid frequency + sequence length features
    - Output: LLPS probability logit
    This is a placeholder and should be replaced with a true diffusion model.
    """

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return logits.squeeze(-1)
