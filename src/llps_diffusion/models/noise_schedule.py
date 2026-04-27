from __future__ import annotations

import torch


class NoiseSchedule:
    def __init__(
        self,
        num_steps: int,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        device: torch.device | None = None,
    ) -> None:
        if num_steps <= 0:
            raise ValueError("num_steps must be positive.")
        if beta_start <= 0 or beta_end <= 0:
            raise ValueError("beta_start and beta_end must be positive.")
        if beta_start >= beta_end:
            raise ValueError("beta_start must be smaller than beta_end.")
        self.num_steps = int(num_steps)
        self.device = device or torch.device("cpu")
        self.betas = torch.linspace(
            beta_start, beta_end, self.num_steps, dtype=torch.float32, device=self.device
        )
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)

    def to(self, device: torch.device) -> NoiseSchedule:
        self.device = device
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        self.sqrt_alpha_bars = self.sqrt_alpha_bars.to(device)
        self.sqrt_one_minus_alpha_bars = self.sqrt_one_minus_alpha_bars.to(device)
        return self

    def gather(self, values: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        # values: [T], timesteps: [B] -> [B, 1]
        gathered = values.index_select(0, timesteps.long())
        return gathered.unsqueeze(-1)
