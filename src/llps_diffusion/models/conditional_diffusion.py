from __future__ import annotations

import math
from typing import cast

import torch
from torch import nn

from llps_diffusion.models.noise_schedule import NoiseSchedule


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, emb_dim: int) -> None:
        super().__init__()
        if emb_dim <= 0:
            raise ValueError("emb_dim must be positive.")
        self.emb_dim = emb_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.emb_dim // 2
        scale = math.log(10000.0) / max(half_dim - 1, 1)
        exponents = torch.exp(torch.arange(half_dim, device=timesteps.device) * -scale)
        args = timesteps.float().unsqueeze(1) * exponents.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.emb_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros((emb.shape[0], 1), device=timesteps.device)], dim=1)
        return emb


class ConditionalDiffusionModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        max_seq_len: int,
        num_diffusion_steps: int,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.num_diffusion_steps = num_diffusion_steps
        self.pad_id = pad_id
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.condition_encoder = nn.GRU(
            input_size=embed_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True
        )
        self.time_embedding = SinusoidalTimeEmbedding(emb_dim=hidden_dim)
        self.cond_proj = nn.Linear(hidden_dim * 2, embed_dim)
        self.time_proj = nn.Linear(hidden_dim, embed_dim)
        self.noise_predictor = nn.Sequential(
            nn.Linear(embed_dim * 3, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.token_decoder = nn.Linear(embed_dim, vocab_size)
        self.schedule = NoiseSchedule(
            num_steps=num_diffusion_steps, beta_start=beta_start, beta_end=beta_end
        )

    def sync_schedule_device(self) -> None:
        device = next(self.parameters()).device
        self.schedule.to(device)

    def encode_condition(self, cond_tokens: torch.Tensor) -> torch.Tensor:
        cond_emb = self.token_embedding(cond_tokens)
        _, h = self.condition_encoder(cond_emb)
        # bidirectional last state concat: [2, B, H] -> [B, 2H]
        return torch.cat([h[0], h[1]], dim=-1)

    def encode_target(self, target_tokens: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.token_embedding(target_tokens))

    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        eps = noise if noise is not None else torch.randn_like(x0)
        sqrt_ab = self.schedule.gather(self.schedule.sqrt_alpha_bars, t).unsqueeze(1)
        sqrt_omab = self.schedule.gather(self.schedule.sqrt_one_minus_alpha_bars, t).unsqueeze(1)
        xt = sqrt_ab * x0 + sqrt_omab * eps
        return xt, eps

    def _expand_condition_time(
        self, cond_repr: torch.Tensor, t: torch.Tensor, seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cond_vec = self.cond_proj(cond_repr).unsqueeze(1).expand(-1, seq_len, -1)
        t_vec = self.time_proj(self.time_embedding(t)).unsqueeze(1).expand(-1, seq_len, -1)
        return cond_vec, t_vec

    def predict_noise(
        self, cond_repr: torch.Tensor, xt: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        cond_vec, t_vec = self._expand_condition_time(cond_repr, t, xt.shape[1])
        inp = torch.cat([xt, cond_vec, t_vec], dim=-1)
        return cast(torch.Tensor, self.noise_predictor(inp))

    def reconstruct_x0(
        self, xt: torch.Tensor, eps_pred: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        sqrt_ab = self.schedule.gather(self.schedule.sqrt_alpha_bars, t).unsqueeze(1)
        sqrt_omab = self.schedule.gather(self.schedule.sqrt_one_minus_alpha_bars, t).unsqueeze(1)
        return (xt - sqrt_omab * eps_pred) / (sqrt_ab + 1e-8)

    def decode_token_logits(self, x_repr: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.token_decoder(x_repr))

    def decode_tokens(self, x_repr: torch.Tensor) -> torch.Tensor:
        logits = self.decode_token_logits(x_repr)
        return torch.argmax(logits, dim=-1)

    def diffusion_loss(
        self, cond_tokens: torch.Tensor, target_tokens: torch.Tensor
    ) -> torch.Tensor:
        cond_repr = self.encode_condition(cond_tokens)
        x0 = self.encode_target(target_tokens)
        batch = x0.shape[0]
        t = torch.randint(0, self.num_diffusion_steps, (batch,), device=x0.device)
        xt, eps = self.q_sample(x0=x0, t=t)
        eps_pred = self.predict_noise(cond_repr=cond_repr, xt=xt, t=t)
        mask = (target_tokens != self.pad_id).float().unsqueeze(-1)
        sq = (eps - eps_pred) ** 2
        return torch.sum(sq * mask) / torch.clamp(mask.sum() * sq.shape[-1], min=1.0)

    @torch.no_grad()
    def sample_target_repr(
        self, cond_tokens: torch.Tensor, num_samples: int = 1, num_steps: int | None = None
    ) -> torch.Tensor:
        cond_repr = self.encode_condition(cond_tokens)
        seq_len = cond_tokens.shape[1]
        if num_samples > 1:
            cond_repr = cond_repr.repeat_interleave(num_samples, dim=0)
            cond_tokens = cond_tokens.repeat_interleave(num_samples, dim=0)
        steps = num_steps if num_steps is not None else self.num_diffusion_steps
        x = torch.randn(
            cond_repr.shape[0],
            seq_len,
            self.embed_dim,
            device=cond_repr.device,
            dtype=cond_repr.dtype,
        )
        for step in reversed(range(steps)):
            t = torch.full((x.shape[0],), step, device=x.device, dtype=torch.long)
            beta_t = self.schedule.gather(self.schedule.betas, t).unsqueeze(1)
            alpha_t = self.schedule.gather(self.schedule.alphas, t).unsqueeze(1)
            alpha_bar_t = self.schedule.gather(self.schedule.alpha_bars, t).unsqueeze(1)
            eps_pred = self.predict_noise(cond_repr=cond_repr, xt=x, t=t)
            x = (1.0 / torch.sqrt(alpha_t)) * (
                x - ((1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t)) * eps_pred
            )
            if step > 0:
                z = torch.randn_like(x)
                x = x + torch.sqrt(beta_t) * z
        return x

    @torch.no_grad()
    def sample_target_tokens(
        self,
        cond_tokens: torch.Tensor,
        num_samples: int = 1,
        num_steps: int | None = None,
    ) -> torch.Tensor:
        sampled_repr = self.sample_target_repr(
            cond_tokens=cond_tokens, num_samples=num_samples, num_steps=num_steps
        )
        return self.decode_tokens(sampled_repr)
