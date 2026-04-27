from __future__ import annotations

from dataclasses import dataclass

import torch

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


@dataclass(frozen=True)
class TokenizerConfig:
    max_length: int = 256


class SequenceTokenizer:
    def __init__(self, config: TokenizerConfig | None = None) -> None:
        self.config = config or TokenizerConfig()
        vocab_tokens = [PAD_TOKEN, UNK_TOKEN] + list(AMINO_ACIDS)
        self.stoi = {tok: idx for idx, tok in enumerate(vocab_tokens)}
        self.itos = {idx: tok for tok, idx in self.stoi.items()}
        self.pad_id = self.stoi[PAD_TOKEN]
        self.unk_id = self.stoi[UNK_TOKEN]

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, sequence: str) -> list[int]:
        clean = sequence.strip().upper()
        ids = [self.stoi.get(ch, self.unk_id) for ch in clean[: self.config.max_length]]
        if len(ids) < self.config.max_length:
            ids += [self.pad_id] * (self.config.max_length - len(ids))
        return ids

    def encode_tensor(self, sequence: str, device: torch.device | None = None) -> torch.Tensor:
        return torch.tensor(self.encode(sequence), dtype=torch.long, device=device)

    def decode(self, token_ids: list[int] | torch.Tensor) -> str:
        if isinstance(token_ids, torch.Tensor):
            ids = token_ids.detach().cpu().tolist()
        else:
            ids = list(token_ids)
        out: list[str] = []
        for tid in ids:
            tok = self.itos.get(int(tid), UNK_TOKEN)
            if tok == PAD_TOKEN:
                continue
            if tok == UNK_TOKEN:
                continue
            out.append(tok)
        return "".join(out)
