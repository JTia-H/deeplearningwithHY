from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TrainConfig:
    seed: int
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    max_seq_len: int
    diffusion_steps: int
    beta_start: float
    beta_end: float
    sampling_steps: int
    hidden_dim: int
    grad_clip_norm: float
    warmup_epochs: int
    min_learning_rate: float
    early_stopping_patience: int
    early_stopping_min_delta: float
    log_csv_path: str


def load_config(config_path: str | Path) -> TrainConfig:
    path = Path(config_path)
    payload: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    train_block = payload["train"]
    return TrainConfig(
        seed=int(train_block["seed"]),
        batch_size=int(train_block["batch_size"]),
        epochs=int(train_block["epochs"]),
        learning_rate=float(train_block["learning_rate"]),
        weight_decay=float(train_block.get("weight_decay", 1e-2)),
        max_seq_len=int(train_block.get("max_seq_len", 256)),
        diffusion_steps=int(train_block.get("diffusion_steps", 1000)),
        beta_start=float(train_block.get("beta_start", 1e-4)),
        beta_end=float(train_block.get("beta_end", 2e-2)),
        sampling_steps=int(train_block.get("sampling_steps", 200)),
        hidden_dim=int(train_block["hidden_dim"]),
        grad_clip_norm=float(train_block.get("grad_clip_norm", 1.0)),
        warmup_epochs=int(train_block.get("warmup_epochs", 3)),
        min_learning_rate=float(train_block.get("min_learning_rate", 1e-5)),
        early_stopping_patience=int(train_block.get("early_stopping_patience", 10)),
        early_stopping_min_delta=float(train_block.get("early_stopping_min_delta", 1e-4)),
        log_csv_path=str(train_block.get("log_csv_path", "models/checkpoints/train_log.csv")),
    )
