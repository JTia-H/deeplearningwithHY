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
    hidden_dim: int
    temperature: float
    num_negatives: int
    loss_type: str
    bce_pos_weight: float
    bce_balance_sampling: bool
    bce_auto_pos_weight: bool
    bce_pos_weight_grid: list[float]
    bce_pos_weight_tune_epochs: int
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
        hidden_dim=int(train_block["hidden_dim"]),
        temperature=float(train_block["temperature"]),
        num_negatives=int(train_block["num_negatives"]),
        loss_type=str(train_block.get("loss_type", "bce")).lower(),
        bce_pos_weight=float(train_block.get("bce_pos_weight", 1.0)),
        bce_balance_sampling=bool(train_block.get("bce_balance_sampling", True)),
        bce_auto_pos_weight=bool(train_block.get("bce_auto_pos_weight", True)),
        bce_pos_weight_grid=[
            float(x) for x in train_block.get("bce_pos_weight_grid", [1.0, 1.5, 2.0, 3.0])
        ],
        bce_pos_weight_tune_epochs=int(train_block.get("bce_pos_weight_tune_epochs", 3)),
        grad_clip_norm=float(train_block.get("grad_clip_norm", 1.0)),
        warmup_epochs=int(train_block.get("warmup_epochs", 3)),
        min_learning_rate=float(train_block.get("min_learning_rate", 1e-5)),
        early_stopping_patience=int(train_block.get("early_stopping_patience", 10)),
        early_stopping_min_delta=float(train_block.get("early_stopping_min_delta", 1e-4)),
        log_csv_path=str(train_block.get("log_csv_path", "models/checkpoints/train_log.csv")),
    )
