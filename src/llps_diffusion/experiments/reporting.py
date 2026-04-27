from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import yaml


def _load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _last_row_from_csv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    last: dict[str, str] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            last = row
    return last


def generate_experiment_report(
    *,
    config_path: str,
    device: str,
    output_dir: str | Path = "docs/experiments",
    checkpoints_dir: str | Path = "models/checkpoints",
    exp_id: str | None = None,
) -> Path:
    now = datetime.now()
    resolved_exp_id = exp_id or now.strftime("exp_%Y%m%d_%H%M%S")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(checkpoints_dir)

    train_log_path = ckpt_dir / "train_log.csv"
    curves_path = ckpt_dir / "train_curves.png"
    best_ckpt_path = ckpt_dir / "pair_cross_attention_best.pt"
    last_ckpt_path = ckpt_dir / "pair_cross_attention.pt"
    test_metrics_path = ckpt_dir / "test_metrics.json"
    retrieval_metrics_path = ckpt_dir / "retrieval_metrics.json"
    eval_compare_path = ckpt_dir / "final_eval_comparison.json"
    retrieval_compare_path = ckpt_dir / "retrieval_comparison.json"
    best_ckpt_path = ckpt_dir / "conditional_diffusion_best.pt"
    last_ckpt_path = ckpt_dir / "conditional_diffusion.pt"

    cfg_payload: dict[str, Any] = {}
    cfg_path = Path(config_path)
    if cfg_path.exists():
        cfg_payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    train_block = cfg_payload.get("train", {})

    last_train_row = _last_row_from_csv(train_log_path)
    test_metrics = _load_json_if_exists(test_metrics_path)
    retrieval_metrics = _load_json_if_exists(retrieval_metrics_path).get("metrics", {})

    lines = [
        f"# Experiment Report: {resolved_exp_id}",
        "",
        "## Basic Info",
        f"- date: {now.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- exp_id: {resolved_exp_id}",
        f"- config_path: {config_path}",
        f"- device: {device}",
        "",
        "## Diffusion Config",
        f"- max_seq_len: {train_block.get('max_seq_len', '')}",
        f"- hidden_dim: {train_block.get('hidden_dim', '')}",
        f"- diffusion_steps: {train_block.get('diffusion_steps', '')}",
        f"- beta_start: {train_block.get('beta_start', '')}",
        f"- beta_end: {train_block.get('beta_end', '')}",
        f"- sampling_steps: {train_block.get('sampling_steps', '')}",
        "",
        "## Artifacts",
        f"- best_checkpoint: {best_ckpt_path}",
        f"- last_checkpoint: {last_ckpt_path}",
        f"- train_log_csv: {train_log_path}",
        f"- train_curves_png: {curves_path}",
        f"- test_metrics_json: {test_metrics_path}",
        f"- retrieval_metrics_json: {retrieval_metrics_path}",
        f"- eval_comparison_json: {eval_compare_path}",
        f"- retrieval_comparison_json: {retrieval_compare_path}",
        "",
        "## Last Training Log Row",
        f"- epoch: {last_train_row.get('epoch', '')}",
        f"- train_loss: {last_train_row.get('train_loss', '')}",
        f"- val_loss: {last_train_row.get('val_loss', '')}",
        f"- val_acc: {last_train_row.get('val_acc', '')}",
        f"- val_auc: {last_train_row.get('val_auc', '')}",
        f"- lr: {last_train_row.get('lr', '')}",
        f"- grad_norm_mean: {last_train_row.get('grad_norm_mean', '')}",
        f"- grad_norm_max: {last_train_row.get('grad_norm_max', '')}",
        f"- skipped: {last_train_row.get('skipped', '')}",
        "",
        "## Classification Metrics",
        f"- accuracy: {test_metrics.get('accuracy', '')}",
        f"- f1: {test_metrics.get('f1', '')}",
        f"- auc: {test_metrics.get('auc', '')}",
        f"- pr_auc: {test_metrics.get('pr_auc', '')}",
        "",
        "## Retrieval Metrics",
        f"- mrr: {retrieval_metrics.get('mrr', '')}",
        f"- recall@1: {retrieval_metrics.get('recall@1', '')}",
        f"- recall@5: {retrieval_metrics.get('recall@5', '')}",
        f"- recall@10: {retrieval_metrics.get('recall@10', '')}",
        f"- recall@20: {retrieval_metrics.get('recall@20', '')}",
        f"- ndcg@1: {retrieval_metrics.get('ndcg@1', '')}",
        f"- ndcg@5: {retrieval_metrics.get('ndcg@5', '')}",
        f"- ndcg@10: {retrieval_metrics.get('ndcg@10', '')}",
        f"- ndcg@20: {retrieval_metrics.get('ndcg@20', '')}",
        "",
        "## Notes",
        "- This report is auto-generated after training.",
        "- If metrics fields are empty, run evaluation commands first.",
        "",
    ]

    out_path = out_dir / f"{resolved_exp_id}.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path
