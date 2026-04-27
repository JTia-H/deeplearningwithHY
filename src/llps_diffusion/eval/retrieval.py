from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from llps_diffusion.data.tokenization import SequenceTokenizer, TokenizerConfig
from llps_diffusion.models.conditional_diffusion import ConditionalDiffusionModel


def resolve_device(device: str = "auto") -> torch.device:
    normalized = device.lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but not available. Use --device auto or --device cpu."
            )
        return torch.device("cuda")
    if normalized == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device: {device}. Expected one of: auto, cuda, cpu.")


def _dcg_at_k(relevances: list[int], k: int) -> float:
    if k <= 0:
        return 0.0
    gains = np.array(relevances[:k], dtype=np.float64)
    if gains.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, gains.size + 2, dtype=np.float64))
    return float(np.sum(gains / discounts))


def _ndcg_at_k(relevances: list[int], k: int) -> float:
    dcg = _dcg_at_k(relevances, k)
    ideal = sorted(relevances, reverse=True)
    idcg = _dcg_at_k(ideal, k)
    if idcg <= 0:
        return 0.0
    return float(dcg / idcg)


def _recall_at_k(relevances: list[int], k: int) -> float:
    positives = int(np.sum(relevances))
    if positives == 0:
        return 0.0
    return float(np.sum(relevances[:k]) / positives)


def _mrr(relevances: list[int]) -> float:
    for idx, rel in enumerate(relevances, start=1):
        if rel > 0:
            return 1.0 / float(idx)
    return 0.0


def _sequence_match_score(seq1: str, seq2: str) -> float:
    if not seq1 or not seq2:
        return 0.0
    n = min(len(seq1), len(seq2))
    if n <= 0:
        return 0.0
    same = sum(1 for a, b in zip(seq1[:n], seq2[:n], strict=False) if a == b)
    return float(same / n)


def _extract_state_dict(raw: object) -> dict[str, torch.Tensor]:
    if isinstance(raw, dict) and "state_dict" in raw and isinstance(raw["state_dict"], dict):
        return raw["state_dict"]
    if isinstance(raw, dict):
        return raw
    raise RuntimeError("Unsupported checkpoint format.")


def evaluate_retrieval(
    input_csv: str | Path,
    checkpoint: str | Path = "models/checkpoints/conditional_diffusion_best.pt",
    output_json: str | Path = "models/checkpoints/retrieval_metrics.json",
    device: str = "auto",
    k_values: tuple[int, ...] = (1, 5, 10, 20),
    num_samples: int = 8,
    sampling_steps: int = 200,
    max_seq_len: int = 256,
    hidden_dim: int = 64,
    diffusion_steps: int = 1000,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
) -> dict[str, object]:
    df = pd.read_csv(input_csv)
    required = {"id_a", "seq_a", "id_b", "seq_b", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in retrieval csv: {sorted(missing)}")

    resolved_device = resolve_device(device)
    tokenizer = SequenceTokenizer(config=TokenizerConfig(max_length=max_seq_len))
    model = ConditionalDiffusionModel(
        vocab_size=tokenizer.vocab_size,
        embed_dim=hidden_dim,
        hidden_dim=hidden_dim,
        max_seq_len=max_seq_len,
        num_diffusion_steps=diffusion_steps,
        beta_start=beta_start,
        beta_end=beta_end,
        pad_id=tokenizer.pad_id,
    ).to(resolved_device)
    model.sync_schedule_device()
    ckpt = Path(checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    raw = torch.load(ckpt, map_location=resolved_device)
    state = _extract_state_dict(raw)
    try:
        model.load_state_dict(state)
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint architecture mismatch. Please retrain with current token-level diffusion "
            "model via: make train"
        ) from exc
    model.eval()

    scored_parts: list[pd.DataFrame] = []
    with torch.no_grad():
        for _, group in df.groupby("id_a", sort=False):
            seq_a = str(group.iloc[0]["seq_a"])
            cond_tokens = tokenizer.encode_tensor(seq_a, device=resolved_device).unsqueeze(0)
            sample_token_ids = model.sample_target_tokens(
                cond_tokens=cond_tokens, num_samples=num_samples, num_steps=sampling_steps
            )
            generated_sequences = [
                tokenizer.decode(sample_token_ids[i]) for i in range(sample_token_ids.shape[0])
            ]
            generated_sequences = [seq for seq in generated_sequences if seq]
            if not generated_sequences:
                generated_sequences = [""]
            group_scored = group.copy()
            logits: list[float] = []
            for _, row in group.iterrows():
                seq_b = str(row["seq_b"])
                score = max(
                    _sequence_match_score(gen_seq, seq_b) for gen_seq in generated_sequences
                )
                logits.append(float(score))
            group_scored["score_logit"] = logits
            scored_parts.append(group_scored)
    scored = pd.concat(scored_parts, ignore_index=True)

    recall_buckets: dict[int, list[float]] = {k: [] for k in k_values}
    ndcg_buckets: dict[int, list[float]] = {k: [] for k in k_values}
    mrr_values: list[float] = []
    anchor_reports: list[dict[str, object]] = []

    for anchor_id, group in scored.groupby("id_a", sort=False):
        ranked = group.sort_values("score_logit", ascending=False).reset_index(drop=True)
        relevances = [int(v) for v in ranked["label"].tolist()]
        mrr_values.append(_mrr(relevances))
        for k in k_values:
            recall_buckets[k].append(_recall_at_k(relevances, k))
            ndcg_buckets[k].append(_ndcg_at_k(relevances, k))
        top_hit = ranked.iloc[0]
        anchor_reports.append(
            {
                "id_a": str(anchor_id),
                "n_candidates": int(len(ranked)),
                "n_positives": int(np.sum(relevances)),
                "top1_id_b": str(top_hit["id_b"]),
                "top1_label": int(top_hit["label"]),
                "mrr": float(mrr_values[-1]),
            }
        )

    metrics: dict[str, object] = {
        "n_rows": int(len(scored)),
        "n_anchors": int(len(anchor_reports)),
        "device": str(resolved_device),
        "num_samples": int(num_samples),
        "sampling_steps": int(sampling_steps),
        "mrr": float(np.mean(mrr_values)) if mrr_values else 0.0,
    }
    for k in k_values:
        metrics[f"recall@{k}"] = float(np.mean(recall_buckets[k])) if recall_buckets[k] else 0.0
        metrics[f"ndcg@{k}"] = float(np.mean(ndcg_buckets[k])) if ndcg_buckets[k] else 0.0

    report: dict[str, object] = {"metrics": metrics, "per_anchor": anchor_reports}
    out = Path(output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved retrieval metrics to: {out}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate candidate retrieval quality for p(B|A, y=1)."
    )
    parser.add_argument("--input-csv", type=str, required=True)
    parser.add_argument(
        "--checkpoint", type=str, default="models/checkpoints/conditional_diffusion_best.pt"
    )
    parser.add_argument(
        "--output-json", type=str, default="models/checkpoints/retrieval_metrics.json"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Execution device. auto prefers GPU and falls back to CPU.",
    )
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--sampling-steps", type=int, default=200)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--diffusion-steps", type=int, default=1000)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=2e-2)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_retrieval(
        input_csv=args.input_csv,
        checkpoint=args.checkpoint,
        output_json=args.output_json,
        device=args.device,
        num_samples=args.num_samples,
        sampling_steps=args.sampling_steps,
        max_seq_len=args.max_seq_len,
        hidden_dim=args.hidden_dim,
        diffusion_steps=args.diffusion_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
    )
