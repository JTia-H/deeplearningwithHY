from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import cast

import numpy as np
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


def _extract_state_dict(raw: object) -> dict[str, torch.Tensor]:
    if isinstance(raw, dict) and "state_dict" in raw and isinstance(raw["state_dict"], dict):
        return raw["state_dict"]
    if isinstance(raw, dict):
        return raw  # legacy plain state_dict
    raise RuntimeError("Unsupported checkpoint format.")


def _build_model(
    checkpoint: str | Path,
    resolved_device: torch.device,
    max_seq_len: int,
    hidden_dim: int,
    diffusion_steps: int,
    beta_start: float,
    beta_end: float,
) -> tuple[ConditionalDiffusionModel, SequenceTokenizer]:
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
            "Checkpoint architecture mismatch. This checkpoint was trained with an older model "
            "definition and is incompatible with current token-level diffusion model. "
            "Please retrain and regenerate checkpoints via: make train"
        ) from exc
    model.eval()
    return model, tokenizer


def _sequence_match_score(seq1: str, seq2: str) -> float:
    if not seq1 or not seq2:
        return 0.0
    n = min(len(seq1), len(seq2))
    if n <= 0:
        return 0.0
    same = sum(1 for a, b in zip(seq1[:n], seq2[:n], strict=False) if a == b)
    return float(same / n)


def predict_b_distribution(
    seq_a: str,
    candidates_csv: str | Path,
    checkpoint: str | Path = "models/checkpoints/conditional_diffusion_best.pt",
    device: str = "auto",
    temperature: float = 0.2,
    top_k: int = 20,
    num_samples: int = 8,
    sampling_steps: int = 200,
    max_seq_len: int = 256,
    hidden_dim: int = 64,
    diffusion_steps: int = 1000,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    output_json: str | Path = "models/checkpoints/b_distribution.json",
) -> dict[str, object]:
    if temperature <= 0:
        raise ValueError("temperature must be > 0.")
    resolved_device = resolve_device(device)
    model, tokenizer = _build_model(
        checkpoint=checkpoint,
        resolved_device=resolved_device,
        max_seq_len=max_seq_len,
        hidden_dim=hidden_dim,
        diffusion_steps=diffusion_steps,
        beta_start=beta_start,
        beta_end=beta_end,
    )

    rows: list[dict[str, str]] = []
    with Path(candidates_csv).open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "seq_b" not in reader.fieldnames:
            raise ValueError("Candidates CSV must contain at least: seq_b")
        for row in reader:
            seq_b = str(row.get("seq_b", "")).strip()
            if not seq_b:
                continue
            rows.append(row)
    if not rows:
        raise ValueError("No valid candidate rows found in candidates CSV.")

    cond_tokens = tokenizer.encode_tensor(seq_a, device=resolved_device).unsqueeze(0)
    with torch.no_grad():
        generated_token_ids = model.sample_target_tokens(
            cond_tokens=cond_tokens, num_samples=num_samples, num_steps=sampling_steps
        )
    generated_sequences = [
        tokenizer.decode(generated_token_ids[i]) for i in range(generated_token_ids.shape[0])
    ]
    generated_sequences = [seq for seq in generated_sequences if seq]
    if not generated_sequences:
        generated_sequences = [""]

    logits: list[float] = []
    pair_probs: list[float] = []
    for row in rows:
        seq_b = str(row["seq_b"])
        score = max(_sequence_match_score(gen_seq, seq_b) for gen_seq in generated_sequences)
        logits.append(float(score))
        pair_probs.append(float(torch.sigmoid(torch.tensor(score, device=resolved_device)).item()))
    logits_arr = np.array(logits, dtype=np.float64) / float(temperature)
    logits_arr = logits_arr - np.max(logits_arr)
    exp_logits = np.exp(logits_arr)
    cond_probs = exp_logits / np.sum(exp_logits)

    ranked: list[dict[str, object]] = []
    for i, row in enumerate(rows):
        ranked.append(
            {
                "rank": i + 1,
                "id_b": str(row.get("id_b", f"candidate_{i+1}")),
                "seq_b": str(row["seq_b"]),
                "score_logit": float(logits[i]),
                "pair_probability": float(pair_probs[i]),
                "conditional_probability": float(cond_probs[i]),
            }
        )
    ranked.sort(
        key=lambda x: cast(float, x["conditional_probability"]),
        reverse=True,
    )
    for idx, item in enumerate(ranked):
        item["rank"] = idx + 1

    result: dict[str, object] = {
        "n_candidates": len(ranked),
        "temperature": float(temperature),
        "device": str(resolved_device),
        "num_samples": int(num_samples),
        "sampling_steps": int(sampling_steps),
        "generated_sequences": generated_sequences[: max(num_samples, 1)],
        "sum_conditional_probability": float(np.sum(cond_probs)),
        "top_k": ranked[: max(top_k, 1)],
    }
    out = Path(output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict p(B|A,y=1) using conditional diffusion.")
    parser.add_argument("--seq-a", type=str, required=True, help="Protein sequence A.")
    parser.add_argument(
        "--candidates-csv",
        type=str,
        required=True,
        help="Candidate B CSV path (requires seq_b column) for p(B|A,y=1) over candidate set.",
    )
    parser.add_argument(
        "--checkpoint", type=str, default="models/checkpoints/conditional_diffusion_best.pt"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Execution device. auto prefers GPU and falls back to CPU.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Softmax temperature for candidate-set conditional distribution.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="How many top candidates to keep in output when using --candidates-csv.",
    )
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--sampling-steps", type=int, default=200)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--diffusion-steps", type=int, default=1000)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=2e-2)
    parser.add_argument(
        "--output-json",
        type=str,
        default="models/checkpoints/b_distribution.json",
        help="Output JSON path for candidate-set conditional distribution.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out = predict_b_distribution(
        seq_a=args.seq_a,
        candidates_csv=args.candidates_csv,
        checkpoint=args.checkpoint,
        device=args.device,
        temperature=args.temperature,
        top_k=args.top_k,
        num_samples=args.num_samples,
        sampling_steps=args.sampling_steps,
        max_seq_len=args.max_seq_len,
        hidden_dim=args.hidden_dim,
        diffusion_steps=args.diffusion_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        output_json=args.output_json,
    )
    print(
        f"Predicted p(B|A,y=1) over {out['n_candidates']} candidates, "
        f"top_k saved to {args.output_json}"
    )
