from __future__ import annotations

import argparse
from pathlib import Path

from llps_diffusion.eval.retrieval import evaluate_retrieval


def run_generative_retrieval_eval(
    input_csv: str | Path,
    checkpoint: str | Path = "models/checkpoints/conditional_diffusion_best.pt",
    output_json: str | Path = "models/checkpoints/retrieval_metrics.json",
    device: str = "auto",
    num_samples: int = 8,
    sampling_steps: int = 200,
) -> dict[str, object]:
    return evaluate_retrieval(
        input_csv=input_csv,
        checkpoint=checkpoint,
        output_json=output_json,
        device=device,
        num_samples=num_samples,
        sampling_steps=sampling_steps,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run generative retrieval evaluation pipeline.")
    parser.add_argument("--input-csv", type=str, required=True)
    parser.add_argument(
        "--checkpoint", type=str, default="models/checkpoints/conditional_diffusion_best.pt"
    )
    parser.add_argument(
        "--output-json", type=str, default="models/checkpoints/retrieval_metrics.json"
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--sampling-steps", type=int, default=200)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_generative_retrieval_eval(
        input_csv=args.input_csv,
        checkpoint=args.checkpoint,
        output_json=args.output_json,
        device=args.device,
        num_samples=args.num_samples,
        sampling_steps=args.sampling_steps,
    )
