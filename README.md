# LLPS Diffusion Predictor

Open-source framework for protein liquid-liquid phase separation (LLPS)
prediction with diffusion-inspired modeling.
The current structure is aligned with the four-stage technical guide.

## Goals

- Build reproducible data, training, and inference pipelines.
- Use strict open-source engineering tooling (lint, type check, tests, hooks, CI).
- Keep modules replaceable for full diffusion model integration.

## Project Structure (Guide-Aligned)

`src/llps_diffusion/`: core code  
`configs/`: configuration files  
`tests/`: tests  
`docs/`: documentation  
`data/`: data directory (raw / interim / processed)

- `data/pairs.py`: pair schema, CSV I/O, and triplet iterator.
- `data/generate_pairs.py`: positive/random-negative/hard-negative generation.
- `data/pairs_qc.py`: dataset quality checks and report output.
- `data/split_pairs.py`: leakage-safe split by anchor protein (`id_a`).
- `features/priors.py`: prior-feature interfaces for IUPred3 / PLAAC.
- `models/conditional_diffusion.py`: conditional diffusion model for `p(B|A,y=1)` approximation.
- `models/noise_schedule.py`: forward/reverse diffusion schedule.
- `losses/infonce.py`: InfoNCE loss.
- `scoring/pspi.py`: CRL, CFG-Gap, and PSPI fusion interfaces.

## Quick Start

1. Install Python 3.11+.
2. Install dependencies:
   - `pip install -e ".[dev]"`
3. Install pre-commit hook:
   - `pre-commit install`
4. Download PhasePro data (example):
   - `python -m llps_diffusion.data.download_phasepro`
5. Generate sample pairs (PhasePro + Swiss-Prot + STRING):
   - `python -m llps_diffusion.data.generate_pairs --output data/processed/protein_pairs.csv`
6. (Optional) Normalize LLPSDB pairs into project schema:
   - `python -m llps_diffusion.data.ingest_llpsdb --input data/raw/llpsdb_pairs.csv --output data/processed/llpsdb_positive_pairs.csv`
7. Build strict positive candidates (PhasePro anchor + STRING high-confidence + UniProt mapping):
   - `python -m llps_diffusion.data.build_strict_positives --output data/processed/strict_positive_candidates.csv --report data/processed/strict_builder_report.txt --required-score 900`
8. Merge multi-source positives (base + strict + LLPSDB):
   - `python -m llps_diffusion.data.merge_positive_sources --base-pairs data/processed/protein_pairs.csv --strict-candidates data/processed/strict_positive_candidates.csv --llpsdb data/processed/llpsdb_positive_pairs.csv --output data/processed/protein_pairs_multi_source.csv --report data/processed/multi_source_merge_report.txt`
   - One-shot prep shortcut: `make prepare-multi-source`
9. Curate high-purity positives and export evidence tiers:
   - `python -m llps_diffusion.data.curate_positives --input data/processed/protein_pairs_multi_source.csv --strict-candidates data/processed/strict_positive_candidates.csv --output-all data/processed/positives_with_tiers.csv --output-strict data/processed/positives_strict.csv --report data/processed/curation_report.txt`
10. Assemble training pairs by positive-source mode (`strict_only`, `strict_plus_supported`, or `all`):
   - `python -m llps_diffusion.data.assemble_training_pairs --all-pairs data/processed/protein_pairs_multi_source.csv --positives-with-tiers data/processed/positives_with_tiers.csv --mode strict_only --output data/processed/protein_pairs_selected.csv --report data/processed/assemble_training_pairs_report.txt`
   - Makefile shortcuts:
     - `make assemble-training-pairs-strict`
     - `make assemble-training-pairs-supported`
     - `make assemble-training-pairs-all`
11. Run pair quality checks (selected latest training set):
   - `python -m llps_diffusion.data.pairs_qc --input data/processed/protein_pairs_selected.csv --report data/processed/pairs_qc_report.txt`
12. Create leakage-safe splits:
   - `python -m llps_diffusion.data.split_pairs --input data/processed/protein_pairs_selected.csv --out-dir data/processed/splits --train-ratio 0.8 --val-ratio 0.1`
13. Train conditional diffusion model:
   - `python -m llps_diffusion.train --config configs/base.yaml --device auto`
   - After each training run, an experiment report is auto-generated under `docs/experiments/exp_YYYYMMDD_HHMMSS.md`
14. Predict conditional distribution over candidate B set (`p(B|A, y=1)` on a finite candidate pool):
   - Build candidate CSV automatically (recommended):
     - `python -m llps_diffusion.data.build_candidate_b --input-csv data/processed/splits/test.csv --fallback-csv data/processed/protein_pairs_selected.csv --output-csv data/processed/candidate_b.csv`
   - Candidate CSV requires `seq_b` column (optional `id_b`).
   - `python -m llps_diffusion.predict --seq-a "MSTNPKPQRKTKRNTNRRPQDVKFPGG" --candidates-csv data/processed/candidate_b.csv --checkpoint models/checkpoints/conditional_diffusion_best.pt --temperature 0.2 --top-k 50 --num-samples 8 --sampling-steps 200 --output-json models/checkpoints/b_distribution.json --device auto`
   - Makefile shortcut (`make predict-distribution`) now auto-builds `candidate_b.csv` before prediction.
15. Evaluate retrieval quality for candidate ranking (`Recall@K`, `MRR`, `NDCG@K`):
   - Build retrieval eval CSV from test split:
     - `python -m llps_diffusion.data.build_retrieval_eval --input-csv data/processed/splits/test.csv --output-csv data/processed/retrieval_eval.csv --report-path data/processed/retrieval_eval_report.txt --max-candidates-per-anchor 0 --seed 42`
   - Prepare retrieval eval CSV with columns: `id_a, seq_a, id_b, seq_b, label`
   - `python -m llps_diffusion.eval.generative_retrieval --input-csv data/processed/retrieval_eval.csv --checkpoint models/checkpoints/conditional_diffusion_best.pt --output-json models/checkpoints/retrieval_metrics.json --device auto --num-samples 8 --sampling-steps 200`
   - One-shot pipeline shortcut:
     - `make retrieval-pipeline`
   - Compare two retrieval reports:
     - `python -m llps_diffusion.eval.compare_retrieval --baseline-json models/checkpoints/retrieval_metrics_baseline.json --candidate-json models/checkpoints/retrieval_metrics.json --output-json models/checkpoints/retrieval_comparison.json --baseline-name baseline --candidate-name candidate`
16. (Optional baseline) keep classification threshold/evaluate/calibration scripts for old discriminative checkpoints.
17. Plot training curves:
   - `python -m llps_diffusion.visualization.plot_training --log-csv models/checkpoints/train_log.csv --output-png models/checkpoints/train_curves.png`
18. Plot retrieval metrics (`Recall@K`, `NDCG@K`, `MRR`):
   - `python -m llps_diffusion.visualization.plot_retrieval --retrieval-json models/checkpoints/retrieval_metrics.json --output-png models/checkpoints/retrieval_curves.png`

## Development Tooling

- Formatting / lint: `ruff`
- Type check: `mypy`
- Tests: `pytest`
- Pre-commit: `pre-commit`
- CI: GitHub Actions (`lint + test`)

## Reproducibility Note

- `data/raw/`, `data/interim/`, `data/processed/`, and `models/checkpoints/` are ignored by Git.
- GitHub repository stores pipeline code, not generated local data artifacts.
- To ensure training uses the latest dataset, always run:
  - `make prepare-multi-source`
  - `make assemble-training-pairs`
  - `make split-pairs`
  - `make train`

## Mapping to the PDF Stages

- Stage 1 (data): `download_phasepro.py` + `generate_pairs.py` + `pairs.py` + `features/priors.py`
- Stage 2 (architecture): `models/conditional_diffusion.py` + `models/noise_schedule.py`
- Stage 3 (fine-tuning): `train.py`
- Stage 4 (inference/scoring): `predict.py` + `eval/generative_retrieval.py`

