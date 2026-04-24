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
- `models/cross_attention.py`: two-tower + cross-attention scorer skeleton.
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
13. Train (contrastive placeholder):
   - `python -m llps_diffusion.train --config configs/base.yaml`
14. Predict on a pair:
   - `python -m llps_diffusion.predict --seq-a "MSTNPKPQRKTKRNTNRRPQDVKFPGG" --seq-b "GGGGSSSSQQQQNNNNKKKK"`
15. Sweep threshold on validation split:
   - `python -m llps_diffusion.eval.threshold_sweep --val-csv data/processed/splits/val.csv --checkpoint models/checkpoints/pair_cross_attention_best.pt --output-json models/checkpoints/best_threshold.json --step 0.01`
16. Evaluate best checkpoint on test split:
   - `python -m llps_diffusion.eval.evaluate --test-csv data/processed/splits/test.csv --checkpoint models/checkpoints/pair_cross_attention_best.pt --output-json models/checkpoints/test_metrics.json --threshold-json models/checkpoints/best_threshold.json`
   - Optional manual override: add `--threshold 0.5`
   - Optional calibrated eval: add `--calibration-method isotonic --val-csv-for-calibration data/processed/splits/val.csv`
   - Makefile shortcuts: `make evaluate-test`, `make evaluate-test-platt`, `make evaluate-test-isotonic`, `make evaluate-compare`
17. Calibrate probabilities on val split and compare test metrics:
   - `python -m llps_diffusion.eval.calibrate --val-csv data/processed/splits/val.csv --test-csv data/processed/splits/test.csv --checkpoint models/checkpoints/pair_cross_attention_best.pt --threshold-json models/checkpoints/best_threshold.json --output-json models/checkpoints/calibration_report.json --method platt`
   - Compare both methods with Makefile: `make calibrate-compare`
18. Plot training curves:
   - `python -m llps_diffusion.visualization.plot_training --log-csv models/checkpoints/train_log.csv --output-png models/checkpoints/train_curves.png`

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
- Stage 2 (architecture): `models/cross_attention.py`
- Stage 3 (fine-tuning): `train.py` + `losses/infonce.py`
- Stage 4 (inference/scoring): `predict.py` + `scoring/pspi.py`

