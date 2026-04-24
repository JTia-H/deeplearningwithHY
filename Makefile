.PHONY: install-dev lint format typecheck test precommit generate-pairs ingest-llpsdb merge-multi-source prepare-multi-source build-strict-positives curate-positives assemble-training-pairs assemble-training-pairs-strict assemble-training-pairs-supported assemble-training-pairs-all pairs-qc split-pairs split-pairs-selected train predict threshold-sweep evaluate-test evaluate-test-platt evaluate-test-isotonic evaluate-compare calibrate-test calibrate-test-isotonic calibrate-compare plot-training

install-dev:
	pip install -e ".[dev]"

lint:
	ruff check src tests

format:
	ruff format src tests

typecheck:
	mypy src

test:
	pytest

precommit:
	pre-commit run --all-files

generate-pairs:
	python -m llps_diffusion.data.generate_pairs --output data/processed/protein_pairs.csv

ingest-llpsdb:
	python -m llps_diffusion.data.ingest_llpsdb --input data/raw/llpsdb_pairs.csv --output data/processed/llpsdb_positive_pairs.csv

merge-multi-source:
	python -m llps_diffusion.data.merge_positive_sources --base-pairs data/processed/protein_pairs.csv --strict-candidates data/processed/strict_positive_candidates.csv --llpsdb data/processed/llpsdb_positive_pairs.csv --output data/processed/protein_pairs_multi_source.csv --report data/processed/multi_source_merge_report.txt

build-strict-positives:
	python -m llps_diffusion.data.build_strict_positives --output data/processed/strict_positive_candidates.csv --report data/processed/strict_builder_report.txt --required-score 900 --max-pairs 300 --max-partners-per-anchor 3

curate-positives:
	python -m llps_diffusion.data.curate_positives --input data/processed/protein_pairs_multi_source.csv --strict-candidates data/processed/strict_positive_candidates.csv --output-all data/processed/positives_with_tiers.csv --output-strict data/processed/positives_strict.csv --report data/processed/curation_report.txt

assemble-training-pairs:
	python -m llps_diffusion.data.assemble_training_pairs --all-pairs data/processed/protein_pairs_multi_source.csv --positives-with-tiers data/processed/positives_with_tiers.csv --mode strict_only --output data/processed/protein_pairs_selected.csv --report data/processed/assemble_training_pairs_report.txt

assemble-training-pairs-strict:
	python -m llps_diffusion.data.assemble_training_pairs --all-pairs data/processed/protein_pairs_multi_source.csv --positives-with-tiers data/processed/positives_with_tiers.csv --mode strict_only --output data/processed/protein_pairs_selected.csv --report data/processed/assemble_training_pairs_report.txt

assemble-training-pairs-supported:
	python -m llps_diffusion.data.assemble_training_pairs --all-pairs data/processed/protein_pairs_multi_source.csv --positives-with-tiers data/processed/positives_with_tiers.csv --mode strict_plus_supported --output data/processed/protein_pairs_selected.csv --report data/processed/assemble_training_pairs_report.txt

assemble-training-pairs-all:
	python -m llps_diffusion.data.assemble_training_pairs --all-pairs data/processed/protein_pairs_multi_source.csv --positives-with-tiers data/processed/positives_with_tiers.csv --mode all --output data/processed/protein_pairs_selected.csv --report data/processed/assemble_training_pairs_report.txt

prepare-multi-source:
	python -m llps_diffusion.data.merge_positive_sources --base-pairs data/processed/protein_pairs.csv --strict-candidates data/processed/strict_positive_candidates.csv --llpsdb data/processed/llpsdb_positive_pairs.csv --output data/processed/protein_pairs_multi_source.csv --report data/processed/multi_source_merge_report.txt && python -m llps_diffusion.data.curate_positives --input data/processed/protein_pairs_multi_source.csv --strict-candidates data/processed/strict_positive_candidates.csv --output-all data/processed/positives_with_tiers.csv --output-strict data/processed/positives_strict.csv --report data/processed/curation_report.txt

pairs-qc:
	python -m llps_diffusion.data.pairs_qc --input data/processed/protein_pairs_selected.csv --report data/processed/pairs_qc_report.txt

split-pairs:
	python -m llps_diffusion.data.split_pairs --input data/processed/protein_pairs_selected.csv --out-dir data/processed/splits --train-ratio 0.8 --val-ratio 0.1

split-pairs-selected:
	python -m llps_diffusion.data.split_pairs --input data/processed/protein_pairs_selected.csv --out-dir data/processed/splits --train-ratio 0.8 --val-ratio 0.1

train:
	python -m llps_diffusion.train --config configs/base.yaml

predict:
	python -m llps_diffusion.predict --seq-a "MSTNPKPQRKTKRNTNRRPQDVKFPGG" --seq-b "GGGGSSSSQQQQNNNNKKKK"

threshold-sweep:
	python -m llps_diffusion.eval.threshold_sweep --val-csv data/processed/splits/val.csv --checkpoint models/checkpoints/pair_cross_attention_best.pt --output-json models/checkpoints/best_threshold.json --step 0.01

evaluate-test:
	python -m llps_diffusion.eval.evaluate --test-csv data/processed/splits/test.csv --checkpoint models/checkpoints/pair_cross_attention_best.pt --output-json models/checkpoints/test_metrics.json --threshold-json models/checkpoints/best_threshold.json

evaluate-test-platt:
	python -m llps_diffusion.eval.evaluate --test-csv data/processed/splits/test.csv --checkpoint models/checkpoints/pair_cross_attention_best.pt --output-json models/checkpoints/test_metrics_platt.json --threshold-json models/checkpoints/best_threshold.json --calibration-method platt --val-csv-for-calibration data/processed/splits/val.csv

evaluate-test-isotonic:
	python -m llps_diffusion.eval.evaluate --test-csv data/processed/splits/test.csv --checkpoint models/checkpoints/pair_cross_attention_best.pt --output-json models/checkpoints/test_metrics_isotonic.json --threshold-json models/checkpoints/best_threshold.json --calibration-method isotonic --val-csv-for-calibration data/processed/splits/val.csv

evaluate-compare:
	python -m llps_diffusion.eval.compare_evals --none-json models/checkpoints/test_metrics.json --platt-json models/checkpoints/test_metrics_platt.json --isotonic-json models/checkpoints/test_metrics_isotonic.json --output-json models/checkpoints/final_eval_comparison.json

calibrate-test:
	python -m llps_diffusion.eval.calibrate --val-csv data/processed/splits/val.csv --test-csv data/processed/splits/test.csv --checkpoint models/checkpoints/pair_cross_attention_best.pt --threshold-json models/checkpoints/best_threshold.json --output-json models/checkpoints/calibration_report.json --method platt

calibrate-test-isotonic:
	python -m llps_diffusion.eval.calibrate --val-csv data/processed/splits/val.csv --test-csv data/processed/splits/test.csv --checkpoint models/checkpoints/pair_cross_attention_best.pt --threshold-json models/checkpoints/best_threshold.json --output-json models/checkpoints/calibration_report_isotonic.json --method isotonic

calibrate-compare:
	python -m llps_diffusion.eval.calibrate --val-csv data/processed/splits/val.csv --test-csv data/processed/splits/test.csv --checkpoint models/checkpoints/pair_cross_attention_best.pt --threshold-json models/checkpoints/best_threshold.json --output-json models/checkpoints/calibration_report_platt.json --method platt && python -m llps_diffusion.eval.calibrate --val-csv data/processed/splits/val.csv --test-csv data/processed/splits/test.csv --checkpoint models/checkpoints/pair_cross_attention_best.pt --threshold-json models/checkpoints/best_threshold.json --output-json models/checkpoints/calibration_report_isotonic.json --method isotonic

plot-training:
	python -m llps_diffusion.visualization.plot_training --log-csv models/checkpoints/train_log.csv --output-png models/checkpoints/train_curves.png

