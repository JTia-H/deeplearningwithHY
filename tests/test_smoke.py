from pathlib import Path

from llps_diffusion.data.assemble_training_pairs import assemble_training_pairs
from llps_diffusion.data.build_retrieval_eval import build_retrieval_eval_csv
from llps_diffusion.data.build_strict_positives import write_report
from llps_diffusion.data.curate_positives import curate_positives
from llps_diffusion.data.datasets import sequence_to_features
from llps_diffusion.data.ingest_llpsdb import normalize_llpsdb_pairs
from llps_diffusion.data.merge_positive_sources import merge_positive_sources
from llps_diffusion.data.pairs import ProteinPair, build_demo_pairs, load_pairs_csv, save_pairs_csv
from llps_diffusion.data.pairs_qc import run_qc
from llps_diffusion.data.split_pairs import split_by_anchor_protein
from llps_diffusion.eval.calibrate import compute_metrics
from llps_diffusion.eval.compare_retrieval import compare_retrieval_reports
from llps_diffusion.eval.retrieval import evaluate_retrieval
from llps_diffusion.eval.threshold_sweep import sweep_thresholds
from llps_diffusion.experiments.reporting import generate_experiment_report
from llps_diffusion.features.priors import estimate_idr_ratio, estimate_prld_score
from llps_diffusion.losses.infonce import infonce_loss
from llps_diffusion.predict import predict_b_distribution


def test_sequence_to_features_length() -> None:
    feat = sequence_to_features("ACDEFGHIK")
    assert len(feat) == 21


def test_priors_range() -> None:
    idr = estimate_idr_ratio("GGQQNNSSKK")
    prld = estimate_prld_score("GGQQNNSSKK")
    assert 0.0 <= idr <= 1.0
    assert 0.0 <= prld <= 1.0


def test_infonce_loss_positive() -> None:
    import torch

    loss = infonce_loss(torch.tensor(1.2), torch.tensor([0.2, 0.4]), temperature=0.07)
    assert loss.item() >= 0.0


def test_pairs_csv_roundtrip(tmp_path: Path) -> None:
    pairs = build_demo_pairs()
    out = save_pairs_csv(pairs, tmp_path / "pairs.csv")
    loaded = load_pairs_csv(out)
    assert len(loaded) == len(pairs)
    assert loaded[0].label == 1


def test_pairs_qc_and_split(tmp_path: Path) -> None:
    pairs = [
        ProteinPair("A1", "B1", "AAAA", "CCCC", 1, "pos"),
        ProteinPair("A2", "B2", "GGGG", "TTTT", 0, "neg"),
        ProteinPair("A3", "B3", "NNNN", "QQQQ", 0, "neg"),
    ]
    csv_path = save_pairs_csv(pairs, tmp_path / "pairs.csv")
    stats = run_qc(csv_path, tmp_path / "report.txt")
    assert stats["total_pairs"] == 3.0
    train_path, val_path, test_path = split_by_anchor_protein(
        input_csv=csv_path, out_dir=tmp_path / "splits", train_ratio=0.67, val_ratio=0.0, seed=7
    )
    assert train_path.exists()
    assert val_path.exists()
    assert test_path.exists()


def test_curate_positives_outputs(tmp_path: Path) -> None:
    pairs = [
        ProteinPair("P1", "P2", "A" * 50, "C" * 50, 1, "phasepro_positive"),
        ProteinPair("P2", "P1", "A" * 50, "C" * 50, 1, "phasepro_positive"),
        ProteinPair("P3", "P4", "G" * 50, "T" * 50, 1, "phasepro_string_proxy_positive"),
        ProteinPair("N1", "N2", "A" * 50, "C" * 50, 0, "swissprot_random_negative"),
    ]
    input_csv = save_pairs_csv(pairs, tmp_path / "pairs.csv")
    out_all, out_strict, out_report = curate_positives(
        input_csv=input_csv,
        output_all=tmp_path / "positives_with_tiers.csv",
        output_strict=tmp_path / "positives_strict.csv",
        report_path=tmp_path / "curation_report.txt",
    )
    assert out_all.exists()
    assert out_strict.exists()
    assert out_report.exists()


def test_strict_builder_report_write(tmp_path: Path) -> None:
    out = write_report(
        report_path=tmp_path / "strict_builder_report.txt",
        strict_count=12,
        required_score=900,
        max_pairs=300,
        max_partners_per_anchor=3,
    )
    assert out.exists()


def test_assemble_training_pairs_modes(tmp_path: Path) -> None:
    all_pairs = [
        ProteinPair("P1", "P2", "A" * 50, "C" * 50, 1, "phasepro_positive"),
        ProteinPair("P3", "P4", "G" * 50, "T" * 50, 1, "phasepro_string_proxy_positive"),
        ProteinPair("N1", "N2", "V" * 50, "W" * 50, 0, "swissprot_random_negative"),
    ]
    all_csv = save_pairs_csv(all_pairs, tmp_path / "protein_pairs.csv")
    pos_all, _, _ = curate_positives(
        input_csv=all_csv,
        output_all=tmp_path / "positives_with_tiers.csv",
        output_strict=tmp_path / "positives_strict.csv",
        report_path=tmp_path / "curation_report.txt",
    )
    out_csv, report = assemble_training_pairs(
        all_pairs_csv=all_csv,
        positives_with_tiers_csv=pos_all,
        mode="strict_only",
        output_csv=tmp_path / "selected.csv",
        report_path=tmp_path / "assemble_report.txt",
    )
    assert out_csv.exists()
    assert report.exists()


def test_threshold_sweep_validation(tmp_path: Path) -> None:
    import pytest

    with pytest.raises(ValueError):
        sweep_thresholds(
            val_csv=tmp_path / "missing.csv",
            checkpoint=tmp_path / "missing.pt",
            output_json=tmp_path / "best_threshold.json",
            step=1.2,
        )


def test_ingest_llpsdb_and_merge_sources(tmp_path: Path) -> None:
    llpsdb = tmp_path / "llpsdb.csv"
    llpsdb.write_text(
        "uniprot_a,uniprot_b,sequence_a,sequence_b,label\n"
        "P1,P2,AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA,CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC,1\n",
        encoding="utf-8",
    )
    llpsdb_out = normalize_llpsdb_pairs(llpsdb, tmp_path / "llpsdb_out.csv")
    assert llpsdb_out.exists()

    base_pairs = [
        ProteinPair("N1", "N2", "V" * 50, "W" * 50, 0, "swissprot_random_negative"),
    ]
    base_csv = save_pairs_csv(base_pairs, tmp_path / "base_pairs.csv")
    strict_csv = save_pairs_csv([], tmp_path / "strict_candidates.csv")
    merged_csv, merged_report = merge_positive_sources(
        base_pairs_csv=base_csv,
        strict_candidates_csv=strict_csv,
        llpsdb_csv=llpsdb_out,
        output_csv=tmp_path / "merged.csv",
        report_path=tmp_path / "merged_report.txt",
    )
    assert merged_csv.exists()
    assert merged_report.exists()


def test_compute_metrics_basic() -> None:
    import numpy as np

    y_true = np.array([0, 1, 1, 0], dtype=np.int64)
    y_prob = np.array([0.1, 0.9, 0.7, 0.3], dtype=np.float64)
    out = compute_metrics(y_true, y_prob, threshold=0.5)
    assert 0.0 <= out["accuracy"] <= 1.0
    assert 0.0 <= out["f1"] <= 1.0


def test_predict_b_distribution_candidate_set(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates.csv"
    candidates.write_text(
        "id_b,seq_b\nB1,ACDEFGHIK\nB2,GGGGSSSSQ\nB3,NNNNKKKKQ\n",
        encoding="utf-8",
    )
    ckpt = tmp_path / "missing.pt"
    import pytest

    with pytest.raises(FileNotFoundError):
        predict_b_distribution(
            seq_a="MSTNPKPQRKTKRNTNRRPQDVKFPGG",
            candidates_csv=candidates,
            checkpoint=ckpt,
            device="cpu",
            output_json=tmp_path / "b_dist.json",
        )


def test_evaluate_retrieval_missing_checkpoint(tmp_path: Path) -> None:
    retrieval_csv = tmp_path / "retrieval_eval.csv"
    retrieval_csv.write_text(
        "id_a,seq_a,id_b,seq_b,label\n" "A1,AAAAA,B1,CCCCC,1\n" "A1,AAAAA,B2,DDDDD,0\n",
        encoding="utf-8",
    )
    import pytest

    with pytest.raises(FileNotFoundError):
        evaluate_retrieval(
            input_csv=retrieval_csv,
            checkpoint=tmp_path / "missing.pt",
            output_json=tmp_path / "retrieval_metrics.json",
            device="cpu",
        )


def test_build_retrieval_eval_csv(tmp_path: Path) -> None:
    src = tmp_path / "test_split.csv"
    src.write_text(
        "id_a,seq_a,id_b,seq_b,label,source\n"
        "A1,AAAAA,B1,CCCCC,1,pos\n"
        "A1,AAAAA,B2,DDDDD,0,neg\n"
        "A2,GGGGG,B3,EEEEE,0,neg\n",
        encoding="utf-8",
    )
    out_csv, out_report = build_retrieval_eval_csv(
        input_csv=src,
        output_csv=tmp_path / "retrieval_eval.csv",
        report_path=tmp_path / "retrieval_eval_report.txt",
        max_candidates_per_anchor=0,
        seed=7,
    )
    assert out_csv.exists()
    assert out_report.exists()


def test_compare_retrieval_reports(tmp_path: Path) -> None:
    base_json = tmp_path / "base.json"
    cand_json = tmp_path / "cand.json"
    base_json.write_text(
        '{"metrics":{"mrr":0.2,"recall@1":0.1,"recall@5":0.3,"recall@10":0.4,"recall@20":0.5,"ndcg@1":0.1,"ndcg@5":0.2,"ndcg@10":0.3,"ndcg@20":0.4}}',
        encoding="utf-8",
    )
    cand_json.write_text(
        '{"metrics":{"mrr":0.3,"recall@1":0.2,"recall@5":0.35,"recall@10":0.45,"recall@20":0.55,"ndcg@1":0.2,"ndcg@5":0.25,"ndcg@10":0.35,"ndcg@20":0.45}}',
        encoding="utf-8",
    )
    out = compare_retrieval_reports(
        baseline_json=base_json,
        candidate_json=cand_json,
        output_json=tmp_path / "cmp.json",
        baseline_name="old",
        candidate_name="new",
    )
    assert out.exists()


def test_generate_experiment_report(tmp_path: Path) -> None:
    ckpt = tmp_path / "checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "train_log.csv").write_text(
        "epoch,train_loss,val_loss,val_acc,val_auc,lr,grad_norm_mean,grad_norm_max,skipped\n"
        "1,0.5,0.4,0.8,0.9,0.001,0.3,0.9,0\n",
        encoding="utf-8",
    )
    (ckpt / "test_metrics.json").write_text(
        '{"accuracy":0.8,"f1":0.75,"auc":0.88,"pr_auc":0.84}', encoding="utf-8"
    )
    (ckpt / "retrieval_metrics.json").write_text(
        '{"metrics":{"mrr":0.5,"recall@1":0.4,"recall@5":0.7,"recall@10":0.8,"recall@20":0.9,"ndcg@1":0.4,"ndcg@5":0.6,"ndcg@10":0.7,"ndcg@20":0.8}}',
        encoding="utf-8",
    )
    out = generate_experiment_report(
        config_path="configs/base.yaml",
        device="cpu",
        output_dir=tmp_path / "experiments",
        checkpoints_dir=ckpt,
        exp_id="exp_test",
    )
    assert out.exists()
