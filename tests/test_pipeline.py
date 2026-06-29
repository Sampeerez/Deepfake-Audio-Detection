# -*- coding: utf-8 -*-
"""
tests/test_pipeline.py — End-to-end pipeline glue on synthetic audio.

Covers the orchestration layer that turns audio files into leaderboard rows:
feature-matrix extraction, classic train/eval, scoring pre-fitted estimators,
and the inference-only scorers for the CNN and the raw-waveform model. Every
result dict is checked against the shared ``COL_*`` schema.
"""

import numpy as np
import torch
import torch.nn as nn

from src.models import CNN_5Block
from src.pipeline import (
    evaluate_cnn_on_set, evaluate_raw_on_set, extract_feature_matrix,
    run_classic_models, score_fitted_classic,
)
from src.reporting import COL_EER, COL_MIN_DCF, COL_MODEL, RESULT_COLUMNS


# ---------------------------------------------------------------------------
# extract_feature_matrix
# ---------------------------------------------------------------------------

def test_extract_feature_matrix_shape(extractor, labelled_samples):
    samples = labelled_samples(n=6)
    X, y, ms = extract_feature_matrix(samples, extractor, "2", "unit-test")
    assert X.shape == (6, 40)                    # MFCC → 40 dims
    assert X.dtype == np.float32
    assert y.tolist() == [0, 1, 0, 1, 0, 1]
    assert ms >= 0.0


def test_extract_feature_matrix_parallel_matches_serial(extractor, labelled_samples):
    """Threaded extraction must produce the same matrix as the serial path."""
    samples = labelled_samples(n=4)
    serial, _, _ = extract_feature_matrix(samples, extractor, "3", "serial")
    parallel, _, _ = extract_feature_matrix(samples, extractor, "3", "parallel",
                                            n_workers=2)
    assert np.allclose(serial, parallel)


# ---------------------------------------------------------------------------
# run_classic_models
# ---------------------------------------------------------------------------

def _toy_matrices(seed=0):
    rng = np.random.default_rng(seed)
    # Separable: class 1 shifted, so the model produces non-trivial scores.
    x_train = np.vstack([rng.normal(0, 1, (30, 8)), rng.normal(3, 1, (30, 8))])
    y_train = np.array([0] * 30 + [1] * 30)
    x_dev = np.vstack([rng.normal(0, 1, (10, 8)), rng.normal(3, 1, (10, 8))])
    y_dev = np.array([0] * 10 + [1] * 10)
    return x_train.astype(np.float32), y_train, x_dev.astype(np.float32), y_dev


def test_run_classic_models_dev_rows():
    xtr, ytr, xdv, ydv = _toy_matrices()
    rows = run_classic_models(["logistic_regression"], xtr, ytr, xdv, ydv,
                              feature_label="MFCC", ms_dsp_dev=1.0, seed=42)
    assert len(rows) == 1
    row = rows[0]
    for col in RESULT_COLUMNS:
        assert col in row
    assert "[CPU]" in row[COL_MODEL]
    assert 0.0 <= float(row[COL_EER]) <= 100.0
    assert 0.0 <= float(row[COL_MIN_DCF])


def test_run_classic_models_with_eval_set():
    xtr, ytr, xdv, ydv = _toy_matrices()
    rows = run_classic_models(["logistic_regression"], xtr, ytr, xdv, ydv,
                              feature_label="MFCC", ms_dsp_dev=1.0, seed=42,
                              eval_sets=[("2021 LA", xdv, ydv)])
    eval_rows = [r for r in rows if r.get("Corpus") == "2021 LA"]
    assert len(eval_rows) == 1
    assert "[EVAL]" in eval_rows[0][COL_MODEL]


def test_model_sink_receives_fitted_model():
    xtr, ytr, xdv, ydv = _toy_matrices()
    captured = {}
    run_classic_models(["logistic_regression"], xtr, ytr, xdv, ydv,
                       feature_label="MFCC", ms_dsp_dev=1.0, seed=42,
                       model_sink=lambda name, m: captured.__setitem__(name, m))
    assert "logistic_regression" in captured
    assert hasattr(captured["logistic_regression"], "predict_proba")


# ---------------------------------------------------------------------------
# score_fitted_classic
# ---------------------------------------------------------------------------

def test_score_fitted_classic():
    from src.models import get_classic_model
    xtr, ytr, xdv, ydv = _toy_matrices()
    model = get_classic_model("logistic_regression", seed=1).fit(xtr, ytr)
    rows = score_fitted_classic({"logistic_regression": model}, xdv, ydv,
                                feature_label="MFCC", corpus_label="dev")
    assert len(rows) == 1
    assert "[EVAL]" in rows[0][COL_MODEL]


# ---------------------------------------------------------------------------
# evaluate_cnn_on_set — inference-only CNN scorer
# ---------------------------------------------------------------------------

def test_evaluate_cnn_on_set(extractor, labelled_samples, tmp_path, monkeypatch):
    # Redirect the spectrogram cache into tmp so the test leaves no artefacts.
    monkeypatch.setattr("src.pipeline.CACHE_DIR", str(tmp_path / "cache"))
    samples = labelled_samples(n=6)
    model = CNN_5Block().eval()                 # random weights — shape test only
    rows = evaluate_cnn_on_set(model, samples, extractor,
                               params={"num_workers": 0, "batch_size": 4},
                               corpus_label="unit", arch_label="5-Block CNN")
    assert len(rows) == 1
    assert rows[0]["Corpus"] == "unit"
    assert 0.0 <= float(rows[0][COL_EER]) <= 100.0


# ---------------------------------------------------------------------------
# evaluate_raw_on_set — inference-only raw-waveform scorer
# ---------------------------------------------------------------------------

class _DummyRaw(nn.Module):
    """Stand-in for the wav2vec 2.0 detector: exposes the prob_spoof contract
    without the 95 M-parameter backbone, so the pipeline logic is tested cheaply."""
    @torch.no_grad()
    def prob_spoof(self, x):
        return torch.sigmoid(x.mean(dim=-1))    # deterministic per-clip score


def test_evaluate_raw_on_set(labelled_samples):
    samples = labelled_samples(n=6)
    rows = evaluate_raw_on_set(_DummyRaw(), samples, sample_rate=16_000,
                               params={"num_workers": 0, "raw_batch_size": 3},
                               corpus_label="unit")
    assert len(rows) == 1
    row = rows[0]
    assert "Raw waveform" in row["Feature Configuration"]
    assert row["Corpus"] == "unit"
    assert 0.0 <= float(row[COL_EER]) <= 100.0


# ---------------------------------------------------------------------------
# _aggregate_seed_rows — cross-seed mean ± std for the multi-seed CNN sweep
# ---------------------------------------------------------------------------

def test_aggregate_seed_rows_means_and_std():
    from src.jobs import _aggregate_seed_rows
    from src.reporting import COL_MIN_DCF as _DCF

    def _row(model, eer, dcf, corpus=""):
        return {COL_MODEL: model, COL_EER: f"{eer}", _DCF: f"{dcf}",
                "Accuracy": "0.9", "Corpus": corpus}

    # Two seeds, each with a dev row and one eval row.
    seed_a = [_row("5-Block CNN [CUDA]", 10.0, 0.90),
              _row("5-Block CNN [CUDA][EVAL]", 20.0, 0.95, "2021 LA")]
    seed_b = [_row("5-Block CNN [CUDA]", 12.0, 0.94),
              _row("5-Block CNN [CUDA][EVAL]", 24.0, 0.99, "2021 LA")]
    agg = _aggregate_seed_rows([seed_a, seed_b], n_seeds=2)

    assert len(agg) == 2                              # one dev + one eval row
    dev = next(r for r in agg if "[EVAL]" not in r[COL_MODEL])
    ev  = next(r for r in agg if "[EVAL]" in r[COL_MODEL])
    assert float(dev[COL_EER]) == 11.0               # mean(10, 12)
    assert float(ev[COL_EER]) == 22.0                # mean(20, 24)
    assert dev["Seeds"] == 2
    assert float(dev["EER std"]) > 0                 # std recorded
    assert "minDCF std" in ev


def test_aggregate_seed_rows_single_seed():
    from src.jobs import _aggregate_seed_rows
    one = [{COL_MODEL: "CRNN [CUDA]", COL_EER: "8.0", COL_MIN_DCF: "0.5",
            "Accuracy": "0.9", "Corpus": ""}]
    agg = _aggregate_seed_rows([one], n_seeds=1)
    assert len(agg) == 1
    assert agg[0]["Seeds"] == 1
    assert float(agg[0][COL_EER]) == 8.0
