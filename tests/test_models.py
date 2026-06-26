# -*- coding: utf-8 -*-
"""
tests/test_models.py — The three detector families.

  A) Classic sklearn/xgboost estimators from the ``get_classic_model`` factory.
  B) The 2-D spectrogram CNNs (AudioDeepfakeCNN, ResNetCNN).
  C) The self-supervised raw-waveform detector (Wav2Vec2Classifier) — skipped
     automatically if ``transformers`` is not installed.

Deep models are exercised on CPU with random weights: we assert the forward
contract (output shapes, probability ranges, activation taps), not accuracy.
"""

import numpy as np
import pytest
import torch

from src.models import (
    CLASSIC_MODELS, AudioDeepfakeCNN, ResNetCNN, get_classic_model,
)


# ---------------------------------------------------------------------------
# A) Classic models
# ---------------------------------------------------------------------------

def _make_xy(n=100, dim=40, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, dim)).astype(np.float32)
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    return X, y


@pytest.mark.parametrize("name", CLASSIC_MODELS)
def test_classic_model_probabilities(name):
    """Every classic model must fit and emit calibrated 2-column probabilities."""
    n = 60 if name == "svm_lineal" else 100   # SVM calibration CV is slower
    X, y = _make_xy(n=n)
    model = get_classic_model(name, seed=42)
    model.fit(X, y)
    p = model.predict_proba(X)
    assert p.shape == (n, 2)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-5)
    assert ((p >= 0.0) & (p <= 1.0)).all()


def test_xgboost_scale_pos_weight_accepted():
    """The imbalance compensation argument must be honoured without error."""
    X, y = _make_xy()
    model = get_classic_model("xgboost", seed=0, scale_pos_weight=9.0)
    model.fit(X, y)
    assert model.predict_proba(X).shape == (100, 2)


def test_classic_models_are_seeded():
    """Same seed → identical predictions (reproducibility guarantee)."""
    X, y = _make_xy()
    p1 = get_classic_model("logistic_regression", seed=7).fit(X, y).predict_proba(X)
    p2 = get_classic_model("logistic_regression", seed=7).fit(X, y).predict_proba(X)
    assert np.array_equal(p1, p2)


def test_unknown_classic_model():
    with pytest.raises(ValueError, match="Unknown"):
        get_classic_model("magic_neural_network")


# ---------------------------------------------------------------------------
# B) Spectrogram CNNs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("Arch", [AudioDeepfakeCNN, ResNetCNN])
def test_cnn_forward_shape(Arch):
    """Forward pass returns one raw logit per item in the batch."""
    model = Arch().eval()
    x = torch.randn(4, 1, 128, 300)             # (batch, channel, freq, time)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4,)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("Arch,n_blocks", [(AudioDeepfakeCNN, 3), (ResNetCNN, 4)])
def test_cnn_activation_taps(Arch, n_blocks):
    """forward_with_activations exposes one feature map per conv/residual block."""
    model = Arch().eval()
    x = torch.randn(2, 1, 128, 300)
    logits, acts = model.forward_with_activations(x)
    assert logits.shape == (2,)
    assert len(acts) == n_blocks
    for a in acts:
        assert a.dim() == 4                     # (batch, channels, H, W)


def test_cnn_robust_to_input_size():
    """AdaptiveAvgPool decouples the head from the exact spectrogram size."""
    model = AudioDeepfakeCNN().eval()
    with torch.no_grad():
        out = model(torch.randn(1, 1, 128, 200))   # non-default time frames
    assert out.shape == (1,)


def test_cnn_probability_after_sigmoid():
    """sigmoid(logit) yields a valid p(spoof) in [0, 1]."""
    model = AudioDeepfakeCNN().eval()
    with torch.no_grad():
        p = torch.sigmoid(model(torch.randn(3, 1, 128, 300)))
    assert ((p >= 0.0) & (p <= 1.0)).all()


# ---------------------------------------------------------------------------
# C) Self-supervised raw-waveform detector (optional dependency)
# ---------------------------------------------------------------------------

def test_wav2vec2_forward_and_prob():
    """Wav2Vec2Classifier: (B,2) logits and a p(spoof) in [0,1] from raw audio.

    Skipped when ``transformers`` is unavailable. Uses random weights and a
    short clip — we test the forward contract, not detection quality.
    """
    pytest.importorskip("transformers")
    from src.models import Wav2Vec2Classifier

    model = Wav2Vec2Classifier().eval()
    wave = torch.randn(2, 8000)                 # 2 clips, 0.5 s @ 16 kHz
    with torch.no_grad():
        logits = model(wave)
        probs = model.prob_spoof(wave)
    assert logits.shape == (2, 2)
    assert probs.shape == (2,)
    assert ((probs >= 0.0) & (probs <= 1.0)).all()


def test_wav2vec2_temperature_preserves_ranking():
    """Temperature scaling is monotonic → it must not reorder p(spoof)."""
    pytest.importorskip("transformers")
    from src.models import Wav2Vec2Classifier

    model = Wav2Vec2Classifier().eval()
    wave = torch.randn(3, 8000)
    with torch.no_grad():
        raw = torch.softmax(model(wave), dim=-1)[:, 1]
        cal = model.prob_spoof(wave)
    # Same argsort ordering despite the softened confidences.
    assert torch.equal(torch.argsort(raw), torch.argsort(cal))
