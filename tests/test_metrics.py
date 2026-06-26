# -*- coding: utf-8 -*-
"""
tests/test_metrics.py — EER and minDCF, the two pure-Python detection metrics.

These are the scientific heart of the benchmark: every leaderboard number is
produced here, so they get the most exhaustive coverage (perfect/random/
degenerate inputs, error guards, parameter sensitivity, ordering invariance).
"""

import numpy as np
import pytest

from src.metrics import (
    calculate_eer,
    calculate_eer_and_min_dcf,
    calculate_min_dcf,
)


# ---------------------------------------------------------------------------
# EER
# ---------------------------------------------------------------------------

def test_eer_perfect_separation():
    """Perfectly discriminative scores → EER must be exactly 0."""
    scores = [0.1] * 50 + [0.9] * 50
    labels = [0]   * 50 + [1]   * 50
    eer, _ = calculate_eer(scores, labels)
    assert eer == pytest.approx(0.0, abs=1e-6)


def test_eer_inverted_separation_is_one():
    """Confidently WRONG scores (spoof low, bonafide high) → EER near 1."""
    scores = [0.9] * 50 + [0.1] * 50   # bonafide high, spoof low (inverted)
    labels = [0]   * 50 + [1]   * 50
    eer, _ = calculate_eer(scores, labels)
    assert eer > 0.9


def test_eer_random_scores_near_half():
    """Non-informative scores → EER should land around 0.5."""
    rng    = np.random.default_rng(42)
    scores = rng.random(400).tolist()
    labels = [0] * 200 + [1] * 200
    eer, _ = calculate_eer(scores, labels)
    assert 0.3 < eer < 0.7


def test_eer_returns_threshold():
    """The second return value is a usable decision threshold."""
    scores = [0.1, 0.2, 0.8, 0.9]
    labels = [0, 0, 1, 1]
    eer, threshold = calculate_eer(scores, labels)
    assert isinstance(threshold, float)
    assert eer == pytest.approx(0.0, abs=1e-6)


def test_eer_invariant_to_input_order():
    """Shuffling the (score, label) pairs must not change the EER."""
    scores = [0.1, 0.4, 0.35, 0.8, 0.7, 0.9]
    labels = [0,   0,   1,    1,   0,   1]
    eer_a, _ = calculate_eer(scores, labels)
    idx = [5, 0, 3, 1, 4, 2]
    eer_b, _ = calculate_eer([scores[i] for i in idx], [labels[i] for i in idx])
    assert eer_a == pytest.approx(eer_b)


def test_eer_handles_tied_scores():
    """Identical scores across classes must cross the threshold as a block."""
    scores = [0.5, 0.5, 0.5, 0.5]
    labels = [0, 1, 0, 1]
    eer, _ = calculate_eer(scores, labels)
    assert 0.0 <= eer <= 1.0


def test_eer_error_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        calculate_eer([0.5, 0.5], [0])


def test_eer_error_single_class():
    with pytest.raises(ValueError, match="BOTH classes"):
        calculate_eer([0.1, 0.9], [1, 1])


# ---------------------------------------------------------------------------
# minDCF
# ---------------------------------------------------------------------------

def test_min_dcf_perfect_separation():
    """Perfectly separable scores → minDCF must be 0."""
    scores = [0.1] * 50 + [0.9] * 50
    labels = [0]   * 50 + [1]   * 50
    assert calculate_min_dcf(scores, labels) == pytest.approx(0.0, abs=1e-6)


def test_min_dcf_normalised_upper_bound():
    """A naïve detector can never beat the C_default normaliser → minDCF ≤ 1."""
    rng    = np.random.default_rng(0)
    scores = rng.random(400).tolist()
    labels = [0] * 200 + [1] * 200
    dcf = calculate_min_dcf(scores, labels)
    assert 0.0 <= dcf <= 1.0


def test_min_dcf_better_than_random_for_good_scores():
    """Well-separated scores must yield a strictly lower minDCF than noise."""
    good = calculate_min_dcf([0.1] * 50 + [0.9] * 50, [0] * 50 + [1] * 50)
    rng  = np.random.default_rng(1)
    noisy = calculate_min_dcf(rng.random(100).tolist(), [0] * 50 + [1] * 50)
    assert good < noisy


def test_min_dcf_monotonic_in_fa_cost():
    """The inner detection cost is non-decreasing in C_fa, so a higher
    false-acceptance penalty can only keep or raise the (normalised) minDCF."""
    scores = [0.2, 0.3, 0.55, 0.6, 0.8, 0.4]
    labels = [0, 0, 1, 0, 1, 1]
    cheap = calculate_min_dcf(scores, labels, c_fa=1.0)
    pricey = calculate_min_dcf(scores, labels, c_fa=100.0)
    assert pricey >= cheap


def test_min_dcf_error_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        calculate_min_dcf([0.5], [0, 1])


def test_min_dcf_error_single_class():
    with pytest.raises(ValueError, match="BOTH classes"):
        calculate_min_dcf([0.5, 0.5], [0, 0])


# ---------------------------------------------------------------------------
# Combined single-sort helper (must match the two standalone functions)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scores, labels", [
    # Perfect separation.
    ([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]),
    # Overlapping / noisy.
    ([0.4, 0.55, 0.45, 0.6, 0.5, 0.5], [0, 1, 0, 1, 0, 1]),
    # Tied scores straddling the boundary (block-advance path).
    ([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]),
    # Larger random-ish set.
    (np.linspace(0, 1, 60).tolist(), [0] * 30 + [1] * 30),
])
def test_combined_matches_standalone(scores, labels):
    eer, thr, dcf = calculate_eer_and_min_dcf(scores, labels)
    ref_eer, ref_thr = calculate_eer(scores, labels)
    ref_dcf = calculate_min_dcf(scores, labels)
    assert eer == pytest.approx(ref_eer)
    assert thr == pytest.approx(ref_thr)
    assert dcf == pytest.approx(ref_dcf)


def test_combined_error_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        calculate_eer_and_min_dcf([0.5], [0, 1])


def test_combined_error_single_class():
    with pytest.raises(ValueError, match="BOTH classes"):
        calculate_eer_and_min_dcf([0.1, 0.9], [1, 1])
