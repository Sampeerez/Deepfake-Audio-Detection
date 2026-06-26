# -*- coding: utf-8 -*-
"""
src/metrics.py — Detection metrics: EER and minDCF.

Pure-Python implementations with no external libraries, ensuring full
academic transparency: every arithmetic step is explicit and auditable.
"""

from typing import Sequence, Tuple


def calculate_eer(scores: Sequence[float],
                  labels: Sequence[int]) -> Tuple[float, float]:
    """Equal Error Rate computed without any external library.

    Score convention: p(spoof); high values indicate the detector classifies
    the audio as a deepfake.  Decision rule at threshold t:
        score >= t  -> declared spoof   (audio rejected)
        score <  t  -> declared bonafide (audio accepted)

    Labels: 0 = bonafide, 1 = spoof.

    Definitions:
        FRR(t) = #(bonafide with score >= t) / #bonafide
                 (false rejection rate: genuine utterances flagged as fake)
        FAR(t) = #(spoof with score < t)    / #spoof
                 (false acceptance rate: deepfakes let through as genuine)

    Boundary behaviour of the sweep:
        t -> -inf : everything declared spoof   => FRR = 1, FAR = 0
        t -> +inf : everything declared bonafide => FRR = 0, FAR = 1
    FRR is monotonically non-increasing and FAR non-decreasing in t, so both
    curves cross exactly once: that equilibrium point is the EER.  Because
    the sweep is discrete (one candidate threshold per observed score), the
    function returns (FAR+FRR)/2 at the threshold that minimises |FAR-FRR|.

    Complexity: O(n log n) sort + one linear pass accumulating counters
    (suitable for the ~25k utterances of the ASVspoof dev subset).

    Returns:
        (eer, threshold) with eer in [0, 1].
    """
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length.")

    total_bonafide = sum(1 for e in labels if e == 0)
    total_spoof    = sum(1 for e in labels if e == 1)
    if total_bonafide == 0 or total_spoof == 0:
        raise ValueError("EER requires samples from BOTH classes.")

    # Pairs (score, label) sorted ascending by score.
    pairs = sorted(zip(scores, labels), key=lambda p: p[0])

    best_diff      = float("inf")
    best_eer       = 1.0
    best_threshold = float("-inf")

    # Running counts of samples strictly below the current threshold.
    bonafide_below = 0
    spoof_below    = 0
    idx = 0
    n   = len(pairs)

    while idx <= n:
        threshold = pairs[idx][0] if idx < n else float("inf")

        frr  = (total_bonafide - bonafide_below) / total_bonafide
        far  = spoof_below / total_spoof
        diff = abs(far - frr)

        if diff < best_diff:
            best_diff      = diff
            best_eer       = (far + frr) / 2.0
            best_threshold = threshold

        if idx == n:
            break

        # Advance the threshold consuming ALL tied scores at once
        # (identical scores cross the threshold in a block, not one by one).
        current_score = pairs[idx][0]
        while idx < n and pairs[idx][0] == current_score:
            if pairs[idx][1] == 0:
                bonafide_below += 1
            else:
                spoof_below += 1
            idx += 1

    return best_eer, best_threshold


def calculate_min_dcf(scores: Sequence[float],
                      labels: Sequence[int],
                      p_target: float = 0.05,
                      c_miss: float = 1.0,
                      c_fa: float = 10.0) -> float:
    """Normalised minimum Detection Cost Function (official ASVspoof 2019 metric).

    Same threshold sweep as EER.  Parameters from the ASVspoof 2019 Evaluation
    Plan:
        P_target = 0.05  (prior probability of genuine speakers)
        C_miss   = 1     (cost of rejecting a genuine utterance)
        C_fa     = 10    (cost of accepting a deepfake)

    C_det(t)  = C_miss * P_target * FRR(t) + C_fa * (1-P_target) * FAR(t)
    minDCF    = min_t[C_det(t)] / C_default
    C_default = min(C_miss*P_target, C_fa*(1-P_target))

    minDCF <= 1 means the detector outperforms the optimal naïve decision.
    """
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length.")
    total_bonafide = sum(1 for e in labels if e == 0)
    total_spoof    = sum(1 for e in labels if e == 1)
    if total_bonafide == 0 or total_spoof == 0:
        raise ValueError("minDCF requires samples from BOTH classes.")

    c_default = min(c_miss * p_target, c_fa * (1.0 - p_target))
    pairs     = sorted(zip(scores, labels), key=lambda p: p[0])

    best_dcf       = float("inf")
    bonafide_below = 0
    spoof_below    = 0
    idx = 0
    n   = len(pairs)

    while idx <= n:
        frr = (total_bonafide - bonafide_below) / total_bonafide
        far = spoof_below / total_spoof
        dcf = c_miss * p_target * frr + c_fa * (1.0 - p_target) * far
        if dcf < best_dcf:
            best_dcf = dcf
        if idx == n:
            break
        current_score = pairs[idx][0]
        while idx < n and pairs[idx][0] == current_score:
            if pairs[idx][1] == 0:
                bonafide_below += 1
            else:
                spoof_below += 1
            idx += 1

    return best_dcf / c_default


def calculate_eer_and_min_dcf(scores: Sequence[float],
                              labels: Sequence[int],
                              p_target: float = 0.05,
                              c_miss: float = 1.0,
                              c_fa: float = 10.0) -> Tuple[float, float, float]:
    """Compute EER (with its threshold) and normalised minDCF in a SINGLE sort
    and SINGLE sweep — both metrics evaluate the exact same discrete thresholds
    (one per observed score), so sorting twice is wasteful when, as in the
    benchmark result rows, both are needed for the same scores.

    Returns (eer, eer_threshold, mindcf), numerically identical to calling
    ``calculate_eer`` and ``calculate_min_dcf`` separately. Those two remain as
    the standalone, independently-auditable references (see their docstrings for
    the FRR/FAR definitions and the ASVspoof 2019 cost parameters)."""
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length.")

    total_bonafide = sum(1 for e in labels if e == 0)
    total_spoof    = sum(1 for e in labels if e == 1)
    if total_bonafide == 0 or total_spoof == 0:
        raise ValueError("EER/minDCF require samples from BOTH classes.")

    pairs     = sorted(zip(scores, labels), key=lambda p: p[0])
    c_default = min(c_miss * p_target, c_fa * (1.0 - p_target))

    best_diff      = float("inf")
    best_eer       = 1.0
    best_threshold = float("-inf")
    best_dcf       = float("inf")

    bonafide_below = 0
    spoof_below    = 0
    idx = 0
    n   = len(pairs)

    while idx <= n:
        threshold = pairs[idx][0] if idx < n else float("inf")

        frr = (total_bonafide - bonafide_below) / total_bonafide
        far = spoof_below / total_spoof

        diff = abs(far - frr)
        if diff < best_diff:
            best_diff      = diff
            best_eer       = (far + frr) / 2.0
            best_threshold = threshold

        dcf = c_miss * p_target * frr + c_fa * (1.0 - p_target) * far
        if dcf < best_dcf:
            best_dcf = dcf

        if idx == n:
            break

        current_score = pairs[idx][0]
        while idx < n and pairs[idx][0] == current_score:
            if pairs[idx][1] == 0:
                bonafide_below += 1
            else:
                spoof_below += 1
            idx += 1

    return best_eer, best_threshold, best_dcf / c_default
