# -*- coding: utf-8 -*-
"""
src/metrics.py, Detection metrics: EER and minDCF.

Pure-Python implementations, no external libraries, so every arithmetic
step stays explicit and auditable.
"""

from typing import Sequence, Tuple


def calculate_eer(scores: Sequence[float],
                  labels: Sequence[int]) -> Tuple[float, float]:
    """Equal Error Rate computed without any external library.

    Score convention: p(spoof); high values mean the detector reads the audio
    as a deepfake. Decision rule at threshold t: score >= t is declared spoof
    (rejected), score < t is declared bonafide (accepted). Labels: 0 = bonafide,
    1 = spoof.

    FRR(t) is #(bonafide with score >= t) / #bonafide, the false rejection rate
    (genuine utterances flagged as fake). FAR(t) is #(spoof with score < t) /
    #spoof, the false acceptance rate (deepfakes let through).

    As t drops to -inf everything is spoof (FRR = 1, FAR = 0); as it rises to
    +inf everything is bonafide (FRR = 0, FAR = 1). FRR is non-increasing and
    FAR non-decreasing in t, so the two curves cross once, and that equilibrium
    is the EER. The sweep is discrete (one candidate threshold per observed
    score), so we return (FAR+FRR)/2 at the threshold that minimises |FAR-FRR|.

    Complexity: O(n log n) sort plus one linear pass accumulating counters.

    Returns:
        (eer, threshold) with eer in [0, 1].
    """
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length.")

    total_bonafide = sum(1 for e in labels if e == 0)
    total_spoof = sum(1 for e in labels if e == 1)
    if total_bonafide == 0 or total_spoof == 0:
        raise ValueError("EER requires samples from both classes.")

    pairs = sorted(zip(scores, labels), key=lambda p: p[0])

    best_diff = float("inf")
    best_eer = 1.0
    best_threshold = float("-inf")

    bonafide_below = 0
    spoof_below = 0
    idx = 0
    n = len(pairs)

    while idx <= n:
        threshold = pairs[idx][0] if idx < n else float("inf")

        frr = (total_bonafide - bonafide_below) / total_bonafide
        far = spoof_below / total_spoof
        diff = abs(far - frr)

        if diff < best_diff:
            best_diff = diff
            best_eer = (far + frr) / 2.0
            best_threshold = threshold

        if idx == n:
            break

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

    Same threshold sweep as EER. Parameters from the ASVspoof 2019 Evaluation
    Plan: P_target = 0.05 (prior of genuine speakers), C_miss = 1 (cost of
    rejecting a genuine utterance), C_fa = 10 (cost of accepting a deepfake).

    C_det(t)  = C_miss * P_target * FRR(t) + C_fa * (1-P_target) * FAR(t)
    minDCF    = min_t[C_det(t)] / C_default
    C_default = min(C_miss*P_target, C_fa*(1-P_target))

    minDCF <= 1 means the detector beats the optimal naive decision.
    """
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length.")
    total_bonafide = sum(1 for e in labels if e == 0)
    total_spoof = sum(1 for e in labels if e == 1)
    if total_bonafide == 0 or total_spoof == 0:
        raise ValueError("minDCF requires samples from both classes.")

    c_default = min(c_miss * p_target, c_fa * (1.0 - p_target))
    pairs = sorted(zip(scores, labels), key=lambda p: p[0])

    best_dcf = float("inf")
    bonafide_below = 0
    spoof_below = 0
    idx = 0
    n = len(pairs)

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
    """Compute EER (with its threshold) and normalised minDCF in one sort and
    one sweep. Both metrics evaluate the same discrete thresholds (one per
    observed score), so sorting twice is wasteful when both are needed for the
    same scores, as in the benchmark result rows.

    Returns (eer, eer_threshold, mindcf), numerically identical to calling
    ``calculate_eer`` and ``calculate_min_dcf`` separately, which remain the
    standalone, independently-auditable references."""
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length.")

    total_bonafide = sum(1 for e in labels if e == 0)
    total_spoof = sum(1 for e in labels if e == 1)
    if total_bonafide == 0 or total_spoof == 0:
        raise ValueError("EER/minDCF require samples from both classes.")

    pairs = sorted(zip(scores, labels), key=lambda p: p[0])
    c_default = min(c_miss * p_target, c_fa * (1.0 - p_target))

    best_diff = float("inf")
    best_eer = 1.0
    best_threshold = float("-inf")
    best_dcf = float("inf")

    bonafide_below = 0
    spoof_below = 0
    idx = 0
    n = len(pairs)

    while idx <= n:
        threshold = pairs[idx][0] if idx < n else float("inf")

        frr = (total_bonafide - bonafide_below) / total_bonafide
        far = spoof_below / total_spoof

        diff = abs(far - frr)
        if diff < best_diff:
            best_diff = diff
            best_eer = (far + frr) / 2.0
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
