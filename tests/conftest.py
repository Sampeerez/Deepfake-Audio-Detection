# -*- coding: utf-8 -*-
"""
tests/conftest.py — Shared fixtures for the whole test suite.

Everything here is synthetic: not a single test needs the multi-GB ASVspoof
corpus or a GPU, so the suite runs anywhere (including CI) in seconds.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

# Make `src` importable without installing the project.
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.features import FeatureExtractor  # noqa: E402

CONFIG_PATH = str(ROOT / "config" / "config.yaml")
SAMPLE_RATE = 16_000  # Hz, matches config.yaml


# ---------------------------------------------------------------------------
# Core fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def config_path():
    """Absolute path to the project's config.yaml."""
    return CONFIG_PATH


@pytest.fixture(scope="session")
def extractor():
    """A single FeatureExtractor reused across the session (it is stateless
    after construction, so sharing it is safe and avoids rebuilding the LFCC
    filter bank for every test)."""
    return FeatureExtractor(CONFIG_PATH)


@pytest.fixture
def signal_1s():
    """440 Hz sine wave, 1 second at 16 kHz — a clean, deterministic tone."""
    t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    return (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


@pytest.fixture
def white_noise():
    """1 second of low-amplitude white noise (broadband, every band excited)."""
    rng = np.random.default_rng(7)
    return (0.1 * rng.standard_normal(SAMPLE_RATE)).astype(np.float32)


@pytest.fixture
def silence():
    """1 second of absolute silence — the degenerate edge case."""
    return np.zeros(SAMPLE_RATE, dtype=np.float32)


# ---------------------------------------------------------------------------
# Synthetic audio-file factories (write real files to tmp_path)
# ---------------------------------------------------------------------------

@pytest.fixture
def make_audio(tmp_path):
    """Factory writing a real audio file to disk and returning its path.

    Usage:  path = make_audio("clip.flac", freq=300, seconds=1.0)
    """
    def _make(name="clip.wav", seconds=1.0, freq=440.0, sr=SAMPLE_RATE):
        t = np.arange(int(seconds * sr)) / sr
        sig = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        path = tmp_path / name
        sf.write(str(path), sig, sr)
        return str(path)
    return _make


@pytest.fixture
def labelled_samples(tmp_path):
    """Build N real audio files with alternating labels and return the
    ``[(path, label), ...]`` list the pipeline/datasets consume.

    Both classes are always present (EER/minDCF need it). A couple of distinct
    tone frequencies per class give the classifiers something separable.
    """
    def _make(n=8, sr=SAMPLE_RATE, seconds=1.0):
        samples = []
        for i in range(n):
            label = i % 2                       # 0,1,0,1,…  → both classes
            freq = 220.0 if label == 0 else 660.0
            t = np.arange(int(seconds * sr)) / sr
            sig = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
            path = tmp_path / f"audio_{i:03d}.wav"
            sf.write(str(path), sig, sr)
            samples.append((str(path), label))
        return samples
    return _make
