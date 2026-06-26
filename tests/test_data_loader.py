# -*- coding: utf-8 -*-
"""
tests/test_data_loader.py — Protocol parsing, stratified subsampling and the
two PyTorch Datasets.

All corpus structure is reproduced synthetically inside tmp_path (real FLAC/WAV
files + protocol/metadata text), so the parsers are exercised end-to-end without
the multi-GB ASVspoof download.
"""

import os

import numpy as np
import pytest
import soundfile as sf
import torch

from src.data_loader import (
    LABEL_BONAFIDE, LABEL_SPOOF, ASVspoofRawWaveDataset, ASVspoofTorchDataset,
    parse_protocol, parse_protocol_2021, stratified_subsample,
)

SR = 16_000


def _write_clip(path, freq=300.0, seconds=0.5, sr=SR):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    t = np.arange(int(seconds * sr)) / sr
    sf.write(path, (0.4 * np.sin(2 * np.pi * freq * t)).astype(np.float32), sr)


# ---------------------------------------------------------------------------
# parse_protocol (ASVspoof 2019 LA)
# ---------------------------------------------------------------------------

def _build_2019(tmp_path, subset="train", rows=None):
    """Create a minimal 2019 LA corpus tree and protocol; return (proto, root)."""
    rows = rows or [("LA_0001", "bonafide"), ("LA_0002", "spoof"),
                    ("LA_0003", "spoof")]
    root = tmp_path / "LA"
    flac_dir = root / f"ASVspoof2019_LA_{subset}" / "flac"
    for audio_id, _ in rows:
        _write_clip(str(flac_dir / f"{audio_id}.flac"))
    proto = tmp_path / "proto.txt"
    proto.write_text(
        "\n".join(f"SPK {aid} - A01 {key}" for aid, key in rows) + "\n",
        encoding="utf-8",
    )
    return str(proto), str(root)


def test_parse_protocol_labels_and_count(tmp_path):
    proto, root = _build_2019(tmp_path)
    samples = parse_protocol(proto, root, "train")
    assert len(samples) == 3
    labels = [lbl for _, lbl in samples]
    assert labels.count(LABEL_BONAFIDE) == 1
    assert labels.count(LABEL_SPOOF) == 2
    assert all(os.path.isabs(p) for p, _ in samples)


def test_parse_protocol_invalid_subset(tmp_path):
    proto, root = _build_2019(tmp_path)
    with pytest.raises(ValueError, match="not recognised"):
        parse_protocol(proto, root, "validation")


def test_parse_protocol_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        parse_protocol(str(tmp_path / "nope.txt"), str(tmp_path), "train")


def test_parse_protocol_malformed_line(tmp_path):
    _, root = _build_2019(tmp_path)
    bad = tmp_path / "bad.txt"
    bad.write_text("only three columns here\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed line"):
        parse_protocol(str(bad), root, "train")


def test_parse_protocol_unknown_key(tmp_path):
    _, root = _build_2019(tmp_path)
    bad = tmp_path / "bad.txt"
    bad.write_text("SPK LA_0001 - A01 maybe\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown key"):
        parse_protocol(str(bad), root, "train")


def test_parse_protocol_skips_missing_audio(tmp_path):
    """A protocol row whose audio file is absent is skipped, not fatal."""
    proto, root = _build_2019(tmp_path)
    with open(proto, "a", encoding="utf-8") as f:
        f.write("SPK LA_9999 - A01 spoof\n")   # listed but no file on disk
    samples = parse_protocol(proto, root, "train")
    assert len(samples) == 3                    # the ghost row is dropped


# ---------------------------------------------------------------------------
# parse_protocol_2021
# ---------------------------------------------------------------------------

def test_parse_protocol_2021(tmp_path):
    audio_root = tmp_path / "DF_part0"
    rows = [("DF_1", "bonafide"), ("DF_2", "spoof"), ("DF_3", "spoof")]
    for aid, _ in rows:
        _write_clip(str(audio_root / "flac" / f"{aid}.flac"))
    meta = tmp_path / "trial_metadata.txt"
    meta.write_text(
        "\n".join(f"SPK {aid} codec src A01 {key} extra" for aid, key in rows) + "\n",
        encoding="utf-8",
    )
    samples = parse_protocol_2021(str(meta), [str(audio_root)])
    assert len(samples) == 3
    assert sum(lbl for _, lbl in samples) == 2   # two spoof


def test_parse_protocol_2021_missing_metadata(tmp_path):
    with pytest.raises(FileNotFoundError):
        parse_protocol_2021(str(tmp_path / "nope.txt"), [str(tmp_path)])


# ---------------------------------------------------------------------------
# stratified_subsample
# ---------------------------------------------------------------------------

def _imbalanced(n_bona=10, n_spoof=90):
    return ([(f"b{i}.flac", LABEL_BONAFIDE) for i in range(n_bona)] +
            [(f"s{i}.flac", LABEL_SPOOF) for i in range(n_spoof)])


def test_stratified_subsample_preserves_ratio():
    samples = _imbalanced()                      # 10% bonafide
    out = stratified_subsample(samples, limit=20, seed=42)
    assert len(out) == 20
    n_bona = sum(1 for _, l in out if l == LABEL_BONAFIDE)
    assert n_bona == 2                            # 10% of 20, both classes kept


def test_stratified_subsample_noop_when_limit_exceeds():
    samples = _imbalanced(5, 5)
    assert stratified_subsample(samples, limit=999, seed=1) is samples
    assert stratified_subsample(samples, limit=0, seed=1) is samples


def test_stratified_subsample_is_seeded():
    samples = _imbalanced()
    a = stratified_subsample(samples, limit=20, seed=123)
    b = stratified_subsample(samples, limit=20, seed=123)
    assert a == b


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

def test_torch_dataset_spectrogram_item(tmp_path, extractor):
    path = str(tmp_path / "a.wav")
    _write_clip(path, seconds=1.0)
    ds = ASVspoofTorchDataset(
        [(path, LABEL_SPOOF)], extractor.get_spectrogram_matrix,
        extractor.sample_rate, augment=False,
    )
    assert len(ds) == 1
    tensor, label = ds[0]
    assert tensor.shape == (1, extractor.freq_bins, extractor.time_frames)
    assert tensor.dtype == torch.float32
    assert float(label) == float(LABEL_SPOOF)


def test_raw_wave_dataset_crops_long_clip(tmp_path):
    path = str(tmp_path / "long.wav")
    _write_clip(path, seconds=3.0)
    ds = ASVspoofRawWaveDataset([(path, LABEL_BONAFIDE)], SR, max_samples=SR)
    wave, label = ds[0]
    assert wave.shape == (SR,)                    # cropped to the 1 s window
    assert float(label) == float(LABEL_BONAFIDE)


def test_raw_wave_dataset_pads_short_clip(tmp_path):
    path = str(tmp_path / "short.wav")
    _write_clip(path, seconds=0.25)               # 4000 samples
    ds = ASVspoofRawWaveDataset([(path, LABEL_SPOOF)], SR, max_samples=SR)
    wave, _ = ds[0]
    assert wave.shape == (SR,)                    # zero-padded up to the window
