# -*- coding: utf-8 -*-
"""
tests/test_features.py, DSP front-ends (FeatureExtractor).

Validates the output contract of every extractor: exact dimensionality, finite
values, the strict deep network spectrogram shape and its
z-score normalisation, plus the audio-loading padding guarantees.
"""

import numpy as np
import pytest
import soundfile as sf



def test_rms_shape(extractor, signal_1s):
    v = extractor.extract_rms(signal_1s)
    assert v.shape == (2,)
    assert np.isfinite(v).all()


def test_mfcc_shape(extractor, signal_1s):
    v = extractor.extract_mfcc(signal_1s)
    assert v.shape == (40,)
    assert np.isfinite(v).all()


def test_lfcc_shape(extractor, signal_1s):
    v = extractor.extract_lfcc(signal_1s)
    assert v.shape == (40,)
    assert np.isfinite(v).all()


def test_dwt_shape(extractor, signal_1s):
    v = extractor.extract_wavelet_energy(signal_1s)
    assert v.shape == (4,)
    assert np.isfinite(v).all()


def test_cqcc_shape(extractor, signal_1s):
    v = extractor.extract_cqcc(signal_1s)
    assert v.shape == (26,)
    assert np.isfinite(v).all()


def test_dtype_is_float32(extractor, signal_1s):
    """Downstream models assume float32 feature vectors."""
    assert extractor.extract_mfcc(signal_1s).dtype == np.float32



@pytest.mark.parametrize("choice,dim", [
    ("1", 2), ("2", 40), ("3", 40), ("4", 4), ("6", 26),
])
def test_flat_vector_dimensions(extractor, signal_1s, choice, dim):
    v = extractor.get_flat_vector(signal_1s, choice)
    assert v.shape == (dim,)
    assert np.isfinite(v).all()


def test_invalid_extractor_choice(extractor, signal_1s):
    with pytest.raises(ValueError, match="not valid"):
        extractor.get_flat_vector(signal_1s, "9")


@pytest.mark.parametrize("choice", ["1", "2", "3", "4", "6"])
def test_no_nan_on_silence(extractor, silence, choice):
    """Absolute silence (zero variance) must never leak NaN/inf to a model."""
    v = extractor.get_flat_vector(silence, choice)
    assert np.isfinite(v).all()


def test_extraction_is_deterministic(extractor, signal_1s):
    """The same signal must always produce the same vector (no hidden RNG)."""
    a = extractor.get_flat_vector(signal_1s, "6")
    b = extractor.get_flat_vector(signal_1s, "6")
    assert np.array_equal(a, b)



def test_spectrogram_strict_shape(extractor, signal_1s):
    m = extractor.get_spectrogram_matrix(signal_1s)
    assert m.shape == (extractor.freq_bins, extractor.time_frames)
    assert m.dtype == np.float32
    assert np.isfinite(m).all()


def test_spectrogram_is_zscore_normalised(extractor, white_noise):
    """Per-utterance z-score: mean ≈ 0 and std ≈ 1 across the matrix."""
    m = extractor.get_spectrogram_matrix(white_noise)
    assert m.mean() == pytest.approx(0.0, abs=1e-3)
    assert m.std() == pytest.approx(1.0, abs=1e-2)


def test_spectrogram_pads_short_signal(extractor):
    """A signal far shorter than time_frames must be padded, not truncated."""
    short = np.zeros(extractor.n_fft, dtype=np.float32)
    m = extractor.get_spectrogram_matrix(short)
    assert m.shape == (extractor.freq_bins, extractor.time_frames)



def test_load_audio_pads_short_signal(extractor, tmp_path):
    """load_audio must zero-pad signals shorter than n_fft."""
    short = np.zeros(100, dtype=np.float32)
    path = str(tmp_path / "short.wav")
    sf.write(path, short, extractor.sample_rate)
    assert len(extractor.load_audio(path)) >= extractor.n_fft


def test_load_audio_preserves_normal_length(extractor, tmp_path):
    """A 1-second clip must be returned unaltered in length."""
    t = np.arange(extractor.sample_rate) / extractor.sample_rate
    sig = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    path = str(tmp_path / "normal.wav")
    sf.write(path, sig, extractor.sample_rate)
    assert len(extractor.load_audio(path)) == extractor.sample_rate


def test_load_audio_resamples_to_project_rate(extractor, tmp_path):
    """A 44.1 kHz file must be resampled to the project's 16 kHz."""
    sr_in = 44_100
    t = np.arange(sr_in) / sr_in
    sig = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    path = str(tmp_path / "hi.wav")
    sf.write(path, sig, sr_in)
    out = extractor.load_audio(path)
    assert abs(len(out) - extractor.sample_rate) < 200


def test_config_values_loaded(extractor):
    """Sanity-check the parameters parsed from config.yaml."""
    assert extractor.sample_rate == 16_000
    assert extractor.n_fft == 1024
    assert extractor.hop_length == 512
    assert extractor.freq_bins == 128
    assert extractor.time_frames == 300
