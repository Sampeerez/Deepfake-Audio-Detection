# -*- coding: utf-8 -*-
"""
src/features.py, Digital Signal Processing (DSP) front-ends.

Implements the full battery of feature extractors for the anti-spoofing
benchmark.  Each method takes a discrete time-domain signal ``y`` (ADC
output: sampled at 16 kHz and quantised to 16 bits, normalised to float
in [-1, 1]) and projects it to a different representation space:

    "1" RMS: temporal-domain energy (power envelope).
    "2" MFCC: perceptual spectral envelope (Mel scale).
    "3" LFCC: linear-frequency cepstrum.
    "4" DWT: multi-resolution wavelet energy.
    "6" CQCC: Constant-Q cepstral coefficients.
    STFT 2D: fixed-size spectral image for the CNN.

All physical hyperparameters (sample_rate, n_fft, hop_length …) are read
from the YAML configuration file to keep a single source of experimental
truth.
"""

import os
import subprocess
import tempfile
import warnings
from typing import Dict

import librosa
import numpy as np
import pywt
import yaml
from scipy.fft import dct

warnings.filterwarnings("ignore", message=r".*n_fft=.* is too large.*",
                        category=UserWarning)


class AudioLoadError(RuntimeError):
    """Raised when an audio file cannot be decoded by any available backend."""

_EPS: float = 1e-10


class FeatureExtractor:
    """DSP façade: encapsulates all signal processing front-ends."""

    OPTION_NAMES: Dict[str, str] = {
        "1": "RMS Temporal",
        "2": "MFCC (Mel-Cepstrum)",
        "3": "Linear LFCC (analytical)",
        "4": "Wavelet Energy DWT (db4)",
        "6": "CQCC (Constant-Q Cepstral Coefficients)",
    }

    def __init__(self, config_path: str) -> None:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        audio = config["audio"]
        self.sample_rate: int = int(audio["sample_rate"])
        self.n_fft: int = int(audio["n_fft"])
        self.hop_length: int = int(audio["hop_length"])
        self.n_mels: int = int(audio["n_mels"])
        self.n_mfcc: int = int(audio["n_mfcc"])
        self.n_lfcc: int = int(audio["n_lfcc"])
        self.n_linear_filters: int = int(audio["n_linear_filters"])
        self.wavelet_mother: str = str(audio["wavelet_mother"])

        self.cqcc_n_bins: int = int(audio.get("cqcc_n_bins", 84))
        self.cqcc_bins_per_octave: int = int(audio.get("cqcc_bins_per_octave", 12))
        self.n_cqcc: int = int(audio.get("n_cqcc", 13))

        cnn = config["cnn_input"]
        self.freq_bins: int = int(cnn["freq_bins"])
        self.time_frames: int = int(cnn["time_frames"])

        self._linear_filterbank = self._build_linear_filterbank()

    @staticmethod
    def _ffmpeg_exe() -> str:
        """Prefer the ffmpeg binary bundled by ``imageio-ffmpeg`` (always present
        as a dependency, so this works on hosts like Streamlit Cloud with no
        system ffmpeg); fall back to a system ``ffmpeg`` on PATH."""
        try:
            import imageio_ffmpeg
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return "ffmpeg"

    def _ffmpeg_load(self, path: str):
        """Decode *path* via ffmpeg subprocess. Returns float32 array or None."""
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            result = subprocess.run(
                [
                    self._ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error",
                    "-i", path,
                    "-ar", str(self.sample_rate),
                    "-ac", "1",
                    "-f", "wav",
                    tmp_path,
                ],
                capture_output=True,
                timeout=60,
            )
            if result.returncode != 0:
                return None
            signal, _ = librosa.load(tmp_path, sr=self.sample_rate, mono=True)
            return signal
        except FileNotFoundError:
            return None
        except Exception:
            return None
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def load_audio(self, path: str) -> np.ndarray:
        """Decode an audio file to a mono float32 signal at the project sample rate.

        If the original file has a different sample rate, librosa applies
        low-pass resampling before downsampling, preventing aliasing
        (spectral folding) above the new Nyquist frequency (sr/2).

        Some ASVspoof 2021 FLAC files have non-standard encoding that
        libsndfile cannot decode.  On failure a zero signal is returned
        so that batch evaluation / training can continue without crashing.
        The caller (Signal Explorer UI) should check for silence and warn
        the user explicitly.

        Raises:
            AudioLoadError: if the file cannot be decoded by any backend.
        """
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning,
                                        message="PySoundFile failed")
                warnings.filterwarnings("ignore", category=FutureWarning,
                                        message=".*audioread.*")
                signal, _ = librosa.load(path, sr=self.sample_rate, mono=True)
        except Exception as first_exc:
            signal = self._ffmpeg_load(path)
            if signal is None:
                raise AudioLoadError(
                    f"Cannot decode '{path}': {first_exc}.  "
                    "Install ffmpeg in WSL (`sudo apt install ffmpeg`) to "
                    "enable fallback decoding for all FLAC variants."
                ) from first_exc
        if len(signal) < self.n_fft:
            signal = np.pad(signal, (0, self.n_fft - len(signal)))
        return signal

    @staticmethod
    def sniff_audio_suffix(name, raw: bytes) -> str:
        """Best file suffix for an audio byte blob: trust the filename extension
        first, then the container magic bytes. Used for decode-error messages and
        to help audioread pick a decoder for temp files."""
        if name and "." in name:
            ext = "." + name.rsplit(".", 1)[1].lower()
            if ext in (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac"):
                return ext
        if raw[:4] == b"RIFF":
            return ".wav"
        if raw[:4] == b"fLaC":
            return ".flac"
        if raw[:4] == b"OggS":
            return ".ogg"
        if raw[:3] == b"ID3" or raw[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
            return ".mp3"
        if raw[4:8] == b"ftyp":
            return ".m4a"
        return ".flac"

    def load_audio_bytes(self, blob: bytes, name=None) -> np.ndarray:
        """Decode an in-memory audio blob (upload) to mono float32 at the project
        sample rate. Tries librosa/libsndfile first; falls back to piping the raw
        container through the bundled ffmpeg binary, which decodes compressed
        formats (mp3/ogg/m4a) and odd FLAC variants that libsndfile rejects.

        Raises:
            AudioLoadError: if no backend can decode the blob.
        """
        import io as _io
        try:
            signal, _ = librosa.load(_io.BytesIO(blob), sr=self.sample_rate,
                                     mono=True)
        except Exception as first_exc:
            try:
                proc = subprocess.run(
                    [self._ffmpeg_exe(), "-nostdin", "-loglevel", "quiet",
                     "-i", "pipe:0", "-f", "f32le", "-acodec", "pcm_f32le",
                     "-ac", "1", "-ar", str(self.sample_rate), "pipe:1"],
                    input=blob, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    check=True,
                )
                signal = np.frombuffer(proc.stdout, dtype=np.float32).copy()
            except Exception as exc:
                suffix = self.sniff_audio_suffix(name, blob)
                raise AudioLoadError(
                    f"unsupported or corrupt audio ({suffix}). [{exc}]"
                ) from first_exc
        if len(signal) < self.n_fft:
            signal = np.pad(signal, (0, self.n_fft - len(signal)))
        return signal

    def _stft_magnitude(self, y: np.ndarray) -> np.ndarray:
        """Short-Time Fourier Transform magnitude: |STFT(y)|.

        Hann window, 50% overlap, n_fft//2+1 frequency bins (Nyquist symmetry
        halves the spectrum of a real signal).  Shared basis for MFCC, LFCC
        and the CNN spectrogram.
        """
        spectrum = librosa.stft(
            y,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window="hann",
            center=True,
        )
        return np.abs(spectrum)

    @staticmethod
    def _mean_and_variance(matrix: np.ndarray) -> np.ndarray:
        """Collapse (coefficients × frames) to a fixed 1-D vector, mean+var per row."""
        return np.concatenate(
            [matrix.mean(axis=-1).ravel(), matrix.var(axis=-1).ravel()]
        ).astype(np.float32)

    def extract_rms(self, y: np.ndarray) -> np.ndarray:
        """Mean and variance of the per-frame Root Mean Square energy.

        RMS is the temporal power indicator: square root of the mean of
        squared samples within each frame.  It is the deliberately "weak"
        baseline of the experiment: if a 2-dimensional descriptor already
        separated bonafide from spoof, everything else would be redundant.
        It serves as a control to quantify the real gain of the spectral
        front-ends and the CNN.
        """
        rms = librosa.feature.rms(
            y=y, frame_length=self.n_fft, hop_length=self.hop_length
        )
        return self._mean_and_variance(rms)

    def extract_mfcc(self, y: np.ndarray) -> np.ndarray:
        """Mean and variance of the MFCCs (via librosa).

        Full chain: waveform, STFT, linear magnitude, log-spectra, Mel filter
        bank, DCT, MFCC.

        The Mel scale follows how the ear perceives frequency logarithmically:
        the triangular filters compress high frequencies and densify low ones.
        The DCT replaces the classic cepstrum's inverse DFT because it returns
        real coefficients, decorrelates energy across Mel bands (which helps the
        ML classifiers) and compresses by keeping only the first n_mfcc values.

        Cepstral view: speech is the convolution of the glottal pulse with the
        vocal-tract response; in the log-spectral domain that becomes a sum, so
        keeping the low cepstral coefficients deconvolves the vocal-tract
        envelope and drops the fine excitation structure.

        The limitation for deepfakes: by modelling only coarse spectral
        structure and compressing high frequencies, MFCCs erase the fine detail
        and high bands where neural vocoders leave their artefacts. That is what
        motivates the LFCC below.
        """
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            window="hann",
        )
        return self._mean_and_variance(mfcc)

    def _build_linear_filterbank(self) -> np.ndarray:
        """Build triangular filters UNIFORMLY spaced in Hz (0 → Nyquist).

        Unlike the Mel bank, equal resolution everywhere preserves the
        high-frequency band where AI vocoders leave their artefacts.

        Returns:
            Matrix (n_linear_filters, n_fft//2 + 1) for dot-product with
            the per-frame power spectrum.
        """
        nyquist = self.sample_rate / 2.0
        n_bins = self.n_fft // 2 + 1

        freq_hz = np.linspace(0.0, nyquist, self.n_linear_filters + 2)
        bin_idx = np.floor(
            (self.n_fft + 1) * freq_hz / self.sample_rate
        ).astype(int)
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)

        bank = np.zeros((self.n_linear_filters, n_bins), dtype=np.float64)
        for m in range(1, self.n_linear_filters + 1):
            left, centre, right = bin_idx[m - 1], bin_idx[m], bin_idx[m + 1]
            for k in range(left, centre):
                bank[m - 1, k] = (k - left) / max(centre - left, 1)
            for k in range(centre, right):
                bank[m - 1, k] = (right - k) / max(right - centre, 1)
        return bank

    def extract_lfcc(self, y: np.ndarray) -> np.ndarray:
        """Analytical LFCC where every step is explicit.

        Pipeline: STFT, power spectrum, linear filter bank, log, orthonormal
        DCT-II, first n_lfcc coefficients.

        Compared with MFCC it keeps the same cepstral skeleton (log + DCT, with
        its deconvolving and decorrelating effect) but on a uniform-resolution
        filter bank, so the high-frequency band stays intact, where TTS/VC
        models leave quantisation noise, band cuts and spurious harmonic grids
        that betray the deepfake.
        """
        power = self._stft_magnitude(y) ** 2

        band_energy = self._linear_filterbank @ power

        log_energy = np.log(band_energy + _EPS)

        cepstrum = dct(log_energy, type=2, axis=0, norm="ortho")[: self.n_lfcc]

        return self._mean_and_variance(cepstrum)

    def extract_wavelet_energy(self, y: np.ndarray) -> np.ndarray:
        """Mean and variance of the DWT coefficient energy (Daubechies-4).

        Motivation: with a fixed STFT window (n_fft) the time-frequency
        uncertainty principle locks temporal and frequency resolution for every
        band at once. Wavelets get around this with multi-resolution analysis:
        instead of infinite sinusoids, the signal is correlated with scaled and
        shifted copies of a compact localised function (the mother wavelet, here
        Daubechies-4). Small scales (high frequencies) give sharp temporal
        precision, good for the fast transients, clicks and brief cuts that AI
        generators leave between frames; large scales (low frequencies) give
        sharp frequency precision, capturing pitch and prosody.

        A one-level DWT (pywt.dwt) is a quadrature mirror filter bank plus
        factor-2 downsampling: cA (approximation) is the low band [0, sr/4] =
        [0, 4 kHz], cD (detail) the high band [sr/4, sr/2] = [4, 8 kHz]. The
        descriptor takes the pointwise energy of each coefficient (coeff²,
        energy-preserving by Parseval) as mean and variance per sub-band, a
        4-dimensional vector.
        """
        approx_coeffs, detail_coeffs = pywt.dwt(y, self.wavelet_mother)

        approx_energy = approx_coeffs.astype(np.float64) ** 2
        detail_energy = detail_coeffs.astype(np.float64) ** 2

        return np.array(
            [
                approx_energy.mean(),
                approx_energy.var(),
                detail_energy.mean(),
                detail_energy.var(),
            ],
            dtype=np.float32,
        )

    def extract_cqcc(self, y: np.ndarray) -> np.ndarray:
        """Analytical CQCC: CQT, log-energy, DCT, aggregation.

        The Constant-Q Transform acts as a filter bank with logarithmic
        frequency resolution; from the per-bin energy, log compression and an
        orthonormal DCT-II give the cepstral CQCC coefficients. Returns mean
        and variance.
        """
        cqt = librosa.cqt(
            y,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            n_bins=self.cqcc_n_bins,
            bins_per_octave=self.cqcc_bins_per_octave,
        )
        magnitude = np.abs(cqt)

        power = magnitude ** 2
        log_energy = np.log(power + _EPS)
        cepstrum = dct(log_energy, type=2, axis=0, norm="ortho")[: self.n_cqcc]

        return self._mean_and_variance(cepstrum)

    def get_flat_vector(self, y: np.ndarray, feature_choice: str) -> np.ndarray:
        """Return the 1-D descriptor vector for the given menu option.

        Options: "1" RMS, "2" MFCC, "3" LFCC, "4" DWT, "6" CQCC.
        """
        if feature_choice == "1":
            vector = self.extract_rms(y)
        elif feature_choice == "2":
            vector = self.extract_mfcc(y)
        elif feature_choice == "3":
            vector = self.extract_lfcc(y)
        elif feature_choice == "4":
            vector = self.extract_wavelet_energy(y)
        elif feature_choice == "6":
            vector = self.extract_cqcc(y)
        else:
            raise ValueError(
                f"Feature choice '{feature_choice}' is not valid "
                f"(expected '1'-'4' or '6')."
            )

        return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)

    def get_spectrogram_matrix(self, y: np.ndarray) -> np.ndarray:
        """STFT-dB spectrogram with a strict shape (freq_bins × time_frames).

        The CNN treats the spectrogram as a single-channel image and PyTorch
        batches need a constant size, so the shape is pinned here (default
        128 × 300).

        Steps:
        1) |STFT| with a Hann window (see :meth:`_stft_magnitude`): 513 bins.
        2) Conversion to decibels: compresses the huge dynamic range of the
           linear magnitude to a stable range for gradients (20*log10, floored
           at -80 dB).
        3) Frequency axis: the exact Nyquist bin (513) is dropped and the
           remaining 512 bins are averaged in groups of 4 down to ``freq_bins``.
           Unlike truncation, this preserves the full 0 to 8 kHz band (the high
           band that betrays vocoders) at the cost of fine frequency resolution.
        4) Time axis: truncated to ``time_frames`` or padded with the dB floor
           (padding with silence, not spurious energy).
        5) Per-utterance z-score: zero mean / unit std stabilises gradient
           descent and cooperates with BatchNorm.
        """
        magnitude = self._stft_magnitude(y)

        ref = float(magnitude.max())
        if ref <= 0.0:
            ref = 1.0
        matrix_db = librosa.amplitude_to_db(magnitude, ref=ref, top_db=80.0)

        matrix_db = matrix_db[:-1, :]
        n_total, n_frames = matrix_db.shape
        factor = n_total // self.freq_bins
        matrix_db = (
            matrix_db[: factor * self.freq_bins, :]
            .reshape(self.freq_bins, factor, n_frames)
            .mean(axis=1)
        )

        floor_db = float(matrix_db.min())
        if n_frames < self.time_frames:
            pad = np.full(
                (self.freq_bins, self.time_frames - n_frames),
                floor_db,
                dtype=matrix_db.dtype,
            )
            matrix_db = np.concatenate([matrix_db, pad], axis=1)
        else:
            matrix_db = matrix_db[:, : self.time_frames]

        mean = matrix_db.mean()
        std = matrix_db.std()
        matrix_db = (matrix_db - mean) / (std + _EPS)

        return matrix_db.astype(np.float32)
