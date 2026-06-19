# -*- coding: utf-8 -*-
"""
src/features.py — Digital Signal Processing (DSP) front-ends.

Implements the full battery of feature extractors for the anti-spoofing
benchmark.  Each method takes a discrete time-domain signal ``y`` (ADC
output: sampled at 16 kHz and quantised to 16 bits, normalised to float
in [-1, 1]) and projects it to a different representation space:

    1) RMS       -> temporal-domain energy (power envelope).
    2) MFCC      -> perceptual spectral envelope (Mel scale).
    3) LFCC      -> LINEAR-frequency cepstrum (analytical, no black boxes).
    4) DWT       -> multi-resolution wavelet energy analysis.
    5) Fusion    -> early fusion of ALL 1-D descriptors, CQCC included
                    (concatenation of RMS + MFCC + LFCC + DWT + CQCC).
    6) CQCC      -> Constant-Q Cepstral Coefficients.
    7) STFT 2D   -> fixed-size spectral "image" for the CNN.

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

# Very short utterances (some 2021 DF clips are < 256 samples) make librosa's
# CQT/STFT warn "n_fft=… is too large for input signal of length=…". The result
# is still valid (padded internally); silence the noise.
warnings.filterwarnings("ignore", message=r".*n_fft=.* is too large.*",
                        category=UserWarning)


class AudioLoadError(RuntimeError):
    """Raised when an audio file cannot be decoded by any available backend."""

# Numerical epsilon: prevents log(0) -> -inf on silent frames.
_EPS: float = 1e-10


class FeatureExtractor:
    """DSP façade: encapsulates all signal processing front-ends."""

    # Maps the feature-extractor option key -> human-readable name.
    OPTION_NAMES: Dict[str, str] = {
        "1": "RMS Temporal",
        "2": "MFCC (Mel-Cepstrum)",
        "3": "Linear LFCC (analytical)",
        "4": "Wavelet Energy DWT (db4)",
        "5": "Full Fusion (RMS+MFCC+LFCC+DWT+CQCC)",
        "6": "CQCC (Constant-Q Cepstral Coefficients)",
    }

    def __init__(self, config_path: str) -> None:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        audio = config["audio"]
        self.sample_rate:    int = int(audio["sample_rate"])
        self.n_fft:          int = int(audio["n_fft"])
        self.hop_length:     int = int(audio["hop_length"])
        self.n_mels:         int = int(audio["n_mels"])
        self.n_mfcc:         int = int(audio["n_mfcc"])
        self.n_lfcc:         int = int(audio["n_lfcc"])
        self.n_linear_filters: int = int(audio["n_filtros_lineales"])
        self.wavelet_mother: str = str(audio["wavelet_mother"])

        # CQCC parameters (defaults overridable from YAML).
        self.cqcc_n_bins:        int = int(audio.get("cqcc_n_bins", 84))
        self.cqcc_bins_per_octave: int = int(audio.get("cqcc_bins_per_octave", 12))
        self.n_cqcc:             int = int(audio.get("n_cqcc", 13))

        cnn = config["cnn_input"]
        self.freq_bins:   int = int(cnn["freq_bins"])
        self.time_frames: int = int(cnn["time_frames"])

        # The linear LFCC filter bank is constant for the entire corpus:
        # pre-computed ONCE here so the per-audio cost is amortised to zero.
        self._linear_filterbank = self._build_linear_filterbank()

    # ------------------------------------------------------------------ #
    # Internal utilities
    # ------------------------------------------------------------------ #
    def _ffmpeg_load(self, path: str):
        """Decode *path* via ffmpeg subprocess. Returns float32 array or None."""
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            result = subprocess.run(
                [
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
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
            return None  # ffmpeg not installed
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
            # libsndfile + audioread both failed (common with ASVspoof 2021
            # FLAC files encoded with libFLAC >= 1.4 on WSL without ffmpeg).
            # Try converting via ffmpeg subprocess if it is installed.
            signal = self._ffmpeg_load(path)
            if signal is None:
                raise AudioLoadError(
                    f"Cannot decode '{path}': {first_exc}.  "
                    "Install ffmpeg in WSL (`sudo apt install ffmpeg`) to "
                    "enable fallback decoding for all FLAC variants."
                ) from first_exc
        # Ensure minimum length == n_fft so STFT and CQT never receive a
        # signal shorter than their analysis window (avoids UserWarning).
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
        # Complex coefficients -> magnitude (phase discarded: the auditory
        # system is nearly phase-deaf for spectral envelope estimation).
        return np.abs(spectrum)

    @staticmethod
    def _mean_and_variance(matrix: np.ndarray) -> np.ndarray:
        """Collapse (coefficients × frames) -> fixed 1-D vector via mean+var per row."""
        return np.concatenate(
            [matrix.mean(axis=-1).ravel(), matrix.var(axis=-1).ravel()]
        ).astype(np.float32)

    # ------------------------------------------------------------------ #
    # 1) RMS — temporal domain
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    # 2) MFCC — perceptual cepstrum on the Mel scale
    # ------------------------------------------------------------------ #
    def extract_mfcc(self, y: np.ndarray) -> np.ndarray:
        """Mean and variance of the MFCCs (via librosa).

        Full chain: Waveform -> STFT(FFT) -> linear magnitude ->
        log-spectra -> Mel filter bank -> DCT -> MFCC.

        * Mel scale: the ear perceives frequency logarithmically; Mel
          triangular filters compress high frequencies and densify low
          ones to emulate that perception.
        * DCT (Discrete Cosine Transform): replaces the IDFT of the
          classic cepstrum because (a) it returns purely REAL coefficients,
          (b) it DECORRELATES energy across Mel bands (benefiting ML
          classifiers), and (c) it compresses information by keeping only
          the first n_mfcc coefficients.
        * Cepstral insight: speech is the CONVOLUTION of the glottal pulse
          with the vocal tract frequency response.  In the log-spectral
          domain that convolution becomes a SUM, and retaining only the low
          cepstral coefficients is equivalent to a DE-CONVOLUTION that
          isolates the vocal tract envelope, discarding fine excitation
          structure.

        DECISIVE LIMITATION FOR DEEPFAKES: by modelling only large spectral
        structures and compressing high frequencies, MFCCs erase precisely
        the fine structure and high-frequency bands where neural vocoders
        leave their digital artefacts and synthetic quantisation noise.
        This limitation motivates the LFCC below.
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

    # ------------------------------------------------------------------ #
    # 3) LFCC — LINEAR cepstrum, implemented analytically
    # ------------------------------------------------------------------ #
    def _build_linear_filterbank(self) -> np.ndarray:
        """Build triangular filters UNIFORMLY spaced in Hz (0 → Nyquist).

        Unlike the Mel bank, equal resolution everywhere preserves the
        high-frequency band where AI vocoders leave their artefacts.

        Returns:
            Matrix (n_linear_filters, n_fft//2 + 1) for dot-product with
            the per-frame power spectrum.
        """
        nyquist = self.sample_rate / 2.0
        n_bins  = self.n_fft // 2 + 1

        # n_filters + 2 anchor points: each triangle needs left/centre/right.
        freq_hz  = np.linspace(0.0, nyquist, self.n_linear_filters + 2)
        bin_idx  = np.floor(
            (self.n_fft + 1) * freq_hz / self.sample_rate
        ).astype(int)
        bin_idx  = np.clip(bin_idx, 0, n_bins - 1)

        bank = np.zeros((self.n_linear_filters, n_bins), dtype=np.float64)
        for m in range(1, self.n_linear_filters + 1):
            left, centre, right = bin_idx[m - 1], bin_idx[m], bin_idx[m + 1]
            for k in range(left, centre):
                bank[m - 1, k] = (k - left) / max(centre - left, 1)
            for k in range(centre, right):
                bank[m - 1, k] = (right - k) / max(right - centre, 1)
        return bank

    def extract_lfcc(self, y: np.ndarray) -> np.ndarray:
        """Analytical LFCC with no black boxes: every step is explicit.

        Pipeline: STFT -> |·|² (power) -> LINEAR filter bank ->
        log -> orthonormal DCT-II -> first n_lfcc coefficients.

        Rationale over MFCC: the cepstral skeleton is preserved (log + DCT,
        with its de-convolving and decorrelating effect) but with a UNIFORM
        frequency-resolution filter bank.  This keeps the high-frequency
        band intact — where TTS/VC generative models introduce quantisation
        noise, band cuts, and spurious harmonic grids that betray the
        deepfake.
        """
        # 1–2) Power spectrum: |STFT|² frame by frame.
        power = self._stft_magnitude(y) ** 2

        # 3) Integrate energy per linear band (matrix product: each bank
        #    row weights the bins of its triangular filter).
        band_energy = self._linear_filterbank @ power

        # 4) Logarithmic compression: converts the source-filter convolution
        #    of speech into a separable sum (prerequisite of cepstral
        #    analysis) and approximates the logarithmic loudness perception.
        log_energy = np.log(band_energy + _EPS)

        # 5) Orthonormal DCT-II over the filter-bank axis: decorrelates
        #    inter-band energy, returns real coefficients, and compacts
        #    information into the leading terms (truncated to n_lfcc —
        #    the "liftering" that retains the spectral envelope).
        cepstrum = dct(log_energy, type=2, axis=0, norm="ortho")[: self.n_lfcc]

        return self._mean_and_variance(cepstrum)

    # ------------------------------------------------------------------ #
    # 4) DWT — multi-resolution wavelet analysis
    # ------------------------------------------------------------------ #
    def extract_wavelet_energy(self, y: np.ndarray) -> np.ndarray:
        """Mean and variance of the DWT coefficient energy (Daubechies-4).

        MATHEMATICAL MOTIVATION: the STFT suffers the time-frequency
        uncertainty principle — once the window (n_fft) is fixed, temporal
        and frequency resolution are simultaneously locked for ALL bands.
        Wavelets overcome this with MULTI-RESOLUTION analysis: instead of
        infinite sinusoids, the signal is correlated with scaled and shifted
        versions of a compact, localised function (the mother wavelet, here
        Daubechies-4):

          * Small scales (high frequencies) -> excellent TEMPORAL precision:
            ideal for detecting fast transients, clicks, and brief cuts
            that AI generators leave between frames.
          * Large scales (low frequencies) -> excellent FREQUENCY precision:
            captures pitch and prosody with fine resolution.

        One-level DWT (pywt.dwt) is equivalent to a quadrature mirror filter
        bank + factor-2 downsampling:
          * cA (approximation): low band  [0, sr/4] = [0, 4 kHz].
          * cD (detail):        high band [sr/4, sr/2] = [4, 8 kHz].

        The descriptor uses the POINTWISE ENERGY of each coefficient
        (coeff², energy-preserving by Parseval's theorem) aggregated as
        mean and variance per sub-band -> 4-dimensional vector.
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

    # ------------------------------------------------------------------ #
    # 5) CQCC — Constant-Q Cepstral Coefficients
    # ------------------------------------------------------------------ #
    def extract_cqcc(self, y: np.ndarray) -> np.ndarray:
        """Analytical CQCC: CQT -> log-energy -> DCT -> aggregation.

        The Constant-Q Transform (CQT) is used as a filter bank with
        logarithmic frequency resolution; after obtaining per-bin energy,
        log compression and orthonormal DCT-II are applied to extract the
        cepstral CQCC coefficients.  Mean and variance are returned.
        """
        # CQT: returns complex coefficients (bins x frames).
        cqt = librosa.cqt(
            y,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            n_bins=self.cqcc_n_bins,
            bins_per_octave=self.cqcc_bins_per_octave,
        )
        magnitude = np.abs(cqt)

        power      = magnitude ** 2
        log_energy = np.log(power + _EPS)
        cepstrum   = dct(log_energy, type=2, axis=0, norm="ortho")[: self.n_cqcc]

        return self._mean_and_variance(cepstrum)

    # ------------------------------------------------------------------ #
    # 1-D vector dispatcher (classic models)
    # ------------------------------------------------------------------ #
    def get_flat_vector(self, y: np.ndarray, feature_choice: str) -> np.ndarray:
        """Return the 1-D descriptor vector for the given menu option.

        Options: "1" RMS | "2" MFCC | "3" LFCC | "4" DWT | "5" fusion of
        ALL descriptors (early concatenation, CQCC included) | "6" CQCC.

        Fusion lets the classifier weight simultaneously temporal energy,
        perceptual envelope, high-band linear detail, multi-resolution
        signature and the constant-Q cepstrum in a single vector.
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
        elif feature_choice == "5":
            vector = np.concatenate([
                self.extract_rms(y),
                self.extract_mfcc(y),
                self.extract_lfcc(y),
                self.extract_wavelet_energy(y),
                self.extract_cqcc(y),
            ])
        else:
            raise ValueError(
                f"Feature choice '{feature_choice}' is not valid "
                f"(expected '1'–'6')."
            )

        # Final numeric sanitisation: no NaN/inf should reach the classifier
        # (e.g. variance of a completely silent audio file).
        return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)

    # ------------------------------------------------------------------ #
    # 2-D representation for the CNN
    # ------------------------------------------------------------------ #
    def get_spectrogram_matrix(self, y: np.ndarray) -> np.ndarray:
        """STFT-dB spectrogram with STRICT dimensions (freq_bins × time_frames).

        The CNN treats the spectrogram as a single-channel image; PyTorch
        batches require constant size, so a rigid dimensional contract is
        enforced here (default 128 × 300).

        Steps:
        1) |STFT| with Hann window (see :meth:`_stft_magnitude`): 513 bins.
        2) Conversion to decibels (log scale): compresses the enormous
           dynamic range of linear magnitude to a stable range for gradients
           (20*log10, floored at −80 dB).
        3) Frequency axis: the exact Nyquist bin (513) is discarded and the
           remaining 512 bins are compressed to ``freq_bins`` by block
           averaging groups of 4.  Unlike truncation, this averaging
           PRESERVES the full 0–8 kHz bandwidth (the high band that betrays
           vocoders) at the cost of fine frequency resolution.
        4) Time axis: truncated to ``time_frames`` or padded with the dB
           floor (equivalent to padding with silence, not spurious energy).
        5) Per-utterance z-score normalisation: zero mean / unit std
           stabilises gradient descent and cooperates with BatchNorm.
        """
        magnitude = self._stft_magnitude(y)   # (n_fft//2 + 1, T) = (513, T)

        ref = float(magnitude.max())
        if ref <= 0.0:
            ref = 1.0   # Completely silent audio: avoid log(0).
        matrix_db = librosa.amplitude_to_db(magnitude, ref=ref, top_db=80.0)

        # --- Frequency axis: 513 -> 512 (drop Nyquist bin) -> freq_bins ---
        matrix_db = matrix_db[:-1, :]
        n_total, n_frames = matrix_db.shape
        factor    = n_total // self.freq_bins           # 512 // 128 = 4
        matrix_db = (
            matrix_db[: factor * self.freq_bins, :]
            .reshape(self.freq_bins, factor, n_frames)
            .mean(axis=1)
        )

        # --- Time axis: strict padding / truncation to time_frames ---
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

        # --- Per-utterance z-score normalisation ---
        mean = matrix_db.mean()
        std  = matrix_db.std()
        matrix_db = (matrix_db - mean) / (std + _EPS)

        return matrix_db.astype(np.float32)
