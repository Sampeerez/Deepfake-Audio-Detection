# -*- coding: utf-8 -*-
"""
src/data_loader.py — Data access layer for the ASVspoof 2019 LA corpus.

Single responsibility: translate the official ASVspoof protocol files into
Python structures usable by the rest of the system, and expose a PyTorch
Dataset fully decoupled from feature extraction (injected as a callback).

Official protocol format — 5 space-separated columns per line:

    SPEAKER_ID  AUDIO_FILE  -  ATTACK_SYSTEM  KEY
    LA_0079     LA_T_1138215   -  -               bonafide
    LA_0079     LA_T_1271820   -  A01             spoof

Binary label convention used throughout this project:
    0 -> bonafide (real human speech)
    1 -> spoof    (deepfake / synthesised or voice-converted utterance)
"""

import os
import random
from typing import Callable, Dict, List, Tuple

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset

# Canonical binary labels (immutable project-wide constants).
LABEL_BONAFIDE: int = 0
LABEL_SPOOF:    int = 1


def _spec_augment(
    matrix: np.ndarray,
    time_mask_max: int = 40,
    freq_mask_max: int = 15,
    n_time_masks: int = 2,
    n_freq_masks: int = 2,
) -> np.ndarray:
    """SpecAugment: randomly zero out time and frequency strips.

    Park et al. (2019) showed that masking contiguous frames/bins during
    training acts as strong regularisation for spectrogram-based models —
    the network must learn to ignore any single region and aggregate evidence
    globally.  Parameters are conservative (< 15% of each axis) to preserve
    enough signal for convergence on small subsets.

    Args:
        matrix:        2-D float32 spectrogram of shape (freq_bins, time_frames).
        time_mask_max: Maximum width of each time mask (frames).
        freq_mask_max: Maximum height of each frequency mask (bins).
        n_time_masks:  Number of independent time masks to apply.
        n_freq_masks:  Number of independent frequency masks to apply.

    Returns:
        Augmented copy of ``matrix`` (original is not modified).
    """
    m = matrix.copy()
    freq_bins, time_frames = m.shape
    fill = float(m.mean())
    rng  = np.random.default_rng()   # OS-seeded → truly random per sample

    for _ in range(n_time_masks):
        t  = int(rng.integers(0, time_mask_max + 1))
        t0 = int(rng.integers(0, max(1, time_frames - t)))
        m[:, t0: t0 + t] = fill

    for _ in range(n_freq_masks):
        f  = int(rng.integers(0, freq_mask_max + 1))
        f0 = int(rng.integers(0, max(1, freq_bins - f)))
        m[f0: f0 + f, :] = fill

    return m


def parse_protocol(
    protocol_path: str,
    dataset_dir: str,
    subset: str,
) -> List[Tuple[str, int]]:
    """Parse an official ASVspoof 2019 LA protocol file.

    Args:
        protocol_path: Path to the protocol file
            (e.g. ``ASVspoof2019.LA.cm.train.trn.txt``).
        dataset_dir: Root directory of the LA corpus.  Audio subdirectories
            ``ASVspoof2019_LA_train/flac``, etc. hang from here.
        subset: Split to resolve: ``"train"``, ``"dev"`` or ``"eval"``.

    Returns:
        List of ``(absolute_audio_path, binary_label)`` tuples.

    Raises:
        FileNotFoundError: if the protocol file does not exist on disk.
        ValueError: if the subset is invalid or the protocol is malformed.
    """
    valid_subsets = ("train", "dev", "eval")
    if subset not in valid_subsets:
        raise ValueError(
            f"Subset '{subset}' not recognised.  Use one of {valid_subsets}."
        )

    if not os.path.isfile(protocol_path):
        raise FileNotFoundError(
            f"Protocol file '{protocol_path}' not found.  Check "
            f"'dataset.path_la2019' in config.yaml and verify that the "
            f"ASVspoof 2019 LA corpus is correctly extracted."
        )

    # LA corpus audio is distributed in FLAC (lossless PCM compression that
    # preserves vocoder quantisation artefacts crucial for detection).
    audio_dir = os.path.join(dataset_dir, f"ASVspoof2019_LA_{subset}", "flac")

    samples: List[Tuple[str, int]] = []
    skipped = 0

    with open(protocol_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            fields = line.strip().split()
            if not fields:
                continue
            if len(fields) < 5:
                raise ValueError(
                    f"Malformed line {line_no} in protocol "
                    f"(expected 5 columns): '{line.strip()}'"
                )

            # fields = [speaker_id, audio_id, env, attack_id, key]
            audio_id, key = fields[1], fields[4].lower()

            if key == "bonafide":
                label = LABEL_BONAFIDE
            elif key == "spoof":
                label = LABEL_SPOOF
            else:
                raise ValueError(
                    f"Unknown key '{key}' on line {line_no}.  "
                    f"Only 'bonafide' and 'spoof' are accepted."
                )

            abs_path = os.path.abspath(
                os.path.join(audio_dir, f"{audio_id}.flac")
            )

            # Robustness: skip missing files instead of aborting the whole
            # experiment (useful with partially downloaded corpora).
            if not os.path.isfile(abs_path):
                skipped += 1
                continue

            samples.append((abs_path, label))

    if skipped > 0:
        print(f"[WARNING] {skipped} audio files listed in the protocol "
              f"do not exist under '{audio_dir}' and were skipped.")

    if not samples:
        raise FileNotFoundError(
            f"Protocol parsed but NO audio files exist under "
            f"'{audio_dir}'.  Check the corpus directory structure."
        )

    n_bonafide = sum(1 for _, e in samples if e == LABEL_BONAFIDE)
    n_spoof    = len(samples) - n_bonafide
    print(f"[DATA] Subset '{subset}': {len(samples)} audio files "
          f"({n_bonafide} bonafide / {n_spoof} spoof).")
    return samples


def parse_protocol_2021(
    metadata_path: str,
    audio_dirs: List[str],
) -> List[Tuple[str, int]]:
    """Parse an ASVspoof 2021 trial_metadata.txt and locate audio on disk.

    Two-pass design for efficiency with large corpora (181 k LA / 459 k DF):
      1. Read the metadata file once into a dict {audio_id → label}.
      2. Scan each ``{audio_dir}/flac/`` directory with os.scandir() — a single
         syscall per directory — and look up each filename in the dict (O(1)).

    This avoids the O(n_metadata × n_dirs) os.path.isfile() calls that
    would be needed if we iterated over metadata rows and searched the
    directories for each one.

    Metadata line format (both LA and DF, whitespace-separated):
        SPEAKER_ID  AUDIO_ID  CODEC  SOURCE  ATTACK_ID  LABEL  ...
    ``LABEL`` is always at index 5 (bonafide / spoof).

    Args:
        metadata_path: Path to the ``trial_metadata.txt`` key file.
        audio_dirs:    Root directories each containing a ``flac/``
                       subdirectory.  Pass all partition roots for DF.

    Returns:
        List of ``(absolute_audio_path, binary_label)`` tuples.
    """
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(
            f"ASVspoof 2021 metadata not found: '{metadata_path}'.  "
            f"Check 'dataset_2021' paths in config.yaml."
        )

    # Pass 1 — build label lookup.
    label_map: Dict[str, int] = {}
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            fields = line.strip().split()
            if len(fields) < 6:
                continue
            audio_id = fields[1]
            key      = fields[5].lower()
            if key == "bonafide":
                label_map[audio_id] = LABEL_BONAFIDE
            elif key == "spoof":
                label_map[audio_id] = LABEL_SPOOF

    # Pass 2 — scan flac/ subdirectories; O(1) dict lookup per file.
    samples:  List[Tuple[str, int]] = []
    no_label = 0

    for audio_dir in audio_dirs:
        flac_dir = os.path.join(audio_dir, "flac")
        if not os.path.isdir(flac_dir):
            print(f"[WARNING] 2021 flac directory not found: '{flac_dir}'")
            continue
        for entry in os.scandir(flac_dir):
            if not entry.name.endswith(".flac"):
                continue
            audio_id = entry.name[:-5]          # strip .flac extension
            if audio_id not in label_map:
                no_label += 1
                continue
            samples.append((os.path.abspath(entry.path), label_map[audio_id]))

    if no_label:
        print(f"[WARNING] {no_label} 2021 files had no label entry in metadata.")

    if not samples:
        raise FileNotFoundError(
            "No 2021 audio files found under the provided directories.  "
            "Check that each path in 'dataset_2021' ends at the directory "
            "that contains a 'flac/' subdirectory."
        )

    n_bonafide = sum(1 for _, e in samples if e == LABEL_BONAFIDE)
    n_spoof    = len(samples) - n_bonafide
    tag        = os.path.basename(audio_dirs[0]) if audio_dirs else "?"
    print(f"[DATA] 2021 ({tag}): {len(samples):,} files  "
          f"({n_bonafide:,} bonafide / {n_spoof:,} spoof).")
    return samples


def stratified_subsample(
    samples: List[Tuple[str, int]],
    limit: int,
    seed: int,
) -> List[Tuple[str, int]]:
    """Reduce a subset while preserving the original bonafide/spoof ratio.

    Plain random sampling could leave the bonafide class almost absent
    (~10% of the corpus), biasing accuracy and EER.  Stratified sampling
    draws a proportional number from each class.

    Args:
        samples: Full list of (path, label) tuples.
        limit:   Maximum number of audio files to keep (0 = no limit).
        seed:    RNG seed for reproducibility.

    Returns:
        Subsampled list (unchanged if limit <= 0 or >= len(samples)).
    """
    if limit <= 0 or limit >= len(samples):
        return samples

    rng      = random.Random(seed)
    bonafide = [s for s in samples if s[1] == LABEL_BONAFIDE]
    spoof    = [s for s in samples if s[1] == LABEL_SPOOF]
    rng.shuffle(bonafide)
    rng.shuffle(spoof)

    ratio      = len(bonafide) / len(samples)
    n_bonafide = max(1, round(limit * ratio))
    n_spoof    = max(1, limit - n_bonafide)

    result = bonafide[:n_bonafide] + spoof[:n_spoof]
    rng.shuffle(result)
    print(f"[DATA] Stratified subsample: {len(result)} audio files "
          f"({n_bonafide} bonafide / {n_spoof} spoof).")
    return result


class ASVspoofTorchDataset(Dataset):
    """PyTorch Dataset with on-the-fly (lazy) audio loading.

    Decoupled design: this class does NOT know which spectral representation
    is computed.  It receives an extraction *callback* (normally
    ``FeatureExtractor.get_spectrogram_matrix``) that maps a 1-D waveform
    to the 2-D matrix consumed by the CNN.  Swapping STFT for Mel or a
    wavelet scalogram requires only injecting a different callback — no
    changes here (dependency inversion).

    Lazy loading avoids materialising tens of thousands of spectrograms in
    RAM: each audio is decoded and transformed on demand by the DataLoader,
    enabling parallelisation via ``num_workers``.
    """

    def __init__(
        self,
        samples: List[Tuple[str, int]],
        extraction_callback: Callable[[np.ndarray], np.ndarray],
        sample_rate: int,
        augment: bool = False,
        cache_tag: str = None,
        cache_dir: str = "cache",
    ) -> None:
        """
        Args:
            samples:             List of (audio_path, label) tuples from
                                 :func:`parse_protocol`.
            extraction_callback: ``f(waveform) -> 2-D float32 matrix``
                                 converting raw audio to CNN input.
            sample_rate:         Target sample rate.  librosa resamples if the
                                 file differs, guaranteeing a homogeneous
                                 Nyquist frequency (sr/2) across the corpus.
            augment:             If True, apply :func:`_spec_augment` (time +
                                 frequency masking) to each spectrogram.
                                 Should be True only for the training split.
            cache_tag:           If given, the BASE spectrogram (pre-augment) of
                                 each file is cached to disk as
                                 ``{cache_dir}/spectrograms/{stem}_{cache_tag}.npy``
                                 and reused on later epochs / runs — this removes
                                 the per-epoch STFT recompute, the main CNN
                                 bottleneck. The tag must encode the spectrogram
                                 config so it self-invalidates when it changes.
            cache_dir:           Root cache directory (default "cache").
        """
        self.samples             = samples
        self.extraction_callback = extraction_callback
        self.sample_rate         = sample_rate
        self.augment             = augment
        self.cache_tag           = cache_tag
        self.cache_dir           = cache_dir

    def __len__(self) -> int:
        return len(self.samples)

    def _spectrogram(self, path: str) -> np.ndarray:
        """Return the BASE (un-augmented) spectrogram, using the disk cache."""
        cache_path = None
        if self.cache_tag:
            stem       = os.path.splitext(os.path.basename(path))[0]
            cache_path = os.path.join(self.cache_dir, "spectrograms",
                                      f"{stem}_{self.cache_tag}.npy")
            if os.path.isfile(cache_path):
                try:
                    return np.load(cache_path)
                except Exception:            # corrupt/partial cache → recompute
                    pass

        # Decode FLAC -> float32 vector in [-1, 1] (mono). Some 2021 DF files
        # have non-standard FLAC encoding libsndfile cannot decode; substitute
        # 3 s of silence so training continues without aborting the epoch.
        try:
            signal, _ = librosa.load(path, sr=self.sample_rate, mono=True)
        except Exception:
            print(f"[WARNING] Could not decode '{path}' — substituting silence.")
            signal = np.zeros(self.sample_rate * 3, dtype=np.float32)

        base = np.asarray(self.extraction_callback(signal), dtype=np.float32)

        if cache_path is not None:           # atomic write (workers run in parallel)
            try:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                # tmp must end in .npy, else np.save appends it and the replace
                # would miss the real file.
                tmp = f"{cache_path}.{os.getpid()}.tmp.npy"
                np.save(tmp, base)
                os.replace(tmp, cache_path)
            except Exception:
                pass
        return base

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path, label = self.samples[idx]
        matrix = self._spectrogram(path)
        if self.augment:                     # masking applied fresh each epoch
            matrix = _spec_augment(matrix)
        # Conv2d expects (channels, height, width): add the single channel axis
        # (grayscale spectral "image").
        tensor = torch.from_numpy(matrix).unsqueeze(0).float()
        return tensor, torch.tensor(float(label), dtype=torch.float32)
