# -*- coding: utf-8 -*-
"""
load_checkpoint.py — Standalone loader for the trained CNN.

After a CNN has been trained in the app (Benchmark → CNN), its best weights are
saved to ``asvspoof_model_checkpoint.pth``. This script reloads that file into
the SAME architecture and puts the model in eval mode, so you can score audio
in a fresh Python session WITHOUT retraining.

Usage:
    python load_checkpoint.py                 # smoke test (prints model info)

    # ...or, from your own code:
    from load_checkpoint import load_model
    model, meta, extractor = load_model()
    # score one .flac / .wav file:
    import librosa, numpy as np, torch
    sig, _ = librosa.load("clip.flac", sr=extractor.sample_rate, mono=True)
    if len(sig) < extractor.n_fft:
        sig = np.pad(sig, (0, extractor.n_fft - len(sig)))
    x = torch.from_numpy(extractor.get_spectrogram_matrix(sig)) \
             .unsqueeze(0).unsqueeze(0).float()
    with torch.no_grad():
        p_spoof = torch.sigmoid(model(x.to(next(model.parameters()).device))).item()
    print("p(spoof) =", p_spoof)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from src.features import FeatureExtractor  # noqa: E402
from src.models import AudioDeepfakeCNN, ResNetCNN  # noqa: E402

CHECKPOINT_PATH = "asvspoof_model_checkpoint.pth"
CONFIG_PATH     = "config/config.yaml"


def load_model(path: str = CHECKPOINT_PATH, device: "torch.device" = None):
    """Load the checkpoint into its architecture and return (model, meta, extractor).

    ``model`` is already on ``device`` and in eval() mode.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Checkpoint '{path}' not found. Train a CNN in the app first "
            "(Benchmark → CNN) — it is saved automatically."
        )
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(path, map_location=device)

    model_cls = ResNetCNN if ckpt.get("arch") == "resnet" else AudioDeepfakeCNN
    model = model_cls(dropout=float(ckpt.get("dropout", 0.3)))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()                      # ← ready for inference

    extractor = FeatureExtractor(CONFIG_PATH)
    return model, ckpt, extractor


if __name__ == "__main__":
    mdl, meta, _ = load_model()
    n_params = sum(p.numel() for p in mdl.parameters())
    print(f"Loaded '{meta.get('arch')}' CNN "
          f"({n_params:,} params) on {next(mdl.parameters()).device} — eval mode.")
