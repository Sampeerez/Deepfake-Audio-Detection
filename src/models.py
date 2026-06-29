# -*- coding: utf-8 -*-
"""
src/models.py — Classifiers: classic ML models and a 2-D CNN.

Two detector families, both with probabilistic output so that EER and
minDCF can be computed on continuous scores:

  A) CLASSIC (CPU, sklearn/xgboost) on aggregated 1-D vectors:
     - Logistic Regression with L2 regularisation.
     - RBF-kernel SVM with probability calibration (Platt scaling).
     - XGBoost with L1/L2 regularisation, subsampling, and bounded depth.

  B) DEEP (CPU/GPU, PyTorch) on 2-D STFT-dB spectrograms:
     - 5-block CNN with BatchNorm and Dropout (optional SE attention).
     - Residual SE CNN (ResNet_SE) and its grouped-conv ResNeXt_SE variant.
     - CRNN: convolutional extractor + bidirectional GRU over the time axis.

The ``get_classic_model`` factory decouples the rest of the app from the
concrete libraries: adding a new model = adding one branch.
"""

from typing import Optional

import torch
import torch.nn as nn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Canonical names accepted by the model factory.
CLASSIC_MODELS = ("logistic_regression", "svm_lineal", "xgboost")


def get_classic_model(
    model_name: str,
    seed: int = 42,
    scale_pos_weight: Optional[float] = None,
):
    """Factory for classic classifiers ready for ``fit`` / ``predict_proba``.

    Args:
        model_name:       One of ``CLASSIC_MODELS``.
        seed:             Reproducibility seed for stochastic components.
        scale_pos_weight: (XGBoost only) ratio #negatives / #positives from
                          the training set, used to compensate the heavy
                          class imbalance of ASVspoof (~1 bonafide per 9
                          spoof).  Defaults to 1.0 if None.

    Returns:
        A sklearn estimator (or Pipeline) with a uniform interface.
    """
    if model_name == "logistic_regression":
        # Logistic Regression models log(p/(1-p)) as a linear combination
        # of the features — the canonical probabilistic linear classifier.
        # Production details:
        #  * StandardScaler up-front: lbfgs gradient descent converges
        #    poorly when features live on very different scales
        #    (RMS variance ~1e-4 vs MFCC_0 ~1e3).
        #  * penalty='l2' (ridge): shrinks weights and prevents the model
        #    from memorising training noise (regularisation controlled by
        #    C = 1/λ).
        #  * class_weight='balanced': re-weights the loss by inverse class
        #    frequency so the ~10% bonafide class is not drowned by the
        #    spoof majority.
        return Pipeline([
            ("scaler",     StandardScaler()),
            ("classifier", LogisticRegression(
                C=1.0,
                solver="lbfgs",
                max_iter=2000,
                class_weight="balanced",
                random_state=seed,
            )),
        ])

    if model_name == "svm_lineal":
        # SVM finds the MAXIMUM-MARGIN boundary between classes: only the
        # support vectors (boundary samples) determine the solution, making
        # it robust in high dimensions with moderate data.
        #  * kernel='rbf' (gamma='scale'): an implicit, infinite-dimensional
        #    feature mapping that can separate classes a linear boundary
        #    cannot — more expressive for the DSP cepstral features, at a
        #    higher (≈ O(n²)) training cost, so keep the subset moderate.
        #  * CalibratedClassifierCV(..., ensemble=False): Platt external
        #    calibration replacing the deprecated probability=True.  Fits a
        #    logistic regression on out-of-fold predictions to emit p(spoof)
        #    in [0,1] for threshold sweeping. cv=3 balances quality vs cost.
        return Pipeline([
            ("scaler",     StandardScaler()),
            ("classifier", CalibratedClassifierCV(
                SVC(
                    kernel="rbf",
                    gamma="scale",
                    C=1.0,
                    class_weight="balanced",
                    random_state=seed,
                ),
                cv=3,
                ensemble=False,
            )),
        ])

    if model_name == "xgboost":
        # Gradient Boosting: committee of trees where each new tree learns
        # to correct the residual (log-loss gradient) of the current
        # committee.  Robust anti-overfitting configuration:
        #  * max_depth=6: shallow trees = weak learners that capture
        #    interactions without memorising individual samples.
        #  * learning_rate=0.1 (shrinkage): dampens each tree's
        #    contribution; combined with n_estimators=300 it smooths fitting.
        #  * subsample / colsample_bytree=0.8: stochastic bagging of rows
        #    and columns that decorrelates the committee trees.
        #  * reg_lambda (L2) and reg_alpha (L1): penalise leaf weights
        #    directly in the regularised objective function.
        #  * tree_method='hist': bin-histogram approach -> dramatically
        #    faster training on CPU with thousands of audio files.
        #  * Trees split by thresholds: scale-invariant, no StandardScaler
        #    needed.
        # Imported lazily: xgboost loads a heavy native lib (~1-2 s), and the
        # web app only needs it when an XGBoost run is actually launched — not
        # every time the page imports this module.
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            reg_alpha=0.1,
            scale_pos_weight=scale_pos_weight if scale_pos_weight else 1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=seed,
        )

    raise ValueError(
        f"Unknown classic model '{model_name}'.  "
        f"Valid options: {CLASSIC_MODELS}."
    )


def _conv_block(in_channels: int, out_channels: int,
                use_se: bool = False) -> nn.Sequential:
    """Conv2d(3×3, same padding, no bias) → BatchNorm2d → ReLU → [SE] → MaxPool2d(2×2).

    When ``use_se`` is True a Squeeze-and-Excitation gate is inserted right
    AFTER the ReLU and BEFORE the MaxPool, re-weighting each channel by its
    discriminative importance.
    """
    layers = [
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,   # 'same' padding: preserves height and width.
            bias=False,  # Bias is redundant: BatchNorm already re-centres.
        ),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
    ]
    if use_se:
        layers.append(_SEBlock(out_channels))
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


class CNN_5Block(nn.Module):
    """2-D CNN with FIVE convolutional blocks for deepfake detection on
    STFT-dB spectrograms.

    The network treats the spectrogram (1 × freq_bins × time_frames) as a
    single-channel image and learns HIERARCHICALLY through 5 blocks with a
    growing receptive field (channels 16 → 32 → 64 → 128 → 256):
      * Early blocks: local time-frequency edges and textures (transients,
        phoneme transitions, harmonic grids).
      * Mid blocks: formants, vocoder noise bands, discontinuities between
        synthesised frames.
      * Deep blocks: global discriminative structures separating natural
        speech from synthetic audio without manual feature engineering.

    Each block follows the canonical pattern Conv2d -> BatchNorm2d -> ReLU
    -> [SE] -> MaxPool2d:
      * Conv2d 3×3: local filters with SHARED weights across the entire
        time-frequency plane -> drastic parameter reduction vs a dense
        layer, plus translation equivariance (an artefact betrays the
        deepfake whether it appears at second 1 or second 8).
      * BatchNorm2d: re-normalises activations of each channel using the
        batch mean/variance, mitigating internal covariate shift; enables
        larger learning rates, accelerates convergence, and adds mild
        regularisation.
      * ReLU: non-linearity max(0, x); without it, stacking convolutions
        would collapse into a single linear transformation.  Its constant
        gradient in the active region prevents gradient vanishing.
      * SE (optional): Squeeze-and-Excitation channel attention re-weighting
        each frequency channel by its discriminative importance.
      * MaxPool2d 2×2: downsampling retaining the dominant activation of
        each neighbourhood: gains local invariance and halves the spatial
        cost of subsequent layers.

    Classification head: AdaptiveAvgPool fixes the spatial dimensions
    (decoupling the network from the exact input size), then an MLP with
    Dropout emits a single logit.  After sigmoid: p(spoof | spectrogram)
    in [0, 1], consistent with the convention 0=bonafide / 1=spoof.
    """

    CHANNELS = (16, 32, 64, 128, 256)

    def __init__(self, dropout: float = 0.3, use_se: bool = False) -> None:
        super().__init__()

        chans = (1,) + self.CHANNELS
        self.conv_extractor = nn.Sequential(*[
            _conv_block(chans[i], chans[i + 1], use_se=use_se)
            for i in range(len(self.CHANNELS))
        ])

        # Adaptive pooling: any residual spatial resolution is collapsed to
        # 4×8 (freq × time) by averaging.  Makes the network robust to
        # future changes of freq_bins/time_frames in the YAML.
        self.adaptive_pool = nn.AdaptiveAvgPool2d(output_size=(4, 8))

        self.classifier = nn.Sequential(
            nn.Flatten(),                    # (B, 256, 4, 8) -> (B, 8192)
            nn.Linear(self.CHANNELS[-1] * 4 * 8, 256),
            nn.ReLU(inplace=True),
            # Dropout: during training, randomly deactivates 30% of neurons
            # at each step.  Prevents co-adaptation (multiple neurons
            # depending on each other's simultaneous presence) and acts as
            # implicit ensembling of sub-networks -> direct overfitting
            # mitigation.  Automatically disabled in .eval() mode.
            nn.Dropout(p=dropout),
            nn.Linear(256, 1),               # Single binary logit.
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (batch, 1, freq_bins, time_frames).

        Returns:
            Tensor of shape (batch,) containing raw logits.  The sigmoid is
            applied OUTSIDE: in BCEWithLogitsLoss during training (for
            numerically stable log-sum-exp), and explicitly during inference
            to obtain p(spoof).
        """
        feature_maps = self.conv_extractor(x)
        feature_maps = self.adaptive_pool(feature_maps)
        logits = self.classifier(feature_maps)
        return logits.squeeze(1)

    @torch.no_grad()
    def forward_with_activations(self, x: torch.Tensor):
        """Forward pass that also returns the output of each convolutional block.

        Used by the GUI to visualise how the network transforms the
        spectrogram layer by layer.

        Args:
            x: Tensor of shape (batch, 1, freq_bins, time_frames).

        Returns:
            (logits, activations) where ``activations`` is a list of one
            tensor per conv block, each shaped (batch, channels, H, W).
        """
        activations = []
        out = x
        for block in self.conv_extractor:
            out = block(out)
            activations.append(out.detach().cpu())
        pooled = self.adaptive_pool(out)
        logits = self.classifier(pooled).squeeze(1)
        return logits, activations


class CNN_5Block_SE(CNN_5Block):
    """``CNN_5Block`` with a Squeeze-and-Excitation gate in every block.

    Identical topology to :class:`CNN_5Block`, but each convolutional block
    inserts an SE channel-attention gate after the ReLU (and before the
    MaxPool), letting the network suppress irrelevant frequency bands and
    amplify attack-specific synthesis artefacts at every resolution level.
    """

    def __init__(self, dropout: float = 0.3) -> None:
        super().__init__(dropout=dropout, use_se=True)


# ===========================================================================
# ResNet-style CNN with Squeeze-and-Excitation attention
# ===========================================================================

class _SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention (Hu et al., 2018).

    Global average-pools each channel map to a scalar, passes the vector
    through a small bottleneck MLP, and uses the output as a per-channel
    multiplicative gate.  The network learns *which frequency bands* are
    most discriminative for a given attack type — critical for generalising
    to unseen TTS/VC systems where artefacts appear in different spectral
    regions.
    """

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.fc(self.pool(x)).view(x.size(0), x.size(1), 1, 1)
        return x * scale


class _ResBlock(nn.Module):
    """Pre-activation residual block with SE attention and spatial downsampling.

    Pattern: Conv(3×3)→BN→ReLU→Conv(3×3)→BN + skip → ReLU → SE → MaxPool(2×2).

    When in_ch != out_ch a 1×1 projection convolution aligns the skip branch
    so that the residual sum is always dimension-compatible.  MaxPool at the
    end halves both spatial dimensions, matching the behaviour of the plain
    CNN blocks and keeping the progressive resolution reduction intact.

    ``groups`` enables ResNeXt-style GROUPED convolutions (cardinality): the
    channels are split into independent groups, each convolved separately,
    which decorrelates feature paths at a lower parameter cost. Grouping is
    applied per-conv only when both its in/out channels are divisible by
    ``groups`` (so the 1→32 stem, not divisible by 32, falls back to a normal
    convolution automatically).
    """

    def __init__(self, in_ch: int, out_ch: int, groups: int = 1) -> None:
        super().__init__()
        g1 = groups if (in_ch % groups == 0 and out_ch % groups == 0) else 1
        g2 = groups if (out_ch % groups == 0) else 1
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, 1, 1, bias=False, groups=g1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False, groups=g2),
            nn.BatchNorm2d(out_ch),
        )
        self.skip = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
            if in_ch != out_ch else nn.Identity()
        )
        self.se   = _SEBlock(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.conv(x) + self.skip(x))
        return self.pool(self.se(out))


class ResidualSECNN(nn.Module):
    """Residual CNN with SE channel attention for deepfake audio detection.

    Designed for better generalisation to unseen spoofing attacks (ASVspoof
    2021 / codecs / new TTS systems) compared to the plain baseline CNN:

    * Residual connections prevent gradient vanishing in deeper networks,
      allowing the model to stack 4 blocks without degradation.
    * SE attention re-weights spectral channels at each resolution level,
      letting the network suppress irrelevant bands (e.g. telephone codec
      roll-off) and amplify band-specific synthesis artefacts.

    ``groups`` selects the convolution cardinality: ``groups=1`` is the plain
    ResNet (:class:`ResNet_SE`); ``groups>1`` turns the residual convolutions
    into ResNeXt-style grouped convolutions (:class:`ResNeXt_SE`).

    Architecture (input 1 × 128 × 300):
        ResBlock(1  → 32)  → (32,  64, 150)
        ResBlock(32 → 64)  → (64,  32,  75)
        ResBlock(64 → 128) → (128, 16,  37)
        ResBlock(128→ 128) → (128,  8,  18)
        AdaptiveAvgPool(4, 8) → (128, 4, 8)
        Flatten → Linear(4096→256) → ReLU → Dropout → Linear(256→1)
    """

    def __init__(self, dropout: float = 0.3, groups: int = 1) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            _ResBlock(1,   32,  groups=groups),
            _ResBlock(32,  64,  groups=groups),
            _ResBlock(64,  128, groups=groups),
            _ResBlock(128, 128, groups=groups),
        ])
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 8))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x — (batch, 1, freq_bins, time_frames). Returns raw logits (batch,)."""
        for block in self.blocks:
            x = block(x)
        x = self.adaptive_pool(x)
        return self.classifier(x).squeeze(1)

    @torch.no_grad()
    def forward_with_activations(self, x: torch.Tensor):
        """Same GUI interface — one activation tensor per residual block."""
        activations = []
        for block in self.blocks:
            x = block(x)
            activations.append(x.detach().cpu())
        pooled = self.adaptive_pool(x)
        logits = self.classifier(pooled).squeeze(1)
        return logits, activations


class ResNet_SE(ResidualSECNN):
    """4-block residual CNN with SE attention (plain convolutions, groups=1)."""

    def __init__(self, dropout: float = 0.3) -> None:
        super().__init__(dropout=dropout, groups=1)


class ResNeXt_SE(ResidualSECNN):
    """ResNeXt variant of :class:`ResNet_SE`.

    An advanced ResNet that introduces CARDINALITY by splitting the residual
    convolution channels into 32 independent groups (grouped convolutions).
    Same 4 residual blocks and SE attention; the grouping decorrelates feature
    paths and tends to improve generalisation at a comparable parameter budget.
    """

    def __init__(self, dropout: float = 0.3, groups: int = 32) -> None:
        super().__init__(dropout=dropout, groups=groups)


# ===========================================================================
# CRNN: convolutional feature extractor + bidirectional recurrent layer
# ===========================================================================

class CRNN_Model(nn.Module):
    """Convolutional-Recurrent network for deepfake audio detection.

    The 5-block convolutional extractor of :class:`CNN_5Block` learns local
    time-frequency patterns; instead of collapsing the time axis with global
    pooling, the per-frame feature vectors are fed to a BIDIRECTIONAL GRU that
    models their TEMPORAL evolution (forward + backward context). The recurrent
    states are mean-pooled over time and projected to a single logit.

    The frequency axis is adaptively pooled to a fixed height (4) so the GRU
    input size stays constant regardless of ``freq_bins`` in the YAML.
    """

    CHANNELS = (16, 32, 64, 128, 256)
    FREQ_OUT = 4

    def __init__(self, dropout: float = 0.3, hidden: int = 128,
                 rnn_layers: int = 1, use_se: bool = False) -> None:
        super().__init__()
        chans = (1,) + self.CHANNELS
        self.conv_extractor = nn.Sequential(*[
            _conv_block(chans[i], chans[i + 1], use_se=use_se)
            for i in range(len(self.CHANNELS))
        ])
        # Collapse the frequency axis to a fixed height, keep the time axis.
        self.freq_pool = nn.AdaptiveAvgPool2d((self.FREQ_OUT, None))
        rnn_in = self.CHANNELS[-1] * self.FREQ_OUT
        self.rnn = nn.GRU(
            input_size=rnn_in, hidden_size=hidden, num_layers=rnn_layers,
            batch_first=True, bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(2 * hidden, 1),        # 2× for bidirectional.
        )

    def _sequence(self, feat: torch.Tensor) -> torch.Tensor:
        """(B, C, F, T) → (B, T, C*F): per-frame feature vectors over time."""
        feat = self.freq_pool(feat)
        b, c, f, t = feat.shape
        return feat.permute(0, 3, 1, 2).reshape(b, t, c * f)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x — (batch, 1, freq_bins, time_frames). Returns raw logits (batch,)."""
        seq = self._sequence(self.conv_extractor(x))
        out, _ = self.rnn(seq)               # (B, T, 2*hidden)
        return self.classifier(out.mean(dim=1)).squeeze(1)

    @torch.no_grad()
    def forward_with_activations(self, x: torch.Tensor):
        """Same GUI interface — one activation tensor per convolutional block."""
        activations = []
        out = x
        for block in self.conv_extractor:
            out = block(out)
            activations.append(out.detach().cpu())
        seq = self._sequence(out)
        rnn_out, _ = self.rnn(seq)
        logits = self.classifier(rnn_out.mean(dim=1)).squeeze(1)
        return logits, activations


# ===========================================================================
# Architecture registry — single source of truth for arch-key → model class
# ===========================================================================

_ARCH_MODELS = {
    "cnn":     CNN_5Block,
    "cnn_se":  CNN_5Block_SE,
    "resnet":  ResNet_SE,
    "resnext": ResNeXt_SE,
    "crnn":    CRNN_Model,
}

_ARCH_LABELS = {
    "cnn":     "5-Block CNN",
    "cnn_se":  "5-Block CNN + SE",
    "resnet":  "ResNet+SE CNN",
    "resnext": "ResNeXt+SE CNN",
    "crnn":    "CRNN",
}


def model_for_arch(arch: str, dropout: float = 0.3) -> nn.Module:
    """Instantiate the deep model for an ``arch`` key (defaults to CNN_5Block)."""
    cls = _ARCH_MODELS.get(arch, CNN_5Block)
    return cls(dropout=float(dropout))


def arch_label(arch: str) -> str:
    """Human-readable label for an ``arch`` key (used on leaderboards)."""
    return _ARCH_LABELS.get(arch, _ARCH_LABELS["cnn"])


class Wav2Vec2Classifier(nn.Module):
    """Self-supervised wav2vec 2.0 detector working on the RAW 16 kHz waveform.

    A third detector family (alongside the classic DSP models and the 2-D
    spectrogram CNNs): a fine-tuned HuggingFace ``Wav2Vec2Model`` (base, 12
    transformer layers, hidden 768, ``feat_extract_norm="group"``) whose
    time-pooled hidden states feed a 2-class linear head (index 0 = bonafide,
    index 1 = spoof — the class order baked into the released checkpoint).

    The backbone is built from the DEFAULT ``Wav2Vec2Config`` (which equals
    wav2vec2-base) so no internet access or pretrained download is needed: the
    fine-tuned weights are loaded straight from our own ``.pth``. ``transformers``
    is imported lazily so the rest of the app still imports when it is absent —
    the pretrained-model loader catches the ImportError and just skips this model.
    """

    SAMPLE_RATE = 16000
    # Temperature for confidence calibration. The fine-tune is pathologically
    # overconfident (logits ≈ ±8 → p ≈ 0/1), which makes it dominate the late-fusion
    # verdict even when it is wrong on out-of-distribution audio. Dividing the logits
    # by T>1 softens the probabilities WITHOUT changing their order, so EER / minDCF
    # on the leaderboard are unaffected (the metric is rank-based), but p(spoof) is
    # no longer saturated. Applied in prob_spoof only.
    TEMPERATURE = 2.0

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        from transformers import Wav2Vec2Config, Wav2Vec2Model  # lazy
        self.backbone = Wav2Vec2Model(Wav2Vec2Config())
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x — (batch, samples) raw 16 kHz waveform. Returns (batch, 2) logits.

        The waveform is per-utterance standardised (zero mean / unit variance),
        matching ``Wav2Vec2FeatureExtractor(do_normalize=True)``; the time axis of
        the transformer output is mean-pooled into a single utterance embedding.
        """
        x = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-7)
        feats = self.backbone(x).last_hidden_state          # (batch, frames, 768)
        return self.classifier(feats.mean(dim=1))           # (batch, 2)

    @torch.no_grad()
    def prob_spoof(self, x: torch.Tensor) -> torch.Tensor:
        """p(spoof) in [0, 1] for each clip — temperature-calibrated softmax over the
        2 logits, spoof column. Temperature is monotonic, so rankings (and therefore
        EER / minDCF) are identical to the raw model — only the confidence softens."""
        return torch.softmax(self.forward(x) / self.TEMPERATURE, dim=-1)[:, 1]
