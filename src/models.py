# -*- coding: utf-8 -*-
"""
src/models.py, Classifiers: classic ML models and 2-D deep networks.

Two detector families, both with probabilistic output so that EER and
minDCF can be computed on continuous scores:

  A) Classic (CPU, sklearn/xgboost) on aggregated 1-D vectors:
     - Logistic Regression with L2 regularisation.
     - RBF-kernel SVM with probability calibration (Platt scaling).
     - XGBoost with L1/L2 regularisation, subsampling, and bounded depth.

  B) Deep (CPU/GPU, PyTorch) on 2-D STFT-dB spectrograms:
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
                          the training set, used to offset the class imbalance
                          of ASVspoof (~1 bonafide per 9 spoof). Defaults to
                          1.0 if None.

    Returns:
        A sklearn estimator (or Pipeline) with a uniform interface.
    """
    if model_name == "logistic_regression":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                C=1.0,
                solver="lbfgs",
                max_iter=2000,
                class_weight="balanced",
                random_state=seed,
            )),
        ])

    if model_name == "svm_lineal":
        return Pipeline([
            ("scaler", StandardScaler()),
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
        f"Unknown classic model '{model_name}'. "
        f"Valid options: {CLASSIC_MODELS}."
    )


def _conv_block(in_channels: int, out_channels: int,
                use_se: bool = False) -> nn.Sequential:
    """Conv2d(3×3, same padding, no bias), BatchNorm2d, ReLU, [SE], MaxPool2d(2×2).

    When ``use_se`` is True a Squeeze-and-Excitation gate is inserted right
    after the ReLU and before the MaxPool, re-weighting each channel by its
    discriminative importance.
    """
    layers = [
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        ),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
    ]
    if use_se:
        layers.append(_SEBlock(out_channels))
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


class CNN_5Block(nn.Module):
    """2-D CNN with five convolutional blocks for deepfake detection on
    STFT-dB spectrograms.

    The network treats the spectrogram (1 × freq_bins × time_frames) as a
    single-channel image and learns hierarchically through 5 blocks with a
    growing receptive field (channels 16, 32, 64, 128, 256): early blocks pick
    up local time-frequency edges and textures (transients, phoneme
    transitions, harmonic grids), middle blocks the formants, vocoder noise
    bands and discontinuities between synthesised frames, and deep blocks the
    global structure that separates natural from synthetic speech.

    Each block is Conv2d, BatchNorm2d, ReLU, optional SE, MaxPool2d:
      * Conv2d 3×3: local filters with weights shared across the whole
        time-frequency plane, so far fewer parameters than a dense layer plus
        translation equivariance (an artefact betrays the deepfake whether it
        lands at second 1 or second 8).
      * BatchNorm2d: re-normalises each channel with the batch mean/variance,
        easing internal covariate shift, allowing larger learning rates and
        adding mild regularisation.
      * ReLU: the max(0, x) non-linearity; without it the stacked convolutions
        would collapse into one linear map, and its constant gradient in the
        active region avoids vanishing gradients.
      * SE (optional): Squeeze-and-Excitation attention re-weighting each
        frequency channel by its discriminative importance.
      * MaxPool2d 2×2: keeps the dominant activation of each neighbourhood,
        giving local invariance and halving the spatial cost downstream.

    Classification head: AdaptiveAvgPool fixes the spatial size (decoupling the
    network from the exact input), then an MLP with Dropout emits one logit.
    After sigmoid this is p(spoof | spectrogram) in [0, 1], with 0 = bonafide,
    1 = spoof.
    """

    CHANNELS = (16, 32, 64, 128, 256)

    def __init__(self, dropout: float = 0.3, use_se: bool = False) -> None:
        super().__init__()

        chans = (1,) + self.CHANNELS
        self.conv_extractor = nn.Sequential(*[
            _conv_block(chans[i], chans[i + 1], use_se=use_se)
            for i in range(len(self.CHANNELS))
        ])

        self.adaptive_pool = nn.AdaptiveAvgPool2d(output_size=(4, 8))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.CHANNELS[-1] * 4 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (batch, 1, freq_bins, time_frames).

        Returns:
            Tensor of shape (batch,) containing raw logits. The sigmoid is
            applied outside: in BCEWithLogitsLoss during training (for
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



class _SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention (Hu et al., 2018).

    Global average-pools each channel map to a scalar, passes the vector
    through a small bottleneck MLP, and uses the output as a per-channel
    multiplicative gate.  The network learns *which frequency bands* are
    most discriminative for a given attack type, critical for generalising
    to unseen TTS/VC systems where artefacts appear in different spectral
    regions.
    """

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
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

    ``groups`` enables ResNeXt-style grouped convolutions (cardinality): the
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
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False, groups=g1),
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
        self.se = _SEBlock(out_ch)
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
            _ResBlock(1, 32, groups=groups),
            _ResBlock(32, 64, groups=groups),
            _ResBlock(64, 128, groups=groups),
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
        """Args: x, (batch, 1, freq_bins, time_frames). Returns raw logits (batch,)."""
        for block in self.blocks:
            x = block(x)
        x = self.adaptive_pool(x)
        return self.classifier(x).squeeze(1)

    @torch.no_grad()
    def forward_with_activations(self, x: torch.Tensor):
        """Same GUI interface, one activation tensor per residual block."""
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

    An advanced ResNet that adds cardinality by splitting the residual
    convolution channels into 32 independent groups (grouped convolutions).
    Same 4 residual blocks and SE attention; the grouping decorrelates feature
    paths and tends to improve generalisation at a comparable parameter budget.
    """

    def __init__(self, dropout: float = 0.3, groups: int = 32) -> None:
        super().__init__(dropout=dropout, groups=groups)



class CRNN_Model(nn.Module):
    """Convolutional-Recurrent network for deepfake audio detection.

    The 5-block convolutional extractor of :class:`CNN_5Block` learns local
    time-frequency patterns; instead of collapsing the time axis with global
    pooling, the per-frame feature vectors are fed to a bidirectional GRU that
    models their temporal evolution (forward + backward context). The recurrent
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
        self.freq_pool = nn.AdaptiveAvgPool2d((self.FREQ_OUT, None))
        rnn_in = self.CHANNELS[-1] * self.FREQ_OUT
        self.rnn = nn.GRU(
            input_size=rnn_in, hidden_size=hidden, num_layers=rnn_layers,
            batch_first=True, bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(2 * hidden, 1),
        )

    def _sequence(self, feat: torch.Tensor) -> torch.Tensor:
        """(B, C, F, T) → (B, T, C*F): per-frame feature vectors over time."""
        feat = self.freq_pool(feat)
        b, c, f, t = feat.shape
        return feat.permute(0, 3, 1, 2).reshape(b, t, c * f)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x, (batch, 1, freq_bins, time_frames). Returns raw logits (batch,)."""
        seq = self._sequence(self.conv_extractor(x))
        out, _ = self.rnn(seq)
        return self.classifier(out.mean(dim=1)).squeeze(1)

    @torch.no_grad()
    def forward_with_activations(self, x: torch.Tensor):
        """Same GUI interface, one activation tensor per convolutional block."""
        activations = []
        out = x
        for block in self.conv_extractor:
            out = block(out)
            activations.append(out.detach().cpu())
        seq = self._sequence(out)
        rnn_out, _ = self.rnn(seq)
        logits = self.classifier(rnn_out.mean(dim=1)).squeeze(1)
        return logits, activations



_ARCH_MODELS = {
    "cnn": CNN_5Block,
    "cnn_se": CNN_5Block_SE,
    "resnet": ResNet_SE,
    "resnext": ResNeXt_SE,
    "crnn": CRNN_Model,
}

_ARCH_LABELS = {
    "cnn": "5-Block CNN",
    "cnn_se": "5-Block CNN + SE",
    "resnet": "ResNet+SE CNN",
    "resnext": "ResNeXt+SE CNN",
    "crnn": "CRNN",
}


def model_for_arch(arch: str, dropout: float = 0.3) -> nn.Module:
    """Instantiate the deep model for an ``arch`` key (defaults to CNN_5Block)."""
    cls = _ARCH_MODELS.get(arch, CNN_5Block)
    return cls(dropout=float(dropout))


def arch_label(arch: str) -> str:
    """Human-readable label for an ``arch`` key (used on leaderboards)."""
    return _ARCH_LABELS.get(arch, _ARCH_LABELS["cnn"])


class Wav2Vec2Classifier(nn.Module):
    """Self-supervised wav2vec 2.0 detector working on the raw 16 kHz waveform.

    A third detector family (alongside the classic DSP models and the 2-D
    spectrogram CNNs): a fine-tuned HuggingFace ``Wav2Vec2Model`` (base, 12
    transformer layers, hidden 768, ``feat_extract_norm="group"``) whose
    time-pooled hidden states feed a 2-class linear head (index 0 = bonafide,
    index 1 = spoof, the class order baked into the released checkpoint).

    The backbone is built from the default ``Wav2Vec2Config`` (which equals
    wav2vec2-base) so no internet access or pretrained download is needed: the
    fine-tuned weights are loaded straight from our own ``.pth``. ``transformers``
    is imported lazily so the rest of the app still imports when it is absent, 
    the pretrained-model loader catches the ImportError and just skips this model.
    """

    SAMPLE_RATE = 16000
    TEMPERATURE = 2.0

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        from transformers import Wav2Vec2Config, Wav2Vec2Model
        self.backbone = Wav2Vec2Model(Wav2Vec2Config())
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x, (batch, samples) raw 16 kHz waveform. Returns (batch, 2) logits.

        The waveform is per-utterance standardised (zero mean / unit variance),
        matching ``Wav2Vec2FeatureExtractor(do_normalize=True)``; the time axis of
        the transformer output is mean-pooled into a single utterance embedding.
        """
        x = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-7)
        feats = self.backbone(x).last_hidden_state
        return self.classifier(feats.mean(dim=1))

    @torch.no_grad()
    def prob_spoof(self, x: torch.Tensor) -> torch.Tensor:
        """p(spoof) in [0, 1] for each clip, temperature-calibrated softmax over the
        2 logits, spoof column. Temperature is monotonic, so rankings (and therefore
        EER / minDCF) are identical to the raw model, only the confidence softens."""
        return torch.softmax(self.forward(x) / self.TEMPERATURE, dim=-1)[:, 1]
