# -*- coding: utf-8 -*-
"""
export_models.py — Generate the pretrained-weight files (and their metrics) for
the public web demo.

The Streamlit Cloud demo cannot train or touch the corpus, so it serves models
that were trained here, on disk. This script regenerates every model FROM THE
CORPUS, scores it, and writes:

  • models/resnet_checkpoint.pth   — ResNet + SE   (torch checkpoint)
  • models/cnn3x3_checkpoint.pth   — 3-Block CNN   (torch checkpoint)
  • models/xgb_<feat>.joblib       — XGBoost on each DSP front-end (joblib dump)
  • demo_leaderboard.json          — dev/eval EER & minDCF per model (committed,
                                     so the web hub shows the real comparison)

File names come from ui_helpers.PRETRAINED_REGISTRY, so the app's loaders find
them. Upload the models/ files to Hugging Face and paste the direct links into
src/ui_helpers.py (RESNET_URL, CNN3X3_URL, XGB_*_URL); commit demo_leaderboard.json.

Usage (Linux, corpus present, ideally a GPU for the CNNs):
    python export_models.py
    python export_models.py --classic-subset 8000 --epochs 20 --only classic
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from src.data_loader import LABEL_BONAFIDE, LABEL_SPOOF, stratified_subsample  # noqa: E402
from src.features import FeatureExtractor  # noqa: E402
from src.metrics import calculate_eer, calculate_min_dcf  # noqa: E402
from src.models import get_classic_model  # noqa: E402
from src.pipeline import extract_feature_matrix, train_and_evaluate_cnn  # noqa: E402
from src.reporting import COL_EER, COL_MIN_DCF, COL_MODEL  # noqa: E402
from src.ui_helpers import (  # noqa: E402
    DEMO_LEADERBOARD_PATH, MODELS_DIR, PRETRAINED_REGISTRY, get_samples,
)

CONFIG_PATH = os.path.join("config", "config.yaml")


def _subset(samples, n, seed):
    """Stratified subsample, or the whole set when n <= 0."""
    return stratified_subsample(samples, n, seed) if (n and n > 0) else samples


def export_cnn(entry, train, dev, eval_set, extractor, params, arch, out_path):
    """Train one CNN (best weights saved to out_path) and return its metrics."""
    print(f"\n=== {entry['name']} → {out_path} ===")
    p = dict(params)
    p.update({"arch": arch, "augment": True})
    _, _, results = train_and_evaluate_cnn(
        train, dev, extractor, p,
        eval_sets=[("2019 LA", eval_set)] if eval_set else None,
        checkpoint_path=out_path,
    )
    dev_row  = next((r for r in results if "[EVAL]" not in str(r[COL_MODEL])), {})
    eval_row = next((r for r in results if "[EVAL]" in str(r[COL_MODEL])), {})
    print(f"[OK] saved {out_path}")
    return {
        "eer_dev":    float(dev_row.get(COL_EER))     if dev_row.get(COL_EER)     else None,
        "mindcf_dev": float(dev_row.get(COL_MIN_DCF)) if dev_row.get(COL_MIN_DCF) else None,
        "eer_eval":    float(eval_row.get(COL_EER))     if eval_row.get(COL_EER)     else None,
        "mindcf_eval": float(eval_row.get(COL_MIN_DCF)) if eval_row.get(COL_MIN_DCF) else None,
    }


def _score_classic(model, samples, extractor, feat, split):
    """EER (%) and minDCF of a fitted classic model on one split."""
    x, y, _ = extract_feature_matrix(samples, extractor, feat, split,
                                     n_workers=4, use_cache=True)
    probs = model.predict_proba(x)[:, 1].tolist()
    eer, _ = calculate_eer(probs, y.tolist())
    return 100.0 * eer, calculate_min_dcf(probs, y.tolist())


def export_classic(entry, train, dev, eval_set, extractor, seed, out_path):
    """Fit one XGBoost detector on its DSP front-end and return its metrics."""
    feat = entry["feat"]
    print(f"\n=== {entry['name']} (feat {feat}) → {out_path} ===")
    x_tr, y_tr, _ = extract_feature_matrix(train, extractor, feat, "train",
                                           n_workers=4, use_cache=True)
    imbalance = float(np.sum(y_tr == LABEL_BONAFIDE)) / max(np.sum(y_tr == LABEL_SPOOF), 1)
    model = get_classic_model("xgboost", seed=seed, scale_pos_weight=imbalance)
    model.fit(x_tr, y_tr)
    joblib.dump(model, out_path)
    print(f"[OK] saved {out_path}")

    eer_dev, mindcf_dev = _score_classic(model, dev, extractor, feat, "dev")
    metrics = {"eer_dev": eer_dev, "mindcf_dev": mindcf_dev,
               "eer_eval": None, "mindcf_eval": None}
    if eval_set:
        metrics["eer_eval"], metrics["mindcf_eval"] = _score_classic(
            model, eval_set, extractor, feat, "eval")
    return metrics


def main():
    ap = argparse.ArgumentParser(description="Export pretrained weights for the demo.")
    ap.add_argument("--classic-subset", type=int, default=6000,
                    help="Files per classic model (0 = full train set).")
    ap.add_argument("--cnn-subset", type=int, default=0,
                    help="Files for CNN training (0 = full train set).")
    ap.add_argument("--eval-subset", type=int, default=2000,
                    help="Files used to score dev/eval metrics (0 = full split).")
    ap.add_argument("--epochs", type=int, default=None, help="Override CNN epochs.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only", choices=["cnn", "classic"], default=None,
                    help="Export only one model family.")
    args = ap.parse_args()

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    extractor = FeatureExtractor(CONFIG_PATH)
    train = get_samples("train")
    dev   = get_samples("dev")
    eval_ = get_samples("eval")
    if not train:
        raise SystemExit("ASVspoof corpus not found — check dataset.path_la2019 "
                         "in config/config.yaml. This script needs the dataset.")

    os.makedirs(MODELS_DIR, exist_ok=True)
    dev_score  = _subset(dev,   args.eval_subset, args.seed + 10)
    eval_score = _subset(eval_, args.eval_subset, args.seed + 11)

    params = dict(config["train_params"])
    params["semilla"] = args.seed
    params.setdefault("num_workers", 2)
    if args.epochs:
        params["epochs"] = int(args.epochs)

    # Merge into any existing leaderboard so a partial --only run keeps the rest.
    board = {}
    if os.path.isfile(DEMO_LEADERBOARD_PATH):
        with open(DEMO_LEADERBOARD_PATH, "r", encoding="utf-8") as f:
            board = json.load(f)
    by_key = {e["key"]: e for e in PRETRAINED_REGISTRY}

    if args.only != "classic":
        cnn_train = _subset(train, args.cnn_subset, args.seed)
        cnn_dev   = _subset(dev,   args.cnn_subset, args.seed + 1)
        for key, arch in (("resnet", "resnet"), ("cnn3x3", "cnn")):
            e = by_key[key]
            board[key] = export_cnn(e, cnn_train, cnn_dev, eval_score, extractor,
                                    params, arch, os.path.join(MODELS_DIR, e["file"]))

    if args.only != "cnn":
        cls_train = _subset(train, args.classic_subset, args.seed)
        for e in PRETRAINED_REGISTRY:
            if e["kind"] != "classic":
                continue
            board[e["key"]] = export_classic(e, cls_train, dev_score, eval_score,
                                             extractor, args.seed,
                                             os.path.join(MODELS_DIR, e["file"]))

    with open(DEMO_LEADERBOARD_PATH, "w", encoding="utf-8") as f:
        json.dump(board, f, indent=2)

    print(f"\nAll done.\n  Weights : {MODELS_DIR}/  → upload to Hugging Face, then "
          "paste the links into src/ui_helpers.py.\n"
          f"  Metrics : {DEMO_LEADERBOARD_PATH}  → commit it.")


if __name__ == "__main__":
    main()
