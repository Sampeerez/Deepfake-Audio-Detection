# -*- coding: utf-8 -*-
"""
tools/make_samples.py — Populate the committed `samples/` tree from the local corpus.

Run once on a machine that has the ASVspoof corpus on disk (paths in
config/config.yaml). It copies a small, balanced, deterministic selection of
real clips into a SUBSET-AWARE layout the web demo reads when the full corpus
is absent:

    samples/2019_la/{train,dev,eval}/{bonafide,spoof}__<i>_<id>.flac
    samples/2021_la/eval/...
    samples/2021_df/eval/...

The label is encoded in the filename prefix (see ui_helpers.bundled_samples).
The eval folders are intentionally SMALL — a robust offline fallback — because
the web demo streams the bulk of the eval audio from the external Hugging Face
dataset instead (see ui_helpers.hf_eval_samples).

Usage:
    python3 tools/make_samples.py            # generate with the default counts
"""

import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml  # noqa: E402

from src.data_loader import (  # noqa: E402
    LABEL_BONAFIDE, parse_protocol, parse_protocol_2021,
)

# How many clips PER CLASS (bonafide + spoof) to copy into each committed folder.
# train/dev are the headline browsable sets; eval is a small offline fallback
# (the web demo pulls the bulk of eval from Hugging Face).
COUNTS = {
    ("2019_la", "train"): 50,
    ("2019_la", "dev"):   50,
    ("2019_la", "eval"):  15,
    ("2021_la", "eval"):  15,
    ("2021_df", "eval"):  15,
}

SEED = 42
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES_DIR = os.path.join(ROOT, "samples")
CONFIG_PATH = os.path.join(ROOT, "config", "config.yaml")


def _balanced_pick(samples, n_per_class, seed):
    """Deterministically take n_per_class bonafide + n_per_class spoof."""
    import random
    rng = random.Random(seed)
    bona  = [p for p, e in samples if e == LABEL_BONAFIDE]
    spoof = [p for p, e in samples if e != LABEL_BONAFIDE]
    rng.shuffle(bona)
    rng.shuffle(spoof)
    return bona[:n_per_class], spoof[:n_per_class]


def _write_folder(corpus_key, subset, bona, spoof):
    out = os.path.join(SAMPLES_DIR, corpus_key, subset)
    # Clean any previous content so reruns are idempotent.
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(out, exist_ok=True)
    n = 0
    for tag, paths in (("bonafide", bona), ("spoof", spoof)):
        for i, src in enumerate(paths):
            dst = os.path.join(out, f"{tag}__{i}_{os.path.basename(src)}")
            try:
                shutil.copy(src, dst)
                n += 1
            except OSError as exc:
                print(f"  ! skip {src}: {exc}")
    print(f"  {corpus_key}/{subset}: wrote {n} files "
          f"({len(bona)} bonafide / {len(spoof)} spoof) -> {out}")


def main():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # ── 2019 LA: train / dev / eval from the official protocols ──────────── #
    root  = cfg["dataset"]["path_la2019"]
    proto = os.path.join(root, cfg["dataset"]["protocols_dir"])
    for subset in ("train", "dev", "eval"):
        n = COUNTS.get(("2019_la", subset))
        if not n:
            continue
        ppath = os.path.join(proto, cfg["dataset"]["protocols"][subset])
        try:
            samples = parse_protocol(ppath, root, subset)
        except Exception as exc:                       # noqa: BLE001
            print(f"  ! 2019 {subset} unavailable: {exc}")
            continue
        bona, spoof = _balanced_pick(samples, n, SEED)
        _write_folder("2019_la", subset, bona, spoof)

    # ── 2021 LA eval ─────────────────────────────────────────────────────── #
    la = cfg.get("dataset_2021", {}).get("la", {})
    try:
        samples = parse_protocol_2021(la.get("keys", ""), [la.get("eval_dir", "")])
        bona, spoof = _balanced_pick(samples, COUNTS[("2021_la", "eval")], SEED)
        _write_folder("2021_la", "eval", bona, spoof)
    except Exception as exc:                            # noqa: BLE001
        print(f"  ! 2021 LA unavailable: {exc}")

    # ── 2021 DF eval ─────────────────────────────────────────────────────── #
    df = cfg.get("dataset_2021", {}).get("df", {})
    try:
        samples = parse_protocol_2021(df.get("keys", ""), df.get("eval_dirs", []))
        bona, spoof = _balanced_pick(samples, COUNTS[("2021_df", "eval")], SEED)
        _write_folder("2021_df", "eval", bona, spoof)
    except Exception as exc:                            # noqa: BLE001
        print(f"  ! 2021 DF unavailable: {exc}")

    print("Done.")


if __name__ == "__main__":
    main()
