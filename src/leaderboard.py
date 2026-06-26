# -*- coding: utf-8 -*-
"""src/leaderboard.py — Load the committed leaderboard.json metrics.

Pure file/JSON logic split out of src/ui_helpers.py and re-exported there for
backward compatibility.
"""

import os
from typing import Dict, List

import streamlit as st

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Pre-computed metrics for the pretrained models, written automatically by a
# local Benchmark → Full comparison and committed, so the web demo can show the
# real EER/minDCF leaderboard without the corpus. Maps model key ->
# {"eer_dev", "mindcf_dev", "eer_eval", "mindcf_eval"}.
LEADERBOARD_PATH = os.path.join(_REPO_ROOT, "leaderboard.json")
# Previous filename — still read as a fallback so an un-regenerated repo keeps
# showing its committed metrics after the rename.
_LEGACY_LEADERBOARD_PATH = os.path.join(_REPO_ROOT, "demo_leaderboard.json")

# Back-compat alias: some imports still reference the old constant name.
DEMO_LEADERBOARD_PATH = LEADERBOARD_PATH


@st.cache_data(show_spinner=False)
def load_leaderboard() -> Dict:
    """Read the committed leaderboard file as the full structured object
    ``{"models": {key: {eer_dev, mindcf_dev, eer_eval, mindcf_eval}}, "rows": [...]}``.

    Migrates the legacy FLAT format (a bare ``{key: metrics}`` dict, no per-corpus
    rows) into ``{"models": <flat>, "rows": []}`` so old files keep working."""
    import json
    path = (LEADERBOARD_PATH if os.path.isfile(LEADERBOARD_PATH)
            else _LEGACY_LEADERBOARD_PATH)
    if not os.path.isfile(path):
        return {"models": {}, "rows": []}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (ValueError, OSError):
        return {"models": {}, "rows": []}
    if isinstance(data, dict) and ("models" in data or "rows" in data):
        data.setdefault("models", {})
        data.setdefault("rows", [])
        return data
    # Legacy flat dict {key: metrics}.
    return {"models": data if isinstance(data, dict) else {}, "rows": []}


def load_leaderboard_models() -> Dict[str, Dict]:
    """Per-model dev/eval metrics map (keyed by registry key). The headline-verdict
    fusion and the web model-hub use this for ranking; empty until generated."""
    return load_leaderboard().get("models", {})


def load_leaderboard_rows() -> List[Dict]:
    """Every leaderboard ROW (one per model × split/corpus) exactly as produced by a
    local Full comparison — so the web demo can render the IDENTICAL filterable
    table/chart, not just a condensed summary."""
    return load_leaderboard().get("rows", [])


# Back-compat alias for the old name (now returns the per-model metrics map).
def load_demo_leaderboard() -> Dict[str, Dict]:
    return load_leaderboard_models()
