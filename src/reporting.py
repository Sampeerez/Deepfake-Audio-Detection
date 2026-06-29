# -*- coding: utf-8 -*-
"""src/reporting.py — Column constants for experiment result rows.

Result dicts produced by ``src.pipeline`` use these ``COL_*`` keys; the web app
reads them to build its results table and CSV export.
"""

from typing import List

# Result dict keys — shared by the pipeline and the GUI results table.
COL_FEATURES   = "Feature Configuration"
COL_MODEL      = "Model"
COL_ACCURACY   = "Accuracy"
COL_EER        = "EER (%)"
COL_MIN_DCF    = "minDCF"
COL_TRAIN_TIME = "Training Time (s)"
COL_FEAT_TIME  = "Feature Extraction per Audio (ms)"
COL_INFER_TIME = "Avg Inference per Audio (ms)"
# Per-model EER threshold (the p(spoof) cut where FAR = FRR). NOT a display column
# — it is carried on the row so the sweep can persist each model's best operating
# point into leaderboard.json, which "Test an audio" then uses per model.
COL_THRESHOLD  = "EER Threshold"

RESULT_COLUMNS: List[str] = [
    COL_FEATURES, COL_MODEL, COL_ACCURACY,
    COL_MIN_DCF, COL_EER, COL_TRAIN_TIME, COL_FEAT_TIME, COL_INFER_TIME,
]
