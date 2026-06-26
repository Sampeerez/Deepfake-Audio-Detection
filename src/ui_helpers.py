# -*- coding: utf-8 -*-
"""src/ui_helpers.py — Backward-compatible facade.

The former 3000-line monolith was split into focused modules:

  src/ui/styles.py      — CSS, the Light/Dark Side theme, HTML wrapping helpers
  src/ui/figures.py     — matplotlib figure builders + signal statistics
  src/ui/components.py   — reusable Streamlit widgets, eval pickers, job banner
  src/model_registry.py  — config, corpus/sample loading, the pretrained model
                           registry, Hugging Face streaming/download, model loaders
  src/leaderboard.py     — leaderboard.json loading

This module re-exports every public name from those modules, so existing
`from src.ui_helpers import ...` call sites keep working unchanged.
"""

from src.ui.styles import *        # noqa: F401,F403
from src.ui.figures import *       # noqa: F401,F403
from src.model_registry import *   # noqa: F401,F403
from src.leaderboard import *      # noqa: F401,F403
from src.ui.components import *     # noqa: F401,F403
