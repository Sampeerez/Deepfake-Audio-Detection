# -*- coding: utf-8 -*-
"""src.ui, UI layer split out of the former monolithic src/ui_helpers.py.

Submodules: styles (CSS/theme), figures (matplotlib builders), components
(reusable Streamlit widgets). All names remain importable from src.ui_helpers,
which now re-exports this package plus src.model_registry and src.leaderboard.
"""

from .styles import *      # noqa: F401,F403
from .figures import *     # noqa: F401,F403
from .components import *  # noqa: F401,F403
