# -*- coding: utf-8 -*-
"""src.ui — UI layer split out of the former monolithic src/ui_helpers.py.

Submodules: styles (CSS/theme), figures (matplotlib builders), components
(reusable Streamlit widgets). All names remain importable from src.ui_helpers,
which now re-exports this package plus src.model_registry and src.leaderboard.
"""

# Re-export the submodule public names so `from src.ui import build_page_css`
# works alongside the fully-qualified `from src.ui.styles import ...`. Order
# follows the dependency direction (styles -> figures -> components); the
# submodules reference each other as `src.ui.<mod>`, never this package, so
# these stars introduce no import cycle.
from .styles import *      # noqa: F401,F403
from .figures import *     # noqa: F401,F403
from .components import *  # noqa: F401,F403
