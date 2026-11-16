"""Import shim: the BazaarDW generator lives in the top-level ``world/`` directory (see CLAUDE.md §5).

``import bazaar_world.generate`` resolves to ``world/generate.py``. The shim keeps the repository
layout from the brief while giving the package a distinctive import name.
"""

from pathlib import Path

_WORLD_DIR = Path(__file__).resolve().parents[2] / "world"
__path__ = [str(_WORLD_DIR)]  # noqa: F821  (package path redirection)

WORLD_DIR = _WORLD_DIR
