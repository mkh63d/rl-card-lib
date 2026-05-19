from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXTRA_PATHS = [
    ROOT / "packages" / "examples" / "src",
    ROOT / "packages" / "core" / "src",
    ROOT / "packages" / "cardgames" / "src",
    ROOT / "packages" / "visualizer" / "src",
]

for path in EXTRA_PATHS:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
