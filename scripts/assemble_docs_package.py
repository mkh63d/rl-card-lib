#!/usr/bin/env python
"""Assemble the split ``rl_card_lib`` namespace into one tree for the docs build.

The library ships as five distributions that all populate the ``rl_card_lib``
namespace via ``pkgutil.extend_path``. That works at runtime, but mkdocstrings'
static analyser (griffe) cannot follow ``extend_path``: it roots ``rl_card_lib``
at the first ``src`` directory it finds and never discovers the subpackages that
live in the other distributions (``cardgames``, ``games``, ``report``,
``visualizer``, ``harness``).

The subpackages are disjoint across the distributions, so we can merge every
``packages/*/src/rl_card_lib`` into a single ``rl_card_lib`` directory under
``build/docs_pkg`` and point griffe at that one root. This script is idempotent:
it rebuilds the merged tree from scratch each run. It is invoked before
``mkdocs build`` both locally and in CI.
"""

from __future__ import annotations

import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGES_DIR = REPO_ROOT / "packages"
OUTPUT_ROOT = REPO_ROOT / "build" / "docs_pkg"
MERGED_PKG = OUTPUT_ROOT / "rl_card_lib"

# The distribution whose top-level __init__.py (with the convenience re-exports)
# becomes the merged package's __init__.py.
PRIMARY = "core"


def _iter_source_roots() -> list[Path]:
    roots = []
    for pkg in sorted(p.name for p in PACKAGES_DIR.iterdir() if p.is_dir()):
        src = PACKAGES_DIR / pkg / "src" / "rl_card_lib"
        if src.is_dir():
            roots.append(src)
    return roots


def _ignore(_dir: str, names: list[str]) -> set[str]:
    return {n for n in names if n == "__pycache__" or n.endswith(".pyc")}


def main() -> None:
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    MERGED_PKG.mkdir(parents=True)

    for root in _iter_source_roots():
        for child in root.iterdir():
            if child.name == "__init__.py" or child.name == "__pycache__":
                continue
            dest = MERGED_PKG / child.name
            if child.is_dir():
                if dest.exists():
                    raise SystemExit(
                        f"Name collision merging docs package: {child.name} "
                        f"already present from another distribution."
                    )
                shutil.copytree(child, dest, ignore=_ignore)
            else:
                shutil.copy2(child, dest)

    primary_init = PACKAGES_DIR / PRIMARY / "src" / "rl_card_lib" / "__init__.py"
    shutil.copy2(primary_init, MERGED_PKG / "__init__.py")

    print(f"Assembled docs package at {MERGED_PKG}")
    print("Subpackages:", ", ".join(sorted(
        p.name for p in MERGED_PKG.iterdir() if p.is_dir()
    )))


if __name__ == "__main__":
    main()
