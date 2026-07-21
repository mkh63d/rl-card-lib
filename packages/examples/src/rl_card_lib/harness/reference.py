"""The committed library reference runs, exposed as a read-only RunStore.

Full run records for the bundled example games (Klondike, Macao) live in-tree
so someone experimenting with their own agent can compare against the library's
learners without retraining them first. `HtmlReport.build(reference_store=...)`
folds these in only for a game the user is already reporting on, so a custom
game -- which the library has no runs for -- never shows them.
"""
from __future__ import annotations

from pathlib import Path

from rl_card_lib.report import RunStore

#: Root of the committed reference store, in the same layout RunStore writes:
#: ``models/<game>__<agent>/run.json`` and ``baselines/<game>.json``.
REFERENCE_DIR = Path(__file__).resolve().parent / "reference_data"


def library_reference_store() -> RunStore:
    """A RunStore over the committed library reference runs and baselines.

    RunStore degrades to no runs and no baselines when the directory is absent
    or empty, so callers need no guard before a checkout that carries the data.
    """
    return RunStore(REFERENCE_DIR)
