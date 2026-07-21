"""Training report helpers.

`TrainingReport` describes how a run was configured; `RunRecord` describes what
it did. Charts and the HTML page live in `figures` and `html_report`, which are
not imported here because they need matplotlib -- an optional extra, so that
the JSON and Markdown half of this package stays importable without it.
"""

from rl_card_lib.report.run_record import (
    GAME_SPEC,
    METRICS,
    SCHEMA_VERSION,
    BaselineSet,
    RunRecord,
    RunStore,
    agent_label,
    format_metric,
    game_spec,
    metric_range,
    metric_spec,
    purge_checkpoints,
    reconstruct_epsilon,
    register_game,
    register_metric,
    run_id_for,
)
from rl_card_lib.report.training_report import TrainingReport

__all__ = [
    "TrainingReport",
    "RunRecord",
    "RunStore",
    "BaselineSet",
    "GAME_SPEC",
    "METRICS",
    "SCHEMA_VERSION",
    "agent_label",
    "format_metric",
    "game_spec",
    "metric_range",
    "metric_spec",
    "purge_checkpoints",
    "reconstruct_epsilon",
    "register_game",
    "register_metric",
    "run_id_for",
]
