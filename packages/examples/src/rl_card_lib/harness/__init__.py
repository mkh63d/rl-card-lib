"""Importable pieces of the training and benchmarking scripts.

The scripts under `packages/examples/scripts/` are not on any import path, so
anything defined only there has to be duplicated to be reused -- which would
give the thesis two definitions of the evaluation protocol and two sets of
hyperparameters. This subpackage holds the one definition of each; the scripts
import from here.
"""

from rl_card_lib.harness.baselines import (
    klondike_baseline_agents,
    macao_baseline_agents,
    measure_baselines,
    run_klondike_baselines,
    run_macao_baselines,
)
from rl_card_lib.harness.evaluation import (
    evaluate_klondike,
    evaluate_macao,
    evaluate_macao_suite,
)
from rl_card_lib.harness.learners import (
    LEARNERS,
    agent_class_name,
    build_learner,
    checkpoint_suffix,
    epsilon_schedule,
)
from rl_card_lib.harness.recording import make_episode_recorder
from rl_card_lib.harness.reference import library_reference_store
from rl_card_lib.harness.registry import (
    SweepGame,
    is_registered,
    register_sweep_game,
    registered_sweep_games,
    sweep_game,
)

__all__ = [
    "LEARNERS",
    "SweepGame",
    "agent_class_name",
    "build_learner",
    "checkpoint_suffix",
    "epsilon_schedule",
    "evaluate_klondike",
    "evaluate_macao",
    "evaluate_macao_suite",
    "is_registered",
    "klondike_baseline_agents",
    "library_reference_store",
    "macao_baseline_agents",
    "make_episode_recorder",
    "measure_baselines",
    "register_sweep_game",
    "registered_sweep_games",
    "run_klondike_baselines",
    "run_macao_baselines",
    "sweep_game",
]
