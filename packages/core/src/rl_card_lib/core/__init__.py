"""Core subpackage exported for the `packages/core` distribution.

This mirrors the project-level `src/rl_card_lib/core` implementation
so the package can be installed independently.
"""

from rl_card_lib.core.game import Game
from rl_card_lib.core.gym_wrapper import GymEnvWrapper
from rl_card_lib.core.trainer import Trainer

__all__ = [
    "Game",
    "GymEnvWrapper",
    "Trainer",
]
