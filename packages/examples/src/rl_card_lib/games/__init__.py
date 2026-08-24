"""Games module containing example card game implementations."""

from rl_card_lib.games.klondike import KlondikeSolitaire
from rl_card_lib.games.klondike_solver import solve_klondike
from rl_card_lib.games.macao import Macao
from rl_card_lib.games.heuristics import (
    KlondikeHeuristicAgent,
    MacaoHeuristicAgent,
)

__all__ = [
    "KlondikeSolitaire",
    "Macao",
    "KlondikeHeuristicAgent",
    "MacaoHeuristicAgent",
    "solve_klondike",
    "register_gym_envs",
    "registered_gym_ids",
]

# Register the bundled games with the sweep and report registries. Done last,
# after the classes above are bound, because registration imports the harness,
# which imports back from this module -- the names must exist first.
from rl_card_lib.games.registration import register_bundled_games  # noqa: E402
from rl_card_lib.games.gym_registration import (  # noqa: E402
    register_gym_envs,
    registered_gym_ids,
)

register_bundled_games()

# Gymnasium's own registry, so `gymnasium.make("rl_card_lib/Macao-v0")` works.
# After register_bundled_games(): gym_registration reads the step-cap constants
# out of registration, which the line above has just finished importing.
register_gym_envs()
