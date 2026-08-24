"""Gymnasium-compatible wrapper for `Game` objects."""
from typing import Any, Optional

from rl_card_lib.env.card_game_env import CardGameEnv


class GymEnvWrapper(CardGameEnv):
    """Wrap a `Game` in a Gymnasium environment, with no step cap.

    This used to be a parallel implementation that re-derived a subset of
    `CardGameEnv` -- the two spaces, the seed-forwarding reset -- and drifted
    from it: it passed actions straight to the game, so a random action from
    `action_space` raised out of `Game.step()` instead of being absorbed by the
    invalid-action penalty, and `gymnasium.utils.env_checker.check_env` could
    not get past its first sampled action.

    It is now a thin `CardGameEnv` with `max_steps=None`, so it inherits the
    illegal-action handling, the 5-tuple contract and the `gymnasium.Env` base
    class rather than approximating them. Prefer `CardGameEnv` directly; this
    name is kept because it is part of the published API.
    """

    def __init__(self, game: Any, render_mode: Optional[str] = "ansi"):
        # render_mode defaults to "ansi" so render() still returns the rendered
        # string, as this class always did (CardGameEnv defaults to None, which
        # renders nothing).
        super().__init__(game, max_steps=None, render_mode=render_mode)
