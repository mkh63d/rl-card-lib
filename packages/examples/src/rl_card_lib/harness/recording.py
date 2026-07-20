"""Per-episode side data the trainer does not record.

`Trainer.train()` accepts a callback that fires once per episode, after the
episode has finished and before the next `env.reset()`, so the game still
holds its terminal position when we look at it. `Trainer.evaluate()` calls
`_run_episode` directly and does not fire the callback, so mid-training
evaluations cannot pollute these series.

This is why no change to the core trainer was needed to chart cards-to-
foundation, the exploration schedule, or Q-table growth.
"""

from __future__ import annotations

import time
from typing import Callable


def make_episode_recorder(env, agent, game_kind: str) -> tuple[Callable, dict]:
    """Return (callback, extras); `extras` fills as training proceeds.

    Args:
        env: The environment being trained on (its `.game` is read)
        agent: The learning agent (its epsilon / table size are read)
        game_kind: "klondike" or "macao"

    Returns:
        A callback for `Trainer.train(callback=...)` and the dict it fills.
    """
    extras: dict = {
        "epsilon": [], "cards_up": [], "wall_clock": [], "table_size": [],
    }
    game = getattr(env, "game", None)
    last = {"at": time.perf_counter()}

    def callback(episode_metrics: dict) -> bool:
        now = time.perf_counter()
        extras["wall_clock"].append(now - last["at"])
        last["at"] = now

        # Absent on PPO, which explores by sampling its policy.
        extras["epsilon"].append(getattr(agent, "epsilon", None))
        # Present only on the tabular agent; this is the series that shows the
        # table growing without bound.
        extras["table_size"].append(getattr(agent, "table_size", None))

        if game_kind == "klondike" and game is not None:
            foundations = getattr(game, "foundations", None)
            extras["cards_up"].append(
                None if foundations is None
                else sum(len(pile) for pile in foundations)
            )
        else:
            extras["cards_up"].append(None)

        return True  # never abort training

    return callback, extras
