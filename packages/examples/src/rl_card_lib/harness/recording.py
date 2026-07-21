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
from typing import Callable, Optional


def make_episode_recorder(
    env, agent, episode_extras: Optional[Callable] = None,
) -> tuple[Callable, dict]:
    """Return (callback, extras); `extras` fills as training proceeds.

    Always captures three duck-typed series -- exploration rate, per-episode
    wall clock, and Q-table size -- for whatever agent supports them. A game
    supplies its own progress signal through `episode_extras`, so the recorder
    is not tied to any particular game's state (Klondike's cards-to-foundation
    is just one such callable, declared in its registration).

    Args:
        env: The environment being trained on (its `.game` is read)
        agent: The learning agent (its epsilon / table size are read)
        episode_extras: Optional `(game, agent) -> dict[str, float|None]`,
            called after each episode while the game still holds its terminal
            state. It must return the same keys every call, or the series
            desync and RunRecord.validate() rejects the run.

    Returns:
        A callback for `Trainer.train(callback=...)` and the dict it fills.
    """
    extras: dict = {"epsilon": [], "wall_clock": [], "table_size": []}
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

        if episode_extras is not None and game is not None:
            for key, value in episode_extras(game, agent).items():
                extras.setdefault(key, []).append(value)

        return True  # never abort training

    return callback, extras
