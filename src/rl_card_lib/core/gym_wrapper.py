"""Gymnasium-compatible wrapper for `Game` objects.

This adapter exposes a minimal `Env`-like interface so games can be
used with typical RL training loops and agents.
"""
from typing import Any, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # pragma: no cover - optional dependency
    gym = None
    spaces = None


class GymEnvWrapper:
    """Wrap a `Game`-compatible instance and expose `reset`/`step`.

    The wrapper will attempt to construct `observation_space` and
    `action_space` from the target game when the corresponding
    methods are available.
    """

    def __init__(self, game: Any):
        self.game = game
        self.observation_space = None
        self.action_space = None

        # Try to construct observation_space
        try:
            shape = self.game.get_observation_shape()
            if spaces is not None and shape is not None:
                self.observation_space = spaces.Box(
                    low=-np.inf, high=np.inf, shape=tuple(shape), dtype=np.float32
                )
        except Exception:
            self.observation_space = None

        # Try to construct action_space
        try:
            n = self.game.get_action_space_size()
            if spaces is not None and n is not None:
                self.action_space = spaces.Discrete(int(n))
        except Exception:
            self.action_space = None

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> Tuple[Any, dict]:
        if seed is not None:
            try:
                np.random.seed(seed)
            except Exception:
                pass
        obs = self.game.reset()
        return obs, {}

    def step(self, action: int) -> Tuple[Any, float, bool, bool, dict]:
        return self.game.step(action)

    def render(self, mode: str = "human") -> Any:
        return self.game.render()

    def close(self) -> None:
        return None
