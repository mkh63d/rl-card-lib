"""Gymnasium-compatible wrapper for `Game` objects."""
from typing import Any, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # pragma: no cover - optional dependency
    gym = None
    spaces = None


class GymEnvWrapper:
    def __init__(self, game: Any):
        self.game = game
        self.observation_space = None
        self.action_space = None

        try:
            shape = self.game.get_observation_shape()
            if spaces is not None and shape is not None:
                self.observation_space = spaces.Box(
                    low=-np.inf, high=np.inf, shape=tuple(shape), dtype=np.float32
                )
        except Exception:
            self.observation_space = None

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
