"""Gymnasium-compatible wrapper for `Game` objects."""
import inspect
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
        # Forward the seed to the game's own reset() when it takes one; never
        # reseed a global RNG (that used to perturb unrelated randomness).
        if seed is not None and self._game_reset_accepts_seed():
            obs = self.game.reset(seed=seed)
        else:
            obs = self.game.reset()
        return obs, {}

    def _game_reset_accepts_seed(self) -> bool:
        """Whether the wrapped game's reset() takes a seed keyword."""
        try:
            return "seed" in inspect.signature(self.game.reset).parameters
        except (TypeError, ValueError):
            return False

    def step(self, action: int) -> Tuple[Any, float, bool, bool, dict]:
        return self.game.step(action)

    def render(self, mode: str = "human") -> Any:
        return self.game.render()

    def close(self) -> None:
        return None
