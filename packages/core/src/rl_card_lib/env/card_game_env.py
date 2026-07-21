"""Gymnasium-compatible wrappers for card game environments."""

from __future__ import annotations

import inspect
from typing import Any, Optional, Tuple
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # pragma: no cover - optional dependency
    gym = None
    spaces = None


class CardGameEnv:
    """Wrap a CardGame instance to provide a Gymnasium-like API."""

    def __init__(
        self,
        game: Any,
        max_steps: Optional[int] = None,
        render_mode: Optional[str] = None,
        invalid_action_reward: float = -1.0,
        repeated_position_penalty: float = 0.0,
    ):
        """
        Wrap a game in a Gymnasium-like environment.

        Args:
            game: Game instance to wrap
            max_steps: Steps after which the episode truncates (None for no cap)
            render_mode: None, "human" (print) or "ansi" (return string)
            invalid_action_reward: Reward returned for an illegal action; the
                game itself is not stepped
            repeated_position_penalty: Added to the reward (use a negative
                value) whenever a step lands in a position already seen this
                episode. Games with reversible moves let an agent shuffle in
                circles forever; this makes each lap cost something. Repeats
                are flagged in info["repeated_position"] either way.
        """
        self.game = game
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.invalid_action_reward = invalid_action_reward
        self.repeated_position_penalty = repeated_position_penalty
        self._step_count = 0
        self._seen_positions: set[int] = set()

        obs_shape = None
        try:
            obs_shape = self.game.get_observation_shape()
        except Exception:
            obs_shape = None

        if spaces is not None and obs_shape is not None:
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=tuple(obs_shape),
                dtype=np.float32,
            )
        else:
            self.observation_space = None

        action_size = None
        try:
            action_size = int(self.game.get_action_space_size())
        except Exception:
            action_size = None

        if spaces is not None and action_size is not None:
            self.action_space = spaces.Discrete(action_size)
        else:
            self.action_space = None

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        """
        Reset the wrapped game.

        A seed is forwarded to the game's own reset() when it accepts one, so
        the deal is reproducible. Nothing global is reseeded: doing so used to
        silently perturb every other RNG consumer in the process (and never
        made the deal reproducible anyway, since the games shuffle with their
        own RNG).
        """
        self._step_count = 0
        if seed is not None and self._game_reset_accepts_seed():
            observation = self.game.reset(seed=seed)
        else:
            observation = self.game.reset()
        observation = np.asarray(observation, dtype=np.float32)
        self._seen_positions = {hash(observation.tobytes())}
        info = {
            "legal_actions": self.get_legal_actions(),
        }
        return observation, info

    def _game_reset_accepts_seed(self) -> bool:
        """Whether the wrapped game's reset() takes a seed keyword."""
        try:
            return "seed" in inspect.signature(self.game.reset).parameters
        except (TypeError, ValueError):
            return False

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        legal = self.get_legal_actions()
        if legal and action not in legal:
            observation = self.game.get_observation()
            observation = np.asarray(observation, dtype=np.float32)
            info = {
                "invalid_action": True,
                "legal_actions": legal,
                "winner": self.game.winner,
            }
            return observation, float(self.invalid_action_reward), False, False, info

        observation, reward, terminated, truncated, info = self.game.step(action)
        self._step_count += 1

        if self.max_steps is not None and self._step_count >= self.max_steps:
            truncated = True

        observation = np.asarray(observation, dtype=np.float32)
        info = dict(info)
        info.setdefault("legal_actions", self.get_legal_actions())
        info.setdefault("winner", self.game.winner)

        # Repeated-position tracking, keyed on what the agent can see. Catches
        # reversible-move loops in any game rather than needing each game to
        # defend against them separately.
        position = hash(observation.tobytes())
        if position in self._seen_positions:
            info["repeated_position"] = True
            reward += self.repeated_position_penalty
        else:
            self._seen_positions.add(position)

        if self.render_mode == "human":
            self.render()

        return observation, float(reward), bool(terminated), bool(truncated), info

    def render(self) -> Optional[str]:
        if self.render_mode is None:
            return None
        rendered = self.game.render()
        if self.render_mode == "human":
            print(rendered)
            return None
        if self.render_mode == "ansi":
            return rendered
        return rendered

    def close(self) -> None:
        return None

    def get_legal_actions(self) -> list[int]:
        try:
            return list(self.game.get_legal_actions())
        except Exception:
            return []

    def get_legal_action_mask(self) -> np.ndarray:
        mask = np.zeros(int(self.game.get_action_space_size()), dtype=bool)
        for action in self.get_legal_actions():
            mask[action] = True
        return mask

    def action_to_string(self, action: int) -> str:
        try:
            return self.game.action_to_string(action)
        except Exception:
            return f"Action {action}"


class MaskedCardGameEnv(CardGameEnv):
    """Environment wrapper that exposes an action mask in the observation."""

    def __init__(
        self,
        game: Any,
        max_steps: Optional[int] = None,
        render_mode: Optional[str] = None,
        invalid_action_reward: float = -1.0,
        repeated_position_penalty: float = 0.0,
    ):
        super().__init__(
            game,
            max_steps=max_steps,
            render_mode=render_mode,
            invalid_action_reward=invalid_action_reward,
            repeated_position_penalty=repeated_position_penalty,
        )

        if spaces is not None and self.observation_space is not None:
            action_size = int(self.action_space.n) if self.action_space is not None else 0
            self.observation_space = spaces.Dict({
                "observation": self.observation_space,
                "action_mask": spaces.MultiBinary(action_size),
            })

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[dict, dict]:
        observation, info = super().reset(seed=seed, options=options)
        mask = self._get_action_mask_int8()
        return {"observation": observation, "action_mask": mask}, info

    def step(self, action: int) -> Tuple[dict, float, bool, bool, dict]:
        observation, reward, terminated, truncated, info = super().step(action)
        mask = self._get_action_mask_int8()
        return {"observation": observation, "action_mask": mask}, reward, terminated, truncated, info

    def _get_action_mask_int8(self) -> np.ndarray:
        mask = np.zeros(int(self.game.get_action_space_size()), dtype=np.int8)
        for action in self.get_legal_actions():
            mask[action] = 1
        return mask
