"""Gymnasium-compatible wrappers for card game environments."""

from __future__ import annotations

import inspect
import random
from typing import Any, Iterable, Optional, Tuple
import numpy as np

# Both pyproject files declare gymnasium>=0.29.0 as a hard dependency, so a
# correct install always has it. It used to be imported under a try/except that
# set gym = None -- a fallback no supported install could reach, and one that
# ruled out subclassing gymnasium.Env, which is the whole point of the API.
import gymnasium as gym
from gymnasium import spaces


#: Accepted values for CardGameEnv's `deal_order`.
DEAL_ORDERS = ("random", "cycle")


class CardGameEnv(gym.Env):
    """Wrap a CardGame instance in a Gymnasium environment.

    A genuine `gymnasium.Env`: `gymnasium.utils.env_checker.check_env` passes,
    Gymnasium wrappers accept it, and Stable-Baselines3 trains on it directly.
    Following the Gymnasium *conventions* was not enough -- every one of those
    consumers tests `isinstance(env, gymnasium.Env)` before anything else.
    """

    #: Render modes `render()` honours. Required by the Gymnasium contract.
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(
        self,
        game: Any,
        max_steps: Optional[int] = None,
        render_mode: Optional[str] = None,
        invalid_action_reward: float = -1.0,
        repeated_position_penalty: float = 0.0,
        deal_seeds: Optional[Iterable[int]] = None,
        deal_rng_seed: int = 0,
        deal_order: str = "random",
    ):
        """
        Wrap a game in a Gymnasium-like environment.

        Args:
            game: Game instance to wrap
            max_steps: Steps after which the episode truncates (None for no
                cap). Illegal actions count against the cap too, so an agent
                that only proposes illegal ones still ends its episode.
            render_mode: None, "human" (print) or "ansi" (return string)
            invalid_action_reward: Reward returned for an illegal action; the
                game itself is not stepped, but the step is still counted
                against `max_steps`
            repeated_position_penalty: Added to the reward (use a negative
                value) whenever a step lands in a position already seen this
                episode. Games with reversible moves let an agent shuffle in
                circles forever; this makes each lap cost something. Repeats
                are flagged in info["repeated_position"] either way.
            deal_seeds: Pool of deal seeds an unseeded reset() draws from (None
                to leave an unseeded reset random, which is what a Gymnasium env
                does by default). A pool makes the whole episode stream
                reproducible from `deal_rng_seed` alone, and lets an experiment
                say which set of deals it trained or evaluated on -- see
                `rl_card_lib.harness.deals` for the bundled TRAIN/TEST split.
            deal_rng_seed: Seeds the private RNG that picks deals out of the
                pool. Nothing global is read or written, so the deal stream is
                independent of the agents' exploration noise.
            deal_order: "random" draws a seed from the pool each episode;
                "cycle" walks the pool in order and wraps, which is what an
                evaluation env wants -- combined with reset_deal_cursor() every
                evaluation then replays the identical deals.
        """
        if deal_order not in DEAL_ORDERS:
            raise ValueError(
                f"deal_order must be one of {DEAL_ORDERS}, got {deal_order!r}"
            )

        super().__init__()

        self.game = game
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.invalid_action_reward = invalid_action_reward
        self.repeated_position_penalty = repeated_position_penalty
        self.deal_seeds = list(deal_seeds) if deal_seeds is not None else None
        self.deal_rng_seed = deal_rng_seed
        self.deal_order = deal_order
        #: Every deal seed this env has dealt, in order. The record of what an
        #: agent was actually trained or measured on.
        self.dealt_seeds: list[int] = []
        self.last_deal_seed: Optional[int] = None
        self._deal_rng = random.Random(deal_rng_seed)
        self._deal_index = 0
        self._step_count = 0
        self._seen_positions: set[int] = set()

        obs_shape = None
        try:
            obs_shape = self.game.get_observation_shape()
        except Exception:
            obs_shape = None

        if obs_shape is not None:
            try:
                low, high = self.game.get_observation_bounds()
            except AttributeError:
                # A duck-typed game predating the hook. Deliberately narrower
                # than the `except Exception` above: a game that *has* the
                # method and raises out of it, or hands back arrays of the
                # wrong shape, has a bug, and letting Box raise is how it gets
                # found rather than silently falling back to (-inf, inf).
                low, high = -np.inf, np.inf
            self.observation_space = spaces.Box(
                low=low,
                high=high,
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

        if action_size is not None:
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

        When no seed is given, a `deal_seeds` pool supplies one. Without a pool
        the game reshuffles from its own RNG, which nothing seeded -- fine for a
        demo, but it makes an experiment's deals unreproducible and leaves them
        belonging to no declared set.
        """
        # Seeds self.np_random, which the Gymnasium contract requires to exist
        # and to be reseeded here. The library's own randomness lives in the
        # game and in _deal_rng; this is for consumers that expect the standard
        # per-env generator.
        super().reset(seed=seed)

        self._step_count = 0
        if seed is None and self.deal_seeds:
            seed = self._next_deal_seed()
        self.last_deal_seed = seed
        if seed is not None:
            self.dealt_seeds.append(seed)
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

    def _next_deal_seed(self) -> int:
        """Take the next deal out of the pool, per `deal_order`."""
        if self.deal_order == "cycle":
            seed = self.deal_seeds[self._deal_index % len(self.deal_seeds)]
            self._deal_index += 1
            return seed
        return self._deal_rng.choice(self.deal_seeds)

    def reset_deal_cursor(self) -> None:
        """Rewind the deal pool to its start.

        An evaluation env built with deal_order="cycle" replays the identical
        deals after this, so two evaluations of the same agent are comparable
        and two agents are measured on the same board. Never call it on a
        training env mid-run: that would restart the deal stream.
        """
        self._deal_index = 0
        self._deal_rng = random.Random(self.deal_rng_seed)

    def _game_reset_accepts_seed(self) -> bool:
        """Whether the wrapped game's reset() takes a seed keyword.

        True for an explicit ``seed`` parameter or a ``**kwargs`` catch-all --
        forwarding a seed to the latter is harmless and keeps determinism for
        games that accept it that way.
        """
        try:
            params = inspect.signature(self.game.reset).parameters
        except (TypeError, ValueError):
            return False
        return "seed" in params or any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
        )

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        legal = self.get_legal_actions()
        if legal and action not in legal:
            # An illegal action costs a step even though the game is not
            # stepped. Leaving it uncounted meant an agent that proposes
            # nothing but illegal actions never reached max_steps -- the
            # episode livelocked rather than truncating. The library's own
            # agents are handed info["legal_actions"] and never hit it; an
            # unmasked external policy hits it immediately.
            self._step_count += 1
            truncated = (
                self.max_steps is not None and self._step_count >= self.max_steps
            )
            observation = self.game.get_observation()
            observation = np.asarray(observation, dtype=np.float32)
            info = {
                "invalid_action": True,
                "legal_actions": legal,
                "winner": self.game.winner,
            }
            return (
                observation,
                float(self.invalid_action_reward),
                False,
                truncated,
                info,
            )

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

    def action_masks(self) -> np.ndarray:
        """The mask, under the name the wider ecosystem looks for.

        `sb3-contrib` finds an env's mask by calling a method named exactly
        `action_masks()` -- `get_action_masks()` reaches it through
        `env.get_wrapper_attr("action_masks")` on a bare env and through
        `VecEnv.env_method("action_masks")` on a vectorised one. It does *not*
        read the `action_mask` field of `MaskedCardGameEnv`'s observation,
        which is why exposing that field alone left `MaskablePPO` sampling
        illegal actions like any unmasked policy.

        Defined on `CardGameEnv` rather than on the masked subclass so both
        halves of the pair are maskable: the `Dict` observation and this method
        are independent, and an algorithm needing only the mask can take the
        plain `Box` env. It delegates rather than rebuilding the mask, so the
        two spellings cannot drift apart.
        """
        return self.get_legal_action_mask()

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
        deal_seeds: Optional[Iterable[int]] = None,
        deal_rng_seed: int = 0,
        deal_order: str = "random",
    ):
        super().__init__(
            game,
            max_steps=max_steps,
            render_mode=render_mode,
            invalid_action_reward=invalid_action_reward,
            repeated_position_penalty=repeated_position_penalty,
            deal_seeds=deal_seeds,
            deal_rng_seed=deal_rng_seed,
            deal_order=deal_order,
        )

        if self.observation_space is not None:
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
