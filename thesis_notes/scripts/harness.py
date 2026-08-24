"""Experiment harness for the thesis_notes re-runs.

Nothing under `packages/` is modified. Everything this module adds is either a
thin subclass or a plain function, so the library still behaves exactly as the
thesis describes it, and each deviation is visible here as a named object:

    PooledEnv                 draws every episode's deal from a named seed pool
    frozen_exploration        evaluation no longer decays the epsilon schedule
    TimeLimitBootstrapMixin   a time-limit truncation stops being treated as a
                              terminal state in the TD target
    evaluate_* on_pool        greedy evaluation over a fixed list of deal seeds

See thesis_notes/protocol.md and thesis_notes/diagnosis.md for why each exists.
"""

from __future__ import annotations

import contextlib
import random
import time
from typing import Any, Callable, Iterable, Optional

import numpy as np

from rl_card_lib.env import CardGameEnv
from rl_card_lib.games import KlondikeSolitaire, Macao
from rl_card_lib.trainer import SelfPlayTrainer, Trainer


# ---------------------------------------------------------------------------
# 1. Deals come from a declared pool
# ---------------------------------------------------------------------------

class PooledEnv(CardGameEnv):
    """A CardGameEnv whose `reset()` always draws a deal from a seed pool.

    The stock `CardGameEnv.reset()` forwards a seed when it is given one and
    otherwise lets the game reshuffle from its own RNG, which the trainer never
    seeds -- so training deals are unreproducible and belong to no declared set.
    This subclass fills that gap: an unseeded `reset()` picks a seed uniformly
    from `pool` using a private `random.Random`, so the deal sequence is
    reproducible from `deal_rng_seed` alone and never touches the global RNG
    the agents draw their exploration noise from.

    It also records `info["_terminated"]`, which `TimeLimitBootstrapMixin`
    needs in order to tell a real terminal state from a time-limit cut.
    """

    def __init__(self, game, pool: Iterable[int], deal_rng_seed: int = 0, **kwargs):
        super().__init__(game, **kwargs)
        self.pool = list(pool)
        self.deal_rng_seed = deal_rng_seed
        self._deal_rng = random.Random(deal_rng_seed)
        self.dealt_seeds: list[int] = []

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is None:
            seed = self._deal_rng.choice(self.pool)
        self.dealt_seeds.append(seed)
        return super().reset(seed=seed, options=options)

    def step(self, action: int):
        obs, reward, terminated, truncated, info = super().step(action)
        info["_terminated"] = bool(terminated)
        info["_truncated"] = bool(truncated)
        return obs, reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# 2. Evaluation must not consume the exploration schedule
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def frozen_exploration(*agents: Any):
    """Restore epsilon and the episode counter after a block.

    `Agent.reset()` is what decays epsilon (once per episode, by design), and
    every evaluation episode calls it -- so an evaluation permanently advances
    the exploration schedule of the agent it is measuring. Wrapping evaluation
    in this makes the measurement free of side effects.
    """
    saved = [
        (a, getattr(a, "epsilon", None), getattr(a, "episodes", None),
         getattr(a, "training", None))
        for a in agents
    ]
    try:
        yield
    finally:
        for agent, epsilon, episodes, training in saved:
            if epsilon is not None:
                agent.epsilon = epsilon
            if episodes is not None:
                agent.episodes = episodes
            if training is True:
                agent.train()
            elif training is False:
                agent.eval()


# ---------------------------------------------------------------------------
# 3. A time-limit cut is not a terminal state
# ---------------------------------------------------------------------------

class TimeLimitBootstrapMixin:
    """Bootstrap through a truncation instead of zeroing the target at it.

    `Trainer._run_episode` computes `done = terminated or truncated` and hands
    that single flag to `agent.learn()`. Every value-based target in the library
    then multiplies the bootstrap by `(1 - done)`, so a step that was cut by the
    step cap is taught that the future is worth exactly zero. On Klondike, where
    essentially every episode ends at the 300-step cap, that is the last
    transition of every episode. This mixin passes the game's own `terminated`
    instead, which is the standard time-limit handling.
    """

    def _learn(self, agent, observation, action, reward, next_observation, done, info):
        if done and info.get("_terminated") is False:
            done = False
        return super()._learn(
            agent, observation, action, reward, next_observation, done, info,
        )


class BootstrapTrainer(TimeLimitBootstrapMixin, Trainer):
    pass


class BootstrapSelfPlayTrainer(TimeLimitBootstrapMixin, SelfPlayTrainer):
    pass


# ---------------------------------------------------------------------------
# 4. Greedy evaluation over a fixed pool of deals
# ---------------------------------------------------------------------------

def evaluate_klondike_on_pool(
    agent, seeds: list[int], max_steps: int = 300, reward_mode: str = "shaped",
) -> dict:
    """Play every deal in `seeds` greedily and report the aggregate.

    Unlike `harness.evaluation.evaluate_klondike`, the deal really is fixed:
    the seed goes to `game.reset(seed=...)`, not to the global `random` module
    (which the game's private RNG never reads).
    """
    game = KlondikeSolitaire(reward_mode=reward_mode)
    env = CardGameEnv(game, max_steps=max_steps)
    if hasattr(agent, "bind"):
        agent.bind(env)

    rewards, cards, steps, wins = [], [], [], 0
    terminated_count = truncated_count = 0
    started = time.perf_counter()

    with frozen_exploration(agent):
        agent.eval()
        for seed in seeds:
            observation, info = env.reset(seed=seed)
            agent.reset()
            total, moves = 0.0, 0
            terminated = truncated = False
            for _ in range(max_steps):
                action = agent.select_action(observation, info.get("legal_actions"))
                observation, reward, terminated, truncated, info = env.step(action)
                total += reward
                moves += 1
                if terminated or truncated:
                    break
            rewards.append(total)
            cards.append(sum(len(pile) for pile in game.foundations))
            steps.append(moves)
            wins += 1 if game.winner == 0 else 0
            terminated_count += int(bool(terminated))
            truncated_count += int(bool(truncated))

    n = len(seeds)
    return {
        "episodes": n,
        "reward": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "cards_up": float(np.mean(cards)),
        "cards_up_std": float(np.std(cards)),
        "steps": float(np.mean(steps)),
        "win_rate": wins / n if n else 0.0,
        "terminated_rate": terminated_count / n if n else 0.0,
        "truncated_rate": truncated_count / n if n else 0.0,
        "seconds": time.perf_counter() - started,
    }


def evaluate_macao_on_pool(
    agent, opponent, seeds: list[int], max_steps: int = 200,
) -> dict:
    """Play every deal in `seeds` greedily against `opponent`, agent seated as 0."""
    game = Macao(num_players=2)
    env = CardGameEnv(game, max_steps=max_steps)
    for participant in (agent, opponent):
        if hasattr(participant, "bind"):
            participant.bind(env)

    wins = draws = losses = 0
    steps, rewards = [], []
    terminated_count = truncated_count = 0
    started = time.perf_counter()

    with frozen_exploration(agent, opponent):
        agent.eval()
        if hasattr(opponent, "eval"):
            opponent.eval()
        for seed in seeds:
            observation, info = env.reset(seed=seed)
            agent.reset()
            if hasattr(opponent, "reset"):
                opponent.reset()
            moves, total = 0, 0.0
            terminated = truncated = False
            for _ in range(max_steps):
                actor = game.current_player_idx
                chooser = agent if actor == 0 else opponent
                action = chooser.select_action(observation, info.get("legal_actions"))
                observation, reward, terminated, truncated, info = env.step(action)
                if actor == 0:
                    total += reward
                moves += 1
                if terminated or truncated:
                    break
            steps.append(moves)
            rewards.append(total)
            if game.winner == 0:
                wins += 1
            elif game.winner is None:
                draws += 1
            else:
                losses += 1
            terminated_count += int(bool(terminated))
            truncated_count += int(bool(truncated))

    n = len(seeds)
    return {
        "episodes": n,
        "win_rate": wins / n if n else 0.0,
        "draw_rate": draws / n if n else 0.0,
        "loss_rate": losses / n if n else 0.0,
        "reward": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "steps": float(np.mean(steps)),
        "terminated_rate": terminated_count / n if n else 0.0,
        "truncated_rate": truncated_count / n if n else 0.0,
        "seconds": time.perf_counter() - started,
    }


def evaluate_macao_suite_on_pool(agent, seeds: list[int], max_steps: int = 200,
                                 opponent_seed: int = 0) -> dict:
    """Evaluate against both scripted opponents on the same deals."""
    from rl_card_lib.agents import RandomAgent
    from rl_card_lib.games.heuristics import MacaoHeuristicAgent

    out: dict = {}
    opponents = {
        "random": RandomAgent(action_size=Macao.MAX_ACTIONS, seed=opponent_seed),
        "heuristic": MacaoHeuristicAgent(seed=opponent_seed),
    }
    for name, opponent in opponents.items():
        result = evaluate_macao_on_pool(agent, opponent, seeds, max_steps)
        for key, value in result.items():
            out[f"{key}_vs_{name}"] = value
    out["episodes"] = len(seeds)
    return out


# ---------------------------------------------------------------------------
# 5. Per-episode TRAIN recorder
# ---------------------------------------------------------------------------

def make_train_recorder(env, agent, game_name: str):
    """Record the per-episode TRAIN series the report needs.

    `Trainer.train(callback=...)` fires once per episode after it finished and
    before the next reset, so the game still holds its terminal position.
    """
    series: dict[str, list] = {
        "reward": [], "steps": [], "win": [], "loss": [],
        "epsilon": [], "cards_up": [], "terminated": [],
    }
    game = getattr(env, "game", None)

    def callback(metrics: dict) -> bool:
        series["reward"].append(float(metrics.get("reward", 0.0)))
        series["steps"].append(int(metrics.get("steps", 0)))
        series["win"].append(int(metrics.get("win", 0)))
        series["loss"].append(float(metrics.get("loss", 0.0)))
        series["epsilon"].append(getattr(agent, "epsilon", None))
        if game_name == "klondike" and game is not None:
            series["cards_up"].append(
                sum(len(pile) for pile in game.foundations)
            )
        else:
            series["cards_up"].append(None)
        series["terminated"].append(int(bool(getattr(game, "done", False))))
        return True

    return callback, series


def block_average(values: list, block: int) -> list[dict]:
    """Aggregate a per-episode series into blocks of `block` episodes."""
    out = []
    for start in range(0, len(values), block):
        chunk = [v for v in values[start:start + block] if v is not None]
        if not chunk:
            out.append({"episode": start + block, "mean": None, "std": None})
            continue
        out.append({
            "episode": start + len(values[start:start + block]),
            "mean": float(np.mean(chunk)),
            "std": float(np.std(chunk)),
        })
    return out


__all__ = [
    "BootstrapSelfPlayTrainer",
    "BootstrapTrainer",
    "PooledEnv",
    "TimeLimitBootstrapMixin",
    "block_average",
    "evaluate_klondike_on_pool",
    "evaluate_macao_on_pool",
    "evaluate_macao_suite_on_pool",
    "frozen_exploration",
    "make_train_recorder",
]
