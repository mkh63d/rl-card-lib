"""Experiment harness for the thesis_notes re-runs.

Nothing under `packages/` is modified *by this module*, but the library it
calls is no longer the one the thesis text describes: PRs #24-#34 merged the
corrections that used to live here as subclasses. What is left is therefore the
inverse of what this file once held -- the pieces that reconstruct the *pre-fix*
behaviour on top of today's library, so the `asis` arm can still be measured:

    ConflatedTruncationMixin  puts a time-limit truncation back into the TD
                              target as if it were a terminal state (pre-#24)
    frozen_exploration        restores epsilon/eval-mode around a measurement
    evaluate_* on_pool        greedy evaluation over a fixed list of deal seeds,
                              on the same Klondike rules the run trained under

`PooledEnv` is gone: `CardGameEnv(deal_seeds=..., deal_rng_seed=...)` does the
same job in the library now, and records `dealt_seeds` itself.

See thesis_notes/protocol.md and thesis_notes/diagnosis.md for why each exists.
"""

from __future__ import annotations

import contextlib
import time
from typing import Any, Optional

import numpy as np

from rl_card_lib.env import CardGameEnv
from rl_card_lib.games import KlondikeSolitaire, Macao
from rl_card_lib.trainer import SelfPlayTrainer, Trainer


# ---------------------------------------------------------------------------
# 1. Evaluation must not consume the exploration schedule
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def frozen_exploration(*agents: Any):
    """Restore epsilon, the episode counter and the action rule after a block.

    Since #28 the trainer decays epsilon in `on_episode_end()` rather than in
    `Agent.reset()`, so an evaluation no longer advances the schedule on its
    own. This stays because the measurement functions below still flip
    `training` and `eval_greedy` on the agent they are handed, and a caller that
    goes on to train the same object must get it back unchanged.
    """
    saved = [
        (a, getattr(a, "epsilon", None), getattr(a, "episodes", None),
         getattr(a, "training", None), getattr(a, "eval_greedy", None))
        for a in agents
    ]
    try:
        yield
    finally:
        for agent, epsilon, episodes, training, eval_greedy in saved:
            if epsilon is not None:
                agent.epsilon = epsilon
            if episodes is not None:
                agent.episodes = episodes
            if eval_greedy is not None:
                agent.eval_greedy = eval_greedy
            if training is True:
                agent.train()
            elif training is False:
                agent.eval()


def set_eval_action_rule(agent: Any, eval_greedy: Optional[bool]) -> None:
    """Ask `agent` for argmax or sampling at evaluation time, if it has a say.

    Only PPO does: `PPOAgent.eval_greedy` (PR #33, issue #21) picks between the argmax
    of the policy and a sample from it, and on Klondike those are very different
    policies -- see diagnosis.md D11. The DQN family's greedy policy is the one
    it learned, so there is nothing to choose and the attribute is absent.
    """
    if eval_greedy is not None and hasattr(agent, "eval_greedy"):
        agent.eval_greedy = eval_greedy


# ---------------------------------------------------------------------------
# 2. A time-limit cut *was* treated as a terminal state -- the `asis` arm
# ---------------------------------------------------------------------------

class ConflatedTruncationMixin:
    """Teach the learner that a truncated step has no future, as it did pre-#24.

    Before #24, `Trainer._run_episode` computed `done = terminated or truncated`
    and handed that single flag to `agent.learn()`. Every value-based target in
    the library multiplies the bootstrap by `(1 - done)`, so a step cut by the
    step cap was taught that the future is worth exactly zero. On Klondike,
    where essentially every episode ended at the 300-step cap, that was the last
    transition of every episode.

    Master now passes `terminated` and forwards `truncated` separately. This
    mixin puts the old behaviour back, so the `asis` arm measures the library
    the thesis describes while running on the same commit as every other arm: it
    folds `truncated` into `done` and does *not* forward it, so an agent
    declaring `accepts_truncated` (PPO) sees the pre-fix target too.
    """

    def _learn(self, agent, observation, action, reward, next_observation,
               done, info, truncated: bool = False):
        return super()._learn(
            agent, observation, action, reward, next_observation,
            bool(done) or bool(truncated), info,
        )


class LegacyTrainer(ConflatedTruncationMixin, Trainer):
    pass


class LegacySelfPlayTrainer(ConflatedTruncationMixin, SelfPlayTrainer):
    pass


# ---------------------------------------------------------------------------
# 3. Greedy evaluation over a fixed pool of deals
# ---------------------------------------------------------------------------

def evaluate_klondike_on_pool(
    agent, seeds: list[int], max_steps: int = 300, reward_mode: str = "shaped",
    max_passes: Optional[int] = None, eval_greedy: Optional[bool] = None,
) -> dict:
    """Play every deal in `seeds` greedily and report the aggregate.

    `max_passes` has to be the value the run trained under: since #30 the
    bundled Klondike caps passes through the stock, which is what makes a deal
    losable at all. Evaluating on the unlimited-pass default would score the
    agent on a different game than the one it learned.
    """
    game = KlondikeSolitaire(reward_mode=reward_mode, max_passes=max_passes)
    env = CardGameEnv(game, max_steps=max_steps)
    if hasattr(agent, "bind"):
        agent.bind(env)

    rewards, cards, steps, wins = [], [], [], 0
    terminated_count = truncated_count = 0
    started = time.perf_counter()

    with frozen_exploration(agent):
        set_eval_action_rule(agent, eval_greedy)
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
    eval_greedy: Optional[bool] = None,
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
        set_eval_action_rule(agent, eval_greedy)
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
                                 opponent_seed: int = 0,
                                 eval_greedy: Optional[bool] = None) -> dict:
    """Evaluate against both scripted opponents on the same deals."""
    from rl_card_lib.agents import RandomAgent
    from rl_card_lib.games.heuristics import MacaoHeuristicAgent

    out: dict = {}
    opponents = {
        "random": RandomAgent(action_size=Macao.MAX_ACTIONS, seed=opponent_seed),
        "heuristic": MacaoHeuristicAgent(seed=opponent_seed),
    }
    for name, opponent in opponents.items():
        result = evaluate_macao_on_pool(agent, opponent, seeds, max_steps,
                                        eval_greedy=eval_greedy)
        for key, value in result.items():
            out[f"{key}_vs_{name}"] = value
    out["episodes"] = len(seeds)
    return out


# ---------------------------------------------------------------------------
# 4. Per-episode TRAIN recorder
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
    "ConflatedTruncationMixin",
    "LegacySelfPlayTrainer",
    "LegacyTrainer",
    "block_average",
    "evaluate_klondike_on_pool",
    "evaluate_macao_on_pool",
    "evaluate_macao_suite_on_pool",
    "frozen_exploration",
    "make_train_recorder",
    "set_eval_action_rule",
]
