"""Construction of the learning agents, with the hyperparameters the thesis uses.

Moved here from `scripts/train_agents.py` so the sweep, the single-game
scripts and the tests all build agents from one definition. Two copies of
these numbers would mean the report could describe a configuration that no
run actually used.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from rl_card_lib.agents import (
    Agent,
    DoubleDQNAgent,
    DQNAgent,
    PPOAgent,
    QLearningAgent,
)

LEARNERS = ("q_learning", "dqn", "double_dqn", "ppo")


def build_learner(kind: str, state_size: int, action_size: int, seed: int) -> Agent:
    """
    Construct one learning agent by name.

    Args:
        kind: One of LEARNERS
        state_size: Observation width
        action_size: Number of actions
        seed: Random seed

    Returns:
        The constructed agent
    """
    if kind == "q_learning":
        return QLearningAgent(
            action_size=action_size,
            learning_rate=0.1,
            gamma=0.95,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay=0.995,  # per episode: floor at ~600 episodes
            seed=seed,
        )
    if kind == "dqn":
        return DQNAgent(
            state_size=state_size, action_size=action_size,
            hidden_sizes=[256, 128], learning_rate=5e-4, gamma=0.95,
            epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995,
            buffer_size=50_000, batch_size=64, target_update_freq=500,
            device="cpu", seed=seed,
        )
    if kind == "double_dqn":
        return DoubleDQNAgent(
            state_size=state_size, action_size=action_size,
            hidden_sizes=[256, 128], learning_rate=5e-4, gamma=0.95,
            epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995,
            buffer_size=50_000, batch_size=64, target_update_freq=500,
            dueling=True, device="cpu", seed=seed,
        )
    if kind == "ppo":
        return PPOAgent(
            state_size=state_size, action_size=action_size,
            hidden_sizes=[256, 128], learning_rate=3e-4, gamma=0.95,
            gae_lambda=0.95, clip_epsilon=0.2, epochs=4, minibatch_size=64,
            rollout_steps=1024, entropy_coef=0.01, device="cpu", seed=seed,
        )
    raise ValueError(f"Unknown agent {kind!r}, expected one of {LEARNERS}")


def epsilon_schedule(kind: str) -> dict | None:
    """The declared exploration schedule, or None for agents without one.

    Used to reconstruct an epsilon curve for runs recorded before the sweep
    measured it directly. PPO explores by sampling its policy and has no
    epsilon at all.
    """
    if kind in ("q_learning", "dqn", "double_dqn"):
        return {"start": 1.0, "end": 0.05, "decay": 0.995}
    return None


def agent_class_name(kind: str) -> str:
    return {
        "q_learning": "QLearningAgent",
        "dqn": "DQNAgent",
        "double_dqn": "DoubleDQNAgent",
        "ppo": "PPOAgent",
    }.get(kind, kind)


def checkpoint_suffix(kind: str) -> str:
    """Q-learning pickles its table; everything else uses torch.save."""
    return ".pkl" if kind == "q_learning" else ".pt"


def _checkpoint_candidates(
    kind: str, game: str, run_store, checkpoint_dir,
) -> list[Path]:
    """Where a trained checkpoint for (game, kind) might live, best first.

    The run record is authoritative -- it names the exact file the trainer
    wrote -- so it is tried first; the conventional ``<dir>/<game>_<kind>/final``
    path is the fallback for a checkpoint that has no run record yet.
    """
    candidates: list[Path] = []
    if run_store is not None:
        run_json = run_store.run_dir(game, kind) / "run.json"
        if run_json.is_file():
            try:
                with open(run_json, "r", encoding="utf-8") as handle:
                    artifacts = json.load(handle).get("artifacts") or {}
            except (OSError, json.JSONDecodeError):
                artifacts = {}
            recorded = artifacts.get("checkpoint")
            if recorded:
                # Recorded relative to the training cwd; resolve against it.
                candidates.append(Path(recorded))
    if checkpoint_dir is not None:
        candidates.append(
            Path(checkpoint_dir) / f"{game}_{kind}" / f"final{checkpoint_suffix(kind)}"
        )
    return candidates


def load_trained_learner(
    kind: str,
    env,
    *,
    game: str,
    run_store=None,
    checkpoint_dir=None,
    seed: int = 0,
):
    """Rebuild a learner and load its trained weights, or return None.

    The architecture is rebuilt from `build_learner` (so DoubleDQN keeps its
    dueling network and the DQN family matches the trained `hidden_sizes`),
    then the checkpoint is loaded onto it. Returns None when no checkpoint is
    found -- e.g. an agent still training -- so callers can skip it rather than
    crash. Sizes come from the env, matching how the sweep built the agent.

    Args:
        kind: One of LEARNERS
        env: The game's env, for observation/action sizes and a load target
        game: Game name, to locate the run record and checkpoint directory
        run_store: Optional RunStore whose run.json names the checkpoint file
        checkpoint_dir: Optional base dir holding `<game>_<kind>/final.*`
        seed: Seed for the reconstructed agent

    Returns:
        The loaded agent in eval mode, or None if no checkpoint exists
    """
    path: Optional[Path] = next(
        (p for p in _checkpoint_candidates(kind, game, run_store, checkpoint_dir)
         if p.is_file()),
        None,
    )
    if path is None:
        return None

    agent = build_learner(
        kind, env.observation_space.shape[0], env.action_space.n, seed,
    )
    agent.load(str(path))
    agent.eval()
    return agent
