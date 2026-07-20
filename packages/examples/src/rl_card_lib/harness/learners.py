"""Construction of the learning agents, with the hyperparameters the thesis uses.

Moved here from `scripts/train_agents.py` so the sweep, the single-game
scripts and the tests all build agents from one definition. Two copies of
these numbers would mean the report could describe a configuration that no
run actually used.
"""

from __future__ import annotations

from rl_card_lib.agents import DoubleDQNAgent, DQNAgent, PPOAgent, QLearningAgent

LEARNERS = ("q_learning", "dqn", "double_dqn", "ppo")


def build_learner(kind: str, state_size: int, action_size: int, seed: int):
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
