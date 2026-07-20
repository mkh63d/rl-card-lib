"""Training process parameter report helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from rl_card_lib.agents.base import Agent

try:
    from rl_card_lib.agents.dqn_agent import DQNAgent
except Exception:  # pragma: no cover - optional dependency
    DQNAgent = None

try:
    from rl_card_lib.agents.double_dqn_agent import DoubleDQNAgent
except Exception:  # pragma: no cover - optional dependency
    DoubleDQNAgent = None

try:
    from rl_card_lib.agents.ppo_agent import PPOAgent
except Exception:  # pragma: no cover - optional dependency
    PPOAgent = None

try:
    from rl_card_lib.agents.tabular import QLearningAgent
except Exception:  # pragma: no cover - optional dependency
    QLearningAgent = None

try:
    from rl_card_lib.agents.mcts_agent import MCTSAgent
except Exception:  # pragma: no cover - optional dependency
    MCTSAgent = None

try:
    from rl_card_lib.agents.heuristic import GreedyLookaheadAgent
except Exception:  # pragma: no cover - optional dependency
    GreedyLookaheadAgent = None


@dataclass
class TrainingReport:
    """Structured training parameter report."""

    environment: dict[str, Any]
    trainer: dict[str, Any]
    agent: dict[str, Any]
    dqn: Optional[dict[str, Any]] = None
    ppo: Optional[dict[str, Any]] = None
    qlearning: Optional[dict[str, Any]] = None
    search: Optional[dict[str, Any]] = None
    training: Optional[dict[str, Any]] = None

    @classmethod
    def from_trainer(
        cls,
        trainer: Any,
        *,
        episodes: Optional[int] = None,
        max_steps_per_episode: Optional[int] = None,
    ) -> "TrainingReport":
        env_info = _collect_env_info(getattr(trainer, "env", None))
        trainer_info = _collect_trainer_info(trainer)
        agent = getattr(trainer, "agent", None)
        agent_info = _collect_agent_info(agent)
        dqn_info = _collect_dqn_info(agent)
        ppo_info = _collect_ppo_info(agent)
        qlearning_info = _collect_qlearning_info(agent)
        search_info = _collect_search_info(agent)
        training_info = _collect_training_info(episodes, max_steps_per_episode)

        return cls(
            environment=env_info,
            trainer=trainer_info,
            agent=agent_info,
            dqn=dqn_info,
            ppo=ppo_info,
            qlearning=qlearning_info,
            search=search_info,
            training=training_info,
        )

    def as_dict(self) -> dict[str, Any]:
        data = {
            "environment": self.environment,
            "trainer": self.trainer,
            "agent": self.agent,
        }
        if self.training:
            data["training"] = self.training
        if self.dqn:
            data["dqn"] = self.dqn
        if self.ppo:
            data["ppo"] = self.ppo
        if self.qlearning:
            data["qlearning"] = self.qlearning
        if self.search:
            data["search"] = self.search
        return data

    def to_markdown(self) -> str:
        lines: list[str] = []

        _append_section(lines, "Training", self.training)
        _append_section(lines, "Environment", self.environment)
        _append_section(lines, "Trainer", self.trainer)
        _append_section(lines, "Agent", self.agent)
        _append_section(lines, "DQN", self.dqn)
        _append_section(lines, "PPO", self.ppo)
        _append_section(lines, "Q-learning", self.qlearning)
        _append_section(lines, "Search", self.search)

        return "\n".join(lines).strip()

    def to_json(self, indent: int = 2) -> str:
        """
        Render the report as a JSON string.

        Values that JSON has no encoding for (paths, devices, tuples) are
        stringified rather than rejected, since a report is for reading, not
        for round-tripping.

        Args:
            indent: Indentation passed to json.dumps

        Returns:
            JSON document with the same sections as as_dict()
        """
        import json

        return json.dumps(self.as_dict(), indent=indent, default=str)


def _collect_training_info(
    episodes: Optional[int],
    max_steps_per_episode: Optional[int],
) -> dict[str, Any]:
    info: dict[str, Any] = {}
    if episodes is not None:
        info["episodes"] = episodes
    if max_steps_per_episode is not None:
        info["max_steps_per_episode"] = max_steps_per_episode
    return info


def _collect_env_info(env: Any) -> dict[str, Any]:
    info: dict[str, Any] = {}
    if env is None:
        return info

    info["type"] = env.__class__.__name__

    for name in ["max_steps", "invalid_action_reward", "render_mode"]:
        if hasattr(env, name):
            info[name] = getattr(env, name)

    observation_space = getattr(env, "observation_space", None)
    if observation_space is not None:
        shape = getattr(observation_space, "shape", None)
        if shape is not None:
            info["observation_shape"] = tuple(shape)

    action_space = getattr(env, "action_space", None)
    if action_space is not None:
        action_n = getattr(action_space, "n", None)
        if action_n is not None:
            info["action_size"] = int(action_n)

    return info


def _collect_trainer_info(trainer: Any) -> dict[str, Any]:
    info: dict[str, Any] = {}
    if trainer is None:
        return info

    info["type"] = trainer.__class__.__name__

    for name in [
        "log_interval",
        "eval_interval",
        "eval_episodes",
        "checkpoint_interval",
        "checkpoint_dir",
        "opponent_update_interval",
        "self_play",
    ]:
        if hasattr(trainer, name):
            info[name] = getattr(trainer, name)

    opponent = getattr(trainer, "opponent", None)
    if opponent is not None:
        agent = getattr(trainer, "agent", None)
        # In a mirror match the opponent *is* the agent; say so rather than
        # repeating the agent's class name and implying a second policy.
        info["opponent"] = (
            "self" if opponent is agent else opponent.__class__.__name__
        )

    return info


def _collect_agent_info(agent: Optional[Agent]) -> dict[str, Any]:
    info: dict[str, Any] = {}
    if agent is None:
        return info

    info["type"] = agent.__class__.__name__
    info["name"] = getattr(agent, "name", None)
    info["training"] = getattr(agent, "training", None)

    for name in ["steps", "episodes", "train_steps"]:
        if hasattr(agent, name):
            info[name] = getattr(agent, name)

    return info


def _collect_dqn_info(agent: Optional[Agent]) -> Optional[dict[str, Any]]:
    if agent is None:
        return None

    if DQNAgent is not None and isinstance(agent, DQNAgent):
        info: dict[str, Any] = {
            "state_size": agent.state_size,
            "action_size": agent.action_size,
            "gamma": agent.gamma,
            "epsilon_start": getattr(agent, "epsilon_start", None),
            "epsilon": agent.epsilon,
            "epsilon_end": agent.epsilon_end,
            "epsilon_decay": agent.epsilon_decay,
            "batch_size": agent.batch_size,
            "target_update_freq": agent.target_update_freq,
            "device": str(agent.device),
            "seed": getattr(agent, "seed", None),
        }

        info["learning_rate"] = _get_learning_rate(agent)
        info["buffer_size"] = _get_buffer_size(agent)
        info["hidden_sizes"] = _get_hidden_sizes(agent)

        # Dueling is a DoubleDQNAgent-only architectural switch. It has to be
        # read before the None-filter below, and it must survive being False.
        if DoubleDQNAgent is not None and isinstance(agent, DoubleDQNAgent):
            info["dueling"] = bool(getattr(agent, "dueling", False))

        return {k: v for k, v in info.items() if v is not None}

    return None


def _collect_qlearning_info(agent: Optional[Agent]) -> Optional[dict[str, Any]]:
    if agent is None or QLearningAgent is None:
        return None
    if not isinstance(agent, QLearningAgent):
        return None

    info: dict[str, Any] = {
        "action_size": agent.action_size,
        "learning_rate": agent.learning_rate,
        "gamma": agent.gamma,
        "epsilon_start": getattr(agent, "epsilon_start", None),
        "epsilon": agent.epsilon,
        "epsilon_end": agent.epsilon_end,
        "epsilon_decay": agent.epsilon_decay,
        "optimistic_init": getattr(agent, "optimistic_init", None),
        # Observation values are rounded to this many decimals before being
        # used as a table key, so it sets how coarsely states are merged.
        "precision": getattr(agent, "precision", None),
        # The headline cost of tabular Q-learning: one entry per distinct
        # rounded observation, and it only ever grows.
        "table_size": getattr(agent, "table_size", None),
        "seed": getattr(agent, "seed", None),
    }

    return {k: v for k, v in info.items() if v is not None}


def _collect_search_info(agent: Optional[Agent]) -> Optional[dict[str, Any]]:
    if agent is None:
        return None

    if MCTSAgent is not None and isinstance(agent, MCTSAgent):
        info: dict[str, Any] = {
            "type": "MCTSAgent",
            "simulations": agent.simulations,
            "exploration_weight": agent.exploration_weight,
            "rollout_depth": agent.rollout_depth,
            "gamma": agent.gamma,
            "determinizations": agent.determinizations,
            "use_determinization": agent.use_determinization,
        }
        policy = getattr(agent, "rollout_policy", None)
        info["rollout_policy"] = (
            "random" if policy is None else policy.__class__.__name__
        )
        return info

    if GreedyLookaheadAgent is not None and isinstance(agent, GreedyLookaheadAgent):
        return {
            "type": "GreedyLookaheadAgent",
            "depth": agent.depth,
            "gamma": agent.gamma,
        }

    return None


def _collect_ppo_info(agent: Optional[Agent]) -> Optional[dict[str, Any]]:
    if agent is None or PPOAgent is None or not isinstance(agent, PPOAgent):
        return None

    info: dict[str, Any] = {
        "state_size": agent.state_size,
        "action_size": agent.action_size,
        "hidden_sizes": list(agent.hidden_sizes),
        "learning_rate": _get_learning_rate(agent),
        "gamma": agent.gamma,
        "gae_lambda": agent.gae_lambda,
        "clip_epsilon": agent.clip_epsilon,
        "epochs": agent.epochs,
        "minibatch_size": agent.minibatch_size,
        "rollout_steps": agent.rollout_steps,
        "entropy_coef": agent.entropy_coef,
        "value_coef": agent.value_coef,
        "max_grad_norm": agent.max_grad_norm,
        "device": str(agent.device),
        "seed": getattr(agent, "seed", None),
    }

    return {k: v for k, v in info.items() if v is not None}


def _get_learning_rate(agent: Any) -> Optional[float]:
    lr = getattr(agent, "learning_rate", None)
    if lr is not None:
        return float(lr)
    optimizer = getattr(agent, "optimizer", None)
    if optimizer is None:
        return None
    param_groups = getattr(optimizer, "param_groups", [])
    if not param_groups:
        return None
    return float(param_groups[0].get("lr", 0.0))


def _get_buffer_size(agent: Any) -> Optional[int]:
    buffer_size = getattr(agent, "buffer_size", None)
    if buffer_size is not None:
        return int(buffer_size)
    replay_buffer = getattr(agent, "replay_buffer", None)
    if replay_buffer is None:
        return None
    buffer = getattr(replay_buffer, "buffer", None)
    if buffer is None:
        return None
    maxlen = getattr(buffer, "maxlen", None)
    if maxlen is None:
        return None
    return int(maxlen)


def _get_hidden_sizes(agent: Any) -> Optional[list[int]]:
    hidden_sizes = getattr(agent, "hidden_sizes", None)
    if hidden_sizes:
        return list(hidden_sizes)

    q_network = getattr(agent, "q_network", None)
    if q_network is None:
        return None

    try:
        import torch.nn as nn
    except Exception:
        return None

    layers = getattr(q_network, "network", None)
    if layers is None:
        return None

    sizes: list[int] = []
    for layer in layers:
        if isinstance(layer, nn.Linear):
            sizes.append(int(layer.out_features))

    if len(sizes) <= 1:
        return None

    return sizes[:-1]


def _append_section(
    lines: list[str],
    title: str,
    data: Optional[dict[str, Any]],
) -> None:
    if not data:
        return

    lines.append(f"## {title}")
    for key, value in data.items():
        if value is None:
            continue
        lines.append(f"- {key}: {_format_value(value)}")
    lines.append("")


def _format_value(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    return str(value)
