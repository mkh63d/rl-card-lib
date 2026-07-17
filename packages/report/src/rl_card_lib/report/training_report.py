"""Training process parameter report helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from rl_card_lib.agents.base import Agent

try:
    from rl_card_lib.agents.dqn_agent import DQNAgent
except Exception:  # pragma: no cover - optional dependency
    DQNAgent = None


@dataclass
class TrainingReport:
    """Structured training parameter report."""

    environment: dict[str, Any]
    trainer: dict[str, Any]
    agent: dict[str, Any]
    dqn: Optional[dict[str, Any]] = None
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
        training_info = _collect_training_info(episodes, max_steps_per_episode)

        return cls(
            environment=env_info,
            trainer=trainer_info,
            agent=agent_info,
            dqn=dqn_info,
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
        return data

    def to_markdown(self) -> str:
        lines: list[str] = []

        _append_section(lines, "Training", self.training)
        _append_section(lines, "Environment", self.environment)
        _append_section(lines, "Trainer", self.trainer)
        _append_section(lines, "Agent", self.agent)
        _append_section(lines, "DQN", self.dqn)

        return "\n".join(lines).strip()


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

    for name in [
        "log_interval",
        "eval_interval",
        "eval_episodes",
        "checkpoint_interval",
        "checkpoint_dir",
    ]:
        if hasattr(trainer, name):
            info[name] = getattr(trainer, name)

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

        return {k: v for k, v in info.items() if v is not None}

    return None


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
