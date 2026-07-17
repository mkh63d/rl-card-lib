"""Tests for the training report package."""

import json

import pytest

from rl_card_lib.agents import DQNAgent, PPOAgent, RandomAgent
from rl_card_lib.env import CardGameEnv
from rl_card_lib.games import KlondikeSolitaire
from rl_card_lib.report import TrainingReport
from rl_card_lib.trainer import Trainer


@pytest.fixture
def klondike_env():
    return CardGameEnv(KlondikeSolitaire(seed=0), max_steps=100)


def make_trainer(env, agent):
    return Trainer(env, agent, log_interval=10, eval_interval=100, eval_episodes=5)


class TestTrainingReport:
    """Tests for report collection across agent types."""

    def test_collects_dqn_section(self, klondike_env):
        agent = DQNAgent(
            state_size=klondike_env.observation_space.shape[0],
            action_size=klondike_env.action_space.n,
            hidden_sizes=[32], device="cpu", seed=0,
        )
        report = TrainingReport.from_trainer(
            make_trainer(klondike_env, agent), episodes=50
        )

        assert report.dqn is not None
        assert report.ppo is None
        assert report.dqn["gamma"] == agent.gamma
        assert report.dqn["hidden_sizes"] == [32]
        assert report.environment["type"] == "CardGameEnv"
        assert report.environment["action_size"] == 68
        assert report.training == {"episodes": 50}

    def test_collects_ppo_section(self, klondike_env):
        agent = PPOAgent(
            state_size=klondike_env.observation_space.shape[0],
            action_size=klondike_env.action_space.n,
            hidden_sizes=[32], device="cpu", seed=0,
        )
        report = TrainingReport.from_trainer(make_trainer(klondike_env, agent))

        assert report.dqn is None
        assert report.ppo is not None
        assert report.ppo["clip_epsilon"] == agent.clip_epsilon
        assert report.ppo["rollout_steps"] == agent.rollout_steps
        assert report.ppo["learning_rate"] == pytest.approx(agent.learning_rate)

    def test_plain_agent_has_no_algorithm_section(self, klondike_env):
        agent = RandomAgent(action_size=klondike_env.action_space.n, seed=0)
        report = TrainingReport.from_trainer(make_trainer(klondike_env, agent))

        assert report.dqn is None
        assert report.ppo is None
        assert report.agent["type"] == "RandomAgent"

    def test_as_dict_skips_empty_sections(self, klondike_env):
        agent = RandomAgent(action_size=klondike_env.action_space.n, seed=0)
        data = TrainingReport.from_trainer(make_trainer(klondike_env, agent)).as_dict()

        assert "dqn" not in data
        assert "ppo" not in data
        assert set(data) >= {"environment", "trainer", "agent"}

    def test_markdown_lists_the_sections(self, klondike_env):
        agent = DQNAgent(
            state_size=klondike_env.observation_space.shape[0],
            action_size=klondike_env.action_space.n,
            hidden_sizes=[32], device="cpu", seed=0,
        )
        markdown = TrainingReport.from_trainer(
            make_trainer(klondike_env, agent), episodes=10
        ).to_markdown()

        assert "## Training" in markdown
        assert "## Environment" in markdown
        assert "## DQN" in markdown
        assert "- gamma: 0.99" in markdown

    def test_json_round_trips(self, klondike_env):
        agent = DQNAgent(
            state_size=klondike_env.observation_space.shape[0],
            action_size=klondike_env.action_space.n,
            hidden_sizes=[32], device="cpu", seed=0,
        )
        report = TrainingReport.from_trainer(
            make_trainer(klondike_env, agent), episodes=10
        )

        decoded = json.loads(report.to_json())
        assert decoded["dqn"]["gamma"] == agent.gamma
        assert decoded["training"]["episodes"] == 10

    def test_json_stringifies_awkward_values(self, klondike_env):
        """Devices and tuples must serialize rather than raise."""
        agent = DQNAgent(
            state_size=klondike_env.observation_space.shape[0],
            action_size=klondike_env.action_space.n,
            hidden_sizes=[32], device="cpu", seed=0,
        )
        report = TrainingReport.from_trainer(make_trainer(klondike_env, agent))
        decoded = json.loads(report.to_json())
        assert decoded["dqn"]["device"] == "cpu"
        assert "observation_shape" in decoded["environment"]
