"""Tests for the training report package."""

import json

import pytest

from rl_card_lib.agents import (
    DoubleDQNAgent,
    DQNAgent,
    GreedyLookaheadAgent,
    MCTSAgent,
    PPOAgent,
    QLearningAgent,
    RandomAgent,
)
from rl_card_lib.env import CardGameEnv
from rl_card_lib.games import KlondikeSolitaire, Macao, MacaoHeuristicAgent
from rl_card_lib.report import TrainingReport
from rl_card_lib.trainer import SelfPlayTrainer, Trainer


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


class TestDoubleDQNSection:
    """Double DQN reports the dueling switch that plain DQN has no notion of."""

    def make(self, env, **kwargs):
        return DoubleDQNAgent(
            state_size=env.observation_space.shape[0],
            action_size=env.action_space.n,
            hidden_sizes=[32], device="cpu", seed=0, **kwargs,
        )

    def test_reports_dueling(self, klondike_env):
        agent = self.make(klondike_env, dueling=True)
        report = TrainingReport.from_trainer(make_trainer(klondike_env, agent))

        assert report.dqn is not None
        assert report.dqn["dueling"] is True
        assert report.agent["type"] == "DoubleDQNAgent"

    def test_dueling_false_is_not_dropped(self, klondike_env):
        """`False` must survive the None-filter in _collect_dqn_info."""
        agent = self.make(klondike_env, dueling=False)
        report = TrainingReport.from_trainer(make_trainer(klondike_env, agent))

        assert report.dqn["dueling"] is False

    def test_plain_dqn_has_no_dueling_key(self, klondike_env):
        agent = DQNAgent(
            state_size=klondike_env.observation_space.shape[0],
            action_size=klondike_env.action_space.n,
            hidden_sizes=[32], device="cpu", seed=0,
        )
        report = TrainingReport.from_trainer(make_trainer(klondike_env, agent))

        assert "dueling" not in report.dqn


class TestQLearningSection:
    """Tabular Q-learning has no network, so it needs its own section."""

    def test_collects_qlearning_section(self, klondike_env):
        agent = QLearningAgent(action_size=klondike_env.action_space.n, seed=0)
        report = TrainingReport.from_trainer(
            make_trainer(klondike_env, agent), episodes=10
        )

        assert report.qlearning is not None
        assert report.dqn is None
        assert report.ppo is None
        assert report.qlearning["gamma"] == agent.gamma
        assert report.qlearning["precision"] == agent.precision
        assert report.qlearning["epsilon_decay"] == pytest.approx(agent.epsilon_decay)

    def test_reports_table_growth(self, klondike_env):
        agent = QLearningAgent(action_size=klondike_env.action_space.n, seed=0)
        trainer = make_trainer(klondike_env, agent)
        trainer.train(episodes=2, max_steps_per_episode=20, verbose=False)

        report = TrainingReport.from_trainer(trainer)
        assert report.qlearning["table_size"] == agent.table_size
        assert report.qlearning["table_size"] > 0

    def test_markdown_includes_the_section(self, klondike_env):
        agent = QLearningAgent(action_size=klondike_env.action_space.n, seed=0)
        markdown = TrainingReport.from_trainer(
            make_trainer(klondike_env, agent)
        ).to_markdown()

        assert "## Q-learning" in markdown


class TestSearchSection:
    """Search agents carry a budget, not a learning rate."""

    def test_collects_mcts_section(self, klondike_env):
        agent = MCTSAgent(simulations=7, rollout_depth=3, seed=0)
        report = TrainingReport.from_trainer(make_trainer(klondike_env, agent))

        assert report.search is not None
        assert report.search["type"] == "MCTSAgent"
        assert report.search["simulations"] == 7
        assert report.search["rollout_policy"] == "random"

    def test_names_the_rollout_policy(self, klondike_env):
        from rl_card_lib.games import KlondikeHeuristicAgent

        agent = MCTSAgent(
            simulations=7, rollout_depth=3,
            rollout_policy=KlondikeHeuristicAgent(seed=0), seed=0,
        )
        report = TrainingReport.from_trainer(make_trainer(klondike_env, agent))

        assert report.search["rollout_policy"] == "KlondikeHeuristicAgent"

    def test_collects_lookahead_section(self, klondike_env):
        agent = GreedyLookaheadAgent(depth=2, seed=0)
        report = TrainingReport.from_trainer(make_trainer(klondike_env, agent))

        assert report.search["type"] == "GreedyLookaheadAgent"
        assert report.search["depth"] == 2

    def test_learner_has_no_search_section(self, klondike_env):
        agent = QLearningAgent(action_size=klondike_env.action_space.n, seed=0)
        report = TrainingReport.from_trainer(make_trainer(klondike_env, agent))

        assert report.search is None


class TestSelfPlayTrainerSection:
    """A Macao run needs its opponent recorded to be interpretable."""

    @pytest.fixture
    def macao_env(self):
        return CardGameEnv(Macao(num_players=2, seed=0), max_steps=50)

    def test_records_fixed_opponent(self, macao_env):
        agent = QLearningAgent(action_size=macao_env.action_space.n, seed=0)
        trainer = SelfPlayTrainer(
            macao_env, agent, opponent=MacaoHeuristicAgent(seed=0),
        )
        report = TrainingReport.from_trainer(trainer)

        assert report.trainer["type"] == "SelfPlayTrainer"
        assert report.trainer["opponent"] == "MacaoHeuristicAgent"
        assert report.trainer["self_play"] is False

    def test_mirror_match_is_labelled_self(self, macao_env):
        agent = QLearningAgent(action_size=macao_env.action_space.n, seed=0)
        trainer = SelfPlayTrainer(macao_env, agent, opponent_update_interval=None)
        report = TrainingReport.from_trainer(trainer)

        assert report.trainer["opponent"] == "self"
        assert report.trainer["self_play"] is True

    def test_plain_trainer_reports_its_type(self, klondike_env):
        agent = RandomAgent(action_size=klondike_env.action_space.n, seed=0)
        report = TrainingReport.from_trainer(make_trainer(klondike_env, agent))

        assert report.trainer["type"] == "Trainer"
        assert "opponent" not in report.trainer
