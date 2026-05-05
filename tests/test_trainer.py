"""Tests for trainer module."""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing

import pytest
import numpy as np
import json
import os

from rl_card_lib.trainer.metrics import TrainingMetrics
from rl_card_lib.trainer.trainer import Trainer, SelfPlayTrainer
from rl_card_lib.games import KlondikeSolitaire, Macao
from rl_card_lib.env import CardGameEnv
from rl_card_lib.agents import RandomAgent, DQNAgent


class TestTrainingMetrics:
    @pytest.fixture
    def metrics(self):
        return TrainingMetrics(window_size=10)

    def test_init(self, metrics):
        assert metrics.window_size == 10
        assert len(metrics.rewards) == 0
        assert len(metrics.wins) == 0

    def test_add_episode(self, metrics):
        metrics.add_episode({
            "reward": 10.0,
            "steps": 50,
            "win": 1,
            "loss": 0.1,
        })
        assert len(metrics.rewards) == 1
        assert metrics.rewards[0] == 10.0
        assert metrics.wins[0] == 1

    def test_add_episode_defaults(self, metrics):
        metrics.add_episode({})
        assert metrics.rewards[0] == 0.0
        assert metrics.steps[0] == 0

    def test_add_evaluation(self, metrics):
        metrics.add_evaluation(100, {"mean_reward": 5.0})
        assert len(metrics.evaluations) == 1
        assert metrics.evaluations[0]["episode"] == 100

    def test_get_recent_average(self, metrics):
        for i in range(5):
            metrics.add_episode({"reward": float(i)})
        avg = metrics.get_recent_average("reward", 3)
        assert avg == pytest.approx((2 + 3 + 4) / 3)

    def test_get_recent_average_empty(self, metrics):
        result = metrics.get_recent_average("reward", 10)
        assert result == 0.0

    def test_get_moving_average(self, metrics):
        for i in range(5):
            metrics.add_episode({"reward": float(i)})
        ma = metrics.get_moving_average("reward")
        assert len(ma) == 5

    def test_get_moving_average_empty(self, metrics):
        result = metrics.get_moving_average("reward")
        assert result == []

    def test_summary(self, metrics):
        for i in range(10):
            metrics.add_episode({"reward": 1.0, "steps": 10, "win": 1})
        metrics.training_time = 100.0
        summary = metrics.summary()
        assert summary["total_episodes"] == 10
        assert summary["total_wins"] == 10
        assert summary["win_rate"] == 1.0
        assert summary["training_time"] == 100.0

    def test_summary_empty(self, metrics):
        summary = metrics.summary()
        assert summary["total_episodes"] == 0
        assert summary["win_rate"] == 0.0

    def test_save_and_load(self, metrics, tmp_path):
        metrics.add_episode({"reward": 10.0, "steps": 50, "win": 1, "loss": 0.1})
        metrics.add_evaluation(1, {"mean_reward": 10.0})
        
        path = str(tmp_path / "metrics.json")
        metrics.save(path)
        
        loaded = TrainingMetrics()
        loaded.load(path)
        
        assert loaded.rewards == metrics.rewards
        assert loaded.wins == metrics.wins

    def test_plot(self, metrics):
        for i in range(10):
            metrics.add_episode({"reward": float(i), "win": i % 2})
        
        # Just verify it doesn't crash
        fig = metrics.plot(metrics=["reward", "win"])
        assert fig is not None

    def test_plot_single_metric(self, metrics):
        for i in range(10):
            metrics.add_episode({"reward": float(i)})
        fig = metrics.plot(metrics=["reward"])
        assert fig is not None

    def test_plot_empty(self, metrics):
        fig = metrics.plot(metrics=["reward"])
        assert fig is not None

    def test_plot_save(self, metrics, tmp_path):
        for i in range(10):
            metrics.add_episode({"reward": float(i)})
        path = str(tmp_path / "plot.png")
        metrics.plot(metrics=["reward"], save_path=path)
        assert os.path.exists(path)


class TestTrainer:
    @pytest.fixture
    def trainer(self):
        game = KlondikeSolitaire()
        env = CardGameEnv(game, max_steps=20)
        agent = RandomAgent(action_size=env.action_space.n, seed=42)
        return Trainer(env, agent, log_interval=5, eval_interval=10, eval_episodes=3)

    def test_init(self, trainer):
        assert trainer.env is not None
        assert trainer.agent is not None
        assert trainer.log_interval == 5

    def test_train(self, trainer):
        metrics = trainer.train(episodes=5, verbose=False)
        assert len(metrics.rewards) == 5

    def test_train_with_max_steps(self, trainer):
        metrics = trainer.train(episodes=3, max_steps_per_episode=10, verbose=False)
        assert len(metrics.rewards) == 3

    def test_train_with_callback(self, trainer):
        call_count = [0]
        def callback(metrics):
            call_count[0] += 1
            return call_count[0] < 3  # Stop after 3 episodes
        
        metrics = trainer.train(episodes=10, verbose=False, callback=callback)
        assert len(metrics.rewards) == 3

    def test_evaluate(self, trainer):
        result = trainer.evaluate(episodes=3, verbose=False)
        assert "mean_reward" in result
        assert "win_rate" in result

    def test_run_episode(self, trainer):
        result = trainer._run_episode(training=True, max_steps=20)
        assert "reward" in result
        assert "steps" in result
        assert "win" in result

    def test_checkpoint_saving(self, tmp_path):
        game = KlondikeSolitaire()
        env = CardGameEnv(game, max_steps=20)
        agent = DQNAgent(
            state_size=env.observation_space.shape[0],
            action_size=env.action_space.n,
            hidden_sizes=[32],
            device="cpu"
        )
        trainer = Trainer(
            env, agent,
            checkpoint_dir=str(tmp_path),
            checkpoint_interval=2
        )
        trainer.train(episodes=3, verbose=False)
        # Check that checkpoint was saved
        files = os.listdir(tmp_path)
        assert any("checkpoint" in f for f in files)


class TestSelfPlayTrainer:
    @pytest.fixture
    def trainer(self):
        game = Macao(num_players=2)
        env = CardGameEnv(game, max_steps=20)
        agent = RandomAgent(action_size=env.action_space.n, seed=42)
        return SelfPlayTrainer(env, agent, opponent_update_interval=5)

    def test_init(self, trainer):
        assert trainer.opponent_update_interval == 5
        assert trainer.opponent is not None

    def test_train(self, trainer):
        metrics = trainer.train(episodes=5, verbose=False)
        assert len(metrics.rewards) == 5

    def test_run_episode(self, trainer):
        result = trainer._run_episode(training=True, max_steps=20)
        assert "reward" in result
        assert "steps" in result

    def test_selfplay_with_no_loss_agent(self):
        """Test SelfPlayTrainer with agent that returns no loss."""
        from rl_card_lib.agents.base import Agent
        
        class NoLossAgent(Agent):
            def __init__(self):
                self.action_size = 52
            
            def select_action(self, obs, legal_actions=None):
                return legal_actions[0] if legal_actions else 0
            
            def learn(self, obs, action, reward, next_obs, done):
                return {}  # No loss key
            
            def save(self, path): pass
            def load(self, path): pass
        
        game = Macao(num_players=2)
        env = CardGameEnv(game, max_steps=20)
        agent = NoLossAgent()
        
        trainer = SelfPlayTrainer(env, agent, opponent_update_interval=5)
        metrics = trainer.train(episodes=2)
        
        assert len(metrics.rewards) == 2

    def test_selfplay_with_loss_agent(self):
        """Test SelfPlayTrainer with agent that returns loss."""
        from rl_card_lib.agents.base import Agent
        
        class LossAgent(Agent):
            def __init__(self):
                self.action_size = 52
            
            def select_action(self, obs, legal_actions=None):
                return legal_actions[0] if legal_actions else 0
            
            def learn(self, obs, action, reward, next_obs, done):
                return {"loss": 0.5}  # Return loss
            
            def save(self, path): pass
            def load(self, path): pass
        
        game = Macao(num_players=2)
        env = CardGameEnv(game, max_steps=20)
        agent = LossAgent()
        
        trainer = SelfPlayTrainer(env, agent, opponent_update_interval=5)
        metrics = trainer.train(episodes=2)
        
        assert len(metrics.rewards) == 2
        # Should have tracked losses
        assert len(metrics.losses) > 0


class TestTrainerVerbose:
    """Tests for verbose mode."""

    def test_train_verbose(self, capsys):
        game = KlondikeSolitaire()
        env = CardGameEnv(game, max_steps=10)
        agent = RandomAgent(action_size=env.action_space.n, seed=42)
        trainer = Trainer(env, agent, log_interval=2, eval_interval=100)
        
        trainer.train(episodes=3, verbose=True)
        
        captured = capsys.readouterr()
        assert "Training completed" in captured.out

    def test_train_with_logging(self, capsys):
        game = KlondikeSolitaire()
        env = CardGameEnv(game, max_steps=10)
        agent = RandomAgent(action_size=env.action_space.n, seed=42)
        trainer = Trainer(env, agent, log_interval=1, eval_interval=100)
        
        trainer.train(episodes=2, verbose=False)
        
        captured = capsys.readouterr()
        assert "Episode" in captured.out

    def test_train_with_eval(self):
        game = KlondikeSolitaire()
        env = CardGameEnv(game, max_steps=10)
        agent = RandomAgent(action_size=env.action_space.n, seed=42)
        trainer = Trainer(env, agent, eval_interval=2, eval_episodes=1)
        
        metrics = trainer.train(episodes=3, verbose=False)
        
        # Should have done at least one evaluation
        assert len(metrics.evaluations) >= 1

    def test_evaluate_verbose(self, capsys):
        game = KlondikeSolitaire()
        env = CardGameEnv(game, max_steps=10)
        agent = RandomAgent(action_size=env.action_space.n, seed=42)
        trainer = Trainer(env, agent)
        
        trainer.evaluate(episodes=2, verbose=True)
        # Just verify it doesn't crash with verbose mode


class TestTrainerLearnNoLoss:
    """Tests for trainer with agents that don't return loss."""

    def test_train_with_none_learn_result(self):
        """Test train when agent.learn returns None."""
        game = KlondikeSolitaire()
        env = CardGameEnv(game, max_steps=20)
        agent = RandomAgent(action_size=env.action_space.n, seed=42)
        
        trainer = Trainer(env, agent)
        metrics = trainer.train(episodes=2)
        
        # RandomAgent returns None from learn, should still work
        assert len(metrics.rewards) == 2

    def test_train_with_empty_dict(self):
        """Test train when agent.learn returns empty dict."""
        from rl_card_lib.agents.base import Agent
        
        class NoLossAgent(Agent):
            def __init__(self):
                self.action_size = 52
            
            def select_action(self, obs, legal_actions=None):
                return legal_actions[0] if legal_actions else 0
            
            def learn(self, obs, action, reward, next_obs, done):
                return {}  # No loss key
            
            def save(self, path): pass
            def load(self, path): pass
        
        game = KlondikeSolitaire()
        env = CardGameEnv(game, max_steps=20)
        agent = NoLossAgent()
        
        trainer = Trainer(env, agent)
        metrics = trainer.train(episodes=2)
        
        assert len(metrics.rewards) == 2

    def test_train_with_loss_result(self):
        """Test train when agent.learn returns loss."""
        from rl_card_lib.agents.base import Agent
        
        class LossAgent(Agent):
            def __init__(self):
                self.action_size = 52
            
            def select_action(self, obs, legal_actions=None):
                return legal_actions[0] if legal_actions else 0
            
            def learn(self, obs, action, reward, next_obs, done):
                return {"loss": 0.5}  # Return loss value
            
            def save(self, path): pass
            def load(self, path): pass
        
        game = KlondikeSolitaire()
        env = CardGameEnv(game, max_steps=20)
        agent = LossAgent()
        
        trainer = Trainer(env, agent)
        metrics = trainer.train(episodes=2)
        
        assert len(metrics.rewards) == 2
        # Should have tracked losses
        assert len(metrics.losses) > 0


class TestMetricsPlotNoMatplotlib:
    """Test metrics plotting when matplotlib is not available."""
    
    def test_plot_without_matplotlib(self, monkeypatch, capsys):
        """Test plot_metrics when matplotlib import fails."""
        import builtins
        
        original_import = builtins.__import__
        
        def mock_import(name, *args, **kwargs):
            if name == "matplotlib.pyplot":
                raise ImportError("No matplotlib")
            return original_import(name, *args, **kwargs)
        
        import sys
        
        # Remove matplotlib from sys.modules temporarily
        mpl_modules = [k for k in sys.modules if k.startswith('matplotlib')]
        saved_modules = {}
        for m in mpl_modules:
            saved_modules[m] = sys.modules.pop(m)
        
        # Create a fresh metrics object
        metrics = TrainingMetrics()
        metrics.add_episode({"reward": 1.0, "steps": 10, "win": False})
        
        # Patch the import inside the method
        monkeypatch.setattr(builtins, "__import__", mock_import)
        
        result = metrics.plot(metrics=["rewards"])
        
        # Restore
        monkeypatch.undo()
        sys.modules.update(saved_modules)
        
        captured = capsys.readouterr()
        assert result is None
        assert "matplotlib not installed" in captured.out