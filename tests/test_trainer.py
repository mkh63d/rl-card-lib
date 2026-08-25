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
from rl_card_lib.agents import RandomAgent, DQNAgent, GreedyLookaheadAgent
from rl_card_lib.agents.base import Agent


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


class TestTrainerDeals:
    """Which deals a training run plays, and that it can be replayed."""

    @staticmethod
    def _trainer(env, **kwargs):
        agent = RandomAgent(action_size=env.action_space.n, seed=42)
        return Trainer(env, agent, log_interval=10**9, eval_interval=10**9,
                       **kwargs)

    @staticmethod
    def _pooled(**kwargs):
        return CardGameEnv(KlondikeSolitaire(), max_steps=10,
                           deal_seeds=range(500), **kwargs)

    def test_two_runs_train_on_the_identical_deals(self):
        """The whole training deal stream follows from deal_rng_seed alone."""
        first, second = self._pooled(deal_rng_seed=1), self._pooled(deal_rng_seed=1)
        self._trainer(first).train(episodes=6, verbose=False)
        self._trainer(second).train(episodes=6, verbose=False)

        assert first.dealt_seeds == second.dealt_seeds
        assert len(first.dealt_seeds) == 6

    def test_evaluation_runs_in_the_eval_env(self):
        train_env = self._pooled()
        eval_env = CardGameEnv(KlondikeSolitaire(), max_steps=10,
                               deal_seeds=[900, 901], deal_order="cycle")
        trainer = self._trainer(train_env, eval_env=eval_env)

        trainer.evaluate(episodes=2, verbose=False)

        # The evaluation played held-out deals and left the training deal
        # stream alone -- eval_env used to be stored and never read.
        assert eval_env.dealt_seeds == [900, 901]
        assert train_env.dealt_seeds == []

    def test_every_evaluation_replays_the_same_deals(self):
        train_env = self._pooled()
        eval_env = CardGameEnv(KlondikeSolitaire(), max_steps=10,
                               deal_seeds=[900, 901, 902], deal_order="cycle")
        trainer = self._trainer(train_env, eval_env=eval_env)

        trainer.evaluate(episodes=2, verbose=False)
        trainer.evaluate(episodes=2, verbose=False)

        assert eval_env.dealt_seeds == [900, 901, 900, 901]

    def test_evaluation_does_not_rewind_a_shared_env(self):
        """With no separate eval env, evaluating must not restart training deals."""
        env = self._pooled(deal_order="cycle")
        trainer = self._trainer(env)

        trainer.train(episodes=2, verbose=False)
        trainer.evaluate(episodes=2, verbose=False)

        assert env.dealt_seeds == [0, 1, 2, 3]

    def test_bound_agents_follow_the_eval_env_and_come_back(self):
        """A game-reading agent must not judge the eval board from the training one."""
        train_env = self._pooled()
        eval_env = CardGameEnv(KlondikeSolitaire(), max_steps=10,
                               deal_seeds=[900], deal_order="cycle")
        agent = GreedyLookaheadAgent(depth=1, seed=0)
        trainer = Trainer(train_env, agent, eval_env=eval_env,
                          log_interval=10**9, eval_interval=10**9)
        assert agent.game is train_env.game

        trainer.evaluate(episodes=1, verbose=False)

        assert eval_env.dealt_seeds == [900]      # it really played there
        assert agent.game is train_env.game       # and was handed back


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

    def test_evaluation_episodes_accumulate_reward(self, trainer):
        """Evaluation must pay the agent for its own plays.

        The accumulator used to sit behind `if training and ...`, so every
        evaluation episode reported exactly 0.0 and each recorded Macao
        evaluation had mean/std/min/max reward of 0.0.
        """
        rewards = [
            trainer._run_episode(training=False, max_steps=40)["reward"]
            for _ in range(10)
        ]
        assert any(reward != 0.0 for reward in rewards)

    def test_evaluate_reports_nonzero_reward(self, trainer):
        result = trainer.evaluate(episodes=10, verbose=False)
        assert result["mean_reward"] != 0.0

    def test_training_episodes_still_accumulate_reward(self, trainer):
        rewards = [
            trainer._run_episode(training=True, max_steps=40)["reward"]
            for _ in range(10)
        ]
        assert any(reward != 0.0 for reward in rewards)

    def test_evaluation_does_not_learn(self):
        """Reward accumulates outside training, but learning must not.

        Without this, dropping the `training` guard entirely would still pass
        the reward tests above.
        """
        from rl_card_lib.agents.base import Agent

        class CountingAgent(Agent):
            def __init__(self):
                super().__init__(name="CountingAgent")
                self.learn_calls = 0

            def select_action(self, obs, legal_actions=None):
                return legal_actions[0] if legal_actions else 0

            def learn(self, obs, action, reward, next_obs, done):
                self.learn_calls += 1
                return {"loss": 0.5}

            def save(self, path): pass
            def load(self, path): pass

        game = Macao(num_players=2)
        env = CardGameEnv(game, max_steps=20)
        agent = CountingAgent()
        trainer = SelfPlayTrainer(env, agent, opponent_update_interval=5)

        trainer._run_episode(training=False, max_steps=20)
        assert agent.learn_calls == 0

        trainer._run_episode(training=True, max_steps=20)
        assert agent.learn_calls > 0


class TestTimeLimitBootstrap:
    """What the trainer reports as `done`.

    The loop ends on `terminated or truncated`, but only a termination is a
    terminal state. Collapsing the two taught every learner that the state at
    the step cap is worth its immediate reward and nothing more -- on Klondike,
    where every episode used to end that way, the last transition of every
    single episode.
    """

    class RecordingAgent(Agent):
        """Stores the flags every transition arrived with."""

        def __init__(self, **flags):
            super().__init__(name="RecordingAgent")
            self.action_size = 52
            self.calls = []
            for name, value in flags.items():
                setattr(self, name, value)

        def select_action(self, obs, legal_actions=None):
            return legal_actions[0] if legal_actions else 0

        def learn(self, obs, action, reward, next_obs, done, **kwargs):
            self.calls.append({"done": done, **kwargs})
            return None

        def save(self, path): pass
        def load(self, path): pass

    def test_a_capped_episode_reports_no_terminal_transition(self):
        env = CardGameEnv(KlondikeSolitaire(seed=0), max_steps=15)
        agent = self.RecordingAgent()
        trainer = Trainer(env, agent, log_interval=10**9, eval_interval=10**9)

        trainer.train(episodes=4, verbose=False)

        # The episodes really did run and really did stop at the cap.
        assert len(agent.calls) == 60
        assert not any(call["done"] for call in agent.calls)

    def test_a_real_termination_still_reports_done(self):
        """Guard against fixing the bias by never reporting terminal at all."""
        env = CardGameEnv(KlondikeSolitaire(seed=0), max_steps=15)
        agent = self.RecordingAgent()
        trainer = Trainer(env, agent, log_interval=10**9, eval_interval=10**9)

        # Terminate the game two steps in, without truncating.
        real_step = env.step
        countdown = [2]

        def terminating_step(action):
            observation, reward, terminated, truncated, info = real_step(action)
            countdown[0] -= 1
            return observation, reward, countdown[0] <= 0, truncated, info

        env.step = terminating_step
        trainer.train(episodes=1, verbose=False)

        assert [call["done"] for call in agent.calls] == [False, True]

    def test_the_truncation_flag_reaches_agents_that_ask_for_it(self):
        env = CardGameEnv(KlondikeSolitaire(seed=0), max_steps=6)
        agent = self.RecordingAgent(accepts_truncated=True)
        trainer = Trainer(env, agent, log_interval=10**9, eval_interval=10**9)

        trainer.train(episodes=1, verbose=False)

        assert [call["truncated"] for call in agent.calls] == [False] * 5 + [True]
        assert not any(call["done"] for call in agent.calls)

    def test_agents_that_do_not_ask_never_see_the_keyword(self):
        """A plain five-argument learn() must keep working untouched."""
        env = CardGameEnv(KlondikeSolitaire(seed=0), max_steps=6)
        agent = self.RecordingAgent()
        trainer = Trainer(env, agent, log_interval=10**9, eval_interval=10**9)

        trainer.train(episodes=1, verbose=False)

        assert all(call.keys() == {"done"} for call in agent.calls)

    def test_both_opt_ins_compose(self):
        env = CardGameEnv(KlondikeSolitaire(seed=0), max_steps=6)
        agent = self.RecordingAgent(
            accepts_truncated=True, accepts_next_legal_actions=True,
        )
        trainer = Trainer(env, agent, log_interval=10**9, eval_interval=10**9)

        trainer.train(episodes=1, verbose=False)

        last = agent.calls[-1]
        assert last["truncated"] is True
        assert last["next_legal_actions"] is not None

    def test_selfplay_capped_episodes_report_no_terminal_transition(self):
        # A deal pool, not a constructor seed: an unseeded Macao deals a hand a
        # player can empty inside the 10-step cap often enough to flake, and
        # both assertions below are about how the episodes ended. cycle gives
        # each of the four episodes its own reproducible deal -- the form
        # CardGameEnv's deal_seeds docstring recommends for a repeatable stream.
        env = CardGameEnv(Macao(num_players=2), max_steps=10,
                          deal_seeds=[0, 1, 2, 3], deal_order="cycle")
        agent = self.RecordingAgent(accepts_truncated=True)
        trainer = SelfPlayTrainer(env, agent, opponent_update_interval=10**9)

        trainer.train(episodes=4, verbose=False)

        # 22, not 4 x max_steps: SelfPlayTrainer records only the learning
        # agent's transitions, and the opponent takes roughly half the turns.
        assert len(agent.calls) == 22
        assert not any(call["done"] for call in agent.calls)
        assert any(call["truncated"] for call in agent.calls)


class TestEvaluationIsSideEffectFree:
    """Measuring an agent must not advance its exploration schedule.

    Epsilon decayed in `Agent.reset()`, which every episode calls -- evaluation
    episodes included. So an evaluation permanently moved the schedule of the
    agent it was measuring, and the `epsilon` recorded in every run record as
    the *start* value was really the value after the pre-training evaluation:
    0.8647 rather than 1.0 for a 30-deal Klondike protocol.
    """

    @staticmethod
    def _agent(env, **kwargs):
        return DQNAgent(
            state_size=env.observation_space.shape[0],
            action_size=env.action_space.n,
            hidden_sizes=[16], device="cpu", batch_size=4, buffer_size=100,
            epsilon_start=1.0, epsilon_end=0.0, epsilon_decay=0.9, **kwargs
        )

    def test_evaluate_does_not_move_epsilon(self):
        env = CardGameEnv(KlondikeSolitaire(), max_steps=10)
        agent = self._agent(env)
        trainer = Trainer(env, agent, log_interval=10**9, eval_interval=10**9)

        trainer.evaluate(episodes=8, verbose=False)

        assert agent.epsilon == 1.0
        assert agent.episodes == 0

    def test_evaluation_episode_count_does_not_change_the_schedule(self):
        """The whole point: the schedule is a training setting, not a reporting one."""
        schedules = []
        for eval_episodes in (2, 20):
            env = CardGameEnv(KlondikeSolitaire(), max_steps=10)
            agent = self._agent(env)
            trainer = Trainer(env, agent, log_interval=10**9, eval_interval=10**9)

            trainer.evaluate(episodes=eval_episodes, verbose=False)
            trainer.train(episodes=5, max_steps_per_episode=10, verbose=False)
            schedules.append(agent.epsilon)

        assert schedules[0] == pytest.approx(schedules[1])
        assert schedules[0] == pytest.approx(0.9 ** 5)

    def test_training_still_decays(self):
        """The fix must not have quietly disabled exploration decay."""
        env = CardGameEnv(KlondikeSolitaire(), max_steps=10)
        agent = self._agent(env)
        trainer = Trainer(env, agent, log_interval=10**9, eval_interval=10**9)

        for _ in range(4):
            trainer._run_episode(training=True, max_steps=10)

        assert agent.epsilon == pytest.approx(0.9 ** 4)
        assert agent.episodes == 4

    def test_selfplay_training_still_decays(self):
        env = CardGameEnv(Macao(num_players=2), max_steps=10)
        agent = self._agent(env)
        trainer = SelfPlayTrainer(env, agent, opponent_update_interval=10**9)

        for _ in range(4):
            trainer._run_episode(training=True, max_steps=10)

        assert agent.epsilon == pytest.approx(0.9 ** 4)
        assert agent.episodes == 4

    def test_selfplay_evaluation_does_not_move_epsilon(self):
        env = CardGameEnv(Macao(num_players=2), max_steps=10)
        agent = self._agent(env)
        trainer = SelfPlayTrainer(env, agent, opponent_update_interval=10**9)

        for _ in range(4):
            trainer._run_episode(training=False, max_steps=10)

        assert agent.epsilon == 1.0
        assert agent.episodes == 0

    def test_frozen_opponent_does_not_decay(self):
        """A snapshot playing its part in the agent's episode is not training."""
        env = CardGameEnv(Macao(num_players=2), max_steps=10)
        agent = self._agent(env)
        opponent = self._agent(env)
        trainer = SelfPlayTrainer(env, agent, opponent=opponent,
                                  opponent_update_interval=10**9)

        for _ in range(4):
            trainer._run_episode(training=True, max_steps=10)

        assert opponent.epsilon == 1.0
        assert opponent.episodes == 0


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
