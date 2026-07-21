"""Tests for RL agents."""

import pytest
import numpy as np

from rl_card_lib.agents import RandomAgent, DQNAgent


class TestRandomAgent:
    """Tests for Random Agent."""
    
    def test_agent_creation(self):
        """Test agent creation."""
        agent = RandomAgent(action_size=10)
        assert agent.action_size == 10
    
    def test_action_selection(self):
        """Test action selection."""
        agent = RandomAgent(action_size=10, seed=42)
        obs = np.zeros(20, dtype=np.float32)
        
        action = agent.select_action(obs)
        assert 0 <= action < 10
    
    def test_action_with_legal_mask(self):
        """Test action selection with legal actions."""
        agent = RandomAgent(action_size=10, seed=42)
        obs = np.zeros(20, dtype=np.float32)
        legal_actions = [3, 5, 7]
        
        action = agent.select_action(obs, legal_actions)
        assert action in legal_actions


class TestDQNAgent:
    """Tests for DQN Agent."""
    
    @pytest.fixture
    def agent(self):
        """Create a test DQN agent."""
        return DQNAgent(
            state_size=20,
            action_size=10,
            hidden_sizes=[32, 32],
            buffer_size=1000,
            batch_size=8,
            device="cpu",
            seed=42
        )
    
    def test_agent_creation(self, agent):
        """Test agent creation."""
        assert agent.state_size == 20
        assert agent.action_size == 10
    
    def test_action_selection(self, agent):
        """Test action selection."""
        obs = np.random.randn(20).astype(np.float32)
        
        # In training mode, should sometimes explore
        action = agent.select_action(obs)
        assert 0 <= action < 10
    
    def test_action_with_legal_mask(self, agent):
        """Test action selection respects legal actions."""
        obs = np.random.randn(20).astype(np.float32)
        legal_actions = [2, 4, 6]
        
        # In eval mode, should always pick from legal actions
        agent.eval()
        for _ in range(100):
            action = agent.select_action(obs, legal_actions)
            assert action in legal_actions
    
    def test_learning(self, agent):
        """Test learning from transitions."""
        # Fill replay buffer
        for _ in range(100):
            obs = np.random.randn(20).astype(np.float32)
            action = np.random.randint(10)
            reward = np.random.randn()
            next_obs = np.random.randn(20).astype(np.float32)
            done = np.random.random() < 0.1
            
            result = agent.learn(obs, action, reward, next_obs, done)
        
        # Should have learned after enough samples
        assert result is not None
        assert "loss" in result
    
    def test_save_load(self, agent, tmp_path):
        """Test saving and loading agent."""
        # Train a bit
        for _ in range(100):
            obs = np.random.randn(20).astype(np.float32)
            action = agent.select_action(obs)
            reward = 1.0
            next_obs = np.random.randn(20).astype(np.float32)
            agent.learn(obs, action, reward, next_obs, False)
        
        # Save
        save_path = str(tmp_path / "agent.pt")
        agent.save(save_path)
        
        # Load into new agent
        new_agent = DQNAgent(
            state_size=20,
            action_size=10,
            hidden_sizes=[32, 32],
            device="cpu"
        )
        new_agent.load(save_path)
        
        # Should have same epsilon
        assert new_agent.epsilon == agent.epsilon


class TestDQNAgentExtended:
    """Extended DQN Agent tests."""

    def test_get_q_values(self):
        """Test getting Q-values."""
        agent = DQNAgent(
            state_size=10, action_size=5,
            hidden_sizes=[16], device="cpu"
        )
        obs = np.random.randn(10).astype(np.float32)
        q_values = agent.get_q_values(obs)
        assert q_values.shape == (5,)

    def test_train_eval_modes(self):
        """Test switching between train and eval modes."""
        agent = DQNAgent(
            state_size=10, action_size=5,
            hidden_sizes=[16], device="cpu"
        )
        assert agent.training is True
        
        agent.eval()
        assert agent.training is False
        
        agent.train()
        assert agent.training is True

    def test_reset(self):
        """Test episode reset."""
        agent = DQNAgent(
            state_size=10, action_size=5,
            hidden_sizes=[16], device="cpu"
        )
        initial_episodes = agent.episodes
        agent.reset()
        assert agent.episodes == initial_episodes + 1

    def test_epsilon_decays_per_episode_not_per_step(self):
        """Epsilon must decay once per episode (reset), not per gradient step."""
        agent = DQNAgent(
            state_size=10, action_size=5,
            hidden_sizes=[16], device="cpu",
            epsilon_start=1.0, epsilon_decay=0.9
        )
        initial_epsilon = agent.epsilon

        # Learning steps alone must not move epsilon, however many there are.
        for _ in range(100):
            obs = np.random.randn(10).astype(np.float32)
            next_obs = np.random.randn(10).astype(np.float32)
            agent.learn(obs, 0, 1.0, next_obs, False)
        assert agent.epsilon == initial_epsilon

        # The first reset opens the first episode and must not decay either.
        agent.reset()
        assert agent.epsilon == initial_epsilon

        agent.reset()
        assert agent.epsilon == pytest.approx(0.9)
        agent.reset()
        assert agent.epsilon == pytest.approx(0.81)

    def test_dropout_network(self):
        """Test network with dropout."""
        from rl_card_lib.agents.dqn_agent import QNetwork
        net = QNetwork(10, 5, hidden_sizes=[16], dropout=0.5)
        import torch
        x = torch.randn(1, 10)
        output = net(x)
        assert output.shape == (1, 5)

    def test_auto_device(self):
        """Test automatic device selection."""
        agent = DQNAgent(
            state_size=10, action_size=5,
            hidden_sizes=[16], device=None
        )
        # Should have selected a device
        assert agent.device is not None

    def test_explicit_cpu_device(self):
        """Test explicit CPU device."""
        agent = DQNAgent(
            state_size=10, action_size=5,
            hidden_sizes=[16], device="cpu"
        )
        assert "cpu" in str(agent.device)

    def test_no_seed(self):
        """Test agent without seed."""
        agent = DQNAgent(
            state_size=10, action_size=5,
            hidden_sizes=[16], device="cpu", seed=None
        )
        assert agent is not None

    def test_target_network_update(self):
        """Test target network updates at correct interval."""
        agent = DQNAgent(
            state_size=10, action_size=5,
            hidden_sizes=[16], device="cpu",
            target_update_freq=10,  # Update every 10 steps
            batch_size=4,
            buffer_size=100
        )
        
        # Fill buffer and train enough to trigger target update
        for i in range(50):
            obs = np.random.randn(10).astype(np.float32)
            next_obs = np.random.randn(10).astype(np.float32)
            agent.learn(obs, i % 5, 1.0, next_obs, False)
        
        # Should have triggered target updates
        assert agent.train_steps > agent.target_update_freq


class TestRandomAgentExtended:
    """Extended Random Agent tests."""

    def test_save_load_no_op(self):
        """Test save/load are no-ops."""
        agent = RandomAgent(action_size=10)
        agent.save("dummy_path")  # Should not crash
        agent.load("dummy_path")  # Should not crash

    def test_str_repr(self):
        """Test string representation."""
        agent = RandomAgent(action_size=10)
        assert "RandomAgent" in str(agent)


class TestBaseAgent:
    """Tests for base Agent class."""

    def test_learn_default(self):
        """Test default learn returns None."""
        agent = RandomAgent(action_size=10)
        result = agent.learn(
            np.zeros(10), 0, 1.0, np.zeros(10), False
        )
        assert result is None

    def test_reset_default(self):
        """Test default reset is no-op."""
        agent = RandomAgent(action_size=10)
        agent.reset()  # Should not crash

    def test_save_load_not_implemented(self):
        """Test base agent save/load raise errors."""
        from rl_card_lib.agents.base import Agent
        
        class MinimalAgent(Agent):
            def select_action(self, obs, legal_actions=None):
                return 0
        
        agent = MinimalAgent()
        with pytest.raises(NotImplementedError):
            agent.save("path")
        with pytest.raises(NotImplementedError):
            agent.load("path")


class TestDQNEpsilonEdgeCases:
    """Tests for DQN epsilon edge cases."""

    def test_epsilon_already_at_minimum(self):
        """Test no decay when epsilon equals epsilon_end."""
        agent = DQNAgent(
            state_size=10, action_size=5,
            hidden_sizes=[16], device="cpu",
            epsilon_start=0.01, epsilon_end=0.01,  # Same value
            epsilon_decay=0.9, batch_size=4, buffer_size=100
        )
        
        initial_epsilon = agent.epsilon

        # Episodes pass, but epsilon is already at the floor.
        for _ in range(5):
            agent.reset()

        # Epsilon should not have changed
        assert agent.epsilon == initial_epsilon

    def test_epsilon_below_minimum(self):
        """Test no decay when epsilon is below epsilon_end."""
        agent = DQNAgent(
            state_size=10, action_size=5,
            hidden_sizes=[16], device="cpu",
            epsilon_start=0.05, epsilon_end=0.1,  # Start below end
            epsilon_decay=0.9, batch_size=4, buffer_size=100
        )
        
        initial_epsilon = agent.epsilon

        # Episodes pass, but epsilon started below the floor.
        for _ in range(5):
            agent.reset()

        # Epsilon should not have changed since it was already below end
        assert agent.epsilon == initial_epsilon
