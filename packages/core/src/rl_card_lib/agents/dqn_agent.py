"""Deep Q-Network (DQN) agent implementation."""

from typing import Optional
from collections import deque
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from rl_card_lib.agents.base import Agent

#: Stand-in for -inf when masking illegal actions. Real -inf survives an argmax
#: but turns into NaN the moment a fully-masked row reaches the target: with no
#: legal next action the `max` is -inf, and the terminal guard's 0 * -inf is NaN.
MASK_VALUE = -1e8


class QNetwork(nn.Module):
    """Q-value neural network (MLP)."""
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_sizes: list[int] = [256, 256],
        dropout: float = 0.0
    ):
        """
        Initialize the Q-network.
        
        Args:
            state_size: Dimension of state observation
            action_size: Number of possible actions
            hidden_sizes: List of hidden layer sizes
            dropout: Dropout probability
        """
        super().__init__()
        
        layers = []
        prev_size = state_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, action_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ReplayBuffer:
    """Experience replay buffer for DQN training."""
    
    def __init__(self, capacity: int = 100000):
        """
        Initialize the buffer.
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> tuple:
        """
        Sample a batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self) -> int:
        return len(self.buffer)


class MaskedReplayBuffer:
    """
    Replay buffer that also remembers which actions the next state allowed.

    Without the mask the TD target can bootstrap off an action that is illegal in
    the next state, which inflates the target with a value the policy can never
    collect. In these games most actions are illegal in any given position, so
    that is the common case rather than an edge case.
    """

    def __init__(self, capacity: int, action_size: int):
        """
        Initialize the buffer.

        Args:
            capacity: Maximum number of transitions to store
            action_size: Number of possible actions, sets the mask width
        """
        self.buffer: deque = deque(maxlen=capacity)
        self.action_size = action_size

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_legal_actions: Optional[list[int]] = None,
    ) -> None:
        """
        Store a transition.

        Args:
            state: State before action
            action: Action taken
            reward: Reward received
            next_state: State after action
            done: Whether episode ended
            next_legal_actions: Actions legal in next_state; None means unknown,
                which is stored as "all legal"
        """
        if next_legal_actions is None:
            mask = np.ones(self.action_size, dtype=bool)
        else:
            mask = np.zeros(self.action_size, dtype=bool)
            mask[np.asarray(next_legal_actions, dtype=np.int64)] = True

        self.buffer.append((state, action, reward, next_state, done, mask))

    def sample(self, batch_size: int) -> tuple:
        """
        Sample a batch of transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Tuple of (states, actions, rewards, next_states, dones, next_masks)
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, masks = zip(*batch)

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            np.array(masks, dtype=bool),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent(Agent):
    """DQN agent with experience replay, a target network and legal-action masking.

    The target maximizes only over actions the next state actually allows. Without
    that mask the bootstrap can pick an action illegal in the next position, and
    since most actions are illegal in any given card-game position, noise in those
    unreachable Q-values is copied into the target network every update and
    compounds until the loss diverges. `DoubleDQNAgent` adds double-Q selection, a
    dueling head and a Huber loss on top; this stays single-network with an MSE
    loss so the two make a clean before/after teaching contrast.
    """

    accepts_next_legal_actions = True

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_sizes: list[int] = [256, 256],
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 100000,
        batch_size: int = 64,
        target_update_freq: int = 1000,
        device: Optional[str] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize the DQN agent.
        
        Args:
            state_size: Dimension of state observation
            action_size: Number of possible actions
            hidden_sizes: List of hidden layer sizes for Q-network
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for future rewards
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Decay factor applied to epsilon once per episode
                (in reset()), so the schedule is independent of episode length
            buffer_size: Capacity of replay buffer
            batch_size: Batch size for training
            target_update_freq: Steps between target network updates
            device: Device to use ("cuda", "cpu", or None for auto)
            seed: Random seed for reproducibility
        """
        super().__init__(name="DQNAgent")
        
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_sizes = list(hidden_sizes)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.seed = seed
        
        # Set random seeds
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Create networks
        self.q_network = self._create_network().to(self.device)
        self.target_network = self._create_network().to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer. The masked variant is what lets learn() bootstrap only
        # over the next state's legal actions.
        self.replay_buffer = MaskedReplayBuffer(buffer_size, action_size)
        
        # Counters
        self.steps = 0
        self.episodes = 0
        self.train_steps = 0

    def _create_network(self) -> nn.Module:
        """
        Build one Q-network. Called twice, for the online and target copies.

        Subclasses override this to change the architecture without
        reimplementing __init__.

        Returns:
            An un-moved network module
        """
        return QNetwork(self.state_size, self.action_size, self.hidden_sizes)

    def select_action(
        self,
        observation: np.ndarray,
        legal_actions: Optional[list[int]] = None
    ) -> int:
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            observation: Current state observation
            legal_actions: List of valid action indices
            
        Returns:
            Selected action index
        """
        self.steps += 1
        
        # Epsilon-greedy exploration
        if self.training and random.random() < self.epsilon:
            if legal_actions is not None and len(legal_actions) > 0:
                return random.choice(legal_actions)
            return random.randint(0, self.action_size - 1)
        
        # Greedy action selection
        with torch.no_grad():
            state = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            q_values = self.q_network(state).squeeze(0)
            
            if legal_actions is not None and len(legal_actions) > 0:
                # Mask illegal actions with very negative values
                mask = torch.full((self.action_size,), float("-inf"), device=self.device)
                for action in legal_actions:
                    mask[action] = 0
                q_values = q_values + mask
            
            return q_values.argmax().item()
    
    def learn(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
        next_legal_actions: Optional[list[int]] = None,
    ) -> Optional[dict]:
        """
        Store transition and perform learning step.

        Args:
            observation: State before action
            action: Action taken
            reward: Reward received
            next_observation: State after action
            done: Whether episode ended
            next_legal_actions: Actions legal in next_observation; None means
                unknown, and every action is bootstrapped over

        Returns:
            Dict with loss if learning occurred, None otherwise
        """
        # Store transition
        self.replay_buffer.push(
            observation, action, reward, next_observation, done, next_legal_actions
        )

        # Only train if enough samples
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample batch
        states, actions, rewards, next_states, dones, next_masks = (
            self.replay_buffer.sample(self.batch_size)
        )

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        next_masks = torch.as_tensor(next_masks, device=self.device)

        # Compute current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q values. The max ranges only over the next state's legal
        # actions; illegal ones are pushed to MASK_VALUE so they can never be the
        # bootstrap target. A state with no legal action has nothing to bootstrap
        # from, so has_actions drops the term (and keeps 0 * MASK_VALUE out of it).
        with torch.no_grad():
            next_q_all = self.target_network(next_states)
            next_q_all = next_q_all.masked_fill(~next_masks, MASK_VALUE)
            next_q = next_q_all.max(1)[0]
            has_actions = next_masks.any(dim=1).float()
            target_q = rewards + (1 - dones) * has_actions * self.gamma * next_q

        # MSE loss, kept deliberately: the Huber loss is a DoubleDQN-only change.
        loss = F.mse_loss(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()
        
        self.train_steps += 1

        # Update target network
        if self.train_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return {"loss": loss.item()}

    def reset(self) -> None:
        """
        Count the episode and decay epsilon.

        Epsilon decays here, once per episode, not per gradient step. Per-step
        decay made the effective schedule depend on episode length: a 300-step
        Klondike episode used to burn 300 decays, so a "9000-step" schedule was
        exhausted after 20 episodes and training was greedy for the other 99.6%.
        """
        if self.episodes > 0 and self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
        self.episodes += 1
    
    def save(self, path: str) -> None:
        """
        Save agent state to file.
        
        Args:
            path: File path to save to
        """
        torch.save({
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "steps": self.steps,
            "episodes": self.episodes,
            "train_steps": self.train_steps,
        }, path)
    
    def load(self, path: str) -> None:
        """
        Load agent state from file.
        
        Args:
            path: File path to load from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.steps = checkpoint["steps"]
        self.episodes = checkpoint["episodes"]
        self.train_steps = checkpoint["train_steps"]
    
    def get_q_values(self, observation: np.ndarray) -> np.ndarray:
        """
        Get Q-values for all actions given an observation.
        
        Args:
            observation: State observation
            
        Returns:
            Array of Q-values for each action
        """
        with torch.no_grad():
            state = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            q_values = self.q_network(state).squeeze(0)
            return q_values.cpu().numpy()
