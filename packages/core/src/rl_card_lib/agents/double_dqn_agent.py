"""Double DQN agent with a dueling network and legal-action masking."""

from typing import Optional
from collections import deque
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from rl_card_lib.agents.dqn_agent import DQNAgent, QNetwork

#: Stand-in for -inf when masking illegal actions. Real -inf survives argmax but
#: turns into NaN the moment a masked value reaches a gradient or an entropy term.
MASK_VALUE = -1e8


class DuelingQNetwork(nn.Module):
    """
    Q-network that estimates state value and action advantages separately.

    Q(s,a) = V(s) + A(s,a) - mean_a' A(s,a')

    Splitting the two lets the network learn "this position is bad" from a single
    sample, rather than having to discover it once per action. That pays off in
    card games where most actions in a bad position are equally bad and the
    action space is large relative to how often each action is tried.

    Subtracting the mean advantage is what keeps the split identifiable: without
    it, any constant could shift between V and A and leave Q unchanged.
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_sizes: list[int] = [256, 256],
        dropout: float = 0.0
    ):
        """
        Initialize the dueling network.

        Args:
            state_size: Dimension of state observation
            action_size: Number of possible actions
            hidden_sizes: Hidden layer sizes for the shared trunk; the last entry
                also sets the width of the value and advantage heads
            dropout: Dropout probability
        """
        super().__init__()

        if not hidden_sizes:
            raise ValueError("DuelingQNetwork needs at least one hidden layer")

        layers: list[nn.Module] = []
        prev_size = state_size
        for hidden_size in hidden_sizes[:-1]:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_size = hidden_size

        self.trunk = nn.Sequential(*layers) if layers else nn.Identity()

        head_size = hidden_sizes[-1]
        self.value_head = nn.Sequential(
            nn.Linear(prev_size, head_size),
            nn.ReLU(),
            nn.Linear(head_size, 1),
        )
        self.advantage_head = nn.Sequential(
            nn.Linear(prev_size, head_size),
            nn.ReLU(),
            nn.Linear(head_size, action_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.trunk(x)
        value = self.value_head(features)
        advantage = self.advantage_head(features)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)


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


class DoubleDQNAgent(DQNAgent):
    """
    DQN with three fixes to the vanilla version, each aimed at a specific failure.

    - Double Q-learning: the online network picks the next action, the target
      network scores it. Vanilla DQN uses one network for both, so noise that
      happens to raise a Q-value gets selected *and* trusted, and the resulting
      overestimation compounds through the bootstrap.
    - Dueling head: see DuelingQNetwork. Disable with dueling=False.
    - Legal-action masking in the target: only actions the next state allows can
      be bootstrapped from.

    Also uses a Huber loss instead of MSE, so a single bad TD error produces a
    bounded gradient rather than a spike that wrecks the weights.

    Everything else, including the epsilon schedule and checkpoint format, is
    inherited from DQNAgent.
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
        dueling: bool = True,
        device: Optional[str] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize the Double DQN agent.

        Args:
            state_size: Dimension of state observation
            action_size: Number of possible actions
            hidden_sizes: List of hidden layer sizes for Q-network
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for future rewards
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Decay factor for epsilon per learning step
            buffer_size: Capacity of replay buffer
            batch_size: Batch size for training
            target_update_freq: Learning steps between target network updates
            dueling: Whether to use the dueling architecture
            device: Device to use ("cuda", "cpu", or None for auto)
            seed: Random seed for reproducibility
        """
        # _create_network() runs inside the parent constructor and reads this.
        self.dueling = dueling

        super().__init__(
            state_size=state_size,
            action_size=action_size,
            hidden_sizes=hidden_sizes,
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay,
            buffer_size=buffer_size,
            batch_size=batch_size,
            target_update_freq=target_update_freq,
            device=device,
            seed=seed,
        )

        self.name = "DoubleDQNAgent"
        self.replay_buffer = MaskedReplayBuffer(buffer_size, action_size)

    def _create_network(self) -> nn.Module:
        """Build a dueling or plain Q-network, per the `dueling` flag."""
        if self.dueling:
            return DuelingQNetwork(
                self.state_size, self.action_size, self.hidden_sizes
            )
        return QNetwork(self.state_size, self.action_size, self.hidden_sizes)

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
        Store the transition and take one gradient step on a sampled batch.

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
        self.replay_buffer.push(
            observation, action, reward, next_observation, done, next_legal_actions
        )

        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones, next_masks = (
            self.replay_buffer.sample(self.batch_size)
        )

        states = torch.as_tensor(states, device=self.device)
        actions = torch.as_tensor(actions, device=self.device)
        rewards = torch.as_tensor(rewards, device=self.device)
        next_states = torch.as_tensor(next_states, device=self.device)
        dones = torch.as_tensor(dones, device=self.device)
        next_masks = torch.as_tensor(next_masks, device=self.device)

        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Online network selects, target network evaluates: that split is
            # what stops selection noise from feeding straight into the target.
            online_next_q = self.q_network(next_states)
            online_next_q = online_next_q.masked_fill(~next_masks, MASK_VALUE)
            best_actions = online_next_q.argmax(dim=1, keepdim=True)

            next_q = self.target_network(next_states).gather(1, best_actions).squeeze(1)

            # A state with no legal actions has nothing to bootstrap from; its
            # argmax above is meaningless, so drop the term entirely.
            has_actions = next_masks.any(dim=1).float()
            target_q = rewards + (1 - dones) * has_actions * self.gamma * next_q

        loss = F.smooth_l1_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()

        self.train_steps += 1

        if self.train_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

        return {"loss": loss.item()}
