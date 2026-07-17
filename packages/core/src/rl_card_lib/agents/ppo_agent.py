"""Proximal Policy Optimization (PPO) agent with legal-action masking."""

from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

from rl_card_lib.agents.base import Agent

#: Stand-in for -inf when masking illegal actions. Real -inf makes the entropy
#: term evaluate 0 * -inf = NaN, which silently poisons the whole update.
MASK_VALUE = -1e8


class ActorCritic(nn.Module):
    """
    Shared-trunk network with a policy head and a value head.

    The trunk is shared because both heads need the same thing from a card game
    position, roughly "what can I do and how good is this", and sharing gives the
    value head far more gradient signal than it would get alone.
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_sizes: list[int] = [128, 128],
    ):
        """
        Initialize the network.

        Args:
            state_size: Dimension of state observation
            action_size: Number of possible actions
            hidden_sizes: List of hidden layer sizes for the shared trunk
        """
        super().__init__()

        layers: list[nn.Module] = []
        prev_size = state_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.Tanh())
            prev_size = hidden_size

        self.trunk = nn.Sequential(*layers) if layers else nn.Identity()
        self.policy_head = nn.Linear(prev_size, action_size)
        self.value_head = nn.Linear(prev_size, 1)

        # Small policy-head weights keep the initial distribution near uniform,
        # so early updates explore instead of locking onto one arbitrary action.
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.constant_(self.policy_head.bias, 0.0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run both heads.

        Args:
            x: Batch of observations

        Returns:
            Tuple of (action logits, state values)
        """
        features = self.trunk(x)
        return self.policy_head(features), self.value_head(features).squeeze(-1)


class PPOAgent(Agent):
    """
    On-policy actor-critic trained with the PPO clipped surrogate objective.

    The contrast with the DQN agents is the point of having this here. DQN learns
    a value for every action and derives a policy from it; PPO learns the policy
    directly and keeps a value function only to judge whether an action beat
    expectations. For card games with big action spaces where most actions are
    illegal most of the time, learning the policy directly avoids spending
    capacity on Q-values for moves that are rarely available.

    "Proximal" refers to the clipped ratio: an update may only move the
    probability of an action so far from the policy that collected the data,
    because past that point the sampled advantages no longer describe the new
    policy and the step is not trustworthy.

    Being on-policy, this agent cannot use a replay buffer: it collects
    rollout_steps of fresh experience, updates a few times over it, and throws it
    away. That is less sample-efficient than DQN but much more stable.

    Args:
        state_size: Dimension of state observation
        action_size: Number of possible actions
        hidden_sizes: List of hidden layer sizes for the shared trunk
        learning_rate: Learning rate for optimizer
        gamma: Discount factor for future rewards
        gae_lambda: GAE trace decay, trading advantage bias against variance
            (1.0 is the plain discounted return, 0.0 is the one-step TD error)
        clip_epsilon: How far the probability ratio may stray from 1
        epochs: Passes over each rollout
        minibatch_size: Minibatch size within an epoch
        rollout_steps: Steps collected before each update
        entropy_coef: Weight on the entropy bonus that discourages premature
            collapse onto one action
        value_coef: Weight on the value loss
        max_grad_norm: Gradient clipping threshold
        device: Device to use ("cuda", "cpu", or None for auto)
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_sizes: list[int] = [128, 128],
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        epochs: int = 4,
        minibatch_size: int = 64,
        rollout_steps: int = 1024,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(name="PPOAgent")

        self.state_size = state_size
        self.action_size = action_size
        self.hidden_sizes = list(hidden_sizes)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.minibatch_size = minibatch_size
        self.rollout_steps = rollout_steps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.seed = seed

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        self.rng = np.random.RandomState(seed)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.network = ActorCritic(state_size, action_size, self.hidden_sizes)
        self.network = self.network.to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)

        # Current rollout. States are appended by select_action, rewards by
        # learn(), so the two run one call out of step by design.
        self._states: list[np.ndarray] = []
        self._actions: list[int] = []
        self._log_probs: list[float] = []
        self._values: list[float] = []
        self._masks: list[np.ndarray] = []
        self._rewards: list[float] = []
        self._dones: list[float] = []

        self.steps = 0
        self.episodes = 0
        self.train_steps = 0
        self.updates = 0

    def _build_mask(self, legal_actions: Optional[list[int]]) -> np.ndarray:
        """Turn a legal-action list into a boolean mask; None or empty means all."""
        if not legal_actions:
            return np.ones(self.action_size, dtype=bool)
        mask = np.zeros(self.action_size, dtype=bool)
        mask[np.asarray(legal_actions, dtype=np.int64)] = True
        return mask

    def select_action(
        self,
        observation: np.ndarray,
        legal_actions: Optional[list[int]] = None
    ) -> int:
        """
        Sample an action from the masked policy, or take the best one in eval mode.

        Args:
            observation: Current state observation
            legal_actions: List of valid action indices

        Returns:
            Selected action index
        """
        self.steps += 1

        mask = self._build_mask(legal_actions)
        observation = np.asarray(observation, dtype=np.float32)

        with torch.no_grad():
            state = torch.as_tensor(observation, device=self.device).unsqueeze(0)
            logits, value = self.network(state)
            mask_tensor = torch.as_tensor(mask, device=self.device).unsqueeze(0)
            logits = logits.masked_fill(~mask_tensor, MASK_VALUE)

            if not self.training:
                return int(logits.argmax(dim=1).item())

            distribution = Categorical(logits=logits)
            action = distribution.sample()
            log_prob = distribution.log_prob(action)

        # An episode that ended between select_action and learn leaves a step
        # with no reward. Drop it rather than let rewards shift onto the wrong
        # state for the rest of the rollout.
        if len(self._states) > len(self._rewards):
            self._discard_incomplete_step()

        self._states.append(observation)
        self._actions.append(int(action.item()))
        self._log_probs.append(float(log_prob.item()))
        self._values.append(float(value.item()))
        self._masks.append(mask)

        return int(action.item())

    def _discard_incomplete_step(self) -> None:
        """Drop trailing steps that never received a reward."""
        keep = len(self._rewards)
        del self._states[keep:]
        del self._actions[keep:]
        del self._log_probs[keep:]
        del self._values[keep:]
        del self._masks[keep:]

    def learn(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool
    ) -> Optional[dict]:
        """
        Record the outcome of the last action, updating once a rollout fills up.

        Args:
            observation: State before action
            action: Action taken
            reward: Reward received
            next_observation: State after action
            done: Whether episode ended

        Returns:
            Dict with losses on update steps, None on the steps in between
        """
        if len(self._states) <= len(self._rewards):
            # learn() without a matching select_action; nothing to attach to.
            return None

        self._rewards.append(float(reward))
        self._dones.append(float(done))

        if len(self._rewards) < self.rollout_steps:
            return None

        return self._update(next_observation, done)

    def _compute_advantages(
        self,
        last_observation: np.ndarray,
        last_done: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute GAE advantages and the value targets that go with them.

        Args:
            last_observation: State the rollout stopped at, used to bootstrap
            last_done: Whether the rollout ended on a terminal state

        Returns:
            Tuple of (advantages, returns)
        """
        with torch.no_grad():
            state = torch.as_tensor(
                np.asarray(last_observation, dtype=np.float32), device=self.device
            ).unsqueeze(0)
            _, last_value = self.network(state)
            last_value = float(last_value.item()) * (1.0 - float(last_done))

        rewards = np.asarray(self._rewards, dtype=np.float32)
        values = np.asarray(self._values, dtype=np.float32)
        dones = np.asarray(self._dones, dtype=np.float32)

        advantages = np.zeros_like(rewards)
        running = 0.0
        for t in reversed(range(len(rewards))):
            next_value = values[t + 1] if t + 1 < len(rewards) else last_value
            # After a terminal step the next stored state belongs to a new
            # episode, so this zeroes both the bootstrap and the trace.
            non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * non_terminal - values[t]
            running = delta + self.gamma * self.gae_lambda * non_terminal * running
            advantages[t] = running

        return advantages, advantages + values

    def _update(self, last_observation: np.ndarray, last_done: bool) -> dict:
        """
        Run the PPO update over the collected rollout, then clear it.

        Args:
            last_observation: State the rollout stopped at
            last_done: Whether the rollout ended on a terminal state

        Returns:
            Dict of averaged losses over the update
        """
        advantages_np, returns_np = self._compute_advantages(last_observation, last_done)

        states = torch.as_tensor(np.asarray(self._states, dtype=np.float32), device=self.device)
        actions = torch.as_tensor(np.asarray(self._actions, dtype=np.int64), device=self.device)
        old_log_probs = torch.as_tensor(
            np.asarray(self._log_probs, dtype=np.float32), device=self.device
        )
        masks = torch.as_tensor(np.asarray(self._masks, dtype=bool), device=self.device)
        advantages = torch.as_tensor(advantages_np, device=self.device)
        returns = torch.as_tensor(returns_np, device=self.device)

        # Normalizing advantages keeps the step size comparable across rollouts
        # whose reward magnitudes differ.
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        sample_count = len(self._rewards)
        policy_losses, value_losses, entropies = [], [], []

        for _ in range(self.epochs):
            order = self.rng.permutation(sample_count)
            for start in range(0, sample_count, self.minibatch_size):
                batch = order[start:start + self.minibatch_size]
                if len(batch) < 2:
                    continue
                index = torch.as_tensor(batch, dtype=torch.long, device=self.device)

                logits, values_pred = self.network(states[index])
                logits = logits.masked_fill(~masks[index], MASK_VALUE)
                distribution = Categorical(logits=logits)

                new_log_probs = distribution.log_prob(actions[index])
                ratio = (new_log_probs - old_log_probs[index]).exp()

                batch_advantages = advantages[index]
                unclipped = ratio * batch_advantages
                clipped = torch.clamp(
                    ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon
                ) * batch_advantages

                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = F.mse_loss(values_pred, returns[index])
                entropy = distribution.entropy().mean()

                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.item())

        self._clear_rollout()
        self.updates += 1
        self.train_steps += 1

        return {
            "loss": float(np.mean(policy_losses) + np.mean(value_losses)),
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropies)),
        }

    def _clear_rollout(self) -> None:
        """Drop the collected rollout; PPO cannot reuse data across updates."""
        self._states.clear()
        self._actions.clear()
        self._log_probs.clear()
        self._values.clear()
        self._masks.clear()
        self._rewards.clear()
        self._dones.clear()

    def reset(self) -> None:
        """Reset episode counter. The rollout deliberately survives episodes."""
        self.episodes += 1

    def save(self, path: str) -> None:
        """
        Save agent state to file.

        Args:
            path: File path to save to
        """
        torch.save({
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "steps": self.steps,
            "episodes": self.episodes,
            "train_steps": self.train_steps,
            "updates": self.updates,
        }, path)

    def load(self, path: str) -> None:
        """
        Load agent state from file.

        Args:
            path: File path to load from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.steps = checkpoint["steps"]
        self.episodes = checkpoint["episodes"]
        self.train_steps = checkpoint["train_steps"]
        self.updates = checkpoint["updates"]

    def get_action_probabilities(
        self,
        observation: np.ndarray,
        legal_actions: Optional[list[int]] = None,
    ) -> np.ndarray:
        """
        Get the policy's action distribution for a state.

        Args:
            observation: State observation
            legal_actions: List of valid action indices

        Returns:
            Array of probabilities for each action, zero for illegal ones
        """
        mask = self._build_mask(legal_actions)
        with torch.no_grad():
            state = torch.as_tensor(
                np.asarray(observation, dtype=np.float32), device=self.device
            ).unsqueeze(0)
            logits, _ = self.network(state)
            mask_tensor = torch.as_tensor(mask, device=self.device).unsqueeze(0)
            logits = logits.masked_fill(~mask_tensor, MASK_VALUE)
            return F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
