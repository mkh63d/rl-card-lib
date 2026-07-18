"""Tabular Q-learning agent."""

from typing import Optional
import pickle
import numpy as np

from rl_card_lib.agents.base import Agent


class QLearningAgent(Agent):
    """
    Classic tabular Q-learning with an epsilon-greedy policy.

    Keeps one Q-value per (state, action) pair in a dictionary keyed by the raw
    observation bytes, and updates them with the off-policy TD rule
    ``Q(s,a) <- Q(s,a) + lr * (r + gamma * max_a' Q(s',a') - Q(s,a))``,
    where the max only ranges over actions the next state actually allows.

    This is here as the didactic reference point, not a contender: it can only
    reuse a Q-value when it sees a bit-identical observation again, and the games
    in this library have far too many distinct states for that to happen often.
    Expect the table to grow roughly one entry per step and the policy to stay
    near random. Watching that failure is the point, and it is what motivates the
    function approximation the DQN agents use.

    Args:
        action_size: Total number of possible actions
        learning_rate: Step size for the TD update
        gamma: Discount factor for future rewards
        epsilon_start: Initial exploration rate
        epsilon_end: Minimum exploration rate
        epsilon_decay: Multiplicative decay applied to epsilon once per episode
            (in reset()), so the schedule is independent of episode length
        optimistic_init: Q-value given to unseen states, above the reward scale
            to encourage trying unexplored actions
        precision: Decimal places the observation is rounded to before it is used
            as a key; coarser values merge more states into the same entry
        seed: Random seed for reproducibility
    """

    accepts_next_legal_actions = True

    def __init__(
        self,
        action_size: int,
        learning_rate: float = 0.1,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        optimistic_init: float = 0.0,
        precision: int = 2,
        seed: Optional[int] = None,
    ):
        super().__init__(name="QLearningAgent")

        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.optimistic_init = optimistic_init
        self.precision = precision
        self.seed = seed

        self.rng = np.random.RandomState(seed)
        self.q_table: dict[bytes, np.ndarray] = {}

        self.steps = 0
        self.episodes = 0
        self.train_steps = 0

    def _key(self, observation: np.ndarray) -> bytes:
        """
        Turn an observation into a dictionary key.

        Rounding first means observations that differ only in float noise share
        an entry; without it, continuous features would make every state unique.

        Args:
            observation: State observation

        Returns:
            Hashable key for the Q-table
        """
        rounded = np.round(np.asarray(observation, dtype=np.float32), self.precision)
        return rounded.tobytes()

    def _q_values(self, observation: np.ndarray) -> np.ndarray:
        """Return the Q-row for a state, creating it on first visit."""
        key = self._key(observation)
        row = self.q_table.get(key)
        if row is None:
            row = np.full(self.action_size, self.optimistic_init, dtype=np.float64)
            self.q_table[key] = row
        return row

    def select_action(
        self,
        observation: np.ndarray,
        legal_actions: Optional[list[int]] = None
    ) -> int:
        """
        Select an action using an epsilon-greedy policy over the legal actions.

        Args:
            observation: Current state observation
            legal_actions: List of valid action indices

        Returns:
            Selected action index
        """
        self.steps += 1

        if not legal_actions:
            legal_actions = list(range(self.action_size))

        if self.training and self.rng.random_sample() < self.epsilon:
            return int(self.rng.choice(legal_actions))

        q_values = self._q_values(observation)
        legal = np.asarray(legal_actions, dtype=np.int64)
        legal_q = q_values[legal]
        best = np.flatnonzero(legal_q >= legal_q.max() - 1e-12)
        return int(legal[self.rng.choice(best)])

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
        Apply one Q-learning update.

        Args:
            observation: State before action
            action: Action taken
            reward: Reward received
            next_observation: State after action
            done: Whether episode ended
            next_legal_actions: Actions the next state allows; the bootstrap only
                maximizes over these, since the policy could not pick the others

        Returns:
            Dict with the TD error magnitude, reported as "loss"
        """
        q_values = self._q_values(observation)

        if done:
            target = reward
        else:
            next_q = self._q_values(next_observation)
            if next_legal_actions:
                best_next = next_q[np.asarray(next_legal_actions, dtype=np.int64)].max()
            elif next_legal_actions is None:
                best_next = next_q.max()
            else:
                # Empty (not None) means the next state has no moves at all.
                best_next = 0.0
            target = reward + self.gamma * best_next

        td_error = target - q_values[action]
        q_values[action] += self.learning_rate * td_error

        self.train_steps += 1

        return {"loss": abs(float(td_error))}

    def reset(self) -> None:
        """
        Count the episode and decay epsilon.

        Decaying here rather than in learn() keeps the exploration schedule in
        episodes, independent of how many steps each episode happens to take.
        """
        if self.episodes > 0 and self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
        self.episodes += 1

    @property
    def table_size(self) -> int:
        """Number of distinct states the agent has stored."""
        return len(self.q_table)

    def save(self, path: str) -> None:
        """
        Save the Q-table and exploration state to file.

        Args:
            path: File path to save to
        """
        with open(path, "wb") as handle:
            pickle.dump({
                "q_table": self.q_table,
                "epsilon": self.epsilon,
                "steps": self.steps,
                "episodes": self.episodes,
                "train_steps": self.train_steps,
                "action_size": self.action_size,
                "precision": self.precision,
            }, handle)

    def load(self, path: str) -> None:
        """
        Load the Q-table and exploration state from file.

        Args:
            path: File path to load from
        """
        with open(path, "rb") as handle:
            checkpoint = pickle.load(handle)

        if checkpoint["action_size"] != self.action_size:
            raise ValueError(
                f"Checkpoint was trained with action_size="
                f"{checkpoint['action_size']}, this agent has {self.action_size}"
            )

        self.q_table = checkpoint["q_table"]
        self.epsilon = checkpoint["epsilon"]
        self.steps = checkpoint["steps"]
        self.episodes = checkpoint["episodes"]
        self.train_steps = checkpoint["train_steps"]
        self.precision = checkpoint["precision"]

    def get_q_values(self, observation: np.ndarray) -> np.ndarray:
        """
        Get Q-values for all actions in a state.

        Args:
            observation: State observation

        Returns:
            Array of Q-values for each action
        """
        return self._q_values(observation).copy()
