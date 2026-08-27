"""Tabular Q-learning agent."""

from typing import Optional
import pickle
import warnings

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

    Measured on Klondike: 1 253 141 entries after 1 499 936 steps, i.e. **0.836
    new entries per step**, and at epsilon = 0 the trained agent scores the
    random baseline to two decimals. `new_entries_per_step` reports that ratio
    while training runs, so the failure is a number the run records rather than
    one a reader has to derive afterwards.

    The table is unbounded by default, which is the textbook algorithm. Pass
    `max_table_size` to cap it: entries are then evicted least-recently-used and
    the first eviction warns. A cap bounds the memory -- the uncapped Klondike
    checkpoint is 1.26-1.76 GB and `pickle.dump` copies it while writing -- but
    it buys no accuracy. At ~0.84 new entries per step the evicted rows were
    never going to be looked up again either.

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
        max_table_size: Entries to keep before the least recently used one is
            evicted, or None for the unbounded textbook table. Must be at least
            2: one `learn()` call touches the rows for both `observation` and
            `next_observation`, and a cap of 1 would evict the row being updated
        seed: Random seed for reproducibility
    """

    accepts_next_legal_actions = True

    #: `save()` pickles the table rather than calling `torch.save`,
    #: so the checkpoint must not claim to be a torch archive.
    checkpoint_suffix = ".pkl"

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
        max_table_size: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(name="QLearningAgent")

        if max_table_size is not None and max_table_size < 2:
            raise ValueError(
                f"max_table_size must be at least 2, got {max_table_size}: a "
                "single learn() call touches two rows, so a cap of 1 would "
                "evict the row it is updating"
            )

        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.optimistic_init = optimistic_init
        self.precision = precision
        self.max_table_size = max_table_size
        self.seed = seed

        self.rng = np.random.RandomState(seed)
        # A plain dict: it has preserved insertion order since 3.7, which is
        # all the LRU below needs, and it keeps the pickle identical in type
        # to the ones already on disk.
        self.q_table: dict[bytes, np.ndarray] = {}

        self.steps = 0
        self.episodes = 0
        self.train_steps = 0
        # Entries ever created. Under a cap this keeps climbing after table_size
        # has flattened, and that gap is the evidence eviction bought nothing.
        self.new_entries = 0
        self.evictions = 0
        self._warned_full = False

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
        """Return the Q-row for a state, creating it on first visit.

        Under a cap this is also the LRU touch point: a hit moves its key to the
        end, and a miss that overflows the table drops the oldest one. The
        uncapped path skips that bookkeeping entirely, so the default
        configuration runs the same instructions it always did.
        """
        key = self._key(observation)
        row = self.q_table.get(key)
        if row is None:
            row = np.full(self.action_size, self.optimistic_init, dtype=np.float64)
            self.q_table[key] = row
            self.new_entries += 1
            if self.max_table_size is not None:
                self._evict_to_fit()
        elif self.max_table_size is not None:
            # Re-insert to move the key to the newest end of the dict.
            self.q_table[key] = self.q_table.pop(key)
        return row

    def _evict_to_fit(self) -> None:
        """Drop least-recently-used rows until the table is back within its cap.

        This cannot drop the row `learn()` is part-way through updating, because
        the cap is at least 2: the key just inserted is last, the one touched
        immediately before it is second from last, and eviction takes from the
        front.
        """
        while len(self.q_table) > self.max_table_size:
            del self.q_table[next(iter(self.q_table))]
            self.evictions += 1

        if self.evictions and not self._warned_full:
            self._warned_full = True
            warnings.warn(
                f"Q-table hit its {self.max_table_size}-entry cap and is now "
                f"evicting; {self.new_entries_per_step:.2f} of every step has "
                "created a new entry so far. A rate near 1.0 means this state "
                "space is too large to tabulate -- the agent is memorising "
                "rather than generalising, and its policy is at or near random. "
                "The cap bounds the memory, not the error.",
                RuntimeWarning,
                stacklevel=3,
            )

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

    def on_episode_end(self) -> None:
        """
        Count the episode and decay epsilon.

        Decaying here rather than in learn() keeps the exploration schedule in
        episodes, independent of how many steps each episode happens to take;
        decaying here rather than in reset() keeps it independent of how often
        the agent was evaluated, since evaluation resets an episode too.
        """
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
        self.episodes += 1

    @property
    def table_size(self) -> int:
        """Number of distinct states the agent has stored."""
        return len(self.q_table)

    @property
    def new_entries_per_step(self) -> float:
        """Share of environment steps that created a new table entry.

        The headline diagnostic for a tabular agent. Near 1.0 means practically
        every state it meets is one it has never seen, so its Q-row is still at
        `optimistic_init`, every legal action ties, and `select_action` picks
        among them uniformly -- a random policy wearing a Q-table. Klondike
        measures 0.84 here; Macao 0.99.

        Counted per step rather than per lookup so the number stays comparable
        across configurations: `learn()` consults two rows and an exploratory
        move consults none. That also lets the ratio sit a little above 1.0,
        since `learn()` instantiates the successor state's row as well as the
        current one -- so read it as "at or above 1.0 is total memorisation",
        not as a percentage.

        Under a cap this keeps climbing after `table_size` has flattened,
        because a state whose row was evicted counts as new when it returns.
        That is the intended reading: the cap bounded the memory, not the
        number of states the agent has no value for.
        """
        return self.new_entries / self.steps if self.steps else 0.0

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
                "max_table_size": self.max_table_size,
                "new_entries": self.new_entries,
                "evictions": self.evictions,
            }, handle)

    def load(self, path: str) -> None:
        """
        Load the Q-table and exploration state from file.

        The checkpoint's own `max_table_size` wins over this instance's, because
        it records how the table was actually trained: a file written without a
        cap loads back uncapped and untrimmed, whatever the caller constructed.
        That also keeps a pre-cap checkpoint loading exactly as it used to.

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

        # `.get` on the keys added with the cap: checkpoints written before it
        # do not carry them.
        self.max_table_size = checkpoint.get("max_table_size")
        self.q_table = checkpoint["q_table"]
        self.epsilon = checkpoint["epsilon"]
        self.steps = checkpoint["steps"]
        self.episodes = checkpoint["episodes"]
        self.train_steps = checkpoint["train_steps"]
        self.precision = checkpoint["precision"]
        self.new_entries = checkpoint.get("new_entries", len(self.q_table))
        self.evictions = checkpoint.get("evictions", 0)

    def get_q_values(self, observation: np.ndarray) -> np.ndarray:
        """
        Get Q-values for all actions in a state.

        Args:
            observation: State observation

        Returns:
            Array of Q-values for each action
        """
        return self._q_values(observation).copy()
