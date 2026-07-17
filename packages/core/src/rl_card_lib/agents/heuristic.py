"""Heuristic (non-learning) agents and the base class for game-aware agents."""

from abc import abstractmethod
from typing import Any, Optional
import numpy as np

from rl_card_lib.agents.base import Agent


class GameAwareAgent(Agent):
    """
    Base for agents that need the game object, not just the observation vector.

    Rule-based and search-based agents reason over piles, hands and legal moves.
    Recovering those from a flat float vector is possible but pointless, so these
    agents hold a reference to the live game and read it during select_action().

    The reference makes them read-only observers: they must never mutate the game
    they are handed, only copies of it.
    """

    def __init__(
        self,
        game: Optional[Any] = None,
        name: str = "GameAwareAgent",
        seed: Optional[int] = None,
    ):
        """
        Initialize the agent.

        Args:
            game: Game or environment to read state from (can be bound later)
            name: Name identifier for the agent
            seed: Random seed for tie-breaking and rollouts
        """
        super().__init__(name=name)
        self.game: Optional[Any] = None
        self.rng = np.random.RandomState(seed)
        if game is not None:
            self.bind(game)

    def bind(self, source: Any) -> "GameAwareAgent":
        """
        Point the agent at a game, accepting either a game or an env wrapping one.

        Args:
            source: A Game instance, or an environment exposing `.game`

        Returns:
            self, so this can be chained onto the constructor
        """
        self.game = getattr(source, "game", source)
        return self

    def _require_game(self) -> Any:
        """Return the bound game, explaining the fix if there isn't one."""
        if self.game is None:
            raise RuntimeError(
                f"{self.name} needs a game to read. Pass game=... to the "
                f"constructor or call agent.bind(env) before selecting actions."
            )
        return self.game

    def save(self, path: str) -> None:
        """Rule-based agents carry no learned state."""

    def load(self, path: str) -> None:
        """Rule-based agents carry no learned state."""


class HeuristicAgent(GameAwareAgent):
    """
    Picks the legal action with the highest hand-written score.

    Subclasses implement score_action() with the domain knowledge; this class
    handles the plumbing and breaks ties at random so repeated positions do not
    always produce the same move.
    """

    #: Scores this close to the best count as tied.
    tie_tolerance: float = 1e-9

    def __init__(
        self,
        game: Optional[Any] = None,
        name: str = "HeuristicAgent",
        seed: Optional[int] = None,
    ):
        super().__init__(game=game, name=name, seed=seed)

    @abstractmethod
    def score_action(self, game: Any, action: int) -> float:
        """
        Rate a legal action in the given position. Higher is better.

        Args:
            game: Game to read the position from
            action: Legal action index to rate

        Returns:
            Score, on whatever scale the subclass finds convenient
        """

    def select_action(
        self,
        observation: np.ndarray,
        legal_actions: Optional[list[int]] = None
    ) -> int:
        """
        Select the highest scoring legal action, breaking ties at random.

        Args:
            observation: Unused, the score comes from the bound game
            legal_actions: Valid action indices (read from the game if None)

        Returns:
            Selected action index
        """
        game = self._require_game()

        if not legal_actions:
            legal_actions = list(game.get_legal_actions())
        if not legal_actions:
            return 0

        scores = np.array(
            [self.score_action(game, action) for action in legal_actions],
            dtype=np.float64,
        )
        best = np.flatnonzero(scores >= scores.max() - self.tie_tolerance)
        return int(legal_actions[self.rng.choice(best)])


class GreedyLookaheadAgent(GameAwareAgent):
    """
    Plays the action with the best simulated return, searching `depth` moves ahead.

    Game-agnostic: it only needs copy(), step() and get_legal_actions(), so it
    works on any game in the library without domain knowledge. What it optimizes
    is whatever the environment's reward function rewards, which makes it a
    useful probe: if this agent plays badly, the reward shaping is the suspect,
    not the learner.

    It has already earned its keep in that role: it exposed Klondike's old
    reward loop by spending ~139 of 150 moves shuffling tableau cards, since a
    non-revealing tableau move paid +0.05 against a -0.01 step cost and was
    reversible. That payment is gone (such moves now net -0.01), but keep the
    probe in mind whenever a reward function changes.

    Cost is branching^depth simulations per move, so depth beyond 2-3 is slow.
    Depth > 1 also assumes a single player, since summing rewards across players
    would treat an opponent's gain as the agent's own.
    """

    def __init__(
        self,
        game: Optional[Any] = None,
        depth: int = 1,
        gamma: float = 0.95,
        seed: Optional[int] = None,
    ):
        """
        Initialize the agent.

        Args:
            game: Game or environment to read state from (can be bound later)
            depth: Plies to search (1 is plain one-step greedy)
            gamma: Discount applied to rewards further down the search
            seed: Random seed for tie-breaking
        """
        super().__init__(game=game, name="GreedyLookaheadAgent", seed=seed)
        if depth < 1:
            raise ValueError(f"depth must be at least 1, got {depth}")
        self.depth = depth
        self.gamma = gamma

    def select_action(
        self,
        observation: np.ndarray,
        legal_actions: Optional[list[int]] = None
    ) -> int:
        """
        Select the action with the best simulated return.

        Args:
            observation: Unused, the search runs on copies of the bound game
            legal_actions: Valid action indices (read from the game if None)

        Returns:
            Selected action index
        """
        game = self._require_game()

        if self.depth > 1 and getattr(game, "num_players", 1) > 1:
            raise ValueError(
                "GreedyLookaheadAgent only supports depth > 1 for single-player "
                "games; deeper search would credit opponent rewards to this agent. "
                "Use MCTSAgent for multi-player search."
            )

        if not legal_actions:
            legal_actions = list(game.get_legal_actions())
        if not legal_actions:
            return 0

        scores = np.array(
            [self._value_of(game, action, self.depth) for action in legal_actions],
            dtype=np.float64,
        )
        best = np.flatnonzero(scores >= scores.max() - 1e-9)
        return int(legal_actions[self.rng.choice(best)])

    def _value_of(self, game: Any, action: int, depth: int) -> float:
        """
        Return the discounted reward of playing `action`, then continuing greedily.

        Args:
            game: Position to play the action from (never mutated)
            action: Action to simulate
            depth: Remaining plies to search

        Returns:
            Discounted return of the simulated line
        """
        simulated = game.copy()
        _, reward, terminated, truncated, _ = simulated.step(action)
        value = float(reward)

        if depth <= 1 or terminated or truncated:
            return value

        next_actions = simulated.get_legal_actions()
        if not next_actions:
            return value

        best_next = max(
            self._value_of(simulated, next_action, depth - 1)
            for next_action in next_actions
        )
        return value + self.gamma * best_next
