"""Abstract base class for general games."""

from abc import ABC, abstractmethod
from typing import Any, Optional
import numpy as np


class Game(ABC):
    """Game-agnostic base class for turn- and step-based games.

    A `Game` describes one playable game to the rest of the library: how to
    start it (`reset`), advance it (`step`), enumerate legal moves
    (`get_legal_actions`), and encode its state for a neural network
    (`get_observation`). Implement the seven abstract methods and maintain the
    plain attributes below; everything else — the Gymnasium wrapper, the
    agents, the trainer and the report — works on any conforming game unchanged.

    Attributes:
        num_players: Number of seats in the game (1 for solitaire).
        players: Per-seat state objects (game-specific; may be empty).
        current_player_idx: Index of the seat to move next.
        done: True once the game has reached a terminal state.
        winner: Index of the winning player, or None (no winner / not over).

    Provided with sensible defaults (override only when needed): `copy`,
    `determinize`, `get_reward`, `render`, `action_to_string`,
    `get_legal_action_mask`, `get_current_player`, `next_player`, `get_winner`,
    `log_action`, `get_history`.
    """

    def __init__(self, num_players: int = 1):
        self.num_players = num_players
        self.players: list[Any] = []
        self.current_player_idx: int = 0
        self.done: bool = False
        self.winner: Optional[int] = None
        self._turn_count: int = 0
        self._history: list[dict] = []

    @abstractmethod
    def reset(self) -> np.ndarray:
        """Start a new game and return the initial observation.

        Implementations must clear terminal state (`self.done = False`,
        `self.winner = None`) and return `get_observation()`.
        """

    @abstractmethod
    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Apply `action` and return the Gymnasium five-tuple.

        Args:
            action: A legal action index (see `get_legal_actions`).

        Returns:
            `(observation, reward, terminated, truncated, info)` — the same
            contract as a Gymnasium environment's `step`.
        """

    @abstractmethod
    def get_legal_actions(self) -> list[int]:
        """Return the action indices legal in the current state."""

    @abstractmethod
    def get_observation(self) -> np.ndarray:
        """Return the current state encoded as a fixed-shape float array."""

    @abstractmethod
    def get_action_space_size(self) -> int:
        """Return the total number of distinct actions (legal or not)."""

    @abstractmethod
    def get_observation_shape(self) -> tuple[int, ...]:
        """Return the shape of the array `get_observation` produces."""

    @abstractmethod
    def is_game_over(self) -> bool:
        """Return True once the game has reached a terminal state."""

    def get_current_player(self) -> Any:
        """Return the state object for the seat currently to move."""
        return self.players[self.current_player_idx]

    def next_player(self) -> Any:
        """Advance to the next seat (wrapping around) and return it."""
        self.current_player_idx = (self.current_player_idx + 1) % self.num_players
        return self.get_current_player()

    def action_to_string(self, action: int) -> str:
        """Return a human-readable label for an action (cosmetic)."""
        return f"Action {action}"

    def render(self) -> str:
        """Return a printable summary of the current game state."""
        lines = [
            f"=== {self.__class__.__name__} ===",
            f"Turn: {self._turn_count}",
            f"Current Player: {self.current_player_idx}",
            f"Game Over: {self.done}",
        ]
        return "\n".join(lines)

    def get_reward(self, player_idx: int) -> float:
        """Return the terminal payoff for `player_idx`.

        Defaults to 0.0. Multiplayer games must override this so MCTS can see
        each non-actor's outcome — otherwise losses are invisible to the search.
        """
        return 0.0

    def get_winner(self) -> Optional[int]:
        """Return the winning player index, or None."""
        return self.winner

    def log_action(self, action: int, player_idx: int, reward: float) -> None:
        """Append an action to the internal move history."""
        self._history.append({
            "turn": self._turn_count,
            "player": player_idx,
            "action": action,
            "reward": reward,
        })

    def get_history(self) -> list[dict]:
        """Return a copy of the logged move history."""
        return self._history.copy()

    def get_legal_action_mask(self) -> np.ndarray:
        """Return a boolean mask over all actions, True where legal."""
        mask = np.zeros(self.get_action_space_size(), dtype=bool)
        for action in self.get_legal_actions():
            mask[action] = True
        return mask

    def copy(self) -> "Game":
        """Return an independent copy of this game state.

        The default is a deep copy, so search agents (GreedyLookaheadAgent,
        MCTSAgent) work out of the box for any pure-Python game without the
        author writing a clone. Games with large state or an RNG whose stream
        must be reproduced exactly override this for speed and control -- the
        two bundled games do -- but a custom game is not required to.
        """
        import copy as _copy

        return _copy.deepcopy(self)

    def determinize(self, observer_idx: int = 0, rng: Optional[Any] = None) -> "Game":
        """
        Return a copy with the cards hidden from `observer_idx` re-dealt at random.

        Search agents call this so they plan over a state they could actually
        have deduced, instead of reading face-down cards straight out of the
        game object. The default assumes perfect information and just copies.

        Args:
            observer_idx: Player whose knowledge the sample must stay consistent with
            rng: `random.Random` instance to draw from (None for the global one)

        Returns:
            A game whose visible state matches this one
        """
        return self.copy()
