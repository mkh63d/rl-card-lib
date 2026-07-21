"""Abstract base class for general games."""

from abc import ABC, abstractmethod
from typing import Any, Optional
import numpy as np


class Game(ABC):
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
        pass

    @abstractmethod
    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        pass

    @abstractmethod
    def get_legal_actions(self) -> list[int]:
        pass

    @abstractmethod
    def get_observation(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_action_space_size(self) -> int:
        pass

    @abstractmethod
    def get_observation_shape(self) -> tuple[int, ...]:
        pass

    @abstractmethod
    def is_game_over(self) -> bool:
        pass

    def get_current_player(self) -> Any:
        return self.players[self.current_player_idx]

    def next_player(self) -> Any:
        self.current_player_idx = (self.current_player_idx + 1) % self.num_players
        return self.get_current_player()

    def action_to_string(self, action: int) -> str:
        return f"Action {action}"

    def render(self) -> str:
        lines = [
            f"=== {self.__class__.__name__} ===",
            f"Turn: {self._turn_count}",
            f"Current Player: {self.current_player_idx}",
            f"Game Over: {self.done}",
        ]
        return "\n".join(lines)

    def get_reward(self, player_idx: int) -> float:
        return 0.0

    def get_winner(self) -> Optional[int]:
        return self.winner

    def log_action(self, action: int, player_idx: int, reward: float) -> None:
        self._history.append({
            "turn": self._turn_count,
            "player": player_idx,
            "action": action,
            "reward": reward,
        })

    def get_history(self) -> list[dict]:
        return self._history.copy()

    def get_legal_action_mask(self) -> np.ndarray:
        mask = np.zeros(self.get_action_space_size(), dtype=bool)
        for action in self.get_legal_actions():
            mask[action] = True
        return mask

    def copy(self) -> "Game":
        raise NotImplementedError("copy() must be implemented by subclass")

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
