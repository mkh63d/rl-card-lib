"""Abstract base class for card games."""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

from rl_card_lib.core.game import Game
from rl_card_lib.cardgames.deck import Deck
from rl_card_lib.cardgames.player import Player


class CardGame(Game, ABC):
    def __init__(self, num_players: int = 1):
        super().__init__(num_players=num_players)
        self.deck = Deck()
        self.players: list[Player] = []

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

    def get_current_player(self) -> Player:
        return self.players[self.current_player_idx]

    def next_player(self) -> Player:
        self.current_player_idx = (self.current_player_idx + 1) % self.num_players
        return self.get_current_player()

    def get_reward(self, player_idx: int) -> float:
        return 0.0

    def get_winner(self) -> Optional[int]:
        return self.winner

    def copy(self) -> "CardGame":
        """Return an independent copy; deep-copied by default (see Game.copy)."""
        import copy as _copy

        return _copy.deepcopy(self)
