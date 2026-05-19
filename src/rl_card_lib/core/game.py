"""Abstract base class for general games."""

from abc import ABC, abstractmethod
from typing import Any, Optional
import numpy as np


class Game(ABC):
    """
    Abstract base class for general games.

    Subclasses must implement the core game logic methods.
    This class provides a standard interface for RL environments.

    Attributes:
        players: List of players (game-specific types)
        current_player_idx: Index of the current player
        done: Whether the game has ended
        winner: Index of the winning player (None if game not over)
    """
    
    def __init__(self, num_players: int = 1):
        """
        Initialize the game.
        
        Args:
            num_players: Number of players in the game
        """
        self.num_players = num_players
        self.players: list[Any] = []
        self.current_player_idx: int = 0
        self.done: bool = False
        self.winner: Optional[int] = None
        self._turn_count: int = 0
        self._history: list[dict] = []
    
    @abstractmethod
    def reset(self) -> np.ndarray:
        """
        Reset the game to its initial state.
        
        Returns:
            Initial observation (state) as numpy array
        """
        pass
    
    @abstractmethod
    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one action in the game.
        
        Args:
            action: Action index to execute
            
        Returns:
            Tuple of:
            - observation: New state as numpy array
            - reward: Reward for this action
            - terminated: Whether the game is over (win/lose/draw)
            - truncated: Whether episode was cut short (e.g., max steps)
            - info: Additional information dictionary
        """
        pass
    
    @abstractmethod
    def get_legal_actions(self) -> list[int]:
        """
        Get list of valid action indices for current state.
        
        Returns:
            List of valid action indices
        """
        pass
    
    @abstractmethod
    def get_observation(self) -> np.ndarray:
        """
        Get the current game state as an observation.
        
        Returns:
            State observation as numpy array
        """
        pass
    
    @abstractmethod
    def get_action_space_size(self) -> int:
        """
        Get the total number of possible actions.
        
        Returns:
            Size of the action space
        """
        pass
    
    @abstractmethod
    def get_observation_shape(self) -> tuple[int, ...]:
        """
        Get the shape of observation arrays.
        
        Returns:
            Tuple representing observation dimensions
        """
        pass
    
    @abstractmethod
    def is_game_over(self) -> bool:
        """
        Check if the game has ended.
        
        Returns:
            True if game is over
        """
        pass
    
    def get_current_player(self) -> Any:
        return self.players[self.current_player_idx]
    
    def next_player(self) -> Any:
        """
        Advance to the next player.
        
        Returns:
            The new current player
        """
        self.current_player_idx = (self.current_player_idx + 1) % self.num_players
        return self.get_current_player()
    
    def action_to_string(self, action: int) -> str:
        """
        Convert action index to human-readable string.
        
        Args:
            action: Action index
            
        Returns:
            String description of the action
        """
        return f"Action {action}"
    
    def render(self) -> str:
        """
        Render the current game state as a string.
        
        Returns:
            String representation of the game state
        """
        lines = [
            f"=== {self.__class__.__name__} ===",
            f"Turn: {self._turn_count}",
            f"Current Player: {self.current_player_idx}",
            f"Game Over: {self.done}",
        ]
        return "\n".join(lines)
    
    def get_reward(self, player_idx: int) -> float:
        """
        Get the reward for a specific player.
        
        Override in subclasses for custom reward functions.
        
        Args:
            player_idx: Index of the player
            
        Returns:
            Reward value
        """
        return 0.0
    
    def get_winner(self) -> Optional[int]:
        """
        Get the winning player index.
        
        Returns:
            Index of winner, or None if no winner yet
        """
        return self.winner
    
    def log_action(self, action: int, player_idx: int, reward: float) -> None:
        """
        Log an action to the game history.
        
        Args:
            action: Action taken
            player_idx: Player who took the action
            reward: Reward received
        """
        self._history.append({
            "turn": self._turn_count,
            "player": player_idx,
            "action": action,
            "reward": reward,
        })
    
    def get_history(self) -> list[dict]:
        return self._history.copy()
    
    def get_legal_action_mask(self) -> np.ndarray:
        """
        Get a boolean mask of legal actions.
        
        Returns:
            Boolean array where True indicates legal action
        """
        mask = np.zeros(self.get_action_space_size(), dtype=bool)
        for action in self.get_legal_actions():
            mask[action] = True
        return mask
    
    def copy(self) -> "Game":
        """
        Create a copy of the current game state.
        
        Override in subclasses for proper deep copying.
        """
        raise NotImplementedError("copy() must be implemented by subclass")
