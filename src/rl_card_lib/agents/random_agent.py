"""Random agent implementation."""

from typing import Optional
import numpy as np

from rl_card_lib.agents.base import Agent


class RandomAgent(Agent):
    """
    Agent that selects random legal actions.
    
    Useful as a baseline for comparison with learning agents.
    """
    
    def __init__(self, action_size: int, seed: Optional[int] = None):
        """
        Initialize the random agent.
        
        Args:
            action_size: Total number of possible actions
            seed: Random seed for reproducibility
        """
        super().__init__(name="RandomAgent")
        self.action_size = action_size
        self.rng = np.random.RandomState(seed)
    
    def select_action(
        self,
        observation: np.ndarray,
        legal_actions: Optional[list[int]] = None
    ) -> int:
        """Select a random legal action."""
        if legal_actions is None or len(legal_actions) == 0:
            return self.rng.randint(0, self.action_size)
        return int(self.rng.choice(legal_actions))
    
    def save(self, path: str) -> None:
        """Random agent has no state to save."""
        pass
    
    def load(self, path: str) -> None:
        """Random agent has no state to load."""
        pass
