"""Base agent class."""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class Agent(ABC):
    """
    Abstract base class for all agents.

    Agents interact with card game environments by selecting actions
    based on observations and optionally learning from experience.
    """

    #: Set True by agents whose learn() takes a `next_legal_actions` keyword.
    #: Bootstrapping over actions the next state does not actually allow biases
    #: the TD target, but most agents ignore this, so Trainer only passes the
    #: argument to those that opt in.
    accepts_next_legal_actions: bool = False

    def __init__(self, name: str = "Agent"):
        """
        Initialize the agent.
        
        Args:
            name: Name identifier for the agent
        """
        self.name = name
        self.training = True
    
    @abstractmethod
    def select_action(
        self,
        observation: np.ndarray,
        legal_actions: Optional[list[int]] = None
    ) -> int:
        """
        Select an action based on the current observation.
        
        Args:
            observation: Current state observation
            legal_actions: List of valid action indices (if None, all actions valid)
            
        Returns:
            Selected action index
        """
        pass
    
    def learn(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool
    ) -> Optional[dict]:
        """
        Learn from a transition.
        
        Override in learning agents.
        
        Args:
            observation: State before action
            action: Action taken
            reward: Reward received
            next_observation: State after action
            done: Whether episode ended
            
        Returns:
            Optional dict with learning metrics
        """
        return None
    
    def train(self) -> None:
        self.training = True
    
    def eval(self) -> None:
        self.training = False
    
    def save(self, path: str) -> None:
        """
        Save agent to file.
        
        Args:
            path: File path to save to
        """
        raise NotImplementedError("save() not implemented")
    
    def load(self, path: str) -> None:
        """
        Load agent from file.
        
        Args:
            path: File path to load from
        """
        raise NotImplementedError("load() not implemented")
    
    def reset(self) -> None:
        """Reset agent state (called at start of episode)."""
        pass
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"
