"""
RL Card Library - Training agents for card-game play using reinforcement learning.

A universal library enabling the creation and training of agents playing card games.
"""

__version__ = "0.1.0"
__author__ = "Michał Hołyński"

from rl_card_lib.core import Card, Deck, CardGame, Player
from rl_card_lib.env import CardGameEnv
from rl_card_lib.agents import DQNAgent, RandomAgent
from rl_card_lib.trainer import Trainer

__all__ = [
    "Card",
    "Deck", 
    "CardGame",
    "Player",
    "CardGameEnv",
    "DQNAgent",
    "RandomAgent",
    "Trainer",
]
