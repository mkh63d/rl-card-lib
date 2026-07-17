"""RL Card Library - Training agents for card-game play using reinforcement learning."""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

__version__ = "0.1.0"
__author__ = "Michal Holynski"

from rl_card_lib.cardgames import Card, Deck, CardGame, Player
from rl_card_lib.core import Game, GymEnvWrapper
from rl_card_lib.env import CardGameEnv
from rl_card_lib.agents import (
	Agent,
	DoubleDQNAgent,
	DQNAgent,
	GameAwareAgent,
	GreedyLookaheadAgent,
	HeuristicAgent,
	MCTSAgent,
	PPOAgent,
	QLearningAgent,
	RandomAgent,
)
from rl_card_lib.trainer import Trainer

__all__ = [
	"Card",
	"Deck",
	"CardGame",
	"Player",
	"Game",
	"GymEnvWrapper",
	"CardGameEnv",
	"Agent",
	"GameAwareAgent",
	"RandomAgent",
	"HeuristicAgent",
	"GreedyLookaheadAgent",
	"MCTSAgent",
	"QLearningAgent",
	"DQNAgent",
	"DoubleDQNAgent",
	"PPOAgent",
	"Trainer",
]
