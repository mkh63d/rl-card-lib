"""Agents module containing RL agents.

The agents span three families, which is deliberate: the interesting comparison
in a card game is not between two learners but between learning, hand-written
rules and search.

Baselines (no learning)
    RandomAgent          uniform over legal actions; the floor
    HeuristicAgent       base class for hand-written rules
    GreedyLookaheadAgent maximizes simulated reward `depth` moves ahead
    MCTSAgent            UCT search with determinized hidden cards

Learners
    QLearningAgent       tabular Q-learning; the didactic reference point
    DQNAgent             vanilla DQN
    DoubleDQNAgent       double + dueling + masked targets
    PPOAgent             on-policy actor-critic with masked policy

Search and rule agents need the game object rather than the observation vector,
so they derive from GameAwareAgent and must be bound to a game or env first.
"""

from rl_card_lib.agents.base import Agent
from rl_card_lib.agents.random_agent import RandomAgent
from rl_card_lib.agents.heuristic import (
    GameAwareAgent,
    GreedyLookaheadAgent,
    HeuristicAgent,
)
from rl_card_lib.agents.tabular import QLearningAgent
from rl_card_lib.agents.dqn_agent import DQNAgent, QNetwork
from rl_card_lib.agents.double_dqn_agent import (
    DoubleDQNAgent,
    DuelingQNetwork,
    MaskedReplayBuffer,
)
from rl_card_lib.agents.ppo_agent import ActorCritic, PPOAgent
from rl_card_lib.agents.mcts_agent import MCTSAgent

__all__ = [
    "Agent",
    "GameAwareAgent",
    "RandomAgent",
    "HeuristicAgent",
    "GreedyLookaheadAgent",
    "MCTSAgent",
    "QLearningAgent",
    "DQNAgent",
    "QNetwork",
    "DoubleDQNAgent",
    "DuelingQNetwork",
    "MaskedReplayBuffer",
    "PPOAgent",
    "ActorCritic",
]
