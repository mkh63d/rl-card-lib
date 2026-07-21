# The agent zoo

The agents span **three families**, and that is deliberate: the interesting
comparison in a card game is not between two learners, but between *learning*,
*hand-written rules*, and *search*.

## Baselines (no learning)

| Agent | What it does |
|---|---|
| `RandomAgent` | Uniform over legal actions — the floor every other agent must beat. |
| `HeuristicAgent` | Base class for hand-written rules; subclass it and implement `score_action(game, action)`. |
| `GreedyLookaheadAgent` | Maximizes simulated reward `depth` moves ahead. |

## Search

| Agent | What it does |
|---|---|
| `MCTSAgent` | UCT search with **determinized** hidden cards — it re-samples the unknown state each simulation rather than cheating by reading it. |

## Learners

| Agent | What it does |
|---|---|
| `QLearningAgent` | Tabular Q-learning — the didactic reference point. |
| `DQNAgent` | Vanilla DQN with **masked TD targets** (illegal actions cannot leak into the bootstrap). |
| `DoubleDQNAgent` | Adds double-Q, a dueling network head, and Huber loss on top of the shared masking. |
| `PPOAgent` | On-policy actor-critic with a masked policy. |

!!! note "Why masking matters"
    In any card-game position most actions are illegal. An unmasked DQN target
    maximizes over illegal actions too, so their garbage Q-values leak into the
    bootstrap and compound through the target network — historically enough to
    make the trained greedy policy *worse* than an untrained one. Every learner
    here masks to the next state's legal actions.

## Game-aware vs. observation-only agents

Search and rule agents need the **game object** (to copy it, step it, and read
its legal actions), not just the observation vector. They derive from
`GameAwareAgent` and must be **bound to a game or environment** before use. The
learners, by contrast, only read `get_observation_shape()` and
`get_action_space_size()`, so they work on any game unchanged.

| | Reads | Works on a new game out of the box? |
|---|---|---|
| `RandomAgent`, `QLearningAgent`, `DQNAgent`, `DoubleDQNAgent`, `PPOAgent` | the observation vector | Yes |
| `GreedyLookaheadAgent`, `MCTSAgent` | the game object (`copy`, `step`, `get_legal_actions`; MCTS also `get_reward`) | Yes (MCTS needs `get_reward` for multiplayer) |
| `HeuristicAgent` | game-specific knowledge | Only if you write the `score_action` rules |

## Example

```python
from rl_card_lib.games import KlondikeSolitaire
from rl_card_lib.env import CardGameEnv
from rl_card_lib.agents import MCTSAgent, DoubleDQNAgent

env = CardGameEnv(KlondikeSolitaire(), max_steps=200)

# A learner: needs only the shapes.
learner = DoubleDQNAgent(
    state_size=env.observation_space.shape[0],
    action_size=env.action_space.n,
)

# A search agent: bind it to the game/env first.
searcher = MCTSAgent(simulations=200)
searcher.bind(env)   # game-aware agents must be bound before acting
```

See the full [agents API reference](../reference/agents.md) for constructor
arguments and methods.
