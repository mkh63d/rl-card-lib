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
| `QLearningAgent` | Tabular Q-learning — the didactic reference point, and a demonstration of why tabular methods do not reach these state spaces. |
| `DQNAgent` | Vanilla DQN with **masked TD targets** (illegal actions cannot leak into the bootstrap). |
| `DoubleDQNAgent` | Adds double-Q, a dueling network head, and Huber loss on top of the shared masking. |
| `PPOAgent` | On-policy actor-critic with a masked policy. |

!!! warning "`QLearningAgent` does not learn these games, and that is the lesson"
    Its table is keyed on the rounded observation. Klondike's is a 221-dim
    vector of which 208 are card-location bits, so `precision=2` merges nothing
    and the table grows **0.836 entries per step** — practically every position
    is one it has never seen. A fresh row is all `optimistic_init`, so every
    legal action ties and `select_action` picks uniformly. At ε = 0 the trained
    agent scores 9.79 cards on Klondike: the random baseline, to two decimals.
    It is a random policy wearing a Q-table, and it is what motivates the
    function approximation the DQN agents use.

    Watch `agent.new_entries_per_step` to see it happen. Near 1.0 means the
    agent is memorising rather than generalising, and no amount of training will
    change that — only a coarser state abstraction or function approximation
    will.

    The cost is real: an uncapped Klondike table reaches 1.26–1.76 GB per
    checkpoint. Pass `max_table_size` to bound it, and entries are evicted
    least-recently-used with a warning on the first eviction. That bounds the
    memory and nothing else — a capped `table_size` curve flattens while
    `new_entries_per_step` stays near 1.0, which is the honest reading. The
    library's sweep sets `SWEEP_Q_TABLE_LIMIT` (200 000 entries, ~0.3 GB a
    checkpoint); the class itself stays unbounded so that
    `QLearningAgent(action_size=...)` is the textbook algorithm.

!!! note "What `eval()` means, per family"
    Turning exploration off is not one rule. The value-based agents
    (`QLearningAgent`, `DQNAgent`, `DoubleDQNAgent`) learn `Q(s,a)` and *derive*
    a policy from it, so `eval()` takes the argmax — that greedy policy is
    exactly what they learned. `PPOAgent` learns the distribution itself, so
    `eval()` **samples** it; its argmax is a policy that was never trained and
    whose value the critic never estimated.

    On a game with reversible moves the two are far apart. A deterministic
    policy that walks back into a position it has already seen replays its whole
    future from there and cycles until the step cap, which on Klondike is the
    difference between 7.5 and 22.5 cards to the foundation from one set of
    weights. Ask for the deterministic number with
    `PPOAgent(..., eval_greedy=True)`, the mutable `agent.eval_greedy`, or
    `select_action(obs, legal, greedy=True)` for a single call. Sampled
    evaluation stays reproducible: `eval()` rewinds the sampler, so measuring the
    same agent twice gives the same numbers.

!!! note "Why masking matters"
    In any card-game position most actions are illegal. An unmasked DQN target
    maximizes over illegal actions too, so their garbage Q-values leak into the
    bootstrap and compound through the target network — historically enough to
    make the trained greedy policy *worse* than an untrained one. Every learner
    here masks to the next state's legal actions.

!!! note "Why the step cap is not an ending"
    An episode can stop for two different reasons, and only one of them is a
    fact about the game. `terminated` means the game reached a terminal state —
    the future really is worth nothing. `truncated` means the step cap fired
    while the game was still going, and the state it stopped at is worth
    whatever the value function says. `Trainer` reports only `terminated` as
    `done`, so a capped episode keeps its bootstrap. Collapsing the two used to
    zero the target on the last transition of *every* Klondike episode, since
    every one of them ended at the cap.

### Writing a learner of your own

`learn(observation, action, reward, next_observation, done)` is the whole
contract, and `done` is enough for anything that only bootstraps. Two class
flags ask `Trainer` for more:

| Flag | Adds to `learn()` | Ask for it when |
|---|---|---|
| `accepts_next_legal_actions` | `next_legal_actions=[...]` | you maximize over the next state and want the illegal actions masked out |
| `accepts_truncated` | `truncated=bool` | you hold several episodes in one buffer, so you need the boundary itself — `done` no longer marks a capped one |

`PPOAgent` needs the second: its rollout spans episode resets, so it uses
`truncated` to bootstrap through the cap while still cutting the advantage
trace at the boundary. Leave both flags alone and your five-argument `learn()`
is called exactly as before.

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
