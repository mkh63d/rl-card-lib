# RL Card Library

A modular, **game-agnostic** library for training reinforcement-learning agents
to play card games — and, through a small `Game` contract, any turn-based game.
It ships an agent zoo spanning three families (baselines, search, learners), two
worked games (Klondike Solitaire and Macao), a training framework with
self-play, and a self-contained HTML training report.

<div class="grid cards" markdown>

- :material-rocket-launch: **[Get started](getting-started/installation.md)**
  Install the packages and run your first training sweep.

- :material-robot: **[The agent zoo](guides/agents.md)**
  Baselines, MCTS search, and the DQN / Double-DQN / PPO / Q-learning learners.

- :material-cards-playing: **[Add your own game](custom_game.md)**
  Implement the `Game` contract and get the full sweep and report for free.

- :material-sitemap: **[Architecture](architecture.md)**
  Package, use-case and training-sequence diagrams.

</div>

## Package structure

This is a monorepo of five packages that layer on top of a game-agnostic core.

| Package | What it provides | Depends on |
|---|---|---|
| **`rl-card-lib-core`** | `Game` base class, Gymnasium env wrappers (`CardGameEnv`, `MaskedCardGameEnv`), the agent framework, `Trainer` / `SelfPlayTrainer` and `TrainingMetrics` | — |
| **`rl-card-lib-cardgames`** | Card primitives (`Card`, `Suit`, `Rank`, `Deck`, `Player`, `CardGame`) and reusable rule predicates in `cardgames.rules` | core |
| **`rl-card-lib-visualizer`** | Text rendering of cards and tableaux (`render_cards`, `render_tableau`, `create_simple_board_view`) | core |
| **`rl-card-lib-report`** | `TrainingReport` (run configuration), `RunRecord` / `RunStore` (what a run did), `HtmlReport` (a self-contained page) | core; matplotlib via the `charts` extra |
| **`rl-card-lib-examples`** | Klondike (with a perfect-information solver) and Macao, hand-written heuristic agents, and the demo / training / benchmark scripts | core, cardgames |

## Features

- Define custom card games with Gymnasium compatibility.
- An agent zoo spanning three families:
    - **Baselines (no learning)** — `RandomAgent`, `HeuristicAgent`, `GreedyLookaheadAgent`
    - **Search** — `MCTSAgent` (UCT with determinized hidden cards)
    - **Learners** — `QLearningAgent`, `DQNAgent` (masked targets), `DoubleDQNAgent` (double-Q + dueling + Huber loss), `PPOAgent` (masked actor-critic)
- Built-in games: Klondike (with a budgeted perfect-information solver) and Macao.
- Reusable rule helpers in `rl_card_lib.cardgames.rules` for building new games.
- Training framework with self-play against a frozen opponent snapshot.
- Markdown / JSON parameter reports and a self-contained HTML training report.
- Modular design — install only the packages you need.

## License

MIT. Authored by Michał Hołyński
([mholynski@proton.me](mailto:mholynski@proton.me)).
