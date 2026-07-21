# RL Card Library

A modular library for training reinforcement learning agents to play card games.

## 📦 Package Structure

This is a monorepo containing multiple packages:

### Core Packages

- **[`rl-card-lib-core`](packages/core/)** - Foundation library
  - Game base class, Gymnasium environment wrappers (`CardGameEnv`, `MaskedCardGameEnv`)
  - Agent framework: baselines, search agents and learners
  - `Trainer` / `SelfPlayTrainer` and `TrainingMetrics`
  - **Status**: Core package - all others depend on this

### Extension Packages

- **[`rl-card-lib-cardgames`](packages/cardgames/)** - Card primitives and rule helpers
  - `Card`, `Suit`, `Rank`, `Deck`, `Player`, `CardGame`
  - `cardgames.rules` - reusable predicates shared by games
    (`can_stack_alternating_descending`, `is_alternating_color`, `count_by_rank`, ...)
  - **Depends on**: `rl-card-lib-core`

- **[`rl-card-lib-visualizer`](packages/visualizer/)** - Visualization utilities
  - Text rendering of cards and tableaux
    (`render_cards`, `render_tableau`, `create_simple_board_view`)
  - Training curves via `TrainingMetrics.plot()` in the core package
  - **Depends on**: `rl-card-lib-core`

- **[`rl-card-lib-report`](packages/report/)** - Training parameter reports
  - `TrainingReport.from_trainer(...)` collects env, trainer, agent, DQN and PPO settings
  - Renders to Markdown (`to_markdown`) or JSON (`to_json`)
  - **Depends on**: `rl-card-lib-core`

- **[`rl-card-lib-examples`](packages/examples/)** - Games, demos and training scripts
  - Klondike Solitaire (plus a perfect-information solvability search) and Macao
  - Hand-written `KlondikeHeuristicAgent` / `MacaoHeuristicAgent`
  - Quick demo, training and benchmarking scripts
  - **Depends on**: `rl-card-lib-core`, `rl-card-lib-cardgames`
    (uses `rl-card-lib-report` and `rl-card-lib-visualizer` where available)

## 🚀 Quick Start

### Installation (Development)

```bash
# Install all packages in development mode
pip install -e .  # Installs root metapackage

# Or install individual packages
pip install -e ./packages/core
pip install -e ./packages/cardgames
pip install -e ./packages/visualizer
pip install -e ./packages/report
pip install -e ./packages/examples
```

### Quick Demo

```bash
python packages/examples/scripts/quick_demo.py
```

### Training and Benchmarking

```bash
# Train a chosen agent on a chosen game
python packages/examples/scripts/train_agents.py

# Compare the agent zoo on the same deals
python packages/examples/scripts/benchmark_agents.py

# Single-game training scripts
python packages/examples/scripts/train_klondike.py
python packages/examples/scripts/train_macao.py
```

### Run Tests

```bash
# Everything
pytest

# Library tests
pytest tests/ -q

# Example game tests (what CI runs separately)
pytest packages/examples/tests/ -q

# With coverage report
pytest --cov --cov-report=html
```

## 📋 Features

- ✅ Define custom card games with Gymnasium compatibility
- ✅ An agent zoo spanning three families:
  - **Baselines (no learning)**: `RandomAgent`, `HeuristicAgent`, `GreedyLookaheadAgent`
  - **Search**: `MCTSAgent` (UCT with determinized hidden cards)
  - **Learners**: `QLearningAgent`, `DQNAgent`, `DoubleDQNAgent` (double + dueling + masked
    targets), `PPOAgent` (masked actor-critic)
- ✅ Built-in games: Klondike (with a budgeted perfect-information solver), Macao
- ✅ Reusable rule helpers in `rl_card_lib.cardgames.rules` for building new games
- ✅ Training framework with self-play against a frozen opponent snapshot
- ✅ Markdown/JSON training parameter reports
- ✅ Visualization and metrics monitoring
- ✅ Modular design - use only what you need

## 📚 Documentation

- [Core Package](packages/core/README.md) - API and concepts
- [Card Games Package](packages/cardgames/README.md) - Game implementations
- [Visualizer Package](packages/visualizer/README.md) - Visualization tools
- [Report Package](packages/report/README.md) - Training parameter reports
- [Examples Package](packages/examples/README.md) - Usage examples
- [Architecture Diagrams](docx/README.md) - PlantUML package, use case and sequence diagrams

## 🧪 Testing Strategy

### Library Tests

Root-level tests cover the library and the public API contracts:

```
tests/test_core.py                 # Card, Deck, Player, CardGame
tests/test_agents.py               # Baseline and DQN agents
tests/test_new_agents.py           # Heuristic, MCTS, tabular, Double DQN, PPO
tests/test_trainer.py              # Trainer and SelfPlayTrainer
tests/test_reward_design.py        # Reward shaping and terminal conditions
tests/test_klondike_solver.py      # Perfect-information solvability search
tests/test_report.py               # TrainingReport rendering
tests/test_utils.py                # Encoding helpers
tests/test_public_api_imports.py   # Public API contracts
```

### Example Game Tests

The example games ship their own suite, run separately in CI:

```
packages/examples/tests/test_game_base.py   # Shared game contract
packages/examples/tests/test_games.py       # Klondike and Macao rules
```

### Coverage

Coverage is measured across all packages but not exported:

```bash
# Generate coverage report
pytest --cov --cov-report=html

# View report
open htmlcov/index.html
```

Coverage is a **dev dependency** only (not included in installations).

## 🔧 Development

### Setup Development Environment

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Or per-package
pip install -e "./packages/core[dev]"
```

### Code Quality Tools

```bash
# Format code
black .
isort .

# Type checking
mypy packages

# Linting (same command CI runs)
flake8 packages tests
pylint packages
```

### Continuous Integration

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs `flake8 packages tests`,
`pytest tests/ -q` and `pytest packages/examples/tests/ -q`.

## 📦 Publishing

Each package is independently published to PyPI:

```bash
# Build individual package
cd packages/core
python -m build

# Publish
twine upload dist/*
```

### Version Synchronization

All packages maintain synchronized versions (updated in `pyproject.toml`).

## 🎯 Architecture

```
rl-card-lib (root)
├── packages/
│   ├── core/              # Foundation
│   ├── cardgames/         # Extensions
│   ├── visualizer/        # Extensions
│   └── examples/          # Usage examples
├── tests/                 # Integration tests
├── examples/              # Root-level demos
└── pyproject.toml         # Root metapackage
```

## 📝 Dependencies

### Production
- `numpy>=1.21.0` - Numerical computing
- `torch>=2.0.0` - Deep learning
- `gymnasium>=0.29.0` - RL environment API
- `matplotlib>=3.5.0` - Plotting
- `tqdm>=4.64.0` - Progress bars

### Development
- `pytest>=7.0.0` - Testing
- `pytest-cov>=4.0.0` - Coverage measurement
- `black>=23.0.0` - Code formatting
- `isort>=5.12.0` - Import sorting
- `mypy>=1.0.0` - Type checking
- `flake8>=6.0.0` - Linting
- `pylint>=3.0.0` - Static analysis

## 📄 License

MIT - See [LICENSE](LICENSE)

## 👤 Author

Michał Hołyński - [mholynski@proton.me](mailto:mholynski@proton.me)
