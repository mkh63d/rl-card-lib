# RL Card Library

A modular library for training reinforcement learning agents to play card games.

## 📦 Package Structure

This is a monorepo containing multiple packages:

### Core Packages

- **[`rl-card-lib-core`](packages/core/)** - Foundation library
  - Card, Deck, Player, CardGame base classes
  - Agent framework and Trainer
  - Gymnasium-compatible environment wrapper
  - **Status**: Core package - all others depend on this

### Extension Packages

- **[`rl-card-lib-cardgames`](packages/cardgames/)** - Game implementations
  - Klondike Solitaire
  - Macao (multi-player)
  - Framework for adding new games
  - **Depends on**: `rl-card-lib-core`

- **[`rl-card-lib-visualizer`](packages/visualizer/)** - Visualization utilities
  - Game state rendering
  - Training metrics plotting
  - Real-time progress monitoring
  - **Depends on**: `rl-card-lib-core`

- **[`rl-card-lib-examples`](packages/examples/)** - Examples and demonstrations
  - Quick start demo
  - Training scripts
  - Integration examples
  - **Depends on**: `rl-card-lib-core`, `rl-card-lib-cardgames`

## 🚀 Quick Start

### Installation (Development)

```bash
# Install all packages in development mode
pip install -e .  # Installs root metapackage

# Or install individual packages
pip install -e ./packages/core
pip install -e ./packages/cardgames
pip install -e ./packages/visualizer
pip install -e ./packages/examples
```

### Quick Demo

```bash
python examples/quick_demo.py
```

### Run Tests

```bash
# All tests (unit + integration)
pytest

# Only integration tests (root)
pytest tests/

# Only core package tests
pytest packages/core/tests/

# With coverage report
pytest --cov --cov-report=html
```

## 📋 Features

- ✅ Define custom card games with Gymnasium compatibility
- ✅ DQN agent with experience replay
- ✅ Built-in games: Klondike, Macao
- ✅ Comprehensive training framework
- ✅ Visualization and metrics monitoring
- ✅ Modular design - use only what you need

## 📚 Documentation

- [Core Package](packages/core/README.md) - API and concepts
- [Card Games Package](packages/cardgames/README.md) - Game implementations
- [Visualizer Package](packages/visualizer/README.md) - Visualization tools
- [Examples Package](packages/examples/README.md) - Usage examples

## 🧪 Testing Strategy

### Package-Level Tests

Each package contains tests for its own functionality:

```
packages/core/tests/           # Core package tests
packages/cardgames/tests/      # Game implementations tests
packages/visualizer/tests/     # Visualization tests
packages/examples/tests/       # Example validation tests
```

### Integration Tests

Root-level tests verify cross-package integration:

```
tests/                         # Integration tests
tests/test_integration.py      # Cross-package workflows
tests/test_public_api.py       # Public API contracts
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
mypy src/

# Linting
flake8 src/
pylint src/
```

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
