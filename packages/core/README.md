# RL Card Lib - Core

**Foundation library** for building card games with reinforcement learning agent support.

## Overview

This is the core package of the RL Card Library. It provides the essential building blocks for defining card games, training agents, and managing game state. All other packages depend on this core.

## What's Included

### Game Foundation
- **`Card`** - Represents individual cards with suits and ranks
- **`Deck`** - Card deck management and shuffling
- **`Player`** - Player state and hand management
- **`CardGame`** - Abstract base class for defining custom card games

### Agent Framework
- **`Agent`** (base class) - Interface for implementing RL agents
- **`Trainer`** - Training loop orchestration
- **`Metrics`** - Training metrics collection and monitoring

### Utilities
- **`Encoding`** - State encoding utilities for agents
- **`CardGameEnv`** - Gymnasium-compatible environment wrapper

## Installation

```bash
# From root directory (development mode)
pip install -e ./packages/core

# Or with dev dependencies
pip install -e "./packages/core[dev]"
```

## Quick Start

### Define a Custom Game

```python
from rl_card_lib.core import CardGame, Player, Deck, Card

class MyGame(CardGame):
    def __init__(self, num_players=2):
        super().__init__(num_players)
        self.deck = Deck()
    
    def deal_cards(self):
        # Your game logic here
        pass
    
    def get_legal_actions(self, player_id):
        # Return valid actions for player
        pass
    
    def step(self, action):
        # Execute action and return (reward, done)
        pass
```

### Train an Agent

```python
from rl_card_lib.core import Trainer, CardGameEnv
from rl_card_lib.agents import DQNAgent

game = MyGame()
env = CardGameEnv(game)
agent = DQNAgent(state_size=100, action_size=20)

trainer = Trainer(env, agent)
trainer.train(num_episodes=1000)
```

## Dependencies

- `numpy>=1.21.0` - Numerical computing
- `torch>=2.0.0` - Deep learning
- `gymnasium>=0.29.0` - RL environment API
- `tqdm>=4.64.0` - Progress bars

## Optional Dependencies (dev)

- `pytest>=7.0.0` - Testing framework
- `pytest-cov>=4.0.0` - Coverage measurement
- `mypy>=1.0.0` - Type checking
- `black>=23.0.0` - Code formatting
- `isort>=5.12.0` - Import sorting
- `flake8>=6.0.0` - Linting
- `pylint>=3.0.0` - Static analysis

## Testing

```bash
# Run core-specific tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=rl_card_lib --cov-report=html
```

## Architecture

The core package is organized as follows:

```
src/rl_card_lib/
├── core/           # Game foundation classes
├── agents/         # Agent implementations and base classes
├── trainer/        # Training orchestration
└── utils/          # Shared utilities (encoding, environment wrapper)
```

## API Reference

See the [documentation](../../README.md) for detailed API reference and examples.
