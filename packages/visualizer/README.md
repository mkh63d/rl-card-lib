# RL Card Lib - Visualizer

**Visualization utilities** for RL Card Library training and gameplay.

This package provides tools for visualizing game state, training progress, and agent behavior.

## Included Components

### Game State Visualization
- Render card games to visual format
- Support for multiple rendering backends (matplotlib, PIL, etc.)

### Training Visualization
- Plot training metrics (rewards, win rates, etc.)
- Real-time training progress monitoring
- Comparison of multiple training runs

## Installation

```bash
# From root directory (development mode)
pip install -e ./packages/visualizer

# Or with dev dependencies
pip install -e "./packages/visualizer[dev]"
```

## Quick Start

### Visualize Game State

```python
from rl_card_lib.games import KlondikeSolitaire
from rl_card_lib.visualizer.visualization import render_game_state

game = KlondikeSolitaire()
# Play some moves...

render_game_state(game)  # Display game state
```

### Plot Training Metrics

```python
from rl_card_lib.trainer import Trainer
from rl_card_lib.visualizer.visualization import plot_metrics

trainer = Trainer(env, agent)
metrics = trainer.train(num_episodes=1000)

plot_metrics(metrics)  # Visualize training progress
```

## Dependencies

- `rl-card-lib-cardgames>=0.1.0` - Cardgames extension
- `matplotlib>=3.5.0` - Plotting library

## Optional Dependencies (dev)

- `pytest>=7.0.0` - Testing framework
- `pytest-cov>=4.0.0` - Coverage measurement
- `black>=23.0.0` - Code formatting
- `isort>=5.12.0` - Import sorting
- `flake8>=6.0.0` - Linting
- `pylint>=3.0.0` - Static analysis
- `pillow>=9.0.0` - Image processing (optional)

## Testing

```bash
# Run visualization-specific tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=rl_card_lib.utils.visualization --cov-report=html
```

## Architecture

```
src/rl_card_lib/
└── visualizer/
    ├── visualization.py    # Main visualization module
    └── __init__.py
```

## See Also

- [Core Package](../core/README.md)
- [Examples](../examples/README.md)
