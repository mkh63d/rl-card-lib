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
from rl_card_lib.visualizer import render_cards, create_simple_board_view

game = KlondikeSolitaire()
game.reset()

print(render_cards(game.stock[:7]))  # render a row of cards
print(create_simple_board_view({     # labelled board of piles
    "foundations": game.foundations,
    "waste": game.waste,
}))
```

### Watch an Agent Play a Deal

`play_episode` replays one seeded deal and hands back a record of every move:
the action and its readable label, the reward, the flags the env raised, and
the board after it. `print_episode` writes that to a terminal, and
`write_episode_html` writes a self-contained page you can step through.

```python
from rl_card_lib.harness import sweep_game
from rl_card_lib.visualizer import play_episode, print_episode, write_episode_html

env = sweep_game("klondike").env_factory()
record = play_episode(env, agent, seed=0)

print_episode(record)                          # board after every move
write_episode_html(record, "episode.html")     # steppable page
print(record.outcome, record.step_count)       # 'solved', 87
```

To watch it arrive move by move instead of after the fact -- which is what you
want with a slow agent like MCTS -- pass a printer as `on_step`, or iterate:

```python
from rl_card_lib.visualizer import iter_episode, step_printer

record = play_episode(env, agent, seed=0, on_step=step_printer(delay=0.3))
print_episode(iter_episode(env, agent, seed=0))   # or stream it directly
```

Nothing here knows what game it is showing: any env following the
`CardGameEnv` contract and any agent with `select_action` replays.
`packages/examples/scripts/watch_episode.py` is the ready-made entry point.

### Plot Training Metrics

```python
from rl_card_lib.trainer import Trainer

trainer = Trainer(env, agent)
metrics = trainer.train(episodes=1000)  # returns a TrainingMetrics

metrics.plot()  # plot reward / win-rate curves
```

## Dependencies

- `rl-card-lib-core>=0.1.0` - `console_safe`, which every board print goes through
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
pytest tests/ --cov=rl_card_lib.visualizer --cov-report=html
```

## Architecture

```
src/rl_card_lib/
└── visualizer/
    ├── visualization.py    # card / tableau string helpers
    ├── episode.py          # the replay engine (duck-typed on env + agent)
    ├── replay.py           # terminal rendering of a replay
    ├── html_episode.py     # self-contained HTML export of a replay
    └── __init__.py
```

`episode.py` imports nothing from `rl_card_lib.games`, `rl_card_lib.harness` or
`rl_card_lib.report`: the `examples` package depends on this one, so any of
those would close a cycle. Working from the env/agent contract alone is also
what makes the replay generic.

## See Also

- [Core Package](../core/README.md)
- [Examples](../examples/README.md)
