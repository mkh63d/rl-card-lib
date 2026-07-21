# RL Card Lib - Card Games

**Game implementations** for the RL Card Library.

This package contains concrete implementations of card games built on top of `rl-card-lib-core`.

## Included Games

### Klondike Solitaire
Classic solitaire card game where the goal is to build sequences by suit.

- Single player turn-based game
- Episodic with clear win/loss conditions
- Continuous action space (card selection and placement)

### Macao
Traditional trick-taking game with bidding mechanics.

- Multi-player game
- Complex state space with hand management and bidding
- Suitable for testing multi-agent interactions

## Installation

```bash
# From root directory (development mode)
pip install -e ./packages/cardgames

# Or with dev dependencies
pip install -e "./packages/cardgames[dev]"
```

## Quick Start

### Train Agent on Klondike

```python
from rl_card_lib.games import KlondikeSolitaire
from rl_card_lib.core import CardGameEnv, Trainer
from rl_card_lib.agents import DQNAgent

game = KlondikeSolitaire()
env = CardGameEnv(game)
agent = DQNAgent(state_size=100, action_size=20)

trainer = Trainer(env, agent)
trainer.train(num_episodes=5000)
```

### Play Game Manually

```python
from rl_card_lib.games import KlondikeSolitaire

game = KlondikeSolitaire()
done = False
while not done:
    actions = game.get_legal_actions(0)
    # Choose action (manual or agent)
    reward, done = game.step(action)
    print(game.get_state())
```

## Dependencies

- `rl-card-lib-core>=0.1.0` - Core library
- `numpy>=1.21.0` - Numerical computing

## Optional Dependencies (dev)

- `pytest>=7.0.0` - Testing framework
- `pytest-cov>=4.0.0` - Coverage measurement
- `black>=23.0.0` - Code formatting
- `isort>=5.12.0` - Import sorting
- `flake8>=6.0.0` - Linting
- `pylint>=3.0.0` - Static analysis

## Testing

```bash
# Run card games-specific tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=rl_card_lib.games --cov-report=html
```

## Architecture

```
src/rl_card_lib/
├── games/
│   ├── __init__.py
│   ├── klondike.py      # Klondike Solitaire implementation
│   └── macao.py         # Macao game implementation
```

## Adding New Games

To add a new game:

1. Create a new file in `games/` directory
2. Subclass `CardGame` from core
3. Implement required abstract methods:
   - `deal_cards()` - Initialize game state
   - `get_legal_actions(player_id)` - Return available actions
   - `step(action)` - Execute action and return (reward, done)
   - `get_state()` - Return current game state
4. Add tests in `tests/test_games.py`
5. Export in `games/__init__.py`

## See Also

- [Core Package](../core/README.md)
- [Examples](../examples/README.md)
