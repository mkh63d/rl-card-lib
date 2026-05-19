# RL Card Lib - Examples

**Example scripts and notebooks** demonstrating RL Card Library usage.

This package contains practical examples and demonstrations of training agents on different card games.

## Included Examples

### Quick Demo (`quick_demo.py`)
Basic demonstration of:
- Creating cards and decks
- Initializing a game
- Running a short training loop
- Testing with a trained agent

### Klondike Training (`train_klondike.py`)
Complete workflow for training a DQN agent on Klondike Solitaire:
- Environment setup
- Agent configuration
- Training with monitoring
- Performance evaluation

### Macao Training (`train_macao.py`)
Multi-player game training example:
- Multi-agent setup
- Training with different agent types
- Performance comparison

## Installation

```bash
# From root directory (development mode)
pip install -e ./packages/examples

# Or with dev dependencies
pip install -e "./packages/examples[dev]"
```

This will install:
- `rl-card-lib-core` (required)
- `rl-card-lib-cardgames` (required)
- `rl-card-lib-visualizer` (optional but recommended)

## Running Examples

### Quick Demo
```bash
cd packages/examples
python scripts/quick_demo.py
```

### Train on Klondike
```bash
python scripts/train_klondike.py --episodes 1000 --batch-size 32
```

### Train on Macao
```bash
python scripts/train_macao.py --episodes 2000
```

## Dependencies

- `rl-card-lib-core>=0.1.0` - Core library
- `rl-card-lib-cardgames>=0.1.0` - Card games
- `rl-card-lib-visualizer>=0.1.0` (optional) - Visualization tools
- `numpy>=1.21.0`
- `torch>=2.0.0`

## Optional Dependencies (dev)

- `pytest>=7.0.0` - Testing framework
- `pytest-cov>=4.0.0` - Coverage measurement
- `black>=23.0.0` - Code formatting
- `isort>=5.12.0` - Import sorting
- `flake8>=6.0.0` - Linting
- `pylint>=3.0.0` - Static analysis

## Testing

```bash
# Run example validation tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=scripts --cov-report=html
```

## Creating New Examples

When adding a new example:

1. Create script in `scripts/` directory
2. Add necessary docstring and comments
3. Support command-line arguments for configuration
4. Add test in `tests/` validating example runs
5. Update this README with description

## See Also

- [Core Package](../core/README.md)
- [Card Games Package](../cardgames/README.md)
- [Visualizer Package](../visualizer/README.md)
