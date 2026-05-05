# RL Card Library

A library for training reinforcement learning agents to play card games.

## Features

- Define custom card games with states, actions and reward functions
- Compatible with Gymnasium/OpenAI Gym interface
- DQN agent with experience replay
- Example games: Klondike Solitaire, Macao
- PyTorch backend

## Installation

```bash
# Clone the repository
git clone https://github.com/mkh63d/rl-card-lib.git
cd rl-card-lib

# Install in development mode
pip install -e ".[dev]"
```

## Quick Start

### Defining a Custom Game

```python
from rl_card_lib.core import CardGame, Deck

class MyGame(CardGame):
    def __init__(self):
        super().__init__()
        self.deck = Deck()
        self.deck.shuffle()
        
    def get_legal_actions(self):
        # Return list of valid actions
        pass
        
    def step(self, action):
        # Execute action, return (observation, reward, done, info)
        pass
        
    def is_game_over(self):
        # Check if game has ended
        pass
```

### Training an Agent

```python
from rl_card_lib.env import CardGameEnv
from rl_card_lib.agents import DQNAgent
from rl_card_lib.trainer import Trainer
from rl_card_lib.games import KlondikeSolitaire

# Create environment
game = KlondikeSolitaire()
env = CardGameEnv(game)

# Initialize DQN Agent
agent = DQNAgent(
    state_size=env.observation_space.shape[0],
    action_size=env.action_space.n
)

# Train
trainer = Trainer(env, agent)
trainer.train(episodes=1000)
```

## Architecture

The library follows a modular architecture:

- **Game Core**: Pure Python implementation of game rules
- **RL Wrapper**: Adapts games to Gymnasium-compatible interface
- **Agent Module**: Neural network policies (DQN)
- **Trainer**: Manages training loop and hyperparameters

## Example Games

### Klondike Solitaire
Single-player patience card game. The agent learns to move cards between tableaux and foundations.

### Macao
Multiplayer card game (variant of Crazy Eights). Agents learn optimal card play strategies.

## Project Structure

```
src/rl_card_lib/
├── core/           # Base classes (Card, Deck, Game)
├── env/            # Gymnasium environment wrapper
├── agents/         # RL agents (DQN, Random)
├── games/          # Example game implementations
├── trainer/        # Training utilities
└── utils/          # Helper functions
```

## Requirements

- Python >= 3.9
- PyTorch >= 2.0
- Gymnasium >= 0.29
- NumPy >= 1.21

## License

MIT License

## Author

Michał Hołyński - Engineering Thesis, Lodz University of Technology

Supervisor: dr inż. Arkadiusz Tomczyk
