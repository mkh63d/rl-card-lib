# RL Game Lib - Core

Foundation library that provides a minimal, game-agnostic abstraction layer for building and training reinforcement-learning agents on turn-based and step-based games.

## Overview

This package defines the essential interfaces and utilities for describing games as RL environments, collecting metrics, and orchestrating training. It intentionally focuses on generic concepts — games, steps, moves/actions, observations, and win/loss/terminal status — rather than any specific domain (cards, board, etc.).

## Goals

- Provide a small, well-documented `Game` abstraction that describes: initialization, legal moves, applying moves (step), observation/state, reward, and terminal/win-status reporting.
- Offer a thin Gymnasium-compatible wrapper so games can be used directly by standard RL algorithms.
- Keep agent and trainer helpers independent of game internals so users can implement any multi-agent, single-agent, or adversarial setting.

## What's Included

- `Game` (abstract): core interface describing a game's lifecycle and API (see Quick Start).
- `Agent` (base): lightweight interface for agent implementations used by trainers and examples.
- `Trainer`: simple training loop and metrics collection helpers.
- `Env` wrapper: Gymnasium-compatible adapter to expose `Game` as an RL environment.
- `utils.encoding`: helpers to encode observations and actions for neural networks.

## Installation

```bash
# From repository root (development)
pip install -e ./packages/core

# With dev extras
pip install -e "./packages/core[dev]"
```

## Quick Start

A `Game` implements seven abstract methods and maintains a few plain
attributes. `step` returns the five-element Gymnasium tuple.

```python
from rl_card_lib.core import Game
from rl_card_lib.env import CardGameEnv
from rl_card_lib.trainer import Trainer
import numpy as np

class MyGame(Game):
    def __init__(self):
        super().__init__(num_players=1)
        self.reset()

    def reset(self) -> np.ndarray:
        # start a new game; set self.done / self.winner; return the observation
        self.done = False
        self.winner = None
        return self.get_observation()

    def step(self, action: int):
        # apply the action, update state, and return the Gymnasium 5-tuple
        # (observation, reward, terminated, truncated, info)
        ...
        return self.get_observation(), reward, self.done, False, {}

    def get_legal_actions(self) -> list[int]:
        ...

    def get_observation(self) -> np.ndarray:
        ...

    def get_action_space_size(self) -> int:
        ...

    def get_observation_shape(self) -> tuple[int, ...]:
        ...

    def is_game_over(self) -> bool:
        return self.done

# CardGameEnv adapts a Game to the Gymnasium API the agents and Trainer use.
env = CardGameEnv(MyGame(), max_steps=200)
trainer = Trainer(env, agent=my_agent)
trainer.train(episodes=100)
```

## API contract

**Abstract — you must implement these seven:**

- `reset() -> np.ndarray`
- `step(action: int) -> (obs, reward, terminated, truncated, info)`
- `get_legal_actions() -> list[int]`
- `get_observation() -> np.ndarray`
- `get_action_space_size() -> int`
- `get_observation_shape() -> tuple[int, ...]`
- `is_game_over() -> bool`

**Plain attributes you maintain:** `done`, `winner` (winning player index or
`None`), `current_player_idx`, `num_players`.

**Provided with sensible defaults — override only when needed:**

- `copy() -> Game` — deep copy by default; the search agents rely on it
- `determinize(observer_idx, rng) -> Game` — `self.copy()` by default; override
  for hidden-information games
- `get_reward(player_idx) -> float` — `0.0` by default; override for multiplayer
  so MCTS can see each player's terminal payoff
- `render()`, `action_to_string()`, `get_current_player()`, `next_player()`,
  `get_legal_action_mask()`, `get_winner()`, `log_action()`, `get_history()`

See [docs/custom_game.md](../../docs/custom_game.md) for a full walkthrough,
including how to register a game for the training sweep and the HTML report.

## Dependencies

- `numpy`
- `gymnasium` (optional adapter)
- `torch` or `jax` only required by example agents; core aims to be framework agnostic.

## Testing

```bash
pytest tests/ -q
```

## Architecture

```
src/rl_card_lib/
├── core/        # Generic game abstractions, env wrapper, trainer, metrics
├── agents/      # Example agents and base interfaces
├── trainer/     # Training orchestrator and utilities
└── utils/       # Encoding, visualization, helpers
```

## Migration status

The generic core exists: `Game` is the game-agnostic base class (documented
under "Recommended API contract" above), `GymEnvWrapper` adapts any `Game` to
a Gymnasium-style env, and the card-specific classes (`Card`, `Deck`,
`CardGame`) live in the separate `rl-card-lib-cardgames` package, with
`CardGame` a thin subclass of `Game`.

Remaining migration work is tracked in [TODO.md](TODO.md): deprecation shims
for old import paths, non-card example games proving the interface, and a
written migration guide.

## API Reference

See the top-level docs in the repository for detailed references and usage examples.

