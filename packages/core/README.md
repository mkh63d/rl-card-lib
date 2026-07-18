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

The following demonstrates the minimal `Game` interface this package expects. Implementations may be synchronous turn-based games, simultaneous-move games, or episodic environments.

```python
from rl_card_lib.core import Game, Trainer, GymEnvWrapper

class MyGame(Game):
    def __init__(self, *, num_players=1):
        super().__init__(num_players=num_players)
        # initialize state

    def reset(self):
        # return initial observation
        ...

    def legal_actions(self, player_id):
        # return iterable of legal actions for the player
        ...

    def step(self, action, player_id=None):
        # apply action, update state
        # return observation, reward, done, info
        ...

    def win_status(self):
        # return per-player terminal/win/lose/draw status
        ...

# Use Gym wrapper for existing RL algorithms
game = MyGame()
env = GymEnvWrapper(game)

# Trainer accepts either a Gym-like env or a Game directly
trainer = Trainer(env, agent=None)
trainer.train(num_episodes=100)
```

## Recommended API contract

- `reset() -> observation`
- `legal_actions(player_id) -> list[action]`
- `step(action, player_id=None) -> (observation, reward, done, info)`
- `win_status() -> dict[player_id -> {terminal: bool, result: "win"|"loss"|"draw"}]`
- `observation_space` and `action_space` (optional helpers for Gym compatibility)

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

