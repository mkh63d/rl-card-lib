# Core Package - TODO (migrate to game-agnostic core)

Status 2026-07-17: the core of the migration exists and is exercised — `Game`
is the game-agnostic base (card games subclass it via `CardGame`),
`GymEnvWrapper` adapts any `Game`, and both are exported. The remaining items
are the deprecation/compat layer and extra example games, which only matter
once external users exist.

## High-level Migration
- [x] `Game` abstract interface (reset, step, legal_actions, observation and
  action space helpers, winner, copy/determinize) — `core/game.py`
- [x] `GymEnvWrapper` adapting any `Game` to a Gymnasium-style env
- [x] Generic API exported from `rl_card_lib.core` (`Game`, `GymEnvWrapper`,
  `Trainer`)
- [ ] `win_status()` helpers for common patterns (single-winner, zero-sum,
  draws) — `winner`/`get_reward()` cover today's games
- [ ] Compatibility module `extras.card` + deprecation shims aliasing
  `CardGame` -> `Game` — deferred until something outside this repo imports
  the old names

## Code Structure & API
- [x] `CardGame` is a thin subclass of `Game` (lives in cardgames, which is
  the "extras" package in practice)
- [x] Type hints and docstrings on the `Game` API
- [x] `get_legal_action_mask()` / observation-shape helpers for Gym
  compatibility

## Tests
- [x] Game-contract tests (`tests/test_game_base.py`,
  `packages/examples/tests/test_game_base.py`)
- [x] Trainer works against wrapped games (root trainer suite)
- [ ] Non-card example games (grid-world, tic-tac-toe) to prove the interface
  carries beyond cards — worthwhile for the thesis's "universal library" claim

## Documentation & Examples
- [x] `packages/core/README.md` documents the generic `Game` contract
- [ ] `docs/migration.md` step-by-step migration guide
- [ ] Example scripts for a non-card `Game`

## Packaging & Release
- [ ] Update `pyproject.toml` metadata to reflect the generic core
- [x] Changelog: covered by the root `CHANGELOG.md` until packages version
  independently
- [ ] Migration release + notes (blocked on publishing at all)

## Follow-up / Nice-to-have
- [x] CI runs the core tests (root suite, Python 3.10-3.12)
- [ ] mypy gating for the core package
- [ ] Cookiecutter-style template for new `Game` subclasses
