# Core Package - TODO (migrate to game-agnostic core)

This TODO lists the concrete steps to generalize the `core` package from card-specific concepts to a minimal, game-agnostic core. Items are ordered roughly by dependency and recommended implementation order.

## High-level Migration
- [ ] Define `Game` abstract interface and public contract (reset, legal_actions, step, win_status, observation/action spaces)
- [ ] Implement `GymEnvWrapper` to adapt any `Game` to a Gymnasium `Env`
- [ ] Add `win_status()` helpers covering common patterns (single-winner, multi-agent zero-sum, draws)
- [ ] Create compatibility module `extras.card` (or `examples.card`) to host `Card`, `Deck`, and other card-specific helpers
- [ ] Add deprecation shims in the root API that alias `CardGame` -> `Game` with deprecation warnings

## Code Structure & API
- [ ] Audit and rename modules/classes: replace `CardGame` with `Game` or provide `CardGame` as a thin subclass in extras
- [ ] Export the generic API from `rl_card_lib.core.__init__` (`Game`, `Trainer`, `GymEnvWrapper`, `Agent`)
- [ ] Add/verify `observation_space` and `action_space` helpers for Gym compatibility
- [ ] Add type hints and public docstrings to new `Game` API

## Tests
- [ ] Add unit tests that assert the `Game` contract (reset/step/legal_actions/win_status)
- [ ] Create example game tests: simple grid-world, tic-tac-toe, and a card-based example moved into `extras.card`
- [ ] Update `Trainer` tests to work with either a `Game` or a Gym env

## Documentation & Examples
- [ ] Update `packages/core/README.md` to document the generic `Game` contract (done)
- [ ] Add `doc/migration.md` describing API differences and step-by-step migration guide for downstream users
- [ ] Provide example notebooks/scripts for: (a) grid-world, (b) tic-tac-toe, (c) original card example in `extras.card`

## Compatibility & Deprecation
- [ ] Implement runtime deprecation warnings for old APIs and provide migration hints in warnings
- [ ] Keep `extras.card` to avoid immediate breakage for downstream packages; mark it as optional in `pyproject.toml`

## Packaging & Release
- [ ] Update `pyproject.toml` metadata and description to reflect the generic core
- [ ] Create `CHANGELOG.md` documenting the migration and breaking changes
- [ ] Prepare a migration release (major/minor per semantic versioning) and release notes

## Follow-up / Nice-to-have
- [ ] Add CI job(s) to run the core tests against multiple example games
- [ ] Add Lint/typecheck (mypy) gating for the core package
- [ ] Provide a small `cookiecutter`-style template demonstrating how to implement a `Game` subclass

---

If you'd like, I can implement the `Game` interface skeleton and `GymEnvWrapper` now and update tests/examples accordingly. Which item should I start with?

