# Card Games Package - TODO

## Code Structure
- [x] **Correct pathing** — imports verified by the root suite and CI
- [x] **Game base validation** — both games implement the CardGame interface;
  `tests/test_game_base.py` asserts the contract
- [ ] **State serialization** — games have `copy()`/`determinize()`; a
  portable serialization format (for replays/checkpoints) is still open

## Klondike Solitaire (implementation lives in packages/examples)
- [x] **Test game rules** — rules, rewards, terminals and the action-encoding
  invariant are covered in `tests/test_games.py` and
  `tests/test_reward_design.py`
- [x] **Test win conditions** — win, loss (dead deal) and truncation all
  tested; `max_passes` is enforced
- [x] **Test action validity** — legality asserted on every step in the agent
  suites; action space is 68 wide with no unreachable actions below that
- [ ] **Performance** — `get_legal_actions` is called twice per step (once for
  legality, once for the loss check); profile before optimizing
- [ ] **State space analysis** — document observation/state space size for the
  thesis

## Macao Game (implementation lives in packages/examples)
- [x] **Test game rules** — play, draw, penalties, declarations, terminals
- [x] **Test edge cases** — declaration-phase misuse raises; winning Ace/Jack
  skips the declaration; potential-based reward invariants tested
- [ ] **Test multi-player** — 3-4 player games run, but only 2-player is
  systematically tested
- [x] **State normalization** — observation encoding is deterministic and
  shape-checked, including the declaration-phase flags

## New Games Framework
- [ ] **Template/example** — starter template for new games
- [ ] **Validation helpers** — generic game-contract validator (the invariant
  tests in the root suite are a starting point)

## Documentation
- [x] **Reward design** — documented in the game class docstrings and
  CHANGELOG; shaped rewards are progress-only (Klondike) and potential-based
  (Macao), with a sparse mode on both
- [x] **Action space** — encodings documented next to the constants
- [ ] **Game rules** — prose rules writeup per game (docstrings cover the
  mechanics; a standalone document would serve the thesis)
- [ ] **State format** — document the observation layout field by field
