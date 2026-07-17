# RL Card Library - TODO

## Package Migration

### Structure Validation
- [ ] **Correct pathing** across all packages
  - [ ] Core imports work correctly
  - [ ] Cardgames imports core properly
  - [ ] Visualizer imports core properly
  - [ ] Examples import cardgames and core
  - [ ] No circular dependencies
  - [ ] All namespace references consistent

### Package Dependencies
- [ ] Update `pyproject.toml` for each package with correct dependencies
  - [ ] Core: only `numpy`, `torch`, `gymnasium`, `tqdm`
  - [ ] Cardgames: `rl-card-lib-core`, `numpy`
  - [ ] Visualizer: `rl-card-lib-core`, `matplotlib`
  - [ ] Examples: `rl-card-lib-core`, `rl-card-lib-cardgames`, `rl-card-lib-visualizer`
- [ ] Update root `pyproject.toml` with all package references
- [ ] Verify dependency resolution (no version conflicts)

## Testing Infrastructure

### Unit Tests (Per-Package)
- [ ] Core package tests in `packages/core/tests/`
- [ ] Cardgames package tests in `packages/cardgames/tests/`
- [ ] Visualizer package tests in `packages/visualizer/tests/`
- [ ] Examples validation tests in `packages/examples/tests/`

### Integration Tests (Root)
- [ ] Cross-package imports work correctly
- [ ] Public API contracts verified
- [ ] Full training workflows function
- [ ] Multi-agent scenarios work

### Coverage
- [ ] Coverage measured per-package
- [ ] Coverage measured root-level
- [ ] Coverage reports generated (not exported)
- [ ] Minimum coverage threshold defined

## Package Development

### Core Package
- [ ] Complete unit tests
- [ ] Type hints throughout
- [ ] Docstrings for all public APIs
- [ ] API stability review
- [ ] CHANGELOG created

### Cardgames Package
- [ ] Complete game implementations
- [ ] Game-specific tests
- [ ] Game rules validation
- [ ] Performance optimization
- [ ] State space documentation

### Visualizer Package
- [ ] Visualization implementations
- [ ] Rendering backend support
- [ ] Metrics plotting
- [ ] Live dashboard (optional)
- [ ] Export capabilities

### Examples Package
- [ ] All examples functional
- [ ] Example tests pass
- [ ] Documentation complete
- [ ] Expected outputs documented

## Publishing

### Pre-Release
- [ ] Version bumped in all `pyproject.toml` files
- [ ] CHANGELOG updated
- [ ] Documentation reviewed
- [ ] Tests passing (100% coverage)
- [ ] Code quality checks pass

### Release
- [ ] Build artifacts generated
- [ ] PyPI credentials configured
- [ ] Packages published to PyPI
- [ ] GitHub releases created
- [ ] Documentation deployed

## Documentation

### API Documentation
- [ ] Docstrings for all public APIs
- [ ] Type hints in all signatures
- [ ] Usage examples in docstrings

### Repository Documentation
- [ ] CONTRIBUTING guidelines
- [ ] ARCHITECTURE documentation
- [ ] DEVELOPMENT setup guide

## DevOps

### CI/CD Pipeline
- [ ] GitHub Actions workflow
- [ ] Test matrix (Python 3.9, 3.10, 3.11)
- [ ] Coverage reports
- [ ] Linting checks
- [ ] Type checking

### Code Quality
- [ ] Black formatting applied
- [ ] isort import sorting applied
- [ ] Flake8 linting passes
- [ ] Pylint checks pass
- [ ] MyPy type checking passes

## Maintenance

- [ ] Version synchronization strategy defined
- [ ] Dependency update strategy
- [ ] Breaking change policy
- [ ] Support policy defined

## Correctness (found while benchmarking the agents, 2026-07-17)

Ordered by how much they distort results. The first two make current numbers on
their game meaningless; reproduce any of them with
`python packages/examples/scripts/benchmark_agents.py --episodes 20`.

- [ ] **Klondike reward is anti-correlated with winning.** See "reward loop"
  below. Benchmark receipts (20 deals, cards moved to foundation out of 52):
  | agent | reward | cards up | wins |
  |---|---|---|---|
  | Random | 45.08 | 11.3 | 0% |
  | Heuristic (ignores reward) | 55.28 | **30.7** | **50%** |
  | GreedyLookahead (maximizes reward) | 32.33 | **2.1** | 0% |
  The agent that optimizes reward is *5x worse at solitaire than random*, and
  scores less reward than random too, because it locks onto the shuffle loop on
  move one while random play stumbles into real progress. Until this is decided,
  Klondike reward curves measure loop-farming, not skill.

- [ ] **Macao's reward mixes two incompatible perspectives.** `step()` pays the
  *acting* player for playing a card (+0.1/+0.2/+0.3), but the terminal reward is
  written from player 0's seat: `10.0 if current_player.is_agent else -5.0`, and
  `is_agent` is only ever true for player 0 (`macao.py` reset()). So *player 1
  winning is recorded as -5 for player 1*.
  - Breaks any agent that reasons about opponents: `MCTSAgent` wins 5% vs random
    on Macao (identical to random's own 5%) while `GreedyLookahead(1)` wins 80%,
    because MCTS credits the -5 to the winner and concludes its opponent is
    trying to lose. Adding determinizations makes it worse (0%), as expected when
    the search is faithfully propagating an inverted opponent model.
  - Fix one of: make terminal reward actor-relative (`+10` to the winner,
    `-5` to each loser), or declare the reward strictly player-0-centric and have
    `MCTSAgent` minimize instead of maximize at opponent nodes (negamax). The
    first is less surprising; the second matches the truncation reward, which is
    already player-0-centric.
  - `MCTSAgent`'s per-player return tracking assumes the first convention. It is
    correct for Klondike (single player) and for any actor-relative reward.

- [ ] **Decide whether shaped per-step rewards should exist at all.** Both bugs
  above are the same root cause: hand-tuned per-move bonuses that do not compose
  into "win the game". A sparse terminal reward (+1 win / -1 loss / 0 draw) plus
  the existing action masking cannot be farmed and needs no per-game tuning. It
  learns slower, which is the real trade-off worth writing up.

## Library Features

- [ ] **Klondike reward loop (confirmed, decide before trusting any Klondike result)**
  - `_move_tableau_to_tableau` pays `0.05 * cards_moved` (plus 0.2 for a reveal)
    while `step()` charges only -0.01, so a *non-revealing* 1-card move nets
    +0.04 — and it is reversible. Moving a card back and forth is unbounded free
    reward, which is the "deadend liability" below, confirmed and localised.
  - Measured with `GreedyLookaheadAgent`: ~139 of 150 moves are tableau shuffles
    and only ~2.3 cards reach the foundations. `KlondikeHeuristicAgent`, which
    ignores the reward and follows solitaire strategy, gets ~28.4 cards up and
    wins ~43%. Any agent that optimizes this reward well *converges to farming
    the loop*, so shaped reward is currently anti-correlated with winning and
    reward curves on Klondike do not measure solitaire skill.
  - Options: make non-revealing tableau moves cost more than they pay, reward
    reveals/foundations only, or penalize revisiting a position. This changes
    every existing Klondike number, so it is a deliberate call, not a drive-by fix.
- [ ] Deadend liability - if there is like three repeating steps, it is going to go throught them over and over again.
  - Confirmed and localised: see the reward loop item above. Worth handling
    generally too (detect a repeated position hash within an episode and either
    penalize it or drop the repeated action from `get_legal_actions()`), since
    the same trap can appear in any future game with reversible moves.
- [ ] Solvibility check - check full game setup if it is even solvable
  - Now cheap to build: `KlondikeSolitaire.copy()` exists, so a solver can search
    a deal without disturbing the live game. Would let win rate be reported over
    *solvable* deals only, which is the number worth comparing against literature
    (~80% of Klondike deals are winnable with perfect play; ~43% for the current
    heuristic is not comparable to that until the denominators match).
- [ ] Is it full analysis or heuristic? Can I adjust it? What is it dependent on?
  - As built: `KlondikeHeuristicAgent`/`MacaoHeuristicAgent` are pure heuristics
    (fixed priority scores, no search, no learning). `GreedyLookaheadAgent` is
    exhaustive but only to `depth` plies. `MCTSAgent` is sampled search, tunable
    by `simulations`. All the scores live in one `score_action()` per agent so
    they are adjustable in one place.

## Agent work (follow-ups from the first training runs)

- [ ] **Give Double DQN a fair trial.** It underperformed vanilla DQN at 400
  episodes (40% vs 85% vs random) but was still climbing. Its target moves more
  conservatively, so this is likely just a shorter horizon than it needs, not a
  defect. Re-run both at 5k+ episodes before drawing any conclusion for the
  thesis. Do not report the 400-episode number as a DQN-vs-DoubleDQN result.
- [ ] **MCTS is too slow to benchmark honestly on Klondike.** 20 sims/move over
  20 deals takes ~4 minutes; a 300-move deal pays the full budget every move.
  Either cache the tree between moves (reuse the subtree under the chosen action)
  or accept it as a low-episode-count baseline only.
- [ ] **A better rollout policy made MCTS worse on Klondike** (2.9 cards up with
  the heuristic rollout vs 10.2 with random). Consistent with the reward loop:
  sharper value estimates on a broken objective just farm the loop more reliably.
  Recheck once the reward is fixed; it should reverse.
- [ ] Tune the MCTS exploration weight once the rewards are trustworthy. The
  current 1.4 is the textbook value for rewards in [0,1]; `_MinMaxStats`
  rescales into that range, but the constant was never validated on a sane
  objective (the sweep that would have validated it was measuring loop-farming).
- [ ] Consider prioritized experience replay for the DQN family, and n-step
  returns. Both are cheap additions to `MaskedReplayBuffer` and are the standard
  next lever after double + dueling.

