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
- [ ] Solvibility check - check full game setup if it is even solvable
- [ ] Is it full analysis or heuristic? Can I adjust it? What is it dependent on?

