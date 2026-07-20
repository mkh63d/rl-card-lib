# RL Card Library - TODO

## Package Migration

### Structure Validation
- [x] **Correct pathing** across all packages — exercised by
  `tests/test_public_api_imports.py` and the cross-package suites; CI runs them
  on every push.
  - [x] Core imports work correctly
  - [x] Cardgames imports core properly
  - [x] Visualizer imports cardgames properly (it needs `Card`, not core)
  - [x] Examples import cardgames and core
  - [x] No circular dependencies
  - [x] All namespace references consistent

### Package Dependencies
- [x] `pyproject.toml` dependencies audited per package (2026-07-17):
  - Core: `numpy`, `torch`, `gymnasium`, `tqdm`
  - Cardgames: `rl-card-lib-core`, `numpy`
  - Visualizer: `rl-card-lib-cardgames`, `matplotlib` — the original plan said
    core, but the code imports `rl_card_lib.cardgames.card`, so cardgames is
    the true dependency
  - Examples: core + cardgames + visualizer
  - Report: core
- [ ] Update root `pyproject.toml` with all package references once the
  packages are published; until then the root package vendors everything
- [x] Dependency resolution verified locally (editable installs coexist)

## Testing Infrastructure

- [x] Root integration suite (`tests/`, 330+ tests) covering cross-package
  imports, API contracts, training workflows and multi-agent scenarios
- [x] Examples package tests (`packages/examples/tests/`)
- [ ] Per-package unit test directories for core/cardgames/visualizer — the
  root suite covers the code today; split it when the packages publish
  separately
- [ ] Coverage thresholds — measure first, then pick a number that means
  something. `[tool.coverage]` config exists in the root pyproject.

## Correctness (audited 2026-07-17, resolved 2026-07-17)

All four items from the benchmarking audit are fixed, with regression tests in
`tests/test_reward_design.py`. Kept here because the details matter for the
thesis text.

- [x] **Klondike reward is anti-correlated with winning.** Root cause: a
  non-revealing tableau move paid `0.05 * cards_moved` against a `-0.01` step
  cost and was reversible, so shuffling two piles was unbounded free reward.
  Non-revealing tableau moves now pay nothing (net `-0.01`); reveals and
  foundation moves still pay. Any pre-fix Klondike reward curve measures
  loop-farming, not skill — do not compare across the fix.
- [x] **Macao's reward mixed two perspectives.** The terminal reward was
  written from player 0's seat (`+10 if current_player.is_agent else -5`), so
  player 1 winning was recorded as -5 *for player 1* and search agents
  concluded the opponent was trying to lose. Rewards are now actor-relative
  (the winner is always the actor on the winning play); losers' payoffs are
  queryable via `get_reward(player_idx)`, which MCTSAgent feeds into its
  search. Shaped per-move rewards are potential-based on hand size, so no
  move loop can mint reward (the previous flat per-play bonuses made
  hoarding-then-harvesting profitable — measured, MCTS stopped finishing
  games until this was fixed).
- [x] **Shaped vs sparse decided: both, explicitly.** Shaped stays the default
  (now loop-free by construction: Klondike pays only progress, Macao is
  potential-based); `reward_mode="sparse"` (+1 win / -1 loss, nothing else) is
  available on both games for the honest-but-slower baseline. The thesis can
  now compare them as an experiment instead of a leap of faith.
- [x] **`SelfPlayTrainer.opponent_update_interval` did nothing.** Implemented:
  self-play trains against a frozen deep-copied snapshot, refreshed every N
  episodes; `None` restores the zero-lag mirror. Thesis text describing
  periodic opponent updates is accurate again.

While fixing the above, two live defects surfaced in `MCTSAgent` (both fixed,
see `_backpropagate` and `_edge_reward`):

- [x] **Node values excluded the edge reward**, so UCT selection and the root
  choice were blind to immediate rewards — an instantly winning move carried
  Q = 0. This, not the reward bugs alone, is why MCTS played at random
  strength on both games.
- [x] **Losses were invisible to the search**: step() can only pay the acting
  player, so "opponent wins" cost nothing. Terminal edges now credit every
  non-actor's `get_reward()`. After both fixes: Macao vs random went from ~3%
  to ~87% wins at 60 simulations, and strength scales with the budget
  (`test_more_simulations_beat_fewer`, formerly an xfail).

## Action space / environment modelling (resolved 2026-07-17)

- [x] **Macao's Ace/Jack declarations are agent actions** (54-57 suits, 58-64
  ranks, `MAX_ACTIONS = 65`, all reachable). Playing an Ace/Jack with cards in
  hand enters a declaration phase; the same player's next action names the
  request. The observation gained two phase flags. The old hardcoded
  most-common-in-hand rule survives only as `MacaoHeuristicAgent`'s scoring
  for these actions.
- [x] **Klondike "cannot express which card to move" — investigated, no
  ambiguity exists.** The face-up section of a tableau pile is always one
  descending alternating-color run, so for a given destination at most one
  card can legally move (the destination's top card forces rank and color;
  empty piles accept only the run's single king). The pile-pair encoding is
  therefore lossless. Pinned by `test_tableau_move_is_unambiguous`, which
  asserts the invariant over hundreds of random positions.
- [x] **Klondike's action space shrunk from 200 to 68** (highest encodable
  action is 67). The DQN output layer lost its 132 permanently-dead outputs.
  Old checkpoints are incompatible. Macao's 60 → 65, with every action used.

## Library Features

- [x] **Klondike reward loop** — see Correctness above.
- [x] **Deadend liability** — two layers: the loop no longer pays (Klondike),
  and `CardGameEnv` now detects repeated positions generically, flagging them
  in `info["repeated_position"]` and optionally pricing them via
  `repeated_position_penalty`. Works for any future game with reversible moves.
- [x] **Solvability check** — `solve_klondike(game, max_nodes)`: budgeted
  perfect-information DFS with transposition pruning; True / False / None
  (budget exhausted). Lets win rates be reported over solvable deals, which is
  the number comparable to the literature (~80% of deals are winnable with
  perfect play).
- [x] **Full analysis or heuristic?** As built: `KlondikeHeuristicAgent` /
  `MacaoHeuristicAgent` are pure fixed-priority heuristics (adjustable in one
  `score_action()` each), `GreedyLookaheadAgent` is exhaustive to `depth`,
  `MCTSAgent` is sampled search tunable by `simulations`. Documented in each
  class docstring.

## Reporting

- [x] **Training runs are recorded, not just printed.** `RunRecord` / `RunStore`
  in `rl-card-lib-report` persist timestamps, hyperparameters, per-episode
  series (including cards-to-foundation, exploration and Q-table growth) and the
  before/after baseline comparison. Previously the richest numbers — the ones
  `evaluate_klondike` / `evaluate_macao` compute — were printed and discarded,
  and nothing recorded when a run happened.
- [x] **Visual HTML report** — `run_sweep.py` writes `results/index.html`: one
  self-contained page, overview table newest-first, comparison charts per game,
  a detailed section per model, everything exportable to CSV/PNG/SVG.
- [x] **Only the last run of each model is kept.** Structural: a run is keyed
  `{game}__{agent}`, which is also its directory name, and both the run
  directory and the checkpoint directory are purged *before* training so a crash
  leaves an empty directory rather than a mixture of two runs.
- [x] **Metric ranges are stated.** Summary, evaluation and baseline tables mix
  0-1 rates, unbounded shaped rewards and counts; each column and row now
  carries its scale (`0-100%`, `0-52 cards`, `0-300 steps`, `unbounded`) from a
  `METRICS` registry, and rates render as percentages.
- [x] **Figures open full-screen.** Clicking a chart opens it in a modal over a
  dimmed page; clicking anywhere or pressing Escape closes it.
- [x] **Custom games can be reported alone.** `--games` / `--exclude-games` /
  `--exclude-builtin-games` on both `run_sweep.py` (as `--report-*`) and the
  report CLI, with no default — the report covers the whole store unless told
  otherwise. `register_game(...)` lets a custom game declare its headline
  metric, label and bound instead of falling back to the neutral win-rate spec.

### Custom games: what still assumes Klondike or Macao

The reporting layer is generic; the *sweep* is not. Someone using this library
for their own game can already record runs and get a report — `RunRecord`,
`RunStore`, `HtmlReport` and `game_spec()` are game-agnostic, and unknown games
fall back to a neutral spec — but they have to drive training themselves.

- [ ] **`run_sweep.py` hardcodes the two bundled games.** `GAMES`, `MAX_STEPS`,
  `build_env()` and `evaluate()` are literals; a third game cannot be swept
  without editing the script. Wants a small registry — game name to env factory,
  step cap, trainer class and evaluation protocol — that `register_game()` feeds.
- [ ] **Evaluation protocols are game-specific functions.** `evaluate_klondike`
  and `evaluate_macao_suite` are hand-written; a custom game needs its own and
  no interface declares what one must return beyond "a dict of floats".
- [ ] **Baselines are per-game literals.** `klondike_baseline_agents()` /
  `macao_baseline_agents()` name their agents by hand. `RandomAgent` and
  `GreedyLookaheadAgent` are generic and could be derived from any `Game`.
- [ ] **`METRICS` is a module-level dict.** A custom metric renders with a
  neutral "unbounded" range until it is added; there is no `register_metric()`.
- [ ] **No worked example.** A `docs/custom_game.md` walking one third game from
  `Game` subclass to report would prove the seams are actually usable.

- [ ] **`Agent.checkpoint_suffix`.** `Trainer._save_checkpoint` hardcodes `.pt`
  even for `QLearningAgent`, which pickles — so `checkpoint_ep400.pt` in a
  tabular run is a pickle that `torch.load` cannot open. `purge_checkpoints`
  globs both extensions defensively; the real fix is a class attribute the
  trainer consults, kept out of the reporting change to keep its diff reviewable.
- [ ] **Dark-mode figures.** The report is light-only on purpose (matplotlib
  PNGs are baked at render time, and it is printed as an appendix). Supporting
  both would mean rendering every figure twice.

## Experiments to run now that the rewards are trustworthy

The code-level blockers are gone; these are measurement work, not fixes.

- [ ] **Re-run DQN vs Double DQN at 5k+ episodes.** The 400-episode comparison
  (85% vs 40%) predates the epsilon-decay fix and the action-space shrink, so
  it is doubly stale. Do not report it.
- [ ] **Re-check the heuristic rollout policy for MCTS on Klondike.** It
  measured *worse* than random rollouts (2.9 vs 10.2 cards up) when the
  objective was the farmable loop; with the loop gone and the backprop fix in,
  the comparison starts from scratch.
- [ ] **Validate the MCTS exploration weight (1.4) on the fixed rewards.** The
  old sweep was measuring loop-farming. Note the win-reward magnitude
  stretches `_MinMaxStats` normalization, which compresses small Q
  differences; if exploitation looks weak, that interaction is the first
  suspect.
- [ ] **Shaped vs sparse learning-speed comparison** — now a one-flag
  experiment (`reward_mode`), and a natural thesis section.
- [ ] Consider prioritized replay and n-step returns for the DQN family; both
  slot into `MaskedReplayBuffer` and are the standard next lever after
  double + dueling.

## Accepted trade-offs (decided, not forgotten)

- **MCTS rebuilds its tree every move.** Deliberate: each move re-samples the
  hidden cards, and a subtree grown under one sampled world is not valid
  evidence about the next. The cost is real (long Klondike deals pay the full
  budget per move); treat MCTS as a low-episode-count baseline there, or set
  `use_determinization=False` when an upper bound is wanted.
- **Klondike keeps `max_passes=None` by default.** Unlimited passes match
  casual play and keep every historical number comparable; the cost is that a
  dead deal can only end by truncation. Training scripts that want real loss
  terminals should pass a finite `max_passes` (the limit is enforced now).
- **`DQNAgent(seed=...)` still seeds torch globally.** Torch has no clean
  per-instance init seeding; documented rather than half-fixed. The
  library's own components no longer touch global RNGs.

## Publishing / DevOps

- [x] CI: GitHub Actions, Python 3.10-3.12, flake8 + both test suites
- [x] CHANGELOG created (root `CHANGELOG.md`)
- [ ] Version bump + PyPI publishing — blocked on deciding whether the
  packages publish separately at all before the thesis deadline
- [ ] Repo-wide black/isort normalization — deliberately kept out of the
  correctness PR to keep its diff reviewable; run as its own commit
- [ ] mypy in CI — the codebase predates strict typing; adopt per-package,
  core first
- [ ] CONTRIBUTING / ARCHITECTURE docs

## Documentation

- [x] Reward structures documented in the game class docstrings
- [x] Action encodings documented next to the constants they describe
- [ ] `docs/migration.md` for the core `Game` generalization (see
  `packages/core/TODO.md`)
