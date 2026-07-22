# Changelog

## Unreleased

### Added

- **MCTS simulation-budget sweep.**
  `python packages/examples/scripts/sweep_mcts_budget.py` runs MCTS on Macao
  across a range of simulation-per-move budgets and records win rate vs. a
  random opponent at each, writing one `simulations,win_rate` series to
  `results/mcts_budget_sweep/macao_mcts_budget_sweep.csv` and rendering a
  single-line figure (PNG for Word/print, SVG for LaTeX). Where `run_sweep.py`
  treats MCTS as one fixed-budget baseline, this isolates how strength scales
  with the search budget, measuring every point through the same
  `run_macao_baselines` path so the numbers are comparable to the
  agent-comparison run. Defaults to plain MCTS (`--determinizations 1`); the
  x4det variant, a labelled reference line (`--annotate-buggy-backup`) and the
  PNG resolution (`--dpi`) are all options. Only budgets actually run are
  written, and the CSV is flushed point-by-point so an interrupted sweep keeps
  its completed points.
- **Solve-time benchmark over a solvable-deal pool.**
  `python packages/examples/scripts/benchmark_solve_time.py` curates a pool of
  deals a perfect-information solver proves winnable, then plays every agent —
  the non-learning baselines *and* the trained learners loaded from disk — over
  that same pool, recording **solve rate**, **moves to solve** and **wall-clock
  time to solve** (the last two averaged over solved deals only, so a
  faster-looking agent is genuinely faster, not just quicker to give up). This
  is what win rate cannot say: over deals that *are* winnable, how many does the
  agent solve and how long does it take. Results persist to
  `results/solve_benchmark/<game>.json` and render as a "Solve-time benchmark"
  section in the HTML report. It is generic over any single-player game: a game
  opts in by declaring `single_player=True` and a `solver` in its
  `register_sweep_game(...)` call (Klondike does; adversarial Macao has no solve
  oracle and is skipped). New `load_trained_learner()` reconstructs a learner
  from its recorded architecture and loads its checkpoint, skipping any learner
  not yet trained.
- **Custom games are fully supported, end to end.** A user can add their own
  game and get the full training sweep and HTML report without editing library
  code. `register_sweep_game()` (harness) declares how to run a game — env
  factory, step cap, trainer, evaluation protocol, baselines, per-episode
  series — and forwards presentation to `report.register_game()`. Klondike and
  Macao register themselves through this same API and are the worked examples;
  no game-name branch survives in the sweep or report. `Game.copy()` now
  deep-copies by default, so the search agents work for a naive custom game.
  `register_metric()`, palette-cycled colours for custom agents, and
  `higher_is_better=False` headlines round it out. See
  [docs/custom_game.md](https://github.com/mkh63d/rl-card-lib/blob/main/docs/custom_game.md).
- **The report stores a custom game's own metrics.** `RunRecord` previously
  dropped any per-episode series outside a fixed four, and a custom
  `headline_max` never reached the page — both fixed.


- **A visual HTML training report.** `python packages/examples/scripts/run_sweep.py`
  trains every learner on both games and writes `results/index.html`: one
  self-contained page (no CDN, no sibling files, figures embedded as data URIs)
  with an overview table sorted newest-run-first, comparison charts per game,
  and a detailed section per model. Every table exports to CSV/PNG, every figure
  to PNG/SVG, and print rules make it a clean thesis appendix.
- **`RunRecord` / `RunStore`** in `rl-card-lib-report`. `TrainingMetrics` records
  four per-episode arrays — enough to plot a curve, not enough to explain one.
  A record adds timestamps (nothing recorded one before, so "most recent first"
  was not expressible), hyperparameters, the before/after baseline comparison,
  and per-episode cards-to-foundation, exploration and Q-table growth. Stored as
  `run.json` beside the unchanged `metrics.json`. A run is keyed
  `{game}__{agent}`, so re-running a model replaces it rather than accumulating.
  `RunRecord.from_metrics_json()` imports runs recorded before this existed.
- **`rl_card_lib.harness`** — `build_learner`, the evaluation protocols and the
  baseline agent sets, previously defined only inside `scripts/` and therefore
  impossible to reuse without duplicating them.
- **`TrainingReport` covers the remaining agents**: new `qlearning` section
  (including `table_size`, `precision`, `optimistic_init`) and `search` section
  (MCTS, GreedyLookahead), the `dueling` flag for Double DQN, and the trainer's
  class and opponent.

### Fixed

- **Vanilla `DQNAgent` diverged on both games.** Its TD target maximized over an
  unmasked action set, so illegal-action Q-values — the majority in any card-game
  position — leaked into the bootstrap and compounded through the target network
  (loss peaked at 3.1e9 on Klondike, 4.1e14 on Macao) until the trained greedy
  policy was worse than the untrained one. `DQNAgent` now masks its target to the
  next state's legal actions, the same rule `DoubleDQNAgent` and `QLearningAgent`
  already used. `MASK_VALUE`/`MaskedReplayBuffer` moved to `dqn_agent` and are
  re-exported from `double_dqn_agent` so the public import path is unchanged.
  Single-network + MSE are kept, so the DQN-vs-Double-DQN teaching contrast stays
  intact and Double DQN is byte-for-byte unchanged; only the two `dqn` runs need
  retraining.
- **The divergence auto-detector had a blind spot.** The `peak/median > 1000×`
  test — duplicated between the run's text note and the symlog-axis decision —
  missed a blow-up riding on an already-inflated median, so Klondike DQN's 759×
  slipped under the bar and its loss chart drew an unreadable linear spike with no
  caveat. Both call sites now share one `loss_divergence()` helper that also trips
  on a large absolute peak, so the note and the chart axis can never disagree.
- **Klondike was missing the Heuristic baseline** that Macao shows. Its
  `register_sweep_game` now passes a `heuristic_factory`, so the report draws a
  Heuristic reference line and `results/baselines/klondike.json` gains the row
  (re-measure baselines to populate it).
- **The report now flags that "before training" bars are not comparable across
  agents.** A fresh Q-table tie-breaks uniformly at random (its "before" ≈ random
  play) while a fresh network argmaxes a near-constant output; the Configuration
  caveats now say to compare an agent's before→after delta, not one agent's
  "before" against another's.
- **`SelfPlayTrainer` scored every evaluation episode as 0.0 reward.** The
  episode-reward accumulator sat inside `if training and current_player == 0`,
  so it never ran during evaluation. Consequence: every recorded Macao
  evaluation has `mean_reward`, `std_reward`, `min_reward` and `max_reward`
  of exactly `0.0` — those numbers are an artifact, not a measurement, and
  must be re-measured rather than reinterpreted. Only `win_rate` and
  `mean_steps` were ever meaningful there. The agent is still paid only for
  its own plays, and learning is still training-only.
- **Klondike reward loop.** Non-revealing tableau-to-tableau moves no longer
  pay `0.05 * cards_moved`; they now net `-0.01` (the step cost). The old
  payment was reversible and therefore unbounded free reward — agents that
  optimized it farmed the loop instead of playing solitaire, making reward
  anti-correlated with winning. Reveals (+0.2) and foundation moves (+1.0)
  still pay. Existing Klondike reward curves are not comparable across this
  change; cards-to-foundation numbers are.
- **Klondike loss terminal.** A deal with no legal moves left now terminates
  with `LOSS_REWARD` (-1.0) instead of running to truncation. `max_passes` is
  actually enforced: once passes run out, the draw/recycle action becomes
  illegal rather than a penalized no-op.
- **Macao terminal reward perspective.** The winning step now pays the acting
  player (`+10`), whoever that is; previously player 1 winning was recorded as
  `-5` *for player 1*, which taught opponent-modeling agents that the opponent
  was trying to lose. Losers' payoffs are exposed via `get_reward(player_idx)`.
- **Macao shaped rewards are potential-based on hand size.** Every card leaving
  the actor's hand pays `+0.1`, every card entering costs `-0.1`, and nothing
  else pays. The old flat per-play bonuses made hoarding profitable (draw
  cards, then harvest a bonus per play); search agents found that exploit and
  stopped finishing games.
- **MCTSAgent could not see immediate rewards.** Backpropagation credited an
  edge's reward only above the node it led to, so the Q-value UCT used for
  selection — and the root pooling used for the final choice — excluded the
  action's own reward. A move that won on the spot (+10) carried Q = 0. Fixed
  by folding the edge reward into the node's value before accumulation.
- **MCTSAgent now sees losses.** On terminal edges, every non-acting player's
  terminal payoff (from `get_reward()`) enters the search values, so "the
  opponent wins" finally costs something and blocking moves are found. With
  both MCTS fixes, Macao win rate vs a random opponent went from ~3% (below
  random) to ~87% at 60 simulations, and search strength now scales with the
  simulation budget.
- **Epsilon decays per episode, not per learning step.** `DQNAgent`,
  `DoubleDQNAgent` and `QLearningAgent` now apply `epsilon_decay` in `reset()`.
  Previously a 300-step episode burned 300 decays, so documented schedules ran
  ~300x faster than they read; a 5000-episode run was effectively greedy after
  episode 20. Example scripts' decay values retuned accordingly.
- **Global RNG reseeding removed.** `Deck.shuffle(seed)` uses a private
  `random.Random`; games own a per-instance RNG (constructor/`reset(seed=...)`);
  `CardGameEnv.reset(seed=...)` forwards the seed to the game instead of
  calling `np.random.seed()`. Seeded deals are now actually reproducible and
  nothing perturbs other components' randomness.
- **`SelfPlayTrainer.opponent_update_interval` works.** Self-play now trains
  against a frozen deep-copied snapshot of the agent, refreshed every N
  episodes as documented. Pass `None` for the old zero-lag mirror match.
- **Macao illegal card plays raise `ValueError`** instead of penalizing and
  silently advancing the turn; legality enforcement lives in `CardGameEnv`.

### Added

- **Macao Ace/Jack declarations are agent actions** (54-57 declare a suit,
  58-64 declare a rank). The requested suit/rank used to be hardcoded to
  "most common in hand", making the game's two most strategic decisions
  unlearnable. The observation grew two declaration-phase flags;
  `MAX_ACTIONS` is 65 and every action is reachable.
- **`reward_mode="sparse"`** on both games: +1 win / -1 loss (Klondike loss,
  Macao via `get_reward`) and nothing else. Unfarmable by construction, at the
  cost of a slower learning signal.
- **`solve_klondike()`**: budgeted perfect-information solvability search, so
  win rates can be reported over solvable deals only.
- **Repeated-position handling in `CardGameEnv`**: repeats are flagged in
  `info["repeated_position"]` and can be priced with
  `repeated_position_penalty`.
- **`TrainingReport`**: `to_json()`, and a PPO parameter section alongside the
  DQN one.
- **GitHub Actions CI**: flake8 + both test suites on Python 3.10-3.12.

### Changed

- **Klondike `MAX_ACTIONS` shrank from 200 to 68.** Actions 68-199 could never
  be legal; every network carried 132 dead outputs. Checkpoints trained
  against the old action space are incompatible.
