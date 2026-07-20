# Changelog

## Unreleased

### Added

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
