# Training and reports

## The `Trainer`

`Trainer` runs the episode loop, collects `TrainingMetrics`, and evaluates
periodically. Game-aware agents (heuristics, `GreedyLookaheadAgent`,
`MCTSAgent`) are **auto-bound** to the environment when you construct the
trainer, so you rarely call `bind()` yourself.

```python
from rl_card_lib.env import CardGameEnv
from rl_card_lib.trainer import Trainer
from rl_card_lib.agents import DoubleDQNAgent
from rl_card_lib.games import KlondikeSolitaire

env = CardGameEnv(KlondikeSolitaire(), max_steps=200)
agent = DoubleDQNAgent(
    state_size=env.observation_space.shape[0],
    action_size=env.action_space.n,
)

trainer = Trainer(env, agent)
metrics = trainer.train(episodes=1000)
```

## Self-play

`SelfPlayTrainer` trains against a **frozen snapshot** of the agent, refreshed
every `opponent_update_interval` episodes. The lag is deliberate: an opponent
that tracks the learner move-for-move is the least stable target. The opponent
improves as the agent does, so the difficulty tracks it and never becomes
trivial or hopeless.

```python
from rl_card_lib.trainer import SelfPlayTrainer

trainer = SelfPlayTrainer(env, agent, opponent_update_interval=500)
metrics = trainer.train(episodes=5000)

# Or train against a fixed policy for the other seats:
# SelfPlayTrainer(env, agent, opponent=some_heuristic)
```

## The training sweep

`run_sweep.py` trains every learner on every registered game, measures the
`Random` / `GreedyLookahead` / `MCTS` baselines, records a `RunRecord` per
model, and renders one HTML page:

```bash
# Train every learner on both games and write results/index.html
python packages/examples/scripts/run_sweep.py --episodes 200

# Re-render from stored records without training
python packages/examples/scripts/run_sweep.py --html-only
```

A run is keyed by `{game}__{agent}`, which is both its identity and its
directory name — so re-running a pair **replaces** it rather than accumulating
timestamped copies.

## Solve-time benchmark

Win rate answers "how often does this agent win". It cannot answer the two
questions that matter once you care about *efficiency*: over deals that are
**known to be winnable**, how many does the agent actually solve, and — for the
ones it solves — how many moves and how much wall-clock time did it take.
`benchmark_solve_time.py` answers both.

```bash
# Curate 50 winnable deals, then play every agent (baselines + trained
# learners) over that same pool.
python packages/examples/scripts/benchmark_solve_time.py --game klondike --pool-size 50

# Baselines only (skip loading trained checkpoints), quicker:
python packages/examples/scripts/benchmark_solve_time.py --game klondike --pool-size 20 --skip-trained
```

It curates the pool with the game's solver (a deal is kept only if the solver
*proves* it winnable), caches the seeds under
`results/solve_benchmark/<game>_pool.json` so the expensive search runs once,
then measures three things per agent:

| Column | Meaning |
|---|---|
| **Solve rate** | share of the winnable pool the agent actually won |
| **Moves to solve** | mean move count, **over solved deals only** |
| **Time to solve** | mean wall-clock per deal, **over solved deals only** |

Averaging moves and time over *solved* deals only is deliberate: folding in the
move cap of a deal the agent never solved would make a worse agent look faster.
A low solve rate therefore means the agent failed a *winnable* deal, not that it
drew an impossible one.

!!! note "Single-player games only"
    A solvable-deal pool needs a solver, and only a single-player game can have
    one — an adversarial game's outcome depends on the opponent, so "moves to
    solve" is undefined. The benchmark runs for any game that declares
    `single_player=True` and a `solver` (see
    [Add your own game](../custom_game.md#single-player-games-solve-time-benchmark));
    Klondike does, Macao is skipped.

Results write to `results/solve_benchmark/<game>.json` and appear as a
**Solve-time benchmark** section in `results/index.html` on its next build.
Trained learners are loaded from their checkpoints via `load_trained_learner`;
any learner still training (no checkpoint yet) is skipped with a note.

## Two kinds of report

The `rl-card-lib-report` package holds two complementary things.

**`TrainingReport`** describes how a run was *configured* — environment,
trainer, agent and the algorithm's hyperparameters — read straight from a live
`Trainer`:

```python
from rl_card_lib.report import TrainingReport

report = TrainingReport.from_trainer(trainer, episodes=5000, max_steps_per_episode=500)
print(report.to_markdown())
print(report.to_json())
```

**`RunRecord` / `RunStore`** describe what a run *did* — per-episode series,
evaluation history, before/after baseline comparison, timings and provenance —
persisted as `run.json`:

```python
from rl_card_lib.report import RunRecord, RunStore

store = RunStore("./results")
store.reset_run_dir("klondike", "dqn")     # before training, not after

record = RunRecord.from_training(
    game="klondike", agent="dqn", agent_class="DQNAgent",
    metrics=metrics, config=report.as_dict(),
    train_seconds=541.6, episode_extras=extras,
    baseline_before=before, baseline_after=after,
)
store.save_run(record)
```

## The HTML report

```bash
python -m rl_card_lib.report.cli --results-dir ./results
```

```python
from rl_card_lib.report.html_report import HtmlReport

HtmlReport.build(RunStore("./results")).write("./results/index.html")
```

The page is **one self-contained file**: no CDN, no sibling stylesheet, no
remote font, no linked image. Figures are embedded as base64 data URIs *and*
written to disk as PNG/SVG, so the page survives being moved while the images
stay reusable. It contains an overview table (newest run first), comparison
charts per game, and a section per model with stat tiles, figures and
hyperparameter tables. Every table exports to CSV/PNG and every figure to
PNG/SVG, and print rules make `Ctrl-P` a clean thesis appendix.

- `--no-embed` writes a smaller page that references `figures/` relatively.
- `--no-figures` skips chart rendering entirely.

!!! tip "Diverged losses are shown, not hidden"
    A loss whose peak dwarfs its median switches to a symlog axis and says so in
    the caption — it is never clipped, because a divergence is a *result*. The
    chart and the run's text note share one `loss_divergence()` test, so they
    can never disagree.

See the [report API reference](../reference/report.md) for the full record
schema.
