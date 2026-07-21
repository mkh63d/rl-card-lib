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
