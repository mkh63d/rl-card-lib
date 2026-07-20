# RL Card Lib - Report

Summarizing training runs: parameters, records and a self-contained HTML report.

Two complementary things live here.

- **`TrainingReport`** describes how a run was *configured* — environment,
  trainer, agent and the algorithm's hyperparameters. It reads a live `Trainer`.
- **`RunRecord`** describes what a run *did* — per-episode series, evaluation
  history, before/after baseline comparison, timings and provenance. It is
  persisted as JSON and is what the HTML report is built from.

## Installation

```bash
pip install -e ./packages/report            # JSON and Markdown
pip install -e "./packages/report[charts]"  # + matplotlib, for figures and HTML
```

matplotlib is an optional extra. Without it the record and parameter APIs work
normally and figure rendering prints a message and returns nothing, matching how
`TrainingMetrics.plot()` behaves in the core package.

## Training parameters

```python
from rl_card_lib.report import TrainingReport

report = TrainingReport.from_trainer(trainer, episodes=5000, max_steps_per_episode=500)
print(report.to_markdown())
print(report.to_json())
```

Sections: `training`, `environment`, `trainer`, `agent`, and whichever of `dqn`,
`ppo`, `qlearning` or `search` applies to the agent. `trainer` records the
trainer class and its opponent, so a Macao number can be read knowing whether it
came from self-play or from a fixed heuristic.

## Run records

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

for run in store.load_runs():              # newest finished_at first
    print(run.run_id, run.headline["after"])
```

A run is keyed by `{game}__{agent}`, which is both its identity and its
directory name. There are no timestamped run directories, so re-running a pair
necessarily replaces it — "keep only the last run of each model" is a property
of the layout rather than a cleanup step that can be skipped.

`RunRecord.from_metrics_json(...)` adapts a bare `metrics.json` written before
records existed. Sections that were never captured stay `None`, and the report
renders them as "not recorded" rather than inventing a zero.

### Layout

```
results/
  index.html                     # the report: one self-contained file
  baselines/klondike.json
  figures/                       # cross-model comparison charts
  models/klondike__dqn/
    run.json                     # the record
    metrics.json                 # unchanged TrainingMetrics output
    figures/*.png *.svg
```

Named `models/` and `run.json` because the repo's `.gitignore` matches a bare
`runs/` and `metrics.json` at any depth and would otherwise hide the store.

## HTML report

```bash
python -m rl_card_lib.report.cli --results-dir ./results
```

```python
from rl_card_lib.report.html_report import HtmlReport

HtmlReport.build(RunStore("./results")).write("./results/index.html")
```

One file, no network: no CDN, no sibling stylesheet, no remote font, no linked
image. Figures are embedded as base64 data URIs *and* written to disk as PNG and
SVG, so the page survives being moved while the individual images stay directly
reusable.

The page contains an overview table sorted newest-run-first, comparison charts
per game, and a section per model with stat tiles, figures, summary / evaluation
/ hyperparameter tables and a notes callout. Every table carries Copy / CSV /
PNG buttons; CSV is the guaranteed path, and the PNG route falls back to it
where `foreignObject` rasterization is unreliable. Print rules make Ctrl-P a
clean appendix.

`--no-embed` writes a much smaller page that references `figures/` relatively.
`--no-figures` skips chart rendering entirely.

## Figures

```python
from rl_card_lib.report.figures import render_run_figures, render_comparison_figures
```

Per run: cards-to-foundation (Klondike), reward, rolling win rate, loss, episode
length, exploration schedule, Q-table growth, evaluation, before/after. Per game:
learning curves, final standing, cost versus result, evaluation panels.

Charts whose data was never recorded are skipped rather than drawn empty — PPO
has no epsilon, Macao has no cards-up. A loss spanning more than three orders of
magnitude switches to a symlog axis and says so in the caption; it is never
clipped, because a divergence is a result.

Colour follows the agent rather than its rank, so filtering a series never
repaints the others. Non-learning baselines are muted dashed reference lines,
not additional series: they have no learning curve, and the extra categorical
hues would not clear the contrast floors.
