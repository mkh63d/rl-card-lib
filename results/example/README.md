# Example report artifacts

Committed as a reference for the format written by
`packages/examples/scripts/run_sweep.py`. The real store lives in `results/`
and is gitignored -- it is regenerated on every sweep and an embedded
`index.html` runs to several megabytes.

| File | What it is |
|---|---|
| `run.json` | One `RunRecord`: identity, timestamps, hyperparameters, per-episode series, evaluation history, before/after comparison, notes |
| `metrics.json` | The unchanged `TrainingMetrics` output the record is built from |
| `figures/` | The charts for this run, as PNG and SVG |
| `index.html` | A one-run report, so the page structure is browsable here |

Regenerate with:

```bash
python packages/examples/scripts/curate_example_report.py
```
