"""
Copy one run's artifacts into results/example/ as a committed reference.

The results store is regenerated on every sweep and is gitignored, so nothing
in it documents the artifact format for someone reading the repository. This
copies a single representative run -- its record, its metrics, its figures --
plus a small standalone report, so the shape of each file is visible in version
control without committing hundreds of megabytes.

Usage:
    python curate_example_report.py                    # pick a run automatically
    python curate_example_report.py --run klondike__dqn
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

from rl_card_lib.report import RunStore, game_spec
from rl_card_lib.report.html_report import HtmlReport

README = """# Example report artifacts

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
"""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="./results")
    parser.add_argument("--run", default=None,
                        help="Run id to copy, e.g. klondike__dqn")
    parser.add_argument("--out", default=None,
                        help="Destination (default: <results-dir>/example)")
    args = parser.parse_args(argv)

    store = RunStore(args.results_dir)
    runs = store.load_runs()
    if not runs:
        print(f"No runs under {args.results_dir}", file=sys.stderr)
        return 1

    if args.run:
        chosen = next((r for r in runs if r.run_id == args.run), None)
        if chosen is None:
            print(f"No run {args.run!r}. Available: "
                  f"{', '.join(r.run_id for r in runs)}", file=sys.stderr)
            return 1
    else:
        # Prefer a run that exercises the most of the format: real
        # hyperparameters, a game-specific episode curve, and a measured
        # baseline comparison. "Has an episode curve" is game-agnostic --
        # cards_up for Klondike, a custom series for someone else's game.
        def has_episode_curve(record):
            spec = game_spec(record.game)
            return any(record.series(k) for k in spec.get("episode_curves", []))

        chosen = max(runs, key=lambda r: (
            r.config is not None,
            has_episode_curve(r),
            bool(r.baseline_comparison),
            r.episode_count,
        ))

    out = Path(args.out) if args.out else Path(args.results_dir) / "example"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    source = store.run_dir(chosen.game, chosen.agent)
    for name in ("run.json", "metrics.json"):
        if (source / name).is_file():
            shutil.copy2(source / name, out / name)
    if (source / "figures").is_dir():
        shutil.copytree(source / "figures", out / "figures")

    # A one-run report, so the page structure is browsable from the repo
    # without regenerating the whole store.
    single = HtmlReport(
        runs=[chosen],
        baselines={chosen.game: store.load_baselines().get(chosen.game)}
        if store.has_baselines(chosen.game) else {},
        run_figures={}, comparison_figures={},
        generated_at=chosen.timestamps.get("generated_at", ""),
        command="python packages/examples/scripts/run_sweep.py --episodes 200",
    )
    single.write(out / "index.html")
    (out / "README.md").write_text(README, encoding="utf-8")

    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"Curated {chosen.run_id} into {out} ({total / 1024:.0f} KB)")
    for path in sorted(out.rglob("*")):
        if path.is_file():
            print(f"  {path.relative_to(out)}  {path.stat().st_size / 1024:.0f} KB")

    record = json.loads((out / "run.json").read_text(encoding="utf-8"))
    print(f"Record sections: {', '.join(record)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
