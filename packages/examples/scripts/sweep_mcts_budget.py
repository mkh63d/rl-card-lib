"""
Sweep the MCTS simulation budget on Macao and record win-rate vs. random.

This is the missing "budget sweep": it runs the MCTS agent at each of several
simulation-per-move counts, plays a fixed batch of Macao games against a random
opponent at every count, and writes one `simulations,win_rate` series to CSV --
then renders it as a single-line figure suitable for the thesis (Fig. 6.1).

Unlike run_sweep.py (which sweeps games x learners and treats MCTS as a single
fixed-budget baseline), this varies only `simulations` and holds everything else
constant, so the resulting curve isolates the effect of the search budget.

Measurement path
----------------
Each point reuses `run_macao_baselines` -- the exact code behind the
agent-comparison run -- so the numbers are directly comparable to that run's
MCTS figure rather than a second implementation that could drift. Every budget
is measured with the same episodes, seeds, opponent, determinizations and
rollout depth; only the simulation count moves.

The defaults are plain MCTS (determinizations=1, rollout_depth=20, seed=0),
which reproduces the headline anchors: ~77% win rate at 40 simulations and ~90%
at 60. The agent-comparison run's x4det variant (which splits the budget across
4 hidden-card samples and so searches shallower) is available with
--determinizations 4.

Nothing here fabricates or interpolates: only budgets you actually run are
written. The CSV is appended point-by-point and flushed, so a long sweep that
is interrupted still leaves every completed point on disk.

Usage:
    # Full curve (slow -- MCTS pays its whole budget on every move):
    python packages/examples/scripts/sweep_mcts_budget.py --episodes 200

    # Quick smoke test:
    python packages/examples/scripts/sweep_mcts_budget.py \
        --budgets 5,20,60 --episodes 20

    # Re-plot from an existing CSV without re-running the sweep:
    python packages/examples/scripts/sweep_mcts_budget.py --plot-only
"""

import argparse
import csv
import os
import sys

from rl_card_lib.agents import MCTSAgent
from rl_card_lib.harness import run_macao_baselines

DEFAULT_BUDGETS = "1,2,5,10,20,40,60,80,120"


def parse_budgets(text: str) -> list[int]:
    budgets = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        value = int(chunk)
        if value < 1:
            raise SystemExit(f"budget must be >= 1, got {value}")
        budgets.append(value)
    if not budgets:
        raise SystemExit("no budgets to sweep")
    # De-duplicate while keeping ascending order, so the x-axis is monotonic.
    return sorted(set(budgets))


def sweep(args) -> list[dict]:
    """Measure win-rate vs. random at each simulation budget.

    Writes each point to the CSV as it is computed so an interrupted run keeps
    its finished points, and returns the full list of rows for plotting.
    """
    budgets = parse_budgets(args.budgets)
    os.makedirs(args.outdir, exist_ok=True)
    csv_path = os.path.join(args.outdir, "macao_mcts_budget_sweep.csv")

    fields = [
        "simulations", "win_rate", "draw_rate", "episodes",
        "determinizations", "rollout_depth", "seed", "seconds",
    ]

    print(f"Sweeping MCTS on Macao: budgets={budgets}  "
          f"episodes={args.episodes}  determinizations={args.determinizations}  "
          f"rollout_depth={args.rollout_depth}  seed={args.seed}", flush=True)

    rows: list[dict] = []
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        handle.flush()

        for sims in budgets:
            agent = MCTSAgent(
                simulations=sims,
                determinizations=args.determinizations,
                rollout_depth=args.rollout_depth,
                seed=args.seed,
            )
            print(f"\n--- MCTS(simulations={sims}) ---", flush=True)
            result = run_macao_baselines(
                [(f"MCTS({sims})", agent)],
                episodes=args.episodes,
                max_steps=args.max_steps,
                opponent_seed=args.opponent_seed,
                verbose=True,
            )[0]

            row = {
                "simulations": sims,
                "win_rate": round(result["win_rate"], 6),
                "draw_rate": round(result["draw_rate"], 6),
                "episodes": args.episodes,
                "determinizations": args.determinizations,
                "rollout_depth": args.rollout_depth,
                "seed": args.seed,
                "seconds": round(result["seconds"], 1),
            }
            writer.writerow(row)
            handle.flush()
            rows.append(row)

    print(f"\nWrote {csv_path} ({len(rows)} point(s))", flush=True)
    return rows


def read_rows(csv_path: str) -> list[dict]:
    """Load a previously written sweep CSV for --plot-only."""
    if not os.path.exists(csv_path):
        raise SystemExit(
            f"No CSV at {csv_path}. Run the sweep first (drop --plot-only)."
        )
    with open(csv_path, newline="") as handle:
        rows = [
            {"simulations": int(r["simulations"]),
             "win_rate": float(r["win_rate"])}
            for r in csv.DictReader(handle)
        ]
    if not rows:
        raise SystemExit(f"{csv_path} has no data rows to plot.")
    return sorted(rows, key=lambda r: r["simulations"])


def plot(rows: list[dict], args) -> None:
    """Render the budget curve to PNG and SVG.

    One series, so no legend -- the axis labels and (optional) title name it.
    Log x-axis because the budgets span more than a decade; a recessive grid
    and thin marks keep the single line dominant.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    xs = [r["simulations"] for r in rows]
    ys = [r["win_rate"] * 100.0 for r in rows]

    fig, ax = plt.subplots(figsize=(6.5, 4.0), constrained_layout=True)

    ax.plot(xs, ys, color="#2563eb", linewidth=2.0, marker="o",
            markersize=6, markerfacecolor="#2563eb",
            markeredgecolor="white", markeredgewidth=0.8, zorder=3)

    # Optional, clearly-labelled reference line for the pre-fix backup result.
    # It is drawn only when supplied on the command line and is never written
    # to the CSV -- it is an annotation, not a measured point of this sweep.
    if args.annotate_buggy_backup is not None:
        y = args.annotate_buggy_backup * 100.0
        ax.axhline(y, color="#9ca3af", linewidth=1.2, linestyle="--", zorder=1)
        ax.annotate(
            f"pre-fix backup ({y:.0f}%)",
            xy=(xs[0], y), xytext=(0, 4), textcoords="offset points",
            fontsize=8, color="#6b7280", va="bottom",
        )

    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{int(v)}"))
    ax.minorticks_off()

    ax.set_xlabel("Simulations per move")
    ax.set_ylabel("Win rate vs. random (%)")
    ax.set_ylim(0, 100)
    if args.title:
        ax.set_title(args.title)

    ax.grid(True, which="major", color="#e5e7eb", linewidth=0.8, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    png = os.path.join(args.outdir, "macao_mcts_budget_sweep.png")
    svg = os.path.join(args.outdir, "macao_mcts_budget_sweep.svg")
    # A high DPI keeps the PNG crisp when embedded in Word/print; the SVG is
    # resolution-independent and preferred for LaTeX.
    fig.savefig(png, dpi=args.dpi)
    fig.savefig(svg)
    plt.close(fig)
    print(f"Wrote {png}\nWrote {svg}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--budgets", default=DEFAULT_BUDGETS,
                        help="Comma-separated simulation counts to sweep")
    parser.add_argument("--episodes", type=int, default=200,
                        help="Macao games played per budget (more = less noise, "
                             "but MCTS is slow so this dominates runtime)")
    parser.add_argument("--determinizations", type=int, default=1,
                        help="Hidden-card samples per move; 1 is plain MCTS "
                             "(matches the headline anchors). Use 4 for the "
                             "agent-comparison x4det variant")
    parser.add_argument("--rollout-depth", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0,
                        help="Agent seed; episode/opponent seeds are fixed per "
                             "game inside run_macao_baselines")
    parser.add_argument("--opponent-seed", type=int, default=None,
                        help="Fix the random opponent's seed across games; "
                             "default varies it per game")
    parser.add_argument("--outdir", default="./results/mcts_budget_sweep")
    parser.add_argument("--title", default=None,
                        help="Optional figure title (thesis captions externally, "
                             "so this is off by default)")
    parser.add_argument("--annotate-buggy-backup", type=float, default=None,
                        metavar="RATE",
                        help="Draw a labelled reference line at this win rate "
                             "(e.g. 0.03); an annotation only, not a swept point")
    parser.add_argument("--dpi", type=int, default=300,
                        help="PNG resolution; 300 is print/Word quality")
    parser.add_argument("--no-plot", action="store_true",
                        help="Only write the CSV; skip the figure")
    parser.add_argument("--plot-only", action="store_true",
                        help="Re-render the figure from the existing CSV")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    csv_path = os.path.join(args.outdir, "macao_mcts_budget_sweep.csv")

    if args.plot_only:
        rows = read_rows(csv_path)
    else:
        rows = sweep(args)

    if not args.no_plot:
        try:
            plot(rows, args)
        except Exception as exc:  # pragma: no cover - plotting is best-effort
            print(f"Plotting failed ({exc}); the CSV at {csv_path} is intact.",
                  file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
