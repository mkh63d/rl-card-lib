"""Turn thesis_notes/raw/ into the tables and figures the thesis pastes in.

Reads:
    raw/runs/*.json            one per (game, agent, arm, init-seed)
    raw/baselines_on_test.json non-learning agents on the same TEST pool
    raw/protocol_probe.json    hyper-parameters read off constructed agents
    raw/klondike_test_solvable.json
    ../results/mcts_budget_sweep/macao_mcts_budget_sweep.csv

Writes:
    tables/*.csv               every results table, mean +/- std over 3 seeds
    figures/*.png, *.svg       300 dpi, all type >= 10 pt, English axis labels

Colour follows the agent, never its rank, and comes from the validated
four-slot categorical order (blue, orange, aqua, yellow). Three of the four sit
below 3:1 against the surface, so every series is also directly labelled and
every table has a CSV twin -- the relief the contrast check requires.
"""

from __future__ import annotations

import csv
import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "raw")
TABLES = os.path.join(HERE, "..", "tables")
FIGURES = os.path.join(HERE, "..", "figures")
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))

# --- validated categorical order (light surface) ---------------------------
SERIES = {
    "ppo": "#2a78d6",          # slot 1 blue
    "double_dqn": "#eb6834",   # slot 2 orange
    "dqn": "#1baf7a",          # slot 3 aqua
    "q_learning": "#eda100",   # slot 4 yellow
}
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#6f6e69"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BASELINE = "#8a8983"

LABEL = {
    "ppo": "PPO",
    "double_dqn": "Double DQN",
    "dqn": "DQN",
    "q_learning": "Q-learning",
}
# What each arm configures. `fixed` is the library itself since PRs #24-#34, so
# it bundles four changes rather than isolating the bootstrap fix it once meant;
# `asis` is reconstructed on the same commit by switching those four back. Only
# `fixed` -> `noloop` is still a single-factor comparison. See results.md.
# Kept short and comma-free: these strings land in a CSV column, a figure
# legend and the injected diagnosis.md table. The four levers each arm actually
# sets are spelled out in results.md and recorded per run in `arm_config`.
ARM_LABEL = {
    "asis": "before the fixes (pre-#24)",
    "fixed": "library as shipped",
    "noloop": "+ repeated-position penalty",
}
AGENT_ORDER = ["ppo", "double_dqn", "dqn", "q_learning"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Segoe UI", "Arial"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": TEXT_PRIMARY,
    "axes.labelcolor": TEXT_SECONDARY,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "xtick.color": TEXT_MUTED,
    "ytick.color": TEXT_MUTED,
    "legend.frameon": False,
    "lines.linewidth": 2.0,
})


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def load_runs() -> dict:
    """{(game, agent, arm): [record, ...]} sorted by init seed."""
    runs: dict = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(RAW, "runs", "*.json"))):
        with open(path, "r", encoding="utf-8") as handle:
            record = json.load(handle)
        runs[(record["game"], record["agent"], record["arm"])].append(record)
    for key in runs:
        runs[key].sort(key=lambda r: r["init_seed"])
    return runs


def load_json(name: str):
    path = os.path.join(RAW, name)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def emit(fig, name: str) -> None:
    os.makedirs(FIGURES, exist_ok=True)
    for ext in ("png", "svg"):
        path = os.path.join(FIGURES, f"{name}.{ext}")
        fig.savefig(path, dpi=300 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure  {name}.png / .svg")


def write_csv(name: str, header: list[str], rows: list[list]) -> None:
    os.makedirs(TABLES, exist_ok=True)
    path = os.path.join(TABLES, name)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"  table   {name}  ({len(rows)} rows)")


def fmt(mean, std, places=2, percent=False):
    if mean is None:
        return ""
    if percent:
        return f"{mean * 100:.1f} +/- {std * 100:.1f}"
    return f"{mean:.{places}f} +/- {std:.{places}f}"


def agg(records: list, key: str, where: str = "test_after"):
    """Mean and sample std across init seeds of one TEST metric."""
    values = [r[where][key] for r in records if key in r.get(where, {})]
    if not values:
        return None, None, 0
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0, len(arr)


def trailing(values: list, window: int = 100) -> np.ndarray:
    arr = np.asarray([np.nan if v is None else float(v) for v in values],
                     dtype=float)
    out = np.empty_like(arr)
    for i in range(len(arr)):
        chunk = arr[max(0, i - window + 1):i + 1]
        chunk = chunk[~np.isnan(chunk)]
        out[i] = chunk.mean() if len(chunk) else np.nan
    return out


def label_right(ax, entries, min_gap=0.062, fontsize=10):
    """Direct labels at the right edge, pushed apart so they stay legible."""
    if not entries:
        return
    bottom, top = ax.get_ylim()
    span = (top - bottom) or 1.0
    placed = sorted(((t, v, c, (v - bottom) / span) for t, v, c in entries),
                    key=lambda e: e[3])
    fractions, previous = [], -1.0
    for _, _, _, natural in placed:
        f = max(natural, previous + min_gap)
        fractions.append(f)
        previous = f
    overflow = fractions[-1] - 1.0
    if overflow > 0:
        fractions = [f - overflow for f in fractions]
        for i in range(len(fractions) - 2, -1, -1):
            fractions[i] = min(fractions[i], fractions[i + 1] - min_gap)
    for (text, _value, colour, natural), fraction in zip(placed, fractions):
        ax.annotate(text, xy=(1.0, fraction), xycoords="axes fraction",
                    xytext=(7, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=fontsize, color=colour,
                    annotation_clip=False)
        if abs(fraction - natural) > 0.012:
            ax.annotate("", xy=(1.0, natural), xycoords="axes fraction",
                        xytext=(1.0, fraction), textcoords="axes fraction",
                        arrowprops={"arrowstyle": "-", "color": colour,
                                    "alpha": 0.45, "linewidth": 0.8},
                        annotation_clip=False)


def caption(ax, text: str, pad_points: float) -> None:
    """Footnote under the axes, offset in points so it clears xlabel/legend."""
    ax.annotate(text, xy=(0.0, 0.0), xycoords="axes fraction",
                xytext=(0, -pad_points), textcoords="offset points",
                ha="left", va="top", fontsize=9.5, color=TEXT_MUTED,
                linespacing=1.5, annotation_clip=False)

# ---------------------------------------------------------------------------
# TRAIN learning curves
# ---------------------------------------------------------------------------

CURVE = {
    "klondike": ("cards_up", "Cards to foundation (0–52)",
                 "Klondike: training progress, mean of 3 initialisation seeds"),
    "macao": ("win", "Win rate per training episode",
              "Macao: training progress, mean of 3 initialisation seeds"),
}


def baseline_rows(baselines, game: str, arm: str) -> list:
    """The baseline rows an arm should be judged against.

    Klondike's `asis` arm plays `max_passes=None` while `fixed` and `noloop`
    play the bundled three-pass limit, so they are not the same game: random
    scores 11.59 cards under unlimited passes and 9.79 under three. Drawing one
    reference line for both would score an arm against the other arm's
    baseline. Falls back to the bundled rows when the unlimited-pass set has
    not been measured, so an older JSON still renders.
    """
    if not baselines:
        return []
    if game == "klondike" and arm == "asis":
        rows = baselines.get("klondike_unlimited_passes")
        if rows:
            return rows
    return baselines.get(game, [])


def fig_train_curves(runs: dict, game: str, arm: str, baselines) -> None:
    key, ylabel, title = CURVE[game]
    fig, ax = plt.subplots(figsize=(8.4, 4.6))

    endpoints, table_cols, table_data = [], ["Episode"], []
    length = 0
    drawn = 0
    for agent in AGENT_ORDER:
        records = runs.get((game, agent, arm))
        if not records:
            continue
        curves = [trailing(r["train_series"][key], 100) for r in records]
        n = min(len(c) for c in curves)
        stack = np.vstack([c[:n] for c in curves])
        mean, std = np.nanmean(stack, axis=0), np.nanstd(stack, axis=0)
        x = np.arange(n)
        colour = SERIES[agent]
        ax.fill_between(x, mean - std, mean + std, color=colour, alpha=0.13,
                        linewidth=0)
        ax.plot(x, mean, color=colour, label=LABEL[agent])
        endpoints.append((LABEL[agent], float(mean[-1]), colour))
        table_cols += [f"{LABEL[agent]} mean", f"{LABEL[agent]} std"]
        table_data.append((mean, std))
        length = max(length, n)
        drawn += 1

    if not drawn:
        plt.close(fig)
        return

    if baselines:
        for row in baseline_rows(baselines, game, arm):
            if row["agent"] != "Random":
                continue
            value = (row["cards_up"] if game == "klondike"
                     else row["win_rate_vs_heuristic"])
            ax.axhline(value, color=BASELINE, linestyle="--", linewidth=1.2)
            endpoints.append((f"Random baseline\n(greedy, TEST)", value, BASELINE))

    ax.set_xlabel("Training episode")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, max(1, length - 1))
    ax.set_ylim(bottom=0)
    ax.set_title(title, color=TEXT_PRIMARY, loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16),
              ncol=min(4, drawn))
    label_right(ax, endpoints)

    caption(ax,
            "100-episode trailing mean per seed, then averaged over the 3 "
            "seeds; band is ± 1 SD across seeds.\nMeasured during training "
            "with exploration on — not comparable with the greedy TEST "
            "figures.", 96)

    emit(fig, f"train_curve_{game}_{arm}")

    rows = []
    for i in range(0, length, 25):
        row = [i]
        for mean, std in table_data:
            row += ([round(float(mean[i]), 4), round(float(std[i]), 4)]
                    if i < len(mean) else ["", ""])
        rows.append(row)
    write_csv(f"train_curve_{game}_{arm}.csv", table_cols, rows)


# ---------------------------------------------------------------------------
# TEST comparison
# ---------------------------------------------------------------------------

TEST_SPEC = {
    "klondike": ("cards_up", "Cards to foundation (0–52)",
                 "Klondike: greedy evaluation on the 200-deal TEST pool", 1),
    "macao": ("win_rate_vs_heuristic", "Win rate vs the heuristic opponent",
              "Macao: greedy evaluation on the 200-deal TEST pool", 1),
}


def fig_test_comparison(runs: dict, game: str, arm: str, baselines) -> None:
    key, xlabel, title, places = TEST_SPEC[game]
    rows = []
    for row in baseline_rows(baselines, game, arm):
        rows.append((row["agent"], row[key], None, BASELINE, "baseline"))
    rows.sort(key=lambda r: r[1])
    learners = []
    for agent in AGENT_ORDER:
        records = runs.get((game, agent, arm))
        if not records:
            continue
        mean, std, n = agg(records, key)
        learners.append((f"{LABEL[agent]} (n={n})", mean, std, SERIES[agent],
                         "learner"))
    learners.sort(key=lambda r: r[1])
    if not learners:
        return

    # Percentage metrics are plotted in percent, so the axis ticks and the
    # value labels are in the same unit.
    scale = 100.0 if game == "macao" else 1.0
    suffix = "%" if game == "macao" else ""

    allrows = rows + [("", None, None, None, "gap")] + learners if rows else learners
    fig, ax = plt.subplots(figsize=(8.0, 0.46 * len(allrows) + 2.0))
    positions = np.arange(len(allrows))
    values = [0 if r[1] is None else r[1] * scale for r in allrows]
    errors = [0 if r[2] is None else r[2] * scale for r in allrows]
    colours = [r[3] or SURFACE for r in allrows]

    ax.barh(positions, values, xerr=errors, color=colours, height=0.66,
            error_kw={"ecolor": TEXT_MUTED, "elinewidth": 1.2, "capsize": 3})
    for i, row in enumerate(allrows):
        if row[1] is None:
            continue
        text = f"{row[1] * scale:.{places}f}{suffix}"
        if row[2]:
            text += f" ± {row[2] * scale:.{places}f}"
        ax.annotate(text, xy=(values[i] + errors[i], i), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=10,
                    color=TEXT_SECONDARY)

    ax.set_yticks([i for i, r in enumerate(allrows) if r[4] != "gap"])
    ax.set_yticklabels([r[0] for r in allrows if r[4] != "gap"], fontsize=10)
    ax.set_xlabel(xlabel + (" (%)" if game == "macao" else ""))
    ax.set_title(title, color=TEXT_PRIMARY, loc="left")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(values) * 1.30 if max(values) else 1)

    caption(ax,
            "Grey: non-learning baselines (no seed variance). Coloured: "
            "learners, mean ± 1 SD over 3 initialisation seeds.\nSame 200 "
            "deals for every row; exploration off.", 64)
    emit(fig, f"test_comparison_{game}_{arm}")


# ---------------------------------------------------------------------------
# MCTS budget sweep
# ---------------------------------------------------------------------------

def fig_mcts_budget() -> None:
    path = os.path.join(REPO, "results", "mcts_budget_sweep",
                        "macao_mcts_budget_sweep.csv")
    if not os.path.exists(path):
        print("  (no MCTS budget sweep CSV; skipping)")
        return
    with open(path, newline="", encoding="utf-8") as handle:
        data = sorted((
            {"simulations": int(r["simulations"]),
             "win_rate": float(r["win_rate"]),
             "draw_rate": float(r["draw_rate"]),
             "episodes": int(r["episodes"]),
             "seconds": float(r["seconds"])}
            for r in csv.DictReader(handle)), key=lambda r: r["simulations"])

    xs = [r["simulations"] for r in data]
    ys = [r["win_rate"] * 100 for r in data]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(xs, ys, color=SERIES["ppo"], marker="o", markersize=8,
            markerfacecolor=SERIES["ppo"], markeredgecolor=SURFACE,
            markeredgewidth=1.2, zorder=3)
    for x, y in ((xs[0], ys[0]), (xs[-1], ys[-1])):
        ax.annotate(f"{y:.0f}%", xy=(x, y), xytext=(0, 11),
                    textcoords="offset points", ha="center", fontsize=10,
                    color=TEXT_SECONDARY)
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs])
    ax.minorticks_off()
    ax.set_xlabel("Simulations per move (log scale)")
    ax.set_ylabel("Win rate vs a random opponent (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Macao: MCTS strength against the simulation budget",
                 color=TEXT_PRIMARY, loc="left")
    caption(ax,
            f"{data[0]['episodes']} games per budget, determinizations=1, "
            "rollout depth 20, agent seed 0, 200-turn cap.\nMeasured after the "
            "Monte-Carlo backup fix; the pre-fix backup scored about 3% at "
            "every budget.", 62)
    emit(fig, "mcts_budget_sweep")

    write_csv("mcts_budget_sweep.csv",
              ["simulations", "win_rate_vs_random", "draw_rate", "episodes",
               "seconds"],
              [[r["simulations"], round(r["win_rate"], 4),
                round(r["draw_rate"], 4), r["episodes"], round(r["seconds"], 1)]
               for r in data])


# ---------------------------------------------------------------------------
# epsilon schedule
# ---------------------------------------------------------------------------

def fig_epsilon(runs: dict) -> None:
    records = runs.get(("klondike", "dqn", "asis")) or \
        runs.get(("macao", "dqn", "asis"))
    if not records:
        return
    series = [v for v in records[0]["train_series"]["epsilon"] if v is not None]
    if not series:
        return
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.plot(series, color=SERIES["dqn"])
    ax.axhline(0.05, color=BASELINE, linestyle="--", linewidth=1.2)
    floor = next((i for i, v in enumerate(series) if v <= 0.05), None)
    if floor is not None:
        ax.axvline(floor, color=BASELINE, linestyle=":", linewidth=1.2)
        ax.annotate(f"reaches 0.05 at episode {floor + 1}",
                    xy=(floor, 0.05), xytext=(14, 42),
                    textcoords="offset points", fontsize=10,
                    color=TEXT_SECONDARY,
                    arrowprops={"arrowstyle": "->", "color": TEXT_MUTED,
                                "linewidth": 1.0})
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Exploration rate ε")
    ax.set_xlim(0, len(series) - 1)
    ax.set_ylim(0, 1.02)
    ax.set_title("Exploration schedule actually followed during training",
                 color=TEXT_PRIMARY, loc="left")
    caption(ax,
            "ε × 0.995 once per training episode. Recorded from the run, not "
            "reconstructed from the declared schedule.", 60)
    emit(fig, "epsilon_schedule")


# ---------------------------------------------------------------------------
# ablation
# ---------------------------------------------------------------------------

def ablation_rows(runs: dict) -> list[list]:
    rows = []
    for game in ("klondike", "macao"):
        key = TEST_SPEC[game][0]
        for agent in AGENT_ORDER:
            for arm in ("asis", "fixed", "noloop"):
                records = runs.get((game, agent, arm))
                if not records:
                    continue
                before_m, before_s, _ = agg(records, key, "test_before")
                after_m, after_s, n = agg(records, key)
                train = float(np.mean([
                    r["train_summary"]["mean_reward"] for r in records]))
                seconds = float(np.mean([
                    r["duration"]["train_seconds"] for r in records]))
                rows.append([
                    game, LABEL[agent], arm, ARM_LABEL[arm], n,
                    round(before_m, 4), round(before_s, 4),
                    round(after_m, 4), round(after_s, 4),
                    round(after_m - before_m, 4),
                    round(train, 3), round(seconds, 1),
                ])
    return rows


def fig_ablation(runs: dict, game: str) -> None:
    key, xlabel, _title, places = TEST_SPEC[game]
    arms = [a for a in ("asis", "fixed", "noloop")
            if any((game, ag, a) in runs for ag in AGENT_ORDER)]
    agents = [a for a in AGENT_ORDER if (game, a, "asis") in runs
              and any((game, a, arm) in runs for arm in arms[1:])]
    if not agents or len(arms) < 2:
        return

    scale = 100.0 if game == "macao" else 1.0
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    width = 0.8 / len(arms)
    positions = np.arange(len(agents))
    hatches = {"asis": "", "fixed": "//", "noloop": ".."}
    for j, arm in enumerate(arms):
        means, errs = [], []
        for agent in agents:
            records = runs.get((game, agent, arm))
            if not records:
                means.append(0.0)
                errs.append(0.0)
                continue
            m, sd, _ = agg(records, key)
            means.append((m or 0.0) * scale)
            errs.append((sd or 0.0) * scale)
        offset = (j - (len(arms) - 1) / 2) * width
        ax.bar(positions + offset, means, width * 0.9, yerr=errs,
               color=[SERIES[a] for a in agents],
               alpha=1.0 if arm == "asis" else 0.85,
               hatch=hatches[arm], edgecolor=SURFACE, linewidth=1.4,
               error_kw={"ecolor": TEXT_MUTED, "elinewidth": 1.1, "capsize": 3})
        for x, m, e in zip(positions + offset, means, errs):
            ax.annotate(f"{m:.{places}f}", xy=(x, m + e), xytext=(0, 4),
                        textcoords="offset points", ha="center", fontsize=9.5,
                        color=TEXT_SECONDARY)

    ax.set_xticks(positions)
    ax.set_xticklabels([LABEL[a] for a in agents])
    ax.set_ylabel(xlabel + (" (%)" if game == "macao" else ""))
    ax.set_title(f"{game.capitalize()}: effect of each fix on the TEST pool",
                 color=TEXT_PRIMARY, loc="left")
    ax.grid(axis="x", visible=False)
    # Grey legend swatches: the legend encodes the ARM (by texture); colour in
    # this chart belongs to the agent, and a coloured swatch would imply
    # otherwise.
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="#d8d7d1", edgecolor=SURFACE,
                             hatch=hatches[a], label=ARM_LABEL[a])
                       for a in arms],
              loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=len(arms))
    caption(ax,
            "Mean ± 1 SD over 3 initialisation seeds, same 200 TEST deals, "
            "exploration off.\nBar texture separates the arms, so the "
            "comparison does not rest on fill alone.", 92)
    emit(fig, f"ablation_{game}")


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------

def table_results(runs: dict, game: str, baselines, solvable) -> None:
    key = TEST_SPEC[game][0]
    percent = game == "macao"
    unit = "%" if percent else "cards"
    header = ["agent", "kind", "arm", "seeds",
              f"TEST_before_mean_{unit}", f"TEST_before_sd_{unit}",
              f"TEST_after_mean_{unit}", f"TEST_after_sd_{unit}",
              f"delta_{unit}",
              "TEST_before (mean +/- sd)", "TEST_after (mean +/- sd)",
              "TEST_secondary_mean", "TEST_secondary_sd",
              "TRAIN_mean_reward", "train_seconds_mean"]
    rows = []
    scale = 100.0 if percent else 1.0

    # One table spans every arm, and on Klondike the arms do not share a rule
    # set, so a baseline row has to say which rules it was measured under --
    # otherwise `asis` rows sit next to a reference from a different game.
    baseline_sets = [((baselines or {}).get(game, []),
                      "fixed+noloop" if game == "klondike" else "-")]
    if game == "klondike":
        unlimited = (baselines or {}).get("klondike_unlimited_passes")
        if unlimited:
            baseline_sets.append((unlimited, "asis"))

    for source, arm_label in baseline_sets:
        for row in source:
            secondary = (row["win_rate"] if game == "klondike"
                         else row["win_rate_vs_random"])
            value = row[key] * scale
            rows.append([row["agent"], "baseline", arm_label, 1,
                         "", "", round(value, 3), "", "",
                         "", f"{value:.2f}",
                         round(secondary, 4), "", "", ""])

    for arm in ("asis", "fixed", "noloop"):
        for agent in AGENT_ORDER:
            records = runs.get((game, agent, arm))
            if not records:
                continue
            bm, bs, _ = agg(records, key, "test_before")
            am, asd, n = agg(records, key)
            sec_key = "win_rate" if game == "klondike" else "win_rate_vs_random"
            sm, ss, _ = agg(records, sec_key)
            rows.append([
                LABEL[agent], "learner", arm, n,
                round(bm * scale, 3), round(bs * scale, 3),
                round(am * scale, 3), round(asd * scale, 3),
                round((am - bm) * scale, 3),
                fmt(bm, bs, 2, percent), fmt(am, asd, 2, percent),
                round(sm, 4), round(ss, 4),
                round(float(np.mean([r["train_summary"]["mean_reward"]
                                     for r in records])), 3),
                round(float(np.mean([r["duration"]["train_seconds"]
                                     for r in records])), 1),
            ])

    name = ("klondike_cards_to_foundation.csv" if game == "klondike"
            else "macao_win_rate.csv")
    write_csv(name, header, rows)


def table_solve_time(baselines, solvable) -> None:
    """Both halves of the solve-rate table: baselines and trained learners.

    `baselines_on_test.py` measures the non-learning agents and
    `solve_time_learners.py` the trained checkpoints, over the same
    TEST_SOLVABLE pool, so the two sets of rows are one table rather than two.
    A learner row carries a spread because it is three seeds; a baseline is a
    single deterministic measurement, so its sd columns stay empty.
    """
    rows = []
    if baselines and "klondike_solve_time" in baselines:
        for row in baselines["klondike_solve_time"]:
            rows.append([
                row["agent"], row["pool_size"], 1,
                round(row["solve_rate"], 4), "",
                round(row["cards_up"], 2), "",
                "" if row["solve_moves"] is None else round(row["solve_moves"], 1),
                "" if row["solve_seconds"] is None
                else round(row["solve_seconds"], 4),
            ])

    learners = load_json("solve_time_learners.json")
    if learners:
        for row in learners.get("rows", []):
            rows.append([
                row["label"], learners.get("pool_size", ""), row["seeds"],
                round(row["solve_rate_mean"], 4), round(row["solve_rate_sd"], 4),
                round(row["cards_up_mean"], 2), round(row["cards_up_sd"], 2),
                "" if row.get("solve_moves_mean") is None
                else round(row["solve_moves_mean"], 1),
                "",
            ])

    if not rows:
        return
    write_csv("solve_time_benchmark.csv",
              ["agent", "pool_size", "seeds", "solve_rate", "solve_rate_sd",
               "cards_up", "cards_up_sd", "mean_moves_to_solve",
               "mean_seconds_to_solve"], rows)


def table_hyperparameters() -> None:
    probe = load_json("protocol_probe.json")
    if not probe:
        return
    hp = probe["hyperparameters"]
    fields = [
        ("class", "Agent class"),
        ("learning_rate", "Learning rate"),
        ("gamma", "Discount gamma"),
        ("hidden_sizes", "Hidden layers"),
        ("replay_buffer_maxlen", "Replay-buffer capacity"),
        ("batch_size", "Batch size"),
        ("minibatch_size", "Minibatch size"),
        ("rollout_steps", "Rollout length"),
        ("target_update_freq", "Target-update frequency (gradient steps, per game)"),
        ("epsilon_start", "Epsilon start"),
        ("epsilon_end", "Epsilon end"),
        ("epsilon_decay", "Epsilon decay (per episode)"),
        ("gae_lambda", "GAE lambda"),
        ("clip_epsilon", "Clip epsilon"),
        ("epochs", "Epochs per update"),
        ("entropy_coef", "Entropy coefficient"),
        ("value_coef", "Value coefficient"),
        ("max_grad_norm", "Max gradient norm"),
        ("dueling", "Duelling head"),
        ("eval_greedy", "Greedy evaluation (argmax of the policy)"),
        ("precision", "State rounding (decimals)"),
        ("optimistic_init", "Optimistic init"),
    ]
    order = ["q_learning", "dqn", "double_dqn", "ppo"]
    rows = []
    for field, label in fields:
        row = [label]
        for kind in order:
            value = hp[kind].get(field)
            if value is None:
                row.append("-")
            elif isinstance(value, list):
                row.append(", ".join(str(v) for v in value))
            elif isinstance(value, dict):
                # A per-game hyper-parameter (target_update_freq since #19):
                # one agent column, one number per game inside it.
                row.append("; ".join(f"{g}: {v}" for g, v in value.items()))
            else:
                row.append(str(value))
        if any(cell != "-" for cell in row[1:]):
            rows.append(row)
    write_csv("hyperparameters.csv",
              ["hyper-parameter", "Q-learning", "DQN", "Double DQN", "PPO"],
              rows)


def table_episode_shape() -> None:
    probe = load_json("protocol_probe.json")
    if not probe:
        return
    rows = []
    for name, row in probe["episode_shape"].items():
        rows.append([
            name, row["episodes"], round(row["mean_steps"], 1),
            row["median_steps"], row["min_steps"], row["max_steps"],
            round(row["terminated_rate"], 3), round(row["truncated_rate"], 3),
            round(row.get("win_rate", row.get("win_rate_player0", 0.0)), 3),
            round(row.get("mean_cards_up", float("nan")), 2)
            if "mean_cards_up" in row else "",
        ])
    write_csv("episode_shape.csv",
              ["configuration", "episodes", "mean_steps", "median_steps",
               "min_steps", "max_steps", "terminated_rate", "truncated_rate",
               "win_rate", "mean_cards_up"], rows)


def markdown_table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out)


def inject_ablation(runs: dict) -> None:
    rows = ablation_rows(runs)
    if not rows:
        return
    write_csv("ablation_fixes.csv",
              ["game", "agent", "arm", "arm_description", "seeds",
               "TEST_before_mean", "TEST_before_sd", "TEST_after_mean",
               "TEST_after_sd", "delta", "TRAIN_mean_reward",
               "train_seconds_mean"], rows)

    display = []
    for row in rows:
        game, agent, arm, desc, n = row[0], row[1], row[2], row[3], row[4]
        percent = game == "macao"
        before = (f"{row[5] * 100:.1f} ± {row[6] * 100:.1f} %" if percent
                  else f"{row[5]:.2f} ± {row[6]:.2f}")
        after = (f"{row[7] * 100:.1f} ± {row[8] * 100:.1f} %" if percent
                 else f"{row[7]:.2f} ± {row[8]:.2f}")
        delta = (f"{row[9] * 100:+.1f} pp" if percent else f"{row[9]:+.2f}")
        display.append([game, agent, f"`{arm}` — {desc}", n, before, after, delta])

    header = ["gra", "agent", "ramię", "seedy", "TEST przed", "TEST po", "Δ"]
    block = ("Metryka: Klondike — karty na bazach (0–52); Macao — win rate "
             "przeciwko heurystyce.\n\n" + markdown_table(header, display))

    path = os.path.join(HERE, "..", "diagnosis.md")
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()
    marker = "<!-- ABLATION_TABLE -->"
    if marker in text:
        text = text.split(marker)[0] + marker + "\n\n" + block + "\n"
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
        print("  injected ablation table into diagnosis.md")


# ---------------------------------------------------------------------------

def main() -> int:
    runs = load_runs()
    print(f"{sum(len(v) for v in runs.values())} run(s) in "
          f"{len(runs)} (game, agent, arm) group(s)")
    baselines = load_json("baselines_on_test.json")
    # The unlimited-pass reference set lives in its own file so the two
    # measurements can run concurrently; fold it in so `baseline_rows` can
    # reach it by key.
    unlimited = load_json("baselines_unlimited_passes.json")
    if baselines is not None and unlimited:
        rows = unlimited.get("klondike_unlimited_passes")
        if rows:
            baselines["klondike_unlimited_passes"] = rows
    solvable = load_json("klondike_test_solvable.json")

    os.makedirs(TABLES, exist_ok=True)
    os.makedirs(FIGURES, exist_ok=True)

    table_hyperparameters()
    table_episode_shape()
    fig_mcts_budget()

    for game in ("klondike", "macao"):
        if not any(k[0] == game for k in runs):
            continue
        table_results(runs, game, baselines, solvable)
        for arm in ("asis", "fixed", "noloop"):
            if any((game, a, arm) in runs for a in AGENT_ORDER):
                fig_train_curves(runs, game, arm, baselines)
                fig_test_comparison(runs, game, arm, baselines)
        fig_ablation(runs, game)

    table_solve_time(baselines, solvable)
    fig_epsilon(runs)
    inject_ablation(runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
