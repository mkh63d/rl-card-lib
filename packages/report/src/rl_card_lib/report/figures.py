"""Charts for a training run, rendered with matplotlib.

Every figure is written to disk as PNG *and* SVG and also carried in memory as
base64, so the HTML page is one portable file while the individual images stay
directly reusable (dragging a PNG into a thesis document, for instance).

matplotlib is an optional extra here. When it is missing this module degrades
the way ``TrainingMetrics.plot()`` does -- print and return nothing -- so the
JSON and Markdown half of the report package keeps working without it.

Colour follows the *agent*, never its rank, so a chart that drops a series does
not repaint the survivors. The four learner hues are slots 1-4 of the validated
categorical order; the non-learning baselines are deliberately not a fifth
through eighth series -- they have no learning curve, and eight categorical hues
cannot clear the contrast floors. They are drawn as muted dashed reference
lines instead, which is both better reading and a smaller palette.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from rl_card_lib.report.run_record import (
    BaselineSet,
    RunRecord,
    agent_label,
    game_spec,
    moving_average,
)

# Validated categorical slots 1-4 (blue, green, magenta, yellow). Magenta and
# yellow sit below 3:1 on the light surface, so every figure ships a legend and
# a table view -- the relief the contrast warning requires.
AGENT_COLORS = {
    "q_learning": "#2a78d6",
    "dqn": "#008300",
    "double_dqn": "#e87ba4",
    "ppo": "#eda100",
}
FALLBACK_COLOR = "#4a3aa7"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
GOOD = "#006300"
CRITICAL = "#d03b3b"

#: Rows kept in a figure's table twin. Enough to read the shape, few enough
#: that the HTML does not carry 5000 rows per chart.
TABLE_ROWS = 200

_STYLE_APPLIED = False


def agent_color(agent: str) -> str:
    return AGENT_COLORS.get(agent, FALLBACK_COLOR)


def charts_available() -> bool:
    """Whether matplotlib can be imported."""
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        return False
    return True


def _pyplot():
    """Import pyplot on the Agg backend, or None with a message.

    Imported lazily and inside the function: the report package's namespace
    __init__ imports eagerly, and matplotlib is only an extra.
    """
    try:
        import matplotlib
    except ImportError:
        print("matplotlib not installed, cannot render figures")
        return None

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _apply_style(plt)
    return plt


def _apply_style(plt) -> None:
    global _STYLE_APPLIED
    if _STYLE_APPLIED:
        return
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "font.size": 9,
        "text.color": INK,
        "axes.labelcolor": INK_SECONDARY,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.titlesize": 11,
        "axes.titleweight": "600",
        "axes.titlecolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        # Solid hairlines: dashing is reserved for baseline reference lines,
        # which genuinely are thresholds rather than decoration.
        "axes.grid": True,
        # Grid behind the marks; otherwise gridlines slice filled bars.
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linestyle": "-",
        "grid.linewidth": 0.6,
        "grid.alpha": 1.0,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "figure.autolayout": False,
    })
    _STYLE_APPLIED = True


@dataclass
class Figure:
    """One rendered chart, plus the numbers behind it."""

    key: str
    title: str
    caption: str = ""
    png_path: Optional[str] = None
    svg_path: Optional[str] = None
    png_base64: str = ""
    table: Optional[dict] = None
    scope: str = "run"

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "title": self.title,
            "caption": self.caption,
            "png_path": self.png_path,
            "svg_path": self.svg_path,
            "table": self.table,
            "scope": self.scope,
        }


@dataclass
class _Emitter:
    """Saves a figure in every requested form and closes it.

    Centralised so no chart can forget `plt.close`; rendering eight runs of
    nine charts leaks otherwise.
    """

    out_dir: Path
    formats: tuple = ("png", "svg")
    dpi: int = 150
    embed: bool = True
    prefix: str = ""
    written: list = field(default_factory=list)

    def emit(self, plt, fig, key, title, caption="", table=None, scope="run"):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{self.prefix}{key}"
        png_path = svg_path = None

        if "png" in self.formats:
            png_path = self.out_dir / f"{stem}.png"
            fig.savefig(png_path, dpi=self.dpi, bbox_inches="tight")
        if "svg" in self.formats:
            svg_path = self.out_dir / f"{stem}.svg"
            fig.savefig(svg_path, bbox_inches="tight")

        encoded = ""
        if self.embed:
            buffer = io.BytesIO()
            fig.savefig(buffer, format="png", dpi=self.dpi, bbox_inches="tight")
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")

        plt.close(fig)
        return Figure(
            key=key, title=title, caption=caption,
            png_path=str(png_path) if png_path else None,
            svg_path=str(svg_path) if svg_path else None,
            png_base64=encoded, table=table, scope=scope,
        )


def _downsample(values: list, limit: int = TABLE_ROWS) -> list[int]:
    """Indices spread evenly across a series, always including the last."""
    n = len(values)
    if n <= limit:
        return list(range(n))
    step = n / float(limit)
    idx = sorted({int(i * step) for i in range(limit)} | {n - 1})
    return idx


def _series_table(columns: list[str], series: list[list], limit=TABLE_ROWS) -> dict:
    """A table twin for a chart, downsampled for the page."""
    if not series or not series[0]:
        return {"columns": columns, "rows": []}
    indices = _downsample(series[0], limit)
    rows = []
    for i in indices:
        row = []
        for column in series:
            value = column[i] if i < len(column) else None
            row.append(None if value is None else _round(value))
        rows.append(row)
    return {"columns": columns, "rows": rows}


def _round(value, places: int = 4):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    if number.is_integer() and abs(number) < 1e15:
        return int(number)
    return round(number, places)


def _episode_axis(ax, count: int) -> None:
    ax.set_xlabel("Episode")
    ax.set_xlim(0, max(1, count - 1))


def _needs_symlog(values: list) -> tuple[bool, float]:
    """Whether a series spans too many orders of magnitude for a linear axis.

    Clipping would hide a divergence, which is a result; a symlog axis shows
    it and stays readable.
    """
    finite = [abs(float(v)) for v in values if np.isfinite(v)]
    nonzero = [v for v in finite if v > 0]
    if not nonzero:
        return False, 1.0
    peak, typical = max(nonzero), float(np.median(nonzero))
    if typical > 0 and peak / typical > 1e3:
        return True, max(typical, 1e-6)
    return False, 1.0


def _label_right_edge(ax, entries: list, *, fontsize=8, min_gap=0.055) -> None:
    """Direct-label series at the right edge, pushed apart so they stay legible.

    Series that converge -- three learners all pinned at a 0% win rate, which
    is the common case here -- would otherwise stack their labels into an
    unreadable smear. Positions are nudged in axes fraction and the label is
    tethered to its true value by a short leader when it has moved.

    entries: (text, value, colour) tuples.
    """
    if not entries:
        return

    bottom, top = ax.get_ylim()
    span = (top - bottom) or 1.0

    placed = sorted(
        ((text, value, colour, (value - bottom) / span) for text, value, colour in entries),
        key=lambda item: item[3],
    )

    # Push upward from the lowest, then correct any overflow downward, so the
    # stack stays inside the axes even when everything converges.
    fractions: list[float] = []
    previous = -1.0
    for _, _, _, fraction in placed:
        fraction = max(fraction, previous + min_gap)
        fractions.append(fraction)
        previous = fraction

    overflow = fractions[-1] - 1.0
    if overflow > 0:
        fractions = [f - overflow for f in fractions]
        for i in range(len(fractions) - 2, -1, -1):
            fractions[i] = min(fractions[i], fractions[i + 1] - min_gap)

    for (text, value, colour, natural), fraction in zip(placed, fractions):
        ax.annotate(
            text, xy=(1.0, fraction), xycoords="axes fraction",
            xytext=(6, 0), textcoords="offset points",
            va="center", ha="left", fontsize=fontsize, color=colour,
            annotation_clip=False,
        )
        if abs(fraction - natural) > 0.01:
            ax.annotate(
                "", xy=(1.0, natural), xycoords="axes fraction",
                xytext=(1.0, fraction), textcoords="axes fraction",
                arrowprops={
                    "arrowstyle": "-", "color": colour,
                    "alpha": 0.45, "linewidth": 0.7,
                },
                annotation_clip=False,
            )


def _draw_baselines(ax, levels: dict, xmax: float) -> None:
    """Reference lines for the non-learning agents, labelled at the right edge."""
    if not levels:
        return
    for value in levels.values():
        ax.axhline(
            value, color=MUTED, linestyle="--", linewidth=1.0, alpha=0.9, zorder=1,
        )
    _label_right_edge(
        ax, [(name, value, MUTED) for name, value in levels.items()], fontsize=7,
    )


def _raw_and_average(ax, values, color, label, window=100):
    """Raw series faint, trailing average bold -- the shape plus the trend."""
    ax.plot(values, color=color, alpha=0.25, linewidth=1.0, zorder=2)
    smooth = moving_average(values, window)
    ax.plot(smooth, color=color, linewidth=2.0, label=label, zorder=3)
    return smooth


# ---------------------------------------------------------------------------
# Per-run figures
# ---------------------------------------------------------------------------

def render_run_figures(
    record: RunRecord,
    out_dir: str | Path,
    *,
    baselines: Optional[BaselineSet] = None,
    formats: tuple = ("png", "svg"),
    dpi: int = 150,
    embed: bool = True,
) -> list[Figure]:
    """Every chart that applies to one run.

    Charts whose data was never recorded are skipped rather than drawn empty --
    PPO has no epsilon, Macao has no cards-up.
    """
    plt = _pyplot()
    if plt is None:
        return []

    emitter = _Emitter(Path(out_dir), formats, dpi, embed)
    spec = game_spec(record.game)
    colour = agent_color(record.agent)
    window = max(5, min(100, max(1, record.episode_count // 10)))
    figures: list[Figure] = []

    for builder in (
        _fig_headline_curve, _fig_reward, _fig_win_rate, _fig_loss,
        _fig_steps, _fig_epsilon, _fig_table_size, _fig_evaluation,
        _fig_before_after,
    ):
        figure = builder(
            plt, emitter, record,
            colour=colour, window=window, spec=spec, baselines=baselines,
        )
        if figure is not None:
            figures.append(figure)

    # Any declared episode curve that is not the headline (which the builder
    # above already drew). This is what makes register_game(episode_curves=[...])
    # reach the page for a custom game's own progress signal.
    headline_key = spec.get("headline_key")
    for key in spec.get("episode_curves", []):
        if key == headline_key:
            continue
        figure = _fig_episode_curve(plt, emitter, record, key, colour, window)
        if figure is not None:
            figures.append(figure)

    return figures


def _fig_episode_curve(plt, emitter, record, key, colour, window):
    """A game-declared per-episode curve, generically drawn from its metric spec."""
    from rl_card_lib.report.run_record import metric_spec

    values = record.series(key)
    if not values:
        return None

    label = metric_spec(key)["label"]
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    smooth = _raw_and_average(ax, values, colour, f"Avg({window})", window)
    ax.set_ylabel(label)
    ax.set_title(f"{label} per training episode (exploring)")
    _episode_axis(ax, len(values))
    ax.legend(loc="upper left")

    return emitter.emit(
        plt, fig, key, label,
        caption=f"Per-episode {label.lower()} with a {window}-episode trailing average.",
        table=_series_table(["Episode", label, f"Avg({window})"],
                            [list(range(len(values))), values, smooth]),
    )


def _fig_headline_curve(plt, emitter, record, *, colour, window, spec, baselines):
    """Klondike's cards-to-foundation, the metric no reward change can game."""
    key = spec["headline_key"]
    values = record.series(key)
    if not values:
        return None

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    _raw_and_average(ax, values, colour, f"{agent_label(record.agent)} (avg {window})", window)

    ceiling = spec.get("headline_max")
    if ceiling:
        ax.axhline(ceiling, color=AXIS, linewidth=1.0, alpha=0.8)
        ax.annotate(
            f"all {int(ceiling)}", xy=(0, ceiling), xytext=(2, -10),
            textcoords="offset points", fontsize=7, color=MUTED,
        )
        ax.set_ylim(0, ceiling * 1.05)

    ax.set_ylabel(spec["headline_label"])
    ax.set_title(f"{spec['headline_label']} per training episode (exploring)")
    _episode_axis(ax, len(values))
    ax.legend(loc="upper left")

    # Placed last: the label layout needs the final y-limits to convert values
    # into axes fractions.
    _draw_baselines(ax, baselines.headline_values() if baselines else {},
                    len(values) - 1)

    return emitter.emit(
        plt, fig, key, spec["headline_label"],
        caption=(
            f"Raw per-episode value with a {window}-episode trailing average, "
            "measured during training with exploration on. Dashed lines are "
            "the non-learning baselines, which are measured greedily."
        ),
        table=_series_table(["Episode", spec["headline_label"], f"Avg({window})"],
                            [list(range(len(values))), values,
                             moving_average(values, window)]),
    )


def _fig_reward(plt, emitter, record, *, colour, window, spec, **_):
    values = record.series("reward")
    if not values:
        return None

    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    smooth = _raw_and_average(ax, values, colour, f"Avg({window})", window)
    ax.set_ylabel("Episode reward")
    ax.set_title("Reward per episode")
    _episode_axis(ax, len(values))
    ax.legend(loc="upper left")

    return emitter.emit(
        plt, fig, "reward", "Reward",
        caption=f"Shaped reward per episode with a {window}-episode trailing average.",
        table=_series_table(["Episode", "Reward", f"Avg({window})"],
                            [list(range(len(values))), values, smooth]),
    )


def _fig_win_rate(plt, emitter, record, *, colour, window, **_):
    values = record.series("win")
    if not values:
        return None

    rate = moving_average(values, window)
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.plot(rate, color=colour, linewidth=2.0, label=f"Avg({window})")
    ax.set_ylabel("Win rate")
    ax.set_ylim(-0.02, max(0.05, max(rate) * 1.15 if rate else 0.05))
    ax.set_title("Rolling win rate")
    _episode_axis(ax, len(values))
    ax.legend(loc="upper left")

    total = int(sum(values))
    caption = f"{window}-episode trailing win rate. {total} win(s) in {len(values)} episodes."
    if total == 0:
        caption += " The flat line is the measurement, not a missing series."

    return emitter.emit(
        plt, fig, "win_rate", "Win rate", caption=caption,
        table=_series_table(["Episode", "Win", f"Rate({window})"],
                            [list(range(len(values))), values, rate]),
    )


def _fig_loss(plt, emitter, record, *, colour, window, **_):
    values = record.series("loss")
    if not values or all(v == 0 for v in values):
        return None

    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.plot(values, color=colour, alpha=0.35, linewidth=1.0)
    smooth = moving_average(values, window)
    ax.plot(smooth, color=colour, linewidth=2.0, label=f"Avg({window})")

    caption = f"Mean learning loss per episode, {window}-episode trailing average."
    diverged, linthresh = _needs_symlog(values)
    if diverged:
        ax.set_yscale("symlog", linthresh=linthresh)
        caption += (
            " Symlog axis: the loss spans more than three orders of magnitude. "
            "Not clipped -- the divergence is the finding."
        )

    ax.set_ylabel("Loss")
    ax.set_title("Learning loss" + (" (symlog)" if diverged else ""))
    _episode_axis(ax, len(values))
    ax.legend(loc="upper left")

    return emitter.emit(
        plt, fig, "loss", "Loss", caption=caption,
        table=_series_table(["Episode", "Loss", f"Avg({window})"],
                            [list(range(len(values))), values, smooth]),
    )


def _fig_steps(plt, emitter, record, *, colour, window, **_):
    """Episode length exposes step-cap saturation that reward alone conceals."""
    values = record.series("steps")
    if not values:
        return None

    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    smooth = _raw_and_average(ax, values, colour, f"Avg({window})", window)

    caption = f"Episode length with a {window}-episode trailing average."
    cap = record.env_max_steps()
    if cap:
        ax.axhline(cap, color=MUTED, linestyle="--", linewidth=1.0)
        ax.annotate(
            f"cap {cap}", xy=(0, cap), xytext=(2, -10),
            textcoords="offset points", fontsize=7, color=MUTED,
        )
        capped = sum(1 for v in values if v >= cap)
        if capped:
            caption += f" {capped}/{len(values)} episodes hit the {cap}-step cap."

    ax.set_ylabel("Steps")
    ax.set_title("Episode length")
    _episode_axis(ax, len(values))
    ax.legend(loc="lower left")

    return emitter.emit(
        plt, fig, "steps", "Episode length", caption=caption,
        table=_series_table(["Episode", "Steps", f"Avg({window})"],
                            [list(range(len(values))), values, smooth]),
    )


def _fig_epsilon(plt, emitter, record, *, colour, **_):
    """Skipped entirely for PPO, which explores by sampling instead."""
    values = record.series("epsilon")
    if not values:
        return None

    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    ax.plot(values, color=colour, linewidth=2.0, label="Epsilon")
    ax.set_ylabel("Epsilon")
    ax.set_ylim(0, max(values) * 1.08)
    source = record.episodes.get("epsilon_source") or "recorded"
    ax.set_title(f"Exploration schedule ({source})")
    _episode_axis(ax, len(values))
    ax.legend(loc="upper right")

    return emitter.emit(
        plt, fig, "epsilon", "Exploration",
        caption=(
            "Probability of a random action, decayed once per episode. "
            f"Source: {source}."
        ),
        table=_series_table(["Episode", "Epsilon"],
                            [list(range(len(values))), values]),
    )


def _fig_table_size(plt, emitter, record, *, colour, **_):
    """Tabular Q-learning's cost: one entry per distinct state, forever."""
    values = record.series("table_size")
    if not values:
        return None

    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    ax.plot(values, color=colour, linewidth=2.0, label="Distinct states")
    ax.set_ylabel("Q-table entries")
    ax.set_title("Q-table growth")
    _episode_axis(ax, len(values))
    ax.legend(loc="upper left")

    per_episode = values[-1] / max(1, len(values))
    return emitter.emit(
        plt, fig, "table_size", "Q-table growth",
        caption=(
            f"{values[-1]:,} distinct states after {len(values)} episodes "
            f"(~{per_episode:,.0f} new per episode). Near-linear growth means "
            "the table is memorising positions rather than generalising."
        ),
        table=_series_table(["Episode", "Entries"],
                            [list(range(len(values))), values]),
    )


def _fig_evaluation(plt, emitter, record, *, colour, **_):
    """Two stacked panels sharing x -- never a second y-axis."""
    evaluations = record.evaluations
    if not evaluations:
        return None

    episodes = [e.get("episode", i) for i, e in enumerate(evaluations)]
    mean = [e.get("mean_reward", 0.0) for e in evaluations]
    std = [e.get("std_reward", 0.0) for e in evaluations]
    win = [e.get("win_rate", 0.0) for e in evaluations]
    marker = "o" if len(evaluations) < 5 else None

    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(7.2, 4.4), sharex=True,
        gridspec_kw={"height_ratios": [3, 2]},
    )
    top.plot(episodes, mean, color=colour, marker=marker, label="Mean reward")
    top.fill_between(
        episodes,
        [m - s for m, s in zip(mean, std)],
        [m + s for m, s in zip(mean, std)],
        color=colour, alpha=0.15, linewidth=0,
    )
    top.set_ylabel("Mean reward")
    top.set_title("Evaluation over training")
    top.legend(loc="upper left")

    bottom.plot(episodes, win, color=colour, marker=marker, label="Win rate")
    bottom.set_ylabel("Win rate")
    bottom.set_xlabel("Episode")
    bottom.set_ylim(-0.02, max(0.05, max(win) * 1.15 if win else 0.05))
    bottom.legend(loc="upper left")

    caption = "Periodic greedy evaluation. Band is ±1 standard deviation."
    if all(v == 0.0 for v in mean):
        caption += (
            " Reward is exactly zero throughout: this run predates the "
            "SelfPlayTrainer fix and never accumulated reward outside training."
        )

    return emitter.emit(
        plt, fig, "evaluation", "Evaluation", caption=caption,
        table={
            "columns": ["Episode", "Mean reward", "Std", "Win rate"],
            "rows": [[e, _round(m), _round(s), _round(w)]
                     for e, m, s, w in zip(episodes, mean, std, win)],
        },
    )


def _fig_before_after(plt, emitter, record, *, colour, **_):
    """What training actually bought, measured on fixed deals."""
    comparison = record.baseline_comparison or {}
    before, after = comparison.get("before"), comparison.get("after")
    if not before or not after:
        return None

    keys = [k for k in after if isinstance(after.get(k), (int, float))]
    keys = [k for k in keys if isinstance(before.get(k), (int, float))]
    if not keys:
        return None

    positions = np.arange(len(keys))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.bar(positions - width / 2 - 0.01, [before[k] for k in keys], width,
           color=MUTED, label="Before training")
    ax.bar(positions + width / 2 + 0.01, [after[k] for k in keys], width,
           color=colour, label="After training")

    for i, key in enumerate(keys):
        delta = after[key] - before[key]
        ax.annotate(
            f"{'▲' if delta >= 0 else '▼'} {delta:+.3g}",
            xy=(i, max(before[key], after[key])), xytext=(0, 5),
            textcoords="offset points", ha="center", fontsize=8,
            color=GOOD if delta >= 0 else CRITICAL,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels([k.replace("_", " ") for k in keys])
    ax.set_title("Before and after training")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")

    return emitter.emit(
        plt, fig, "before_after", "Before / after",
        caption="Measured on the same fixed deals before and after training.",
        table={
            "columns": ["Metric", "Before", "After", "Delta"],
            "rows": [[k, _round(before[k]), _round(after[k]),
                      _round(after[k] - before[k])] for k in keys],
        },
    )


# ---------------------------------------------------------------------------
# Cross-model comparison figures
# ---------------------------------------------------------------------------

def render_comparison_figures(
    records: list[RunRecord],
    baselines: dict[str, BaselineSet],
    out_dir: str | Path,
    *,
    formats: tuple = ("png", "svg"),
    dpi: int = 150,
    embed: bool = True,
) -> dict[str, list[Figure]]:
    """Comparison charts per game, keyed by game name."""
    plt = _pyplot()
    if plt is None:
        return {}

    by_game: dict[str, list[RunRecord]] = {}
    for record in records:
        by_game.setdefault(record.game, []).append(record)

    out: dict[str, list[Figure]] = {}
    for game, group in by_game.items():
        emitter = _Emitter(Path(out_dir), formats, dpi, embed, prefix=f"{game}_")
        figures = []
        for builder in (
            _cmp_curves, _cmp_headline, _cmp_efficiency, _cmp_evaluation,
        ):
            figure = builder(plt, emitter, game, group, baselines.get(game))
            if figure is not None:
                figures.append(figure)
        if figures:
            out[game] = figures
    return out


def _cmp_curves(plt, emitter, game, records, baselines):
    """Every learner's progress on one axis -- the chart the thesis wants."""
    spec = game_spec(game)
    key = spec["headline_key"]

    series = []
    for record in records:
        values = record.series(key) or record.series("win")
        label = agent_label(record.agent)
        if values:
            series.append((record, label, values, key if record.series(key) else "win"))
    if not series:
        return None

    # If any run lacks the headline series we fall back to win rate for all of
    # them, and the reference lines must follow -- otherwise the axis would
    # carry cards-up baselines against a win-rate curve.
    kinds = {kind for _, _, _, kind in series}
    plotting_headline = kinds == {key}
    measure = spec["headline_label"] if plotting_headline else "Rolling win rate"
    baseline_key = key if plotting_headline else "win_rate"

    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    longest = max(len(values) for _, _, values, _ in series)
    window = max(5, min(100, max(1, longest // 10)))

    endpoints = []
    for record, label, values, _ in series:
        smooth = moving_average(values, window)
        colour = agent_color(record.agent)
        ax.plot(smooth, color=colour, linewidth=2.0, label=label)
        endpoints.append((label, smooth[-1], colour))

    levels = baselines.values_for(baseline_key) if baselines else {}
    for value in levels.values():
        ax.axhline(value, color=MUTED, linestyle="--", linewidth=1.0, zorder=1)

    ax.set_ylabel(f"{measure} (exploring)")
    ax.set_xlabel("Episode")
    ax.set_xlim(0, longest - 1)
    # "while exploring" is not decoration: these are training episodes, so the
    # values are not comparable with the greedy-evaluation headline chart.
    ax.set_title(f"{spec['label']}: {measure.lower()} while exploring")
    # Below the axes: with reference lines near the top a corner legend
    # collides with them, and the direct labels already carry identity.
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.16),
        ncol=min(4, len(series)), frameon=False,
    )

    # After the limits are settled, so the fractions are computed against the
    # axes the labels will actually be drawn on. Baselines join the same
    # layout pass; laying them out separately would let the two sets collide.
    _label_right_edge(
        ax,
        endpoints + [(name, value, MUTED) for name, value in levels.items()],
    )

    columns = ["Episode"] + [label for _, label, _, _ in series]
    smoothed = [moving_average(values, window) for _, _, values, _ in series]
    return emitter.emit(
        plt, fig, "comparison_curves", f"{spec['label']}: learning curves",
        caption=(
            f"{window}-episode trailing average per learner, measured during "
            "training with exploration on. Not comparable with the "
            "greedy-evaluation figures below. Dashed lines are the "
            "non-learning baselines, which have no learning curve."
        ),
        table=_series_table(columns, [list(range(longest))] + smoothed),
        scope="comparison",
    )


def _cmp_headline(plt, emitter, game, records, baselines):
    """Learners then baselines, separated -- not one undifferentiated ranking."""
    spec = game_spec(game)
    learners = [
        (agent_label(r.agent), (r.headline or {}).get("after"), agent_color(r.agent))
        for r in records if (r.headline or {}).get("after") is not None
    ]
    if not learners:
        return None

    learners.sort(key=lambda row: row[1])
    reference = [
        (name, value, MUTED)
        for name, value in sorted(
            (baselines.headline_values() if baselines else {}).items(),
            key=lambda kv: kv[1],
        )
    ]

    rows = reference + [("", None, None)] + learners if reference else learners
    labels = [r[0] for r in rows]
    values = [0 if r[1] is None else r[1] for r in rows]
    colours = [r[2] or SURFACE for r in rows]

    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(rows) + 1.4))
    positions = np.arange(len(rows))
    ax.barh(positions, values, color=colours, height=0.68)
    for i, row in enumerate(rows):
        if row[1] is None:
            continue
        ax.annotate(
            spec["headline_format"].format(row[1]),
            xy=(row[1], i), xytext=(4, 0), textcoords="offset points",
            va="center", fontsize=8, color=INK_SECONDARY,
        )

    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel(f"{spec['headline_label']} (greedy)")
    ax.set_title(
        f"{spec['label']}: {spec['headline_label'].lower()}, greedy evaluation"
    )
    ax.grid(axis="y", visible=False)

    return emitter.emit(
        plt, fig, "comparison_headline", f"{spec['label']}: final standing",
        caption=(
            "Measured after training with exploration off, on fixed deals. "
            "Trained learners in colour, non-learning baselines in grey -- the "
            "baselines play at full strength immediately and are the bar. "
            "These values are lower than the learning curves above, which are "
            "measured while exploring."
        ),
        table={
            "columns": ["Agent", spec["headline_label"], "Kind"],
            "rows": [[r[0], _round(r[1]), "baseline" if r[2] == MUTED else "learner"]
                     for r in rows if r[1] is not None],
        },
        scope="comparison",
    )


def _cmp_efficiency(plt, emitter, game, records, baselines):
    """What each learner cost in wall clock for what it achieved."""
    points = [
        (agent_label(r.agent),
         (r.duration or {}).get("train_seconds"),
         (r.headline or {}).get("after"),
         agent_color(r.agent))
        for r in records
    ]
    points = [p for p in points if p[1] and p[2] is not None]
    if len(points) < 2:
        return None

    spec = game_spec(game)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for label, seconds, value, colour in points:
        ax.scatter([seconds], [value], s=90, color=colour, zorder=3)
        ax.annotate(
            label, xy=(seconds, value), xytext=(6, 4),
            textcoords="offset points", fontsize=8, color=INK_SECONDARY,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Training wall clock (s, log scale)")
    ax.set_ylabel(spec["headline_label"])
    ax.set_title(f"{spec['label']}: cost against result")

    return emitter.emit(
        plt, fig, "comparison_efficiency", f"{spec['label']}: cost vs result",
        caption="Every point is labelled directly; colour repeats the learner's slot.",
        table={
            "columns": ["Agent", "Train seconds", spec["headline_label"]],
            "rows": [[p[0], _round(p[1], 1), _round(p[2])] for p in points],
        },
        scope="comparison",
    )


def _cmp_evaluation(plt, emitter, game, records, baselines):
    """Small multiples: one panel per learner, shared axes."""
    usable = [r for r in records if r.evaluations]
    if len(usable) < 2:
        return None

    spec = game_spec(game)
    columns = 2
    rows = (len(usable) + columns - 1) // columns
    fig, axes = plt.subplots(
        rows, columns, figsize=(7.6, 2.4 * rows), sharex=True, sharey=True,
        squeeze=False,
    )
    flat = [ax for row in axes for ax in row]

    for ax, record in zip(flat, usable):
        episodes = [e.get("episode", i) for i, e in enumerate(record.evaluations)]
        win = [e.get("win_rate", 0.0) for e in record.evaluations]
        ax.plot(episodes, win, color=agent_color(record.agent),
                marker="o" if len(episodes) < 5 else None, linewidth=2.0)
        ax.set_title(agent_label(record.agent), fontsize=9)
        ax.set_ylim(-0.02, None)
    for ax in flat[len(usable):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Episode")
    for row in axes:
        row[0].set_ylabel("Win rate")

    fig.suptitle(f"{spec['label']}: evaluation win rate", y=1.0, fontsize=11)

    return emitter.emit(
        plt, fig, "comparison_evaluation", f"{spec['label']}: evaluation panels",
        caption="Shared axes, so panel heights are directly comparable.",
        table={
            "columns": ["Agent", "Episode", "Win rate"],
            "rows": [[agent_label(r.agent), e.get("episode"), _round(e.get("win_rate"))]
                     for r in usable for e in r.evaluations],
        },
        scope="comparison",
    )


__all__ = [
    "AGENT_COLORS",
    "Figure",
    "agent_color",
    "charts_available",
    "render_comparison_figures",
    "render_run_figures",
]
