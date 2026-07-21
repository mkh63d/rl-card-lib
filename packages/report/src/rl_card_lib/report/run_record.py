"""Persisted records of a single training run, and the store that holds them.

`TrainingMetrics` writes four parallel per-episode arrays and a summary; that is
enough to plot a learning curve but not enough to *explain* one. A `RunRecord`
is the superset: it adds when the run happened, what the hyperparameters were,
how the agent scored against the baselines before and after, and the
game-specific progress signal (cards to the foundation on Klondike) that the
trainer never captures.

The store keys a run by ``{game}__{agent}``, which is both its identity and its
directory name. There are no timestamped run directories, so re-running a pair
necessarily replaces it -- "keep only the last run of every model" is a property
of the layout rather than a cleanup job that can be forgotten.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

SCHEMA_VERSION = 1

# The per-run file is deliberately not called metrics.json: the repo's
# .gitignore matches that name at any depth, and the store is meant to be
# readable evidence rather than a build artifact.
RUN_FILENAME = "run.json"
METRICS_FILENAME = "metrics.json"

#: Everything that differs between games, in one place. A third game should
#: need a new entry here and nothing else in the reporting layer.
GAME_SPEC: dict[str, dict[str, Any]] = {
    "klondike": {
        "label": "Klondike Solitaire",
        "trainer": "Trainer",
        "headline_key": "cards_up",
        "headline_label": "Cards to foundation",
        "headline_unit": "cards",
        "headline_max": 52.0,
        "headline_format": "{:.1f}",
        # Reward is gameable in principle and cards-up is not, so cards-up
        # stays the headline even when the two agree.
        "episode_curves": ["cards_up"],
        "opponents": [],
        "secondary": ["reward", "win_rate"],
    },
    "macao": {
        "label": "Macao",
        "trainer": "SelfPlayTrainer",
        "headline_key": "win_rate_vs_heuristic",
        "headline_label": "Win rate vs heuristic",
        "headline_unit": "",
        "headline_max": 1.0,
        "headline_format": "{:.1%}",
        # Hand sizes and deck size are per-step state, not a per-episode
        # scalar with a story; the rolling win rate is the progress signal.
        "episode_curves": [],
        "opponents": ["random", "heuristic"],
        "secondary": ["win_rate_vs_random", "reward"],
    },
}

#: Per-episode series a record may carry. `None` means "not recorded", which
#: renderers must distinguish from "recorded as zero".
EPISODE_SERIES = (
    "reward",
    "steps",
    "win",
    "loss",
    "epsilon",
    "wall_clock",
    "cards_up",
    "table_size",
)

AGENT_LABELS = {
    "q_learning": "Q-learning",
    "dqn": "DQN",
    "double_dqn": "Double DQN",
    "ppo": "PPO",
}

# The validated 8-slot categorical palette, in order. Lives here rather than in
# figures.py because the HTML chips need it too and this module carries no
# matplotlib dependency. Slots below 3:1 on the light surface (magenta, yellow,
# aqua) are why every figure ships a legend and a table view.
PALETTE = (
    "#2a78d6",  # 1 blue
    "#008300",  # 2 green
    "#e87ba4",  # 3 magenta
    "#eda100",  # 4 yellow
    "#1baf7a",  # 5 aqua
    "#eb6834",  # 6 orange
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
)
# Past the eight hues the palette cannot stay colourblind-distinct, so further
# agents share this muted ink and lean on their direct labels.
FALLBACK_COLOR = "#898781"

# The four bundled learners keep their historical slots 1-4 so existing figures
# are unchanged. Insertion-ordered so a custom agent takes slot 5 onward, and
# colour follows the agent, never its rank.
AGENT_COLORS = {
    "q_learning": PALETTE[0],
    "dqn": PALETTE[1],
    "double_dqn": PALETTE[2],
    "ppo": PALETTE[3],
}
_ASSIGNED_COLORS = dict(AGENT_COLORS)


def agent_color(agent: str) -> str:
    """A stable colour for an agent, cycling the palette for custom names.

    A custom agent takes the next free palette slot the first time it is seen
    and keeps it thereafter, so two custom agents on one axis are
    distinguishable. Beyond eight, agents share the muted ink.
    """
    if agent in _ASSIGNED_COLORS:
        return _ASSIGNED_COLORS[agent]
    used = set(_ASSIGNED_COLORS.values())
    for hue in PALETTE:
        if hue not in used:
            _ASSIGNED_COLORS[agent] = hue
            return hue
    _ASSIGNED_COLORS[agent] = FALLBACK_COLOR
    return FALLBACK_COLOR


def utc_now() -> str:
    """Timestamp with an explicit offset, so sorting never depends on locale."""
    return datetime.now(timezone.utc).isoformat()


#: What each recorded number actually means. Without this a reader cannot tell
#: whether 0.21 is a rate, a reward or a fraction of something -- the report
#: mixes 0-1 rates, unbounded rewards and counts in the same tables.
#:
#: `kind` drives formatting: "rate" renders as a percentage, "duration" as
#: h/m/s, "count" and "value" as plain numbers.
METRICS: dict[str, dict[str, Any]] = {
    "win_rate": {"label": "Win rate", "kind": "rate", "min": 0.0, "max": 1.0},
    "draw_rate": {"label": "Draw rate", "kind": "rate", "min": 0.0, "max": 1.0},
    "win_rate_vs_random": {"label": "Win rate vs random", "kind": "rate",
                           "min": 0.0, "max": 1.0},
    "win_rate_vs_heuristic": {"label": "Win rate vs heuristic", "kind": "rate",
                              "min": 0.0, "max": 1.0},
    "draw_rate_vs_random": {"label": "Draw rate vs random", "kind": "rate",
                            "min": 0.0, "max": 1.0},
    "draw_rate_vs_heuristic": {"label": "Draw rate vs heuristic", "kind": "rate",
                               "min": 0.0, "max": 1.0},
    "cards_up": {"label": "Cards to foundation", "kind": "count",
                 "min": 0, "max": 52, "unit": "cards"},
    "epsilon": {"label": "Exploration rate", "kind": "rate",
                "min": 0.0, "max": 1.0},
    "table_size": {"label": "Q-table entries", "kind": "count", "min": 0,
                   "unit": "states", "note": "unbounded; grows with training"},
    "reward": {"label": "Episode reward", "kind": "value",
               "note": "shaped; unbounded"},
    "mean_reward": {"label": "Mean reward", "kind": "value",
                    "note": "shaped; unbounded"},
    "std_reward": {"label": "Reward std dev", "kind": "value", "min": 0},
    "min_reward": {"label": "Min reward", "kind": "value"},
    "max_reward": {"label": "Max reward", "kind": "value"},
    "loss": {"label": "Learning loss", "kind": "value", "min": 0,
             "note": "unbounded; diverges when unstable"},
    "steps": {"label": "Episode length", "kind": "count", "min": 0,
              "unit": "steps", "note": "capped by the environment"},
    "mean_steps": {"label": "Mean episode length", "kind": "count", "min": 0,
                   "unit": "steps"},
    "total_episodes": {"label": "Episodes", "kind": "count", "min": 0},
    "total_wins": {"label": "Wins", "kind": "count", "min": 0},
    "training_time": {"label": "Training time", "kind": "duration", "min": 0},
    "wall_clock": {"label": "Episode wall clock", "kind": "duration", "min": 0},
    "seconds": {"label": "Wall clock", "kind": "duration", "min": 0},
    "episode": {"label": "Episode", "kind": "count", "min": 0},
    "win": {"label": "Won", "kind": "count", "min": 0, "max": 1,
            "note": "1 = win, 0 = not"},
}


def metric_spec(key: str) -> dict[str, Any]:
    """Describe one metric, falling back to an unbounded value."""
    if key in METRICS:
        return METRICS[key]
    return {"label": str(key).replace("_", " "), "kind": "value"}


def register_metric(key: str, **fields: Any) -> dict[str, Any]:
    """Declare how a custom metric is measured, so it renders with a scale.

    Mirrors register_game. Without it a custom metric renders with a neutral
    "unbounded" range even when a bound is known::

        register_metric("penalty_points", label="Penalty points",
                        kind="count", min=0, max=26, unit="points")

    Returns the stored spec.
    """
    merged = dict(metric_spec(key))
    merged.update({k: v for k, v in fields.items() if v is not None})
    merged.setdefault("label", str(key).replace("_", " "))
    merged.setdefault("kind", "value")
    METRICS[key] = merged
    return merged


def metric_range(key: str, *, env_max_steps: Optional[int] = None) -> str:
    """Human-readable range for a metric, e.g. '0-100%' or '0-52 cards'.

    Returned as text rather than a pair because it is only ever displayed;
    an empty string means "no meaningful bound", which is itself information.
    """
    spec = metric_spec(key)
    kind = spec.get("kind")
    low, high = spec.get("min"), spec.get("max")

    if key in ("steps", "mean_steps") and env_max_steps:
        return f"0-{env_max_steps} steps"

    if kind == "rate":
        return "0-100%"
    if kind == "duration":
        return "seconds"
    if low is not None and high is not None:
        unit = spec.get("unit", "")
        return f"{low:g}-{high:g}{' ' + unit if unit else ''}"
    if low is not None:
        unit = spec.get("unit", "")
        note = spec.get("note", "")
        base = f"{low:g}+{' ' + unit if unit else ''}"
        return f"{base} ({note})" if note else base
    return spec.get("note", "unbounded")


def format_metric(key: str, value: Any) -> str:
    """Render a value in the units its metric is actually measured in."""
    if value is None:
        return ""
    spec = metric_spec(key)
    kind = spec.get("kind")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return str(value)
    if kind == "rate":
        return f"{value:.1%}"
    if kind == "duration":
        return f"{value:,.1f}s"
    if kind == "count":
        return f"{value:,.0f}" if float(value).is_integer() else f"{value:,.2f}"
    if abs(value) >= 1000 or (value != 0 and abs(value) < 0.01):
        return f"{value:,.4g}"
    return f"{value:,.3f}"


def register_game(name: str, **spec: Any) -> dict[str, Any]:
    """Declare how a custom game should be reported.

    Without this a game the library does not ship is reported with a neutral
    default -- win rate as the headline, no game-specific curve. Registering
    it lets the report name the metric the game is actually judged on::

        register_game(
            "hearts",
            label="Hearts",
            headline_key="penalty_points",
            headline_label="Penalty points",
            headline_max=26,
            higher_is_better=False,          # fewer penalty points is better
            episode_curves=["penalty_points"],
        )

    Returns the stored spec. Unspecified fields keep the neutral defaults. The
    headline metric is auto-registered with its bound, and `higher_is_better`
    (default True) flips the delta colour and the standings order.
    """
    merged = dict(game_spec(name))
    merged.update(spec)
    merged.setdefault("label", name.replace("_", " ").title())
    GAME_SPEC[name] = merged

    # Register the headline metric too, so its declared bound reaches the range
    # column instead of showing "unbounded". The kind is inferred from the
    # format and the presence of a bound: a percent format is a rate, a bound
    # without one is a count, otherwise an unbounded value.
    headline_key = merged.get("headline_key")
    if headline_key and headline_key not in METRICS:
        fmt = merged.get("headline_format", "")
        maximum = merged.get("headline_max")
        if "%" in fmt:
            kind = "rate"
        elif maximum is not None:
            kind = "count"
        else:
            kind = "value"
        register_metric(
            headline_key,
            label=merged.get("headline_label"),
            kind=kind,
            min=0 if kind in ("rate", "count") else None,
            max=maximum if kind != "rate" else 1.0,
            unit=merged.get("headline_unit") or None,
        )
    return merged


def game_spec(game: str) -> dict[str, Any]:
    """Look up a game's reporting spec, falling back to a neutral default."""
    if game in GAME_SPEC:
        return GAME_SPEC[game]
    return {
        "label": game.replace("_", " ").title(),
        "trainer": "Trainer",
        "headline_key": "win_rate",
        "headline_label": "Win rate",
        "headline_unit": "",
        "headline_max": 1.0,
        "headline_format": "{:.1%}",
        "higher_is_better": True,
        "episode_curves": [],
        "opponents": [],
        "secondary": ["reward"],
    }


def agent_label(agent: str) -> str:
    return AGENT_LABELS.get(agent, agent.replace("_", " ").title())


def run_id_for(game: str, agent: str) -> str:
    return f"{game}__{agent}"


def _jsonable(value: Any) -> Any:
    """Coerce numpy scalars and arrays into things `json` will accept.

    `np.float64` subclasses `float` and survives on its own, but `np.int64`
    does not subclass `int` and raises. Rather than guess which producer used
    which, normalize everything on the way out.
    """
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def reconstruct_epsilon(
    *,
    start: float,
    end: float,
    decay: float,
    episodes: int,
) -> list[float]:
    """Rebuild an epsilon schedule that was never recorded.

    Exact for the agents in this library because decay is applied once per
    episode in ``reset()``. Only for legacy records -- a live run measures the
    value instead, so a future schedule change cannot silently invalidate it.
    """
    return [max(end, start * (decay ** episode)) for episode in range(episodes)]


def moving_average(values: Iterable[float], window: int = 100) -> list[float]:
    """Trailing mean, matching `TrainingMetrics.get_moving_average`."""
    data = [float(v) for v in values]
    if not data:
        return []
    out: list[float] = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        out.append(float(np.mean(data[start:i + 1])))
    return out


def _exploration_gap_notes(
    episodes: dict[str, Any], headline: Optional[dict],
) -> list[str]:
    """Flag a large gap between exploratory training and greedy evaluation.

    The learning curve is measured while the agent explores; the headline is
    measured with exploration off. When those disagree sharply the two charts
    look contradictory, and the explanation -- the greedy policy is worse than
    the exploring one -- is a finding rather than a plotting error.
    """
    if not headline or headline.get("source") != "baseline_eval":
        return []
    evaluated = headline.get("after")
    series = episodes.get(headline.get("key"))
    if evaluated is None or not series:
        return []

    tail = series[-min(len(series), 100):]
    trained = float(np.mean(tail))
    if trained <= 0 or evaluated < 0:
        return []

    if evaluated < 0.6 * trained:
        return [
            f"Greedy evaluation ({evaluated:.3g}) scores well below the "
            f"exploratory training average ({trained:.3g}). The learning curve "
            "is measured with exploration on and the headline with it off, so "
            "the two charts are not directly comparable -- and the gap means "
            "the learned greedy policy is worse than the behaviour that "
            "produced it."
        ]
    if trained > 0 and evaluated > 1.6 * trained:
        return [
            f"Greedy evaluation ({evaluated:.3g}) scores well above the "
            f"exploratory training average ({trained:.3g}), as expected when "
            "exploration is still costing the agent episodes."
        ]
    return []


def detect_notes(
    *,
    episodes: dict[str, Any],
    evaluations: Optional[list[dict]] = None,
    env_max_steps: Optional[int] = None,
    headline: Optional[dict] = None,
) -> list[str]:
    """Findings worth stating in words, not just drawing.

    A reader who sees a flat line needs to know whether it is a measurement or
    an artifact, so each of these gets said out loud next to the chart.
    """
    notes: list[str] = []

    losses = episodes.get("loss") or []
    finite = [float(v) for v in losses if np.isfinite(v)]
    nonzero = [abs(v) for v in finite if v != 0.0]
    if nonzero:
        peak = max(nonzero)
        typical = float(np.median(nonzero))
        if typical > 0 and peak / typical > 1e3:
            notes.append(
                f"Loss diverged: peak {peak:.3g} against a median of "
                f"{typical:.3g}. Charted on a symlog axis; not clipped."
            )
    if len(finite) != len(losses):
        notes.append("Loss contains non-finite values; those episodes are gapped.")

    steps = episodes.get("steps") or []
    if steps and env_max_steps:
        capped = sum(1 for s in steps if s >= env_max_steps)
        if capped == len(steps):
            notes.append(
                f"Every episode hit the {env_max_steps}-step cap: no game ever "
                f"ended on its own terms, so episode length carries no signal."
            )
        elif capped > 0.9 * len(steps):
            notes.append(
                f"{capped}/{len(steps)} episodes hit the {env_max_steps}-step cap."
            )

    epsilon = episodes.get("epsilon")
    if epsilon:
        floor = min(epsilon)
        at_floor = [i for i, e in enumerate(epsilon) if e <= floor + 1e-12]
        if at_floor and at_floor[0] < len(epsilon) - 1:
            notes.append(
                f"Exploration reached its floor ({floor:.3g}) at episode "
                f"{at_floor[0]}; the remaining "
                f"{len(epsilon) - at_floor[0]} episodes were near-greedy."
            )

    wins = episodes.get("win") or []
    if wins and sum(wins) == 0:
        notes.append("The agent never won during training.")

    notes.extend(_exploration_gap_notes(episodes, headline))

    for entry in evaluations or []:
        reward_fields = ("mean_reward", "std_reward", "min_reward", "max_reward")
        if all(entry.get(k) == 0.0 for k in reward_fields):
            notes.append(
                "Evaluation reward is exactly zero across all fields. Runs "
                "recorded before the SelfPlayTrainer fix never accumulated "
                "reward outside training; re-measure rather than reinterpret."
            )
            break

    return notes


def _git_commit(cwd: Optional[str] = None) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd, capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - env dependent
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def host_info(seed: Optional[int] = None, device: Optional[str] = None) -> dict:
    """Enough about the machine to explain a timing difference later."""
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "seed": seed,
        "device": device,
        "git_commit": _git_commit(),
    }
    try:  # torch is a core dependency, but the report package must not need it
        import torch

        info["torch"] = torch.__version__
    except Exception:  # pragma: no cover - optional
        info["torch"] = None
    return info


@dataclass
class RunRecord:
    """One training run of one agent on one game."""

    run_id: str
    game: str
    agent: str
    agent_class: str = ""
    label: str = ""
    status: str = "completed"
    schema_version: int = SCHEMA_VERSION
    timestamps: dict[str, Any] = field(default_factory=dict)
    duration: dict[str, Any] = field(default_factory=dict)
    host: dict[str, Any] = field(default_factory=dict)
    config: Optional[dict[str, Any]] = None
    episodes: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    evaluations: list[dict] = field(default_factory=list)
    baseline_comparison: Optional[dict[str, Any]] = None
    headline: Optional[dict[str, Any]] = None
    artifacts: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    # -- construction ----------------------------------------------------

    @classmethod
    def from_training(
        cls,
        *,
        game: str,
        agent: str,
        agent_class: str,
        metrics: Any,
        config: Optional[dict] = None,
        started_at: Optional[str] = None,
        finished_at: Optional[str] = None,
        train_seconds: Optional[float] = None,
        eval_seconds: Optional[float] = None,
        episode_extras: Optional[dict[str, list]] = None,
        baseline_before: Optional[dict] = None,
        baseline_after: Optional[dict] = None,
        artifacts: Optional[dict] = None,
        host: Optional[dict] = None,
        status: str = "completed",
        env_max_steps: Optional[int] = None,
    ) -> "RunRecord":
        """Build a record from a finished `TrainingMetrics` plus side data."""
        extras = dict(episode_extras or {})
        summary = _jsonable(metrics.summary())
        count = len(metrics.rewards)

        episodes: dict[str, Any] = {"count": count}
        episodes["reward"] = _jsonable(list(metrics.rewards))
        episodes["steps"] = _jsonable(list(metrics.steps))
        episodes["win"] = _jsonable(list(metrics.wins))
        episodes["loss"] = _jsonable(list(metrics.losses))

        # Store every series a recorder supplied, not a fixed set: a custom
        # game records its own progress signal (say, penalty_points) exactly
        # the way the bundled games record cards_up. The four known extras keep
        # their keys and their leading position for schema stability; anything
        # else is kept verbatim so register_game(episode_curves=[...]) has
        # data to draw.
        known_extras = ("epsilon", "wall_clock", "cards_up", "table_size")
        for name in (*known_extras, *(k for k in extras if k not in known_extras)):
            values = extras.get(name)
            # A series of all-None (PPO has no epsilon) is the same as absent.
            if not values or all(v is None for v in values):
                episodes[name] = None
            else:
                episodes[name] = _jsonable(list(values))
        episodes["epsilon_source"] = "measured" if episodes.get("epsilon") else None

        evaluations = _jsonable(list(metrics.evaluations))
        spec = game_spec(game)

        record = cls(
            run_id=run_id_for(game, agent),
            game=game,
            agent=agent,
            agent_class=agent_class,
            label=f"{spec['label']} / {agent_label(agent)}",
            status=status,
            timestamps={
                "started_at": started_at,
                "finished_at": finished_at or utc_now(),
                "generated_at": utc_now(),
            },
            duration={
                "train_seconds": train_seconds,
                "eval_seconds": eval_seconds,
                "total_seconds": (
                    None if train_seconds is None
                    else train_seconds + (eval_seconds or 0.0)
                ),
                # Kept alongside our own measurement because the trainer only
                # sets it at the very end of train(); a mismatch is a signal.
                "metrics_training_time": summary.get("training_time"),
            },
            host=host or host_info(),
            config=_jsonable(config) if config else None,
            episodes=episodes,
            summary=summary,
            evaluations=evaluations,
            artifacts=_jsonable(artifacts or {}),
        )
        record.baseline_comparison = record._build_baseline_comparison(
            baseline_before, baseline_after
        )
        record.headline = record._build_headline()
        record.notes = detect_notes(
            episodes=episodes,
            evaluations=evaluations,
            env_max_steps=env_max_steps or record.env_max_steps(),
            headline=record.headline,
        )
        record.validate()
        return record

    @classmethod
    def from_metrics_json(
        cls,
        path: str | Path,
        *,
        game: str,
        agent: str,
        agent_class: str = "",
        epsilon_schedule: Optional[dict] = None,
        env_max_steps: Optional[int] = None,
    ) -> "RunRecord":
        """Adapt a bare `metrics.json` written before records existed.

        Absent sections stay `None` so renderers can say "not recorded" rather
        than invent a zero. The timestamp comes from the file's mtime, which is
        the only evidence of when the run happened.
        """
        path = Path(path)
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        finished = datetime.fromtimestamp(
            path.stat().st_mtime, tz=timezone.utc
        ).isoformat()

        episodes: dict[str, Any] = {"count": len(data.get("rewards", []))}
        episodes["reward"] = data.get("rewards", [])
        episodes["steps"] = data.get("steps", [])
        episodes["win"] = data.get("wins", [])
        episodes["loss"] = data.get("losses", [])
        for name in ("wall_clock", "cards_up", "table_size"):
            episodes[name] = None

        if epsilon_schedule and episodes["count"]:
            episodes["epsilon"] = reconstruct_epsilon(
                episodes=episodes["count"], **epsilon_schedule
            )
            episodes["epsilon_source"] = "reconstructed"
        else:
            episodes["epsilon"] = None
            episodes["epsilon_source"] = None

        summary = data.get("summary", {})
        evaluations = data.get("evaluations", [])
        spec = game_spec(game)

        record = cls(
            run_id=run_id_for(game, agent),
            game=game,
            agent=agent,
            agent_class=agent_class,
            label=f"{spec['label']} / {agent_label(agent)}",
            status="completed",
            timestamps={
                "started_at": None,
                "finished_at": finished,
                "generated_at": utc_now(),
            },
            duration={
                "train_seconds": summary.get("training_time"),
                "eval_seconds": None,
                "total_seconds": summary.get("training_time"),
                "metrics_training_time": summary.get("training_time"),
            },
            host={},
            config=None,
            episodes=episodes,
            summary=summary,
            evaluations=evaluations,
            artifacts={"metrics_json": str(path)},
            notes=["Imported from metrics.json; hyperparameters were not recorded."],
        )
        record.headline = record._build_headline()
        record.notes.extend(detect_notes(
            episodes=episodes,
            evaluations=evaluations,
            env_max_steps=env_max_steps,
        ))
        return record

    @classmethod
    def from_dict(cls, data: dict) -> "RunRecord":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def from_json(cls, path: str | Path) -> "RunRecord":
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    # -- derived views ---------------------------------------------------

    def _build_baseline_comparison(
        self, before: Optional[dict], after: Optional[dict]
    ) -> Optional[dict]:
        if before is None and after is None:
            return None
        spec = game_spec(self.game)
        return {
            "protocol": {"opponents": spec["opponents"]},
            "before": _jsonable(before) if before else None,
            "after": _jsonable(after) if after else None,
        }

    def _build_headline(self) -> Optional[dict]:
        """The one number this game is judged on.

        Three sources, in descending order of trustworthiness. Which one was
        used is recorded, because a training-average win rate and a measured
        win rate against a fixed opponent are not the same claim and must not
        be compared as if they were.
        """
        spec = game_spec(self.game)
        key = spec["headline_key"]
        comparison = self.baseline_comparison or {}
        before = (comparison.get("before") or {}).get(key)
        after = (comparison.get("after") or {}).get(key)
        source = "baseline_eval"
        label = spec["headline_label"]
        fmt = spec["headline_format"]
        maximum = spec["headline_max"]

        # No baseline evaluation (legacy records): the tail of the matching
        # episode series is weaker but still measures the same quantity.
        if after is None:
            series = self.episodes.get(key)
            if series:
                after = float(np.mean(series[-min(len(series), 100):]))
                source = "episode_tail"

        # Still nothing: fall back to the training summary, and relabel so the
        # page cannot present it as the headline metric it is not.
        if before is None and after is None:
            win_rate = self.summary.get("win_rate")
            if win_rate is None:
                return None
            key, after, source = "win_rate", float(win_rate), "training_summary"
            label, fmt, maximum = "Win rate in training", "{:.1%}", 1.0

        return {
            "key": key,
            "label": label,
            "unit": spec["headline_unit"] if source != "training_summary" else "",
            "format": fmt,
            "max": maximum,
            "source": source,
            "before": before,
            "after": after,
            "delta": (
                None if (before is None or after is None) else float(after - before)
            ),
            # A game may declare that lower is better (penalty points); the
            # training-summary fallback is always a win rate, where higher wins.
            "higher_is_better": (
                True if source == "training_summary"
                else spec.get("higher_is_better", True)
            ),
        }

    def series(self, name: str) -> Optional[list]:
        """A per-episode series, or None when it was never recorded."""
        value = self.episodes.get(name)
        return value if value else None

    def moving_average(self, name: str, window: int = 100) -> list[float]:
        values = self.series(name)
        return moving_average(values, window) if values else []

    def rolling_win_rate(self, window: int = 100) -> list[float]:
        return self.moving_average("win", window)

    @property
    def episode_count(self) -> int:
        return int(self.episodes.get("count") or 0)

    @property
    def finished_at(self) -> str:
        return self.timestamps.get("finished_at") or ""

    def config_section(self, name: str) -> Optional[dict]:
        return (self.config or {}).get(name)

    def env_max_steps(self) -> Optional[int]:
        return (self.config_section("environment") or {}).get("max_steps")

    def algorithm_section(self) -> tuple[Optional[str], Optional[dict]]:
        """The hyperparameter block that actually applies to this agent."""
        for name, title in (
            ("dqn", "DQN"), ("ppo", "PPO"),
            ("qlearning", "Q-learning"), ("search", "Search"),
        ):
            section = self.config_section(name)
            if section:
                return title, section
        return None, None

    def validate(self) -> None:
        """Parallel series must stay aligned or every chart lies.

        A recorder callback that raised mid-run would otherwise leave a short
        array silently zipped against a full one.
        """
        count = self.episode_count
        # Every list in the episodes dict is a per-episode series and must be
        # aligned, including custom ones a bundled EPISODE_SERIES cannot name.
        # The non-series keys are the count and the epsilon-source tag.
        reserved = {"count", "epsilon_source"}
        for name, values in self.episodes.items():
            if name in reserved or not isinstance(values, list):
                continue
            if len(values) != count:
                raise ValueError(
                    f"{self.run_id}: episode series {name!r} has "
                    f"{len(values)} entries, expected {count}"
                )

    # -- serialization ---------------------------------------------------

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "game": self.game,
            "agent": self.agent,
            "agent_class": self.agent_class,
            "label": self.label,
            "status": self.status,
            "timestamps": self.timestamps,
            "duration": self.duration,
            "host": self.host,
            "config": self.config,
            "episodes": self.episodes,
            "summary": self.summary,
            "evaluations": self.evaluations,
            "baseline_comparison": self.baseline_comparison,
            "headline": self.headline,
            "artifacts": self.artifacts,
            "notes": self.notes,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(_jsonable(self.as_dict()), indent=indent, default=str)


@dataclass
class BaselineSet:
    """Non-learning agents measured on one game.

    They are evaluated, never trained, so they have no learning curve and are
    rendered as reference marks rather than as peers of the learners.
    """

    game: str
    measured_at: str = ""
    protocol: dict[str, Any] = field(default_factory=dict)
    rows: list[dict] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "BaselineSet":
        return cls(
            game=data.get("game", ""),
            measured_at=data.get("measured_at", ""),
            protocol=data.get("protocol", {}),
            rows=data.get("rows", []),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "BaselineSet":
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def as_dict(self) -> dict[str, Any]:
        return {
            "game": self.game,
            "measured_at": self.measured_at,
            "protocol": self.protocol,
            "rows": self.rows,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(_jsonable(self.as_dict()), indent=indent, default=str)

    def values_for(self, key: str) -> dict[str, float]:
        """Reference levels for one measure, keyed by baseline name.

        Strictly the requested measure: a caller plotting win rate must not be
        handed cards-up values, which would put two units on one axis.
        """
        out: dict[str, float] = {}
        for row in self.rows:
            value = row.get(key)
            if value is not None:
                out[str(row.get("agent", "?"))] = float(value)
        return out

    def headline_values(self) -> dict[str, float]:
        """Reference levels for this game's headline metric."""
        return self.values_for(game_spec(self.game)["headline_key"])


class RunStore:
    """Directory of run records, one per (game, agent).

    Layout::

        <root>/models/<game>__<agent>/run.json
        <root>/models/<game>__<agent>/metrics.json
        <root>/models/<game>__<agent>/figures/
        <root>/baselines/<game>.json
        <root>/figures/

    ``models`` rather than ``runs`` because the repo's .gitignore matches a
    bare ``runs/`` at any depth and would hide the whole store.
    """

    MODELS_DIRNAME = "models"
    BASELINES_DIRNAME = "baselines"
    FIGURES_DIRNAME = "figures"

    def __init__(self, root: str | Path):
        self.root = Path(root)

    # -- paths -----------------------------------------------------------

    @property
    def models_dir(self) -> Path:
        return self.root / self.MODELS_DIRNAME

    @property
    def baselines_dir(self) -> Path:
        return self.root / self.BASELINES_DIRNAME

    @property
    def figures_dir(self) -> Path:
        return self.root / self.FIGURES_DIRNAME

    def run_dir(self, game: str, agent: str) -> Path:
        return self.models_dir / run_id_for(game, agent)

    # -- writing ---------------------------------------------------------

    def reset_run_dir(self, game: str, agent: str) -> Path:
        """Empty a run's directory before it is rewritten.

        Done before training rather than after, so a crash leaves an empty
        directory -- honest -- instead of last run's figures sitting beside
        this run's JSON. Also clears figures that a later version no longer
        emits, which a plain overwrite would leave behind forever.
        """
        target = self.run_dir(game, agent)
        if target.exists():
            shutil.rmtree(target)
        (target / self.FIGURES_DIRNAME).mkdir(parents=True, exist_ok=True)
        return target

    def save_run(self, record: RunRecord) -> Path:
        target = self.models_dir / record.run_id
        target.mkdir(parents=True, exist_ok=True)
        path = target / RUN_FILENAME
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(record.to_json())
        return path

    def save_baselines(self, baselines: BaselineSet) -> Path:
        self.baselines_dir.mkdir(parents=True, exist_ok=True)
        path = self.baselines_dir / f"{baselines.game}.json"
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(baselines.to_json())
        return path

    # -- reading ---------------------------------------------------------

    def load_runs(self) -> list[RunRecord]:
        """Every stored run, most recently finished first.

        Ties break on run_id ascending so the order is deterministic and can
        be asserted in a test.
        """
        if not self.models_dir.exists():
            return []

        records: list[RunRecord] = []
        for entry in sorted(self.models_dir.iterdir()):
            path = entry / RUN_FILENAME
            if not path.is_file():
                # A directory with no run.json is a crashed run, not an error.
                continue
            try:
                records.append(RunRecord.from_json(path))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                print(f"Skipping unreadable run record {path}: {exc}")

        records.sort(key=lambda r: r.run_id)
        records.sort(key=lambda r: r.finished_at, reverse=True)
        return records

    def load_baselines(self) -> dict[str, BaselineSet]:
        if not self.baselines_dir.exists():
            return {}
        out: dict[str, BaselineSet] = {}
        for path in sorted(self.baselines_dir.glob("*.json")):
            try:
                baseline = BaselineSet.from_json(path)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                print(f"Skipping unreadable baseline file {path}: {exc}")
                continue
            out[baseline.game or path.stem] = baseline
        return out

    def has_baselines(self, game: str) -> bool:
        return (self.baselines_dir / f"{game}.json").is_file()


# Names a training run is allowed to leave in a checkpoint directory. Anything
# matching is fair game to delete before a rerun; anything else is left alone.
CHECKPOINT_PATTERNS = (
    "checkpoint_ep*.pt",
    "checkpoint_ep*.pkl",
    "final.pt",
    "final.pkl",
    METRICS_FILENAME,
)


def purge_checkpoints(
    checkpoint_dir: str | Path,
    *,
    game: str,
    agent: str,
    root: Optional[str | Path] = None,
) -> list[str]:
    """Delete a previous run's checkpoints so only the last run survives.

    Deletes by allowlist rather than emptying the directory, and never removes
    the directory itself. Both extensions are matched because
    ``Trainer._save_checkpoint`` hardcodes ``.pt`` even for Q-learning, which
    pickles -- so a Q-learning run leaves ``.pt`` files that torch cannot open.
    """
    target = Path(checkpoint_dir)
    expected = f"{game}_{agent}"
    if target.name != expected:
        raise ValueError(
            f"Refusing to purge {target}: expected a directory named {expected!r}"
        )
    if root is not None:
        root_path = Path(root).resolve()
        if not str(target.resolve()).startswith(str(root_path)):
            raise ValueError(f"Refusing to purge {target}: outside {root_path}")

    if not target.is_dir():
        return []

    removed: list[str] = []
    for pattern in CHECKPOINT_PATTERNS:
        for path in sorted(target.glob(pattern)):
            if path.is_file():
                os.remove(path)
                removed.append(path.name)
    return removed


__all__ = [
    "SCHEMA_VERSION",
    "GAME_SPEC",
    "METRICS",
    "EPISODE_SERIES",
    "RunRecord",
    "RunStore",
    "BaselineSet",
    "agent_label",
    "detect_notes",
    "format_metric",
    "game_spec",
    "metric_range",
    "metric_spec",
    "register_game",
    "register_metric",
    "host_info",
    "moving_average",
    "purge_checkpoints",
    "reconstruct_epsilon",
    "run_id_for",
    "utc_now",
]
