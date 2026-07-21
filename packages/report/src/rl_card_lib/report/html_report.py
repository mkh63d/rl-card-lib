"""A single self-contained HTML page describing every stored training run.

Built by string assembly, mirroring `TrainingReport.to_markdown()`, because
adding a templating dependency for one page would be the tail wagging the dog
-- and because the page has to render with nothing but the standard library
available.

The output has no external references at all: no CDN, no sibling stylesheet,
no remote font, no linked image. Figures are embedded as base64 data URIs, so
the file can be mailed, moved or archived on its own and still render.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from rl_card_lib.report.assets import CSS, JS
from rl_card_lib.report.run_record import (
    BaselineSet,
    RunRecord,
    RunStore,
    agent_color,
    format_metric,
    game_spec,
    metric_range,
    metric_spec,
    utc_now,
)

TITLE = "RL Card Library - training report"
NOT_RECORDED = '<span class="none">not recorded</span>'


def _escape(value: Any) -> str:
    """Escape everything interpolated into the page.

    Config sections carry filesystem paths and labels can come from the command
    line, so nothing reaches the document unescaped.
    """
    if value is None:
        return ""
    return html.escape(str(value), quote=True)


def _slug(value: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in str(value))


def _fmt(value, spec: str = "{:.3f}", dash: str = NOT_RECORDED) -> str:
    if value is None:
        return dash
    if isinstance(value, str):
        return _escape(value)
    try:
        return _escape(spec.format(value))
    except (ValueError, TypeError):
        return _escape(value)


def _fmt_seconds(value) -> str:
    if value is None:
        return NOT_RECORDED
    seconds = float(value)
    if seconds < 90:
        return f"{seconds:.1f}s"
    minutes, rest = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {int(rest)}s"
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours}h {minutes}m"


def _fmt_timestamp(value) -> str:
    """Readable local-ish rendering, with the raw ISO value in the tooltip."""
    if not value:
        return NOT_RECORDED
    try:
        moment = datetime.fromisoformat(str(value))
    except ValueError:
        return _escape(value)
    return (
        f'<span title="{_escape(value)}">'
        f'{_escape(moment.strftime("%Y-%m-%d %H:%M"))}</span>'
    )


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _table(
    name: str,
    columns: list,
    rows: list[list],
    *,
    caption: str = "",
    exportable: bool = True,
) -> str:
    """A table with its export toolbar.

    Every table in the report is exportable; CSV is the guaranteed path, PNG
    is best-effort, and Copy emits TSV for pasting straight into a document.
    """
    out = [f'<div class="table-block" data-name="{_escape(name)}">']
    if exportable:
        out.append(
            '<div class="toolbar">'
            '<button type="button" data-export="copy" title="Copy as TSV">Copy</button>'
            '<button type="button" data-export="csv" title="Download CSV">CSV</button>'
            '<button type="button" data-export="png" title="Download PNG">PNG</button>'
            "</div>"
        )
    out.append('<div class="table-scroll"><table>')
    if caption:
        out.append(f"<caption>{caption}</caption>")

    out.append("<thead><tr>")
    numeric_columns = [
        any(_is_number(row[i]) for row in rows if i < len(row))
        for i in range(len(columns))
    ]
    for index, column in enumerate(columns):
        css = ' class="num"' if numeric_columns[index] else ""
        out.append(f"<th{css}>{_escape(column)}</th>")
    out.append("</tr></thead><tbody>")

    for row in rows:
        out.append("<tr>")
        for index, cell in enumerate(row):
            if cell is None:
                out.append(f'<td class="none">{NOT_RECORDED}</td>')
            elif isinstance(cell, str) and cell.startswith("<"):
                # Pre-rendered markup (chips, deltas, timestamps).
                css = ' class="num"' if numeric_columns[index] else ""
                out.append(f"<td{css}>{cell}</td>")
            else:
                css = ' class="num"' if _is_number(cell) else ""
                text = f"{cell:,.4g}" if _is_number(cell) else _escape(cell)
                out.append(f"<td{css}>{text}</td>")
        out.append("</tr>")

    out.append("</tbody></table></div></div>")
    return "\n".join(out)


def _column_header(key: str, env_max_steps=None) -> str:
    """A column header that states the metric's range, e.g. 'Win rate (0-100%)'."""
    label = metric_spec(key)["label"]
    span = metric_range(key, env_max_steps=env_max_steps)
    return f"{label} ({span})" if span else label


def _agent_chip(record: RunRecord) -> str:
    # The same palette-cycling colour the figures use, so a custom agent's chip
    # and its curve match. Inline hex rather than a CSS var, because a custom
    # agent has no var.
    colour = agent_color(record.agent)
    return (
        f'<span class="chip swatch" style="--dot:{colour}">'
        f"{_escape(record.agent_display)}</span>"
    )


def _delta_markup(delta, spec: str = "{:+.3g}", *, higher_is_better: bool = True) -> str:
    if delta is None:
        return NOT_RECORDED
    # The arrow shows the direction of change; the colour shows whether that
    # change is an improvement, which depends on the metric's polarity.
    rising = delta >= 0
    good = rising if higher_is_better else not rising
    css = "delta-up" if good else "delta-down"
    arrow = "▲" if rising else "▼"
    return f'<span class="{css}">{arrow} {_escape(spec.format(delta))}</span>'


def _headline_markup(record: RunRecord) -> str:
    headline = record.headline
    if not headline or headline.get("after") is None:
        return NOT_RECORDED
    spec = headline.get("format", "{:.3f}")
    text = _fmt(headline["after"], spec)
    if headline.get("before") is not None:
        text = f'{_fmt(headline["before"], spec)} → {text}'
    return text


def _figure_block(figure, *, table_name: str) -> str:
    """One chart: the image, its export links, and its numbers underneath."""
    out = ['<figure>']
    out.append('<div class="fig-head">')
    out.append(f"<h4>{_escape(figure.title)}</h4>")

    links = []
    stem = _escape(Path(figure.png_path).name) if figure.png_path else None
    if figure.png_base64:
        links.append(
            f'<a class="btn" download="{stem or figure.key + ".png"}" '
            f'href="data:image/png;base64,{figure.png_base64}">PNG</a>'
        )
    elif figure.png_path:
        links.append(f'<a class="btn" download href="{stem}">PNG</a>')
    if figure.svg_path:
        svg_name = _escape(Path(figure.svg_path).name)
        links.append(f'<a class="btn" download href="figures/{svg_name}">SVG</a>')
    if links:
        out.append(f'<span class="toolbar">{"".join(links)}</span>')
    out.append("</div>")

    if figure.png_base64:
        out.append(
            f'<img alt="{_escape(figure.title)}" '
            f'src="data:image/png;base64,{figure.png_base64}">'
        )
    elif figure.png_path:
        out.append(
            f'<img alt="{_escape(figure.title)}" src="{stem}">'
        )

    if figure.caption:
        out.append(f"<figcaption>{_escape(figure.caption)}</figcaption>")

    table = figure.table
    if table and table.get("rows"):
        out.append("<details><summary>Table view</summary>")
        out.append(_table(table_name, table["columns"], table["rows"]))
        out.append("</details>")

    out.append("</figure>")
    return "\n".join(out)


def _merge_reference(
    runs: list[RunRecord], baselines: dict, reference_store: RunStore,
) -> tuple[list[RunRecord], dict]:
    """Fold committed library runs/baselines in as comparison points.

    Only games the user is already reporting on gain reference data, and only
    runs the user did not produce themselves (their own run of an agent wins).
    The reference store holds nothing but the bundled example games, so a
    custom game -- absent from it -- silently gets no reference and no game
    name is special-cased here.
    """
    games_present = {r.game for r in runs}
    if not games_present:
        return runs, baselines

    have = {r.run_id for r in runs}
    for record in reference_store.load_runs():
        if record.game not in games_present or record.run_id in have:
            continue
        record.reference = True
        runs.append(record)

    for game, baseline in reference_store.load_baselines().items():
        if game in games_present and game not in baselines:
            baselines[game] = baseline

    # The user's own runs first, then the library reference; each group stays
    # newest-first. Stable sort, so the reference key is applied last.
    runs.sort(key=lambda r: r.run_id)
    runs.sort(key=lambda r: r.finished_at, reverse=True)
    runs.sort(key=lambda r: r.reference)
    return runs, baselines


@dataclass
class HtmlReport:
    """The rendered page and everything it was built from."""

    runs: list[RunRecord] = field(default_factory=list)
    baselines: dict = field(default_factory=dict)
    run_figures: dict = field(default_factory=dict)
    comparison_figures: dict = field(default_factory=dict)
    generated_at: str = ""
    command: str = ""

    @classmethod
    def build(
        cls,
        store: RunStore,
        *,
        embed: bool = True,
        formats: tuple = ("png", "svg"),
        with_figures: bool = True,
        command: str = "",
        include_games=None,
        exclude_games=None,
        reference_store: RunStore | None = None,
    ) -> "HtmlReport":
        """Render every stored run, or a chosen subset of games.

        `include_games` / `exclude_games` have no default: a report covers
        everything in the store unless told otherwise. They exist so someone
        using this library for their own game can report on it alone, without
        the bundled Klondike and Macao runs crowding the comparison.

        `reference_store` folds a second, read-only store of committed library
        runs in as comparison points -- but only for games the user is already
        reporting on, so a custom game (absent from that store) gets nothing.
        """
        runs = store.load_runs()
        baselines = store.load_baselines()

        if include_games:
            wanted = set(include_games)
            runs = [r for r in runs if r.game in wanted]
            baselines = {g: b for g, b in baselines.items() if g in wanted}
        if exclude_games:
            unwanted = set(exclude_games)
            runs = [r for r in runs if r.game not in unwanted]
            baselines = {g: b for g, b in baselines.items() if g not in unwanted}
        if reference_store is not None:
            runs, baselines = _merge_reference(runs, baselines, reference_store)
        run_figures: dict = {}
        comparison_figures: dict = {}

        if with_figures:
            from rl_card_lib.report.figures import (
                render_comparison_figures,
                render_run_figures,
            )

            for record in runs:
                out_dir = store.run_dir(record.game, record.agent) / "figures"
                run_figures[record.run_id] = render_run_figures(
                    record, out_dir, baselines=baselines.get(record.game),
                    formats=formats, embed=embed,
                )
            comparison_figures = render_comparison_figures(
                runs, baselines, store.figures_dir, formats=formats, embed=embed,
            )

        return cls(
            runs=runs, baselines=baselines,
            run_figures=run_figures, comparison_figures=comparison_figures,
            generated_at=utc_now(), command=command,
        )

    # -- sections --------------------------------------------------------

    def _header(self) -> str:
        episodes = sum(r.episode_count for r in self.runs)
        games = sorted({r.game for r in self.runs})
        out = ['<header class="page"><div class="wrap">']
        out.append(f"<h1>{_escape(TITLE)}</h1>")
        out.append(
            '<p class="lede">One row per model, most recently trained first. '
            "Only the latest run of each model is kept, so every number here "
            "describes the current state of that agent.</p>"
        )
        out.append('<div class="meta">')
        out.append(f"<span>Generated {_fmt_timestamp(self.generated_at)}</span>")
        out.append(f"<span>{len(self.runs)} run(s)</span>")
        out.append(f"<span>{episodes:,} episodes total</span>")
        if games:
            out.append(f"<span>{_escape(', '.join(games))}</span>")
        if self.command:
            out.append(f"<span>Reproduce: <code>{_escape(self.command)}</code></span>")
        out.append("</div></div></header>")
        return "\n".join(out)

    def _nav(self) -> str:
        links = ['<a href="#overview">Overview</a>']
        by_game: dict = {}
        for record in self.runs:
            by_game.setdefault(record.game, []).append(record)

        for game, group in by_game.items():
            links.append('<span class="sep">|</span>')
            links.append(
                f'<a href="#game-{_slug(game)}">{_escape(game_spec(game)["label"])}</a>'
            )
            for record in group:
                links.append(
                    f'<a href="#{_slug(record.run_id)}">'
                    f"{_escape(record.agent_display)}</a>"
                )
        if self.baselines:
            links.append('<span class="sep">|</span>')
            links.append('<a href="#baselines">Baselines</a>')
        links.append('<span class="sep">|</span>')
        links.append('<a href="#configuration">Configuration</a>')
        return f'<nav class="toc"><div class="wrap">{"".join(links)}</div></nav>'

    def _overview(self) -> str:
        out = ['<section id="overview">', "<h2>Overview</h2>"]
        # Name each present game's headline metric, so the sentence is true for
        # whatever games are in the store rather than the two bundled ones.
        headlines = []
        for game in self._games():
            spec = game_spec(game)
            headlines.append(f"{spec['headline_label'].lower()} for "
                             f"{spec['label']}")
        detail = (f" The headline metric differs per game: "
                  f"{'; '.join(headlines)}.") if headlines else ""
        out.append(
            '<p class="sub">Sorted by finish time, newest first.'
            f"{_escape(detail)}</p>"
        )

        if not self.runs:
            out.append('<p class="empty">No runs recorded yet.</p></section>')
            return "\n".join(out)

        columns = [
            "Finished", "Game", "Agent", "Episodes",
            "Headline metric (range)", "Before → after", "Delta",
            "Train time", "Status",
        ]
        rows = []
        for record in self.runs:
            headline = record.headline or {}
            metric_name = headline.get("label", "")
            # register_game auto-registers the headline metric with its bound,
            # so metric_range knows it for any game -- no per-game special case.
            span = metric_range(headline.get("key", "")) if headline else ""
            status = (
                '<span class="chip failed">failed</span>'
                if record.status != "completed"
                else f'<span class="chip">{_escape(record.status)}</span>'
            )
            rows.append([
                _fmt_timestamp(record.finished_at),
                _escape(game_spec(record.game)["label"]),
                _agent_chip(record),
                record.episode_count,
                (f"{_escape(metric_name)} "
                 f'<span class="chip">{_escape(span)}</span>')
                if metric_name else NOT_RECORDED,
                _headline_markup(record),
                _delta_markup(
                    headline.get("delta"),
                    higher_is_better=headline.get("higher_is_better", True),
                ),
                _escape(_fmt_seconds((record.duration or {}).get("train_seconds"))),
                status,
            ])
        out.append(_table("overview", columns, rows))

        sources = {(r.headline or {}).get("source") for r in self.runs}
        if "training_summary" in sources:
            out.append(
                '<div class="notes"><h4>Reading this table</h4><ul><li>'
                "Some rows fall back to the average win rate during training "
                "because the run predates before/after baseline measurement. "
                "That is a different quantity from a measured win rate against "
                "a fixed opponent and the two must not be compared directly."
                "</li></ul></div>"
            )
        out.append("</section>")
        return "\n".join(out)

    def _games(self) -> list:
        """Games present in the store, in the order their runs appear."""
        seen = []
        for record in self.runs:
            if record.game not in seen:
                seen.append(record.game)
        return seen

    def _comparisons(self) -> str:
        """One section per game, whether or not comparison charts exist.

        Emitted unconditionally so the navigation cannot dangle: the nav links
        to every game, and `--no-figures` must not remove the target.
        """
        out = []
        for game in self._games():
            spec = game_spec(game)
            figures = self.comparison_figures.get(game) or []
            out.append(f'<section id="game-{_slug(game)}">')
            out.append(f'<h2>{_escape(spec["label"])}</h2>')
            out.append(
                f'<p class="sub">All learners on one axis. '
                f'Headline metric: {_escape(spec["headline_label"])}.</p>'
            )
            if figures:
                out.append('<div class="figures">')
                for figure in figures:
                    out.append(
                        _figure_block(figure, table_name=f"{game}_{figure.key}")
                    )
                out.append("</div>")
            else:
                out.append(
                    '<p class="empty">No comparison charts were rendered.</p>'
                )
            out.append("</section>")
        return "\n".join(out)

    def _baselines(self) -> str:
        if not self.baselines:
            return ""
        out = ['<section id="baselines">', "<h2>Baselines</h2>"]
        out.append(
            '<p class="sub">Non-learning agents, evaluated rather than trained. '
            "They play at full strength immediately, so they have no learning "
            "curve -- these are the bar a learner has to clear.</p>"
        )
        for game, baseline in self.baselines.items():
            spec = game_spec(game)
            out.append(f'<h3>{_escape(spec["label"])}</h3>')
            if not baseline.rows:
                out.append('<p class="empty">No baselines measured.</p>')
                continue
            columns: list = []
            for row in baseline.rows:
                for key in row:
                    if key not in columns:
                        columns.append(key)
            cap = (baseline.protocol or {}).get("max_steps")
            headers = [
                "Agent" if key == "agent" else _column_header(key, cap)
                for key in columns
            ]
            rows = [
                [row.get("agent") if key == "agent"
                 else (_escape(format_metric(key, row.get(key))) or NOT_RECORDED)
                 for key in columns]
                for row in baseline.rows
            ]
            protocol = ", ".join(
                f"{k}: {v}" for k, v in (baseline.protocol or {}).items()
                if v is not None
            )
            out.append(_table(
                f"baselines_{game}", headers, rows,
                caption=_escape(protocol) if protocol else "",
            ))
        out.append("</section>")
        return "\n".join(out)

    def _run_section(self, record: RunRecord) -> str:
        spec = game_spec(record.game)
        headline = record.headline or {}
        out = [f'<section id="{_slug(record.run_id)}">']
        # The record's own label, so a run that names itself is honoured rather
        # than having its heading rebuilt from the game and agent keys.
        heading = record.label or f"{spec['label']} / {record.agent_display}"
        out.append(f"<h2>{_escape(heading)}</h2>")
        out.append(
            f'<p class="sub">{_escape(record.agent_class or "")} · '
            f"{record.episode_count:,} episodes · finished "
            f"{_fmt_timestamp(record.finished_at)}</p>"
        )

        # Stat tiles
        out.append('<div class="tiles">')
        out.append(self._tile(
            headline.get("label", "Headline"),
            _fmt(headline.get("after"), headline.get("format", "{:.3f}")),
            f'of {headline["max"]:g} possible' if headline.get("max") else "",
        ))
        out.append(self._tile(
            "Change",
            _delta_markup(
                headline.get("delta"),
                higher_is_better=headline.get("higher_is_better", True),
            ),
            "before → after training",
        ))
        out.append(self._tile("Episodes", f"{record.episode_count:,}",
                              _escape(spec["trainer"])))
        out.append(self._tile(
            "Training time",
            _escape(_fmt_seconds((record.duration or {}).get("train_seconds"))),
            _escape(_fmt_seconds((record.duration or {}).get("eval_seconds"))
                    + " evaluating")
            if (record.duration or {}).get("eval_seconds") is not None else "",
        ))
        out.append("</div>")

        if record.notes:
            out.append('<div class="notes"><h4>Notes on this run</h4><ul>')
            for note in record.notes:
                out.append(f"<li>{_escape(note)}</li>")
            out.append("</ul></div>")

        figures = self.run_figures.get(record.run_id) or []
        if figures:
            out.append('<div class="figures">')
            for figure in figures:
                out.append(_figure_block(
                    figure, table_name=f"{record.run_id}_{figure.key}",
                ))
            out.append("</div>")

        out.append(self._run_tables(record))
        out.append("</section>")
        return "\n".join(out)

    @staticmethod
    def _tile(label: str, value: str, foot: str = "") -> str:
        return (
            f'<div class="tile"><div class="label">{_escape(label)}</div>'
            f'<div class="value">{value}</div>'
            f'<div class="foot">{foot}</div></div>'
        )

    def _run_tables(self, record: RunRecord) -> str:
        cap = record.env_max_steps()
        out = ["<h3>Summary</h3>"]
        out.append(
            '<p class="sub">Every row carries the range it is measured on: '
            "these numbers mix rates, unbounded rewards and counts.</p>"
        )
        summary = record.summary or {}
        out.append(_table(
            f"{record.run_id}_summary", ["Metric", "Value", "Range"],
            [
                [
                    _escape(metric_spec(key)["label"]),
                    _escape(format_metric(key, value)),
                    _escape(metric_range(key, env_max_steps=cap)) or "—",
                ]
                for key, value in summary.items()
            ],
        ))

        if record.evaluations:
            out.append("<h3>Evaluation history</h3>")
            columns: list = []
            for entry in record.evaluations:
                for key in entry:
                    if key not in columns:
                        columns.append(key)
            headers = [_column_header(key, cap) for key in columns]
            out.append(_table(
                f"{record.run_id}_evaluations", headers,
                [
                    [_escape(format_metric(key, entry.get(key))) or NOT_RECORDED
                     for key in columns]
                    for entry in record.evaluations
                ],
            ))

        title, section = record.algorithm_section()
        if section:
            out.append(f"<h3>{_escape(title)} hyperparameters</h3>")
            out.append(_table(
                f"{record.run_id}_hyperparameters", ["Parameter", "Value"],
                [[k, _stringify(v)] for k, v in section.items()],
            ))
        elif record.config is None:
            out.append(
                '<p class="empty">Hyperparameters were not recorded for this '
                "run.</p>"
            )

        artifacts = record.artifacts or {}
        if artifacts:
            out.append("<h3>Artifacts</h3>")
            out.append(_table(
                f"{record.run_id}_artifacts", ["Artifact", "Value"],
                [[k, _stringify(v)] for k, v in artifacts.items()],
            ))
        return "\n".join(out)

    def _configuration(self) -> str:
        out = ['<section id="configuration">', "<h2>Configuration</h2>"]
        out.append(
            '<p class="sub">Environment and trainer settings, and the caveats '
            "that apply to every number in this report.</p>"
        )

        rows = []
        for record in self.runs:
            environment = record.config_section("environment") or {}
            trainer = record.config_section("trainer") or {}
            host = record.host or {}
            rows.append([
                _escape(record.run_id),
                environment.get("max_steps"),
                environment.get("action_size"),
                _escape(trainer.get("type", "")) or NOT_RECORDED,
                _escape(str(trainer.get("opponent", ""))) or NOT_RECORDED,
                trainer.get("eval_episodes"),
                host.get("seed"),
                _escape(host.get("git_commit", "")) or NOT_RECORDED,
            ])
        if rows:
            out.append(_table(
                "configuration",
                ["Run", "Max steps", "Actions", "Trainer", "Opponent",
                 "Eval episodes", "Seed", "Commit"],
                rows,
            ))

        caveats = [
            "Evaluation seeds the global RNG once per episode, so before/after "
            "evaluations perturb the training RNG stream. This is deterministic "
            "and reproducible, but results depend on where evaluations are placed.",
            "Only the most recent run of each model is stored; re-running a "
            "model replaces its record, figures and checkpoints.",
        ]
        # A Klondike-specific caveat, shown only when Klondike is in the store.
        if any(game_spec(g).get("headline_key") == "cards_up"
               for g in self._games()):
            caveats.insert(
                1,
                "Klondike reports cards to the foundation as its headline "
                "because reward shaping has changed since earlier runs and "
                "cards-up is invariant to it.",
            )
        out.append(
            '<div class="notes"><h4>Caveats that apply throughout</h4><ul>'
            + "".join(f"<li>{c}</li>" for c in caveats)
            + "</ul></div>"
        )
        out.append("</section>")
        return "\n".join(out)

    # -- assembly --------------------------------------------------------

    def to_html(self) -> str:
        body = [
            self._header(),
            self._nav(),
            '<div class="wrap">',
            self._overview(),
            self._comparisons(),
            self._baselines(),
        ]
        for record in self.runs:
            body.append(self._run_section(record))
        body.append(self._configuration())
        body.append(
            '<footer class="page">Generated by rl-card-lib-report. '
            "Figures are embedded; this file needs no network and no sibling "
            "files to render.</footer>"
        )
        body.append("</div>")

        payload = json.dumps(
            {"generated_at": self.generated_at,
             "runs": [r.as_dict() for r in self.runs]},
            default=str,
        ).replace("</", "<\\/")

        return "\n".join([
            "<!DOCTYPE html>",
            '<html lang="en"><head>',
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f"<title>{_escape(TITLE)}</title>",
            f"<style>{CSS}</style>",
            "</head><body>",
            "\n".join(body),
            f'<script type="application/json" id="run-data">{payload}</script>',
            f"<script>{JS}</script>",
            "</body></html>",
        ])

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(self.to_html())
        return path


def _stringify(value) -> str:
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    return str(value)


__all__ = ["HtmlReport", "BaselineSet", "RunRecord", "RunStore"]
