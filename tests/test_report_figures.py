"""Tests for the training-report charts."""

import matplotlib
matplotlib.use("Agg")  # noqa: E402 - must precede any pyplot import

import base64  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from rl_card_lib.report import BaselineSet, RunRecord  # noqa: E402
from rl_card_lib.report.figures import (  # noqa: E402
    AGENT_COLORS,
    agent_color,
    charts_available,
    render_comparison_figures,
    render_run_figures,
)
from rl_card_lib.trainer import TrainingMetrics  # noqa: E402

PNG_MAGIC = b"\x89PNG"


def make_metrics(episodes=30, loss=0.5, evaluations=3):
    metrics = TrainingMetrics(window_size=10)
    for i in range(episodes):
        metrics.add_episode({
            "reward": float(i % 7), "steps": 20 + (i % 5),
            "win": 1 if i % 10 == 0 else 0,
            "loss": loss if isinstance(loss, float) else loss[i],
        })
    for e in range(evaluations):
        metrics.add_evaluation((e + 1) * 10, {
            "mean_reward": 1.0 + e, "std_reward": 0.5,
            "win_rate": 0.1 * e, "mean_steps": 25.0,
        })
    metrics.training_time = 12.0
    return metrics


def make_record(game="klondike", agent="dqn", episodes=30, **kwargs):
    kwargs.setdefault("metrics", make_metrics(episodes))
    kwargs.setdefault("agent_class", "DQNAgent")
    kwargs.setdefault("train_seconds", 12.0)
    return RunRecord.from_training(game=game, agent=agent, **kwargs)


def keys(figures):
    return [figure.key for figure in figures]


def render(record, path, **kwargs):
    """PNG-only by default: every save is a full render, and most tests here
    are about which charts exist rather than which formats do."""
    kwargs.setdefault("formats", ("png",))
    kwargs.setdefault("embed", False)
    return render_run_figures(record, path, **kwargs)


class TestRunFigures:
    """What gets drawn, and what is correctly left undrawn."""

    def test_renders_the_core_charts(self, tmp_path):
        figures = render(make_record(), tmp_path)
        assert {"reward", "win_rate", "loss", "steps", "evaluation"} <= set(keys(figures))

    def test_writes_png_and_svg(self, tmp_path):
        figures = render_run_figures(make_record(), tmp_path)
        for figure in figures:
            assert figure.png_path and figure.svg_path
            assert (tmp_path / f"{figure.key}.png").is_file()
            assert (tmp_path / f"{figure.key}.svg").is_file()

    def test_embedded_png_decodes_to_a_png(self, tmp_path):
        for figure in render_run_figures(make_record(), tmp_path):
            assert base64.b64decode(figure.png_base64).startswith(PNG_MAGIC)

    def test_embedding_can_be_turned_off(self, tmp_path):
        figures = render_run_figures(make_record(), tmp_path, embed=False)
        assert all(figure.png_base64 == "" for figure in figures)

    def test_format_selection_is_honoured(self, tmp_path):
        figures = render_run_figures(make_record(), tmp_path, formats=("png",))
        assert all(figure.svg_path is None for figure in figures)
        assert not list(tmp_path.glob("*.svg"))

    def test_klondike_draws_cards_up(self, tmp_path):
        record = make_record(episode_extras={"cards_up": list(range(30))})
        assert "cards_up" in keys(render(record, tmp_path))

    def test_macao_has_no_cards_up_chart(self, tmp_path):
        record = make_record(game="macao", agent="ppo")
        assert "cards_up" not in keys(render(record, tmp_path))

    def test_epsilon_chart_is_skipped_for_ppo(self, tmp_path):
        """PPO explores by sampling, so an epsilon panel would be a lie."""
        record = make_record(agent="ppo", episode_extras={"epsilon": [None] * 30})
        figures = render(record, tmp_path)
        assert "epsilon" not in keys(figures)
        assert figures  # the rest still render

    def test_epsilon_chart_is_drawn_when_measured(self, tmp_path):
        record = make_record(episode_extras={"epsilon": [0.9] * 30})
        assert "epsilon" in keys(render(record, tmp_path))

    def test_table_size_chart_only_for_tabular(self, tmp_path):
        record = make_record(
            agent="q_learning",
            episode_extras={"table_size": list(range(100, 130))},
        )
        assert "table_size" in keys(render(record, tmp_path))
        assert "table_size" not in keys(render(make_record(), tmp_path))

    def test_before_after_needs_both_sides(self, tmp_path):
        without = make_record(baseline_after={"cards_up": 9.0})
        assert "before_after" not in keys(render(without, tmp_path))

        with_both = make_record(
            baseline_before={"cards_up": 3.0}, baseline_after={"cards_up": 9.0},
        )
        assert "before_after" in keys(render(with_both, tmp_path))

    def test_zero_loss_chart_is_omitted(self, tmp_path):
        """A non-learning run has no loss to draw."""
        record = make_record(metrics=make_metrics(loss=0.0))
        assert "loss" not in keys(render(record, tmp_path))


class TestDivergedLoss:
    """A diverged loss is a finding and must not be silently clipped."""

    def record(self):
        losses = [1.0] * 25 + [1e4, 1e6, 1e8, 2.24e9, 2.24e9]
        return make_record(metrics=make_metrics(loss=losses))

    def test_switches_to_symlog(self, tmp_path):
        figures = render(self.record(), tmp_path)
        loss = next(f for f in figures if f.key == "loss")
        assert "symlog" in loss.caption.lower()
        assert "not clipped" in loss.caption.lower()

    def test_the_record_carries_the_note(self):
        assert any("diverged" in note for note in self.record().notes)

    def test_a_tame_loss_stays_linear(self, tmp_path):
        figures = render(make_record(), tmp_path)
        loss = next(f for f in figures if f.key == "loss")
        assert "symlog" not in loss.caption.lower()


class TestTableTwins:
    """Every chart ships its numbers; two palette slots need that relief."""

    def test_every_figure_has_a_table(self, tmp_path):
        for figure in render(make_record(), tmp_path):
            assert figure.table is not None
            assert figure.table["columns"]

    def test_rows_match_the_column_count(self, tmp_path):
        for figure in render(make_record(), tmp_path):
            width = len(figure.table["columns"])
            assert all(len(row) == width for row in figure.table["rows"])

    def test_long_series_are_downsampled(self, tmp_path):
        record = make_record(episodes=1200)
        figures = render(record, tmp_path)
        reward = next(f for f in figures if f.key == "reward")
        assert 0 < len(reward.table["rows"]) <= 201

    def test_downsampling_keeps_the_final_episode(self, tmp_path):
        record = make_record(episodes=1200)
        figures = render(record, tmp_path)
        reward = next(f for f in figures if f.key == "reward")
        assert reward.table["rows"][-1][0] == 1199


class TestComparisonFigures:
    def records(self):
        return [
            make_record(agent=agent, episode_extras={"cards_up": list(range(30))})
            for agent in ("dqn", "double_dqn", "ppo", "q_learning")
        ]

    def test_groups_by_game(self, tmp_path):
        records = self.records() + [make_record(game="macao", agent="ppo")]
        out = render_comparison_figures(records, {}, tmp_path)
        assert set(out) == {"klondike", "macao"}

    def test_renders_the_comparison_charts(self, tmp_path):
        out = render_comparison_figures(self.records(), {}, tmp_path)
        assert "comparison_curves" in keys(out["klondike"])
        assert "comparison_headline" in keys(out["klondike"])

    def test_files_are_prefixed_by_game(self, tmp_path):
        render_comparison_figures(self.records(), {}, tmp_path)
        assert (tmp_path / "klondike_comparison_curves.png").is_file()

    def test_baselines_match_the_plotted_measure(self, tmp_path):
        """Cards-up baselines must never be drawn against a win-rate axis."""
        baselines = BaselineSet(game="klondike", rows=[
            {"agent": "Random", "cards_up": 2.9, "win_rate": 0.01},
        ])
        assert baselines.values_for("cards_up") == {"Random": pytest.approx(2.9)}
        assert baselines.values_for("win_rate") == {"Random": pytest.approx(0.01)}
        assert baselines.values_for("missing") == {}

    def test_single_run_needs_no_evaluation_panels(self, tmp_path):
        out = render_comparison_figures([make_record()], {}, tmp_path)
        assert "comparison_evaluation" not in keys(out.get("klondike", []))


class TestHousekeeping:
    def test_colour_follows_the_agent_not_its_rank(self):
        assert agent_color("dqn") == AGENT_COLORS["dqn"]
        assert agent_color("dqn") != agent_color("ppo")
        assert agent_color("unknown_agent")  # never blank

    def test_no_figures_are_left_open(self, tmp_path):
        render(make_record(), tmp_path)
        render_comparison_figures([make_record()], {}, tmp_path)
        assert plt.get_fignums() == []

    def test_charts_available(self):
        assert charts_available() is True

    def test_degrades_when_matplotlib_is_missing(self, tmp_path, monkeypatch, capsys):
        """Same contract as TrainingMetrics.plot: print and return, never raise."""
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "matplotlib":
                raise ImportError("No matplotlib")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        assert render_run_figures(make_record(), tmp_path) == []
        assert render_comparison_figures([make_record()], {}, tmp_path) == {}
        assert "matplotlib not installed" in capsys.readouterr().out
