"""Tests for the self-contained HTML training report."""

import matplotlib
matplotlib.use("Agg")  # noqa: E402 - must precede any pyplot import

import re  # noqa: E402

import pytest  # noqa: E402

from rl_card_lib.report import BaselineSet, RunRecord, RunStore  # noqa: E402
from rl_card_lib.report.cli import main as cli_main  # noqa: E402
from rl_card_lib.report.html_report import HtmlReport  # noqa: E402
from rl_card_lib.trainer import TrainingMetrics  # noqa: E402


def make_metrics(episodes=12):
    metrics = TrainingMetrics(window_size=5)
    for i in range(episodes):
        metrics.add_episode({
            "reward": float(i), "steps": 20, "win": i % 4 == 0, "loss": 0.5,
        })
    metrics.add_evaluation(episodes, {"mean_reward": 3.0, "win_rate": 0.25})
    metrics.training_time = 5.0
    return metrics


def make_record(game="klondike", agent="dqn", finished=None, **kwargs):
    kwargs.setdefault("metrics", make_metrics())
    kwargs.setdefault("agent_class", "DQNAgent")
    kwargs.setdefault("train_seconds", 5.0)
    record = RunRecord.from_training(game=game, agent=agent, **kwargs)
    if finished:
        record.timestamps["finished_at"] = finished
    return record


@pytest.fixture
def store(tmp_path):
    store = RunStore(tmp_path)
    store.save_run(make_record(
        agent="dqn", finished="2026-01-01T00:00:00+00:00",
        baseline_before={"cards_up": 3.0}, baseline_after={"cards_up": 11.0},
    ))
    store.save_run(make_record(
        game="macao", agent="ppo", finished="2026-06-01T00:00:00+00:00",
    ))
    return store


def build(store, **kwargs):
    kwargs.setdefault("with_figures", False)
    return HtmlReport.build(store, **kwargs).to_html()


class TestDocument:
    """Structural guarantees of the page."""

    def test_is_a_complete_document(self, store):
        page = build(store)
        assert page.startswith("<!DOCTYPE html>")
        assert page.rstrip().endswith("</html>")
        assert "<title>" in page

    def test_names_every_run(self, store):
        page = build(store)
        assert "Klondike Solitaire" in page
        assert "Macao" in page
        assert "Double DQN" not in page  # not in this store

    def test_styles_and_scripts_are_inline(self, store):
        page = build(store)
        assert "<style>" in page and "<script>" in page

    def test_carries_the_run_data_as_json(self, store):
        assert 'id="run-data"' in build(store)


class TestOffline:
    """The page must render with no network and no sibling files."""

    def test_no_external_resources(self, store):
        page = build(store)
        assert not re.search(r'src="https?:', page)
        assert not re.search(r'href="https?:', page)

    def test_no_cdn_or_import_statements(self, store):
        page = build(store)
        assert "cdn." not in page
        assert "@import" not in page

    def test_figures_are_embedded_as_data_uris(self, store):
        page = HtmlReport.build(store, formats=("png",)).to_html()
        assert "data:image/png;base64," in page


class TestOrdering:
    """Most recently run first -- the requirement the store exists to serve."""

    def test_newest_run_appears_first(self, store):
        page = build(store)
        assert page.index("macao__ppo") < page.index("klondike__dqn")

    def test_reordering_follows_the_timestamps(self, tmp_path):
        store = RunStore(tmp_path)
        store.save_run(make_record(agent="dqn", finished="2026-09-01T00:00:00+00:00"))
        store.save_run(make_record(
            game="macao", agent="ppo", finished="2026-02-01T00:00:00+00:00",
        ))
        page = build(store)
        assert page.index("klondike__dqn") < page.index("macao__ppo")


class TestNavigation:
    def test_every_anchor_resolves(self, store):
        page = build(store)
        ids = set(re.findall(r'id="([^"]+)"', page))
        targets = set(re.findall(r'href="#([^"]+)"', page))
        assert targets <= ids

    def test_links_to_each_run_section(self, store):
        page = build(store)
        assert 'href="#klondike__dqn"' in page
        assert 'id="klondike__dqn"' in page


class TestExportAffordances:
    """Requirement: every table and figure exportable to an image."""

    def test_tables_carry_an_export_toolbar(self, store):
        page = build(store)
        assert 'data-export="csv"' in page
        assert 'data-export="png"' in page
        assert 'data-export="copy"' in page

    def test_tables_are_named_for_the_download(self, store):
        assert 'data-name="overview"' in build(store)

    def test_figures_offer_a_download(self, store):
        page = HtmlReport.build(store, formats=("png", "svg")).to_html()
        assert "download=" in page

    def test_print_rules_exist_for_the_appendix(self, store):
        assert "@media print" in build(store)


class TestContent:
    def test_overview_lists_the_headline_metric(self, store):
        page = build(store)
        assert "Cards to foundation" in page

    def test_shows_before_and_after(self, store):
        assert "→" in build(store)

    def test_renders_notes(self, tmp_path):
        store = RunStore(tmp_path)
        record = make_record()
        record.notes = ["Loss diverged spectacularly."]
        store.save_run(record)
        assert "Loss diverged spectacularly." in build(store)

    def test_baselines_are_labelled_as_untrained(self, tmp_path):
        store = RunStore(tmp_path)
        store.save_run(make_record())
        store.save_baselines(BaselineSet(
            game="klondike", rows=[{"agent": "Random", "cards_up": 2.9}],
        ))
        page = build(store)
        assert "no learning curve" in page
        assert "Random" in page

    def test_states_the_caveats(self, store):
        assert "Caveats" in build(store)


class TestMissingData:
    """A record with nothing optional recorded must still render."""

    def test_legacy_record_renders(self, tmp_path):
        path = tmp_path / "metrics.json"
        make_metrics().save(str(path))
        store = RunStore(tmp_path / "store")
        store.save_run(
            RunRecord.from_metrics_json(path, game="klondike", agent="ppo")
        )

        page = build(store)
        assert "not recorded" in page
        assert "Hyperparameters were not recorded" in page

    def test_empty_store_renders_without_raising(self, tmp_path):
        page = build(RunStore(tmp_path / "empty"))
        assert "No runs recorded yet." in page

    def test_failed_run_is_marked(self, tmp_path):
        store = RunStore(tmp_path)
        store.save_run(make_record(status="failed"))
        assert 'class="chip failed"' in build(store)


class TestEscaping:
    def test_markup_in_a_label_is_escaped(self, tmp_path):
        store = RunStore(tmp_path)
        record = make_record()
        record.label = "<script>alert(1)</script>"
        record.notes = ["<img src=x onerror=alert(2)>"]
        store.save_run(record)

        page = build(store)
        # Scoped to the rendered document: the JSON island below it is not
        # parsed as HTML, and is covered by the next test instead.
        body = page.split('<script type="application/json"')[0]

        assert "<script>alert(1)</script>" not in body
        assert "<img src=x" not in body
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in body
        assert "&lt;img src=x" in body

    def test_embedded_json_cannot_close_the_script_tag(self, tmp_path):
        store = RunStore(tmp_path)
        record = make_record()
        record.notes = ["</script><script>alert(3)</script>"]
        store.save_run(record)

        page = build(store)
        payload = page.split('id="run-data">')[1].split("</script>")[0]
        assert "</script>" not in payload


class TestWrite:
    def test_writes_a_single_file(self, tmp_path, store):
        out = tmp_path / "out" / "index.html"
        written = HtmlReport.build(store, with_figures=False).write(out)

        assert written == out
        assert out.is_file()
        assert out.stat().st_size > 10_000
        assert list(out.parent.iterdir()) == [out]


class TestMetricRanges:
    """Rates, rewards and counts share tables; each must state its scale."""

    def test_summary_carries_a_range_column(self, store):
        page = build(store)
        assert "<th>Range</th>" in page or "Range" in page
        assert "0-100%" in page

    def test_rates_render_as_percentages(self, store):
        assert "%" in build(store)

    def test_evaluation_headers_state_the_range(self, store):
        assert "Win rate (0-100%)" in build(store)

    def test_headline_metric_shows_its_bound(self, tmp_path):
        store = RunStore(tmp_path)
        store.save_run(make_record(
            baseline_before={"cards_up": 3.0}, baseline_after={"cards_up": 11.0},
        ))
        assert "0-52 cards" in build(store)


class TestLightbox:
    """Figures open full-screen; clicking anywhere dismisses."""

    def test_ships_the_overlay_code(self, store):
        page = build(store)
        assert "lightbox" in page
        assert "zoom-in" in page and "zoom-out" in page

    def test_escape_closes(self, store):
        assert 'event.key === "Escape"' in build(store)

    def test_hidden_when_printing(self, store):
        assert ".lightbox { display: none !important; }" in build(store)


class TestGameFiltering:
    """No default: the report covers the store unless told otherwise."""

    def test_unfiltered_includes_everything(self, store):
        report = HtmlReport.build(store, with_figures=False)
        assert {r.game for r in report.runs} == {"klondike", "macao"}

    def test_include_selects_one_game(self, store):
        report = HtmlReport.build(
            store, with_figures=False, include_games=["macao"],
        )
        assert {r.game for r in report.runs} == {"macao"}
        assert "Klondike Solitaire" not in report.to_html()

    def test_exclude_drops_the_builtins(self, tmp_path):
        store = RunStore(tmp_path)
        store.save_run(make_record(game="klondike", agent="dqn"))
        store.save_run(make_record(game="hearts", agent="ppo"))

        report = HtmlReport.build(
            store, with_figures=False, exclude_games=["klondike", "macao"],
        )
        assert [r.game for r in report.runs] == ["hearts"]

    def test_baselines_are_filtered_too(self, tmp_path):
        store = RunStore(tmp_path)
        store.save_run(make_record(game="hearts", agent="ppo"))
        store.save_baselines(BaselineSet(
            game="klondike", rows=[{"agent": "Random", "cards_up": 2.9}],
        ))
        report = HtmlReport.build(
            store, with_figures=False, exclude_games=["klondike"],
        )
        assert report.baselines == {}

    def test_a_custom_game_renders_with_a_neutral_spec(self, tmp_path):
        store = RunStore(tmp_path)
        store.save_run(make_record(game="hearts", agent="ppo"))
        page = HtmlReport.build(store, with_figures=False).to_html()
        assert "Hearts" in page


class TestRegisterGame:
    """A custom game can declare the metric it is judged on."""

    def test_registers_a_headline_metric(self):
        from rl_card_lib.report.run_record import GAME_SPEC, game_spec, register_game

        try:
            register_game(
                "hearts", label="Hearts", headline_key="penalty_points",
                headline_label="Penalty points", headline_max=26,
            )
            spec = game_spec("hearts")
            assert spec["headline_key"] == "penalty_points"
            assert spec["headline_label"] == "Penalty points"
            assert spec["episode_curves"] == []  # neutral default kept
        finally:
            GAME_SPEC.pop("hearts", None)

    def test_unregistered_game_falls_back(self):
        from rl_card_lib.report.run_record import game_spec

        spec = game_spec("some_unknown_game")
        assert spec["label"] == "Some Unknown Game"
        assert spec["headline_key"] == "win_rate"


class TestCli:
    def test_renders_from_a_store(self, tmp_path, store, capsys):
        out = tmp_path / "index.html"
        code = cli_main([
            "--results-dir", str(store.root), "--out", str(out),
            "--no-figures",
        ])
        assert code == 0
        assert out.is_file()
        assert "Wrote" in capsys.readouterr().out

    def test_reports_an_empty_store(self, tmp_path, capsys):
        code = cli_main(["--results-dir", str(tmp_path / "nothing")])
        assert code == 1
        assert "No run records" in capsys.readouterr().err
