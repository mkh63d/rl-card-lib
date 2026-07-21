"""Tests for persisted run records and the store that holds them."""

import json

import pytest

from rl_card_lib.report import BaselineSet, RunRecord, RunStore
from rl_card_lib.report.run_record import (
    detect_notes,
    moving_average,
    purge_checkpoints,
    reconstruct_epsilon,
)
from rl_card_lib.trainer import TrainingMetrics


def make_metrics(episodes=6, reward=1.0, win=0, loss=0.5):
    metrics = TrainingMetrics(window_size=3)
    for _ in range(episodes):
        metrics.add_episode(
            {"reward": reward, "steps": 10, "win": win, "loss": loss}
        )
    metrics.add_evaluation(episodes, {"mean_reward": 2.0, "win_rate": 0.5})
    metrics.training_time = 1.25
    return metrics


def make_record(game="klondike", agent="dqn", episodes=6, **kwargs):
    kwargs.setdefault("metrics", make_metrics(episodes))
    kwargs.setdefault("agent_class", "DQNAgent")
    return RunRecord.from_training(game=game, agent=agent, **kwargs)


class TestRunRecord:
    """Construction, derived views and serialization."""

    def test_identity_and_label(self):
        record = make_record()
        assert record.run_id == "klondike__dqn"
        assert record.label == "Klondike Solitaire / DQN"
        assert record.status == "completed"

    def test_copies_the_episode_arrays(self):
        record = make_record(episodes=4)
        assert record.episode_count == 4
        assert record.series("reward") == [1.0, 1.0, 1.0, 1.0]
        assert record.summary["total_episodes"] == 4

    def test_unrecorded_series_are_none_not_empty(self):
        record = make_record()
        assert record.series("cards_up") is None
        assert record.episodes["cards_up"] is None

    def test_all_none_series_is_treated_as_absent(self):
        """PPO has no epsilon, so the recorder hands back a column of None."""
        record = make_record(episode_extras={"epsilon": [None] * 6})
        assert record.episodes["epsilon"] is None
        assert record.episodes["epsilon_source"] is None

    def test_measured_epsilon_is_labelled(self):
        record = make_record(episode_extras={"epsilon": [0.9] * 6})
        assert record.episodes["epsilon_source"] == "measured"

    def test_json_round_trips(self):
        record = make_record(episode_extras={"cards_up": [1, 2, 3, 4, 5, 6]})
        restored = RunRecord.from_dict(json.loads(record.to_json()))
        assert restored.run_id == record.run_id
        assert restored.series("cards_up") == [1, 2, 3, 4, 5, 6]
        assert restored.summary == record.summary

    def test_schema_version_is_recorded(self):
        assert json.loads(make_record().to_json())["schema_version"] >= 1

    def test_records_both_duration_measurements(self):
        """The trainer's own timer is unreliable, so ours is kept beside it."""
        record = make_record(train_seconds=10.0, eval_seconds=2.0)
        assert record.duration["train_seconds"] == 10.0
        assert record.duration["total_seconds"] == 12.0
        assert record.duration["metrics_training_time"] == pytest.approx(1.25)

    def test_moving_average_matches_metrics(self):
        record = make_record()
        assert record.moving_average("reward", window=3) == [1.0] * 6

    def test_validate_rejects_misaligned_series(self):
        record = make_record(episodes=6)
        record.episodes["cards_up"] = [1, 2, 3]
        with pytest.raises(ValueError, match="cards_up"):
            record.validate()

    def test_from_training_validates(self):
        with pytest.raises(ValueError):
            make_record(episodes=6, episode_extras={"cards_up": [1, 2]})

    def test_stores_a_custom_episode_series(self):
        """A custom game's own progress signal must survive, not be dropped."""
        record = make_record(
            episodes=6, episode_extras={"penalty_points": [5, 4, 3, 2, 1, 0]},
        )
        assert record.series("penalty_points") == [5, 4, 3, 2, 1, 0]

    def test_custom_series_round_trips(self):
        record = make_record(
            episodes=4, episode_extras={"pieces_captured": [0, 1, 1, 3]},
        )
        restored = RunRecord.from_dict(json.loads(record.to_json()))
        assert restored.series("pieces_captured") == [0, 1, 1, 3]

    def test_custom_series_is_length_checked(self):
        with pytest.raises(ValueError, match="penalty_points"):
            make_record(episodes=6, episode_extras={"penalty_points": [5, 4, 3]})

    def test_known_extras_keep_their_keys(self):
        record = make_record(
            episodes=3,
            episode_extras={"cards_up": [1, 2, 3], "custom": [4, 5, 6]},
        )
        assert record.episodes["cards_up"] == [1, 2, 3]
        assert record.episodes["custom"] == [4, 5, 6]


class TestHeadline:
    """The headline metric differs per game and must say where it came from."""

    def test_klondike_prefers_measured_cards_up(self):
        record = make_record(
            baseline_before={"cards_up": 4.0},
            baseline_after={"cards_up": 11.5},
        )
        assert record.headline["key"] == "cards_up"
        assert record.headline["source"] == "baseline_eval"
        assert record.headline["delta"] == pytest.approx(7.5)

    def test_klondike_falls_back_to_the_episode_tail(self):
        record = make_record(episode_extras={"cards_up": [2, 4, 6, 8, 10, 12]})
        assert record.headline["key"] == "cards_up"
        assert record.headline["source"] == "episode_tail"

    def test_macao_headline_is_relative_to_the_heuristic(self):
        record = make_record(
            game="macao", agent="ppo",
            baseline_after={"win_rate_vs_heuristic": 0.42},
        )
        assert record.headline["key"] == "win_rate_vs_heuristic"
        assert record.headline["after"] == pytest.approx(0.42)

    def test_summary_fallback_is_relabelled(self):
        """A training win rate must not masquerade as the headline metric."""
        record = make_record(episodes=4)
        assert record.headline["source"] == "training_summary"
        assert record.headline["key"] == "win_rate"
        assert record.headline["label"] == "Win rate in training"

    def test_higher_is_better_defaults_true(self):
        record = make_record(
            baseline_before={"cards_up": 3.0}, baseline_after={"cards_up": 11.0},
        )
        assert record.headline["higher_is_better"] is True

    def test_lower_is_better_from_the_game_spec(self):
        from rl_card_lib.report.run_record import GAME_SPEC, register_game

        try:
            register_game(
                "golf", headline_key="score", headline_label="Score",
                higher_is_better=False, episode_curves=[],
            )
            record = make_record(
                game="golf", agent="dqn",
                baseline_before={"score": 40.0}, baseline_after={"score": 25.0},
            )
            assert record.headline["higher_is_better"] is False
            assert record.headline["delta"] == pytest.approx(-15.0)
        finally:
            GAME_SPEC.pop("golf", None)


class TestEpsilonReconstruction:
    def test_matches_the_per_episode_decay(self):
        values = reconstruct_epsilon(start=1.0, end=0.05, decay=0.9, episodes=4)
        assert values == pytest.approx([1.0, 0.9, 0.81, 0.729])

    def test_clamps_at_the_floor(self):
        values = reconstruct_epsilon(start=1.0, end=0.5, decay=0.5, episodes=5)
        assert values[-1] == pytest.approx(0.5)
        assert min(values) >= 0.5


class TestMovingAverage:
    def test_is_trailing_not_centred(self):
        assert moving_average([0.0, 10.0], window=2) == pytest.approx([0.0, 5.0])

    def test_empty_input(self):
        assert moving_average([]) == []


class TestNotes:
    """Findings that a flat line alone would not communicate."""

    def test_flags_a_diverged_loss(self):
        notes = detect_notes(episodes={"loss": [1.0] * 20 + [2.24e9]})
        assert any("diverged" in note for note in notes)

    def test_ignores_a_well_behaved_loss(self):
        notes = detect_notes(episodes={"loss": [1.0, 1.2, 0.9, 1.1]})
        assert not any("diverged" in note for note in notes)

    def test_flags_full_step_cap_saturation(self):
        notes = detect_notes(episodes={"steps": [300] * 10}, env_max_steps=300)
        assert any("300-step cap" in note for note in notes)

    def test_flags_the_pre_fix_zero_reward_evaluations(self):
        notes = detect_notes(
            episodes={},
            evaluations=[{
                "mean_reward": 0.0, "std_reward": 0.0,
                "min_reward": 0.0, "max_reward": 0.0, "win_rate": 0.3,
            }],
        )
        assert any("SelfPlayTrainer" in note for note in notes)

    def test_flags_an_early_epsilon_floor(self):
        notes = detect_notes(episodes={"epsilon": [1.0, 0.5, 0.05, 0.05, 0.05]})
        assert any("floor" in note for note in notes)


class TestExplorationGap:
    """Training and evaluation measure different policies; say so."""

    def headline(self, after):
        return {"key": "cards_up", "source": "baseline_eval", "after": after}

    def test_flags_a_greedy_policy_worse_than_exploration(self):
        notes = detect_notes(
            episodes={"cards_up": [20.0] * 50},
            headline=self.headline(6.7),
        )
        assert any("worse than the behaviour" in note for note in notes)

    def test_flags_a_greedy_policy_much_better(self):
        notes = detect_notes(
            episodes={"cards_up": [5.0] * 50},
            headline=self.headline(20.0),
        )
        assert any("above the" in note for note in notes)

    def test_silent_when_the_two_agree(self):
        notes = detect_notes(
            episodes={"cards_up": [10.0] * 50},
            headline=self.headline(9.5),
        )
        assert not any("exploratory" in note for note in notes)

    def test_silent_without_a_measured_headline(self):
        """A training-summary fallback is not a greedy measurement."""
        notes = detect_notes(
            episodes={"cards_up": [20.0] * 50},
            headline={"key": "cards_up", "source": "training_summary",
                      "after": 1.0},
        )
        assert notes == []

    def test_appears_on_a_real_record(self):
        record = make_record(
            episode_extras={"cards_up": [20] * 6},
            baseline_before={"cards_up": 2.0}, baseline_after={"cards_up": 5.0},
        )
        assert any("not directly comparable" in note for note in record.notes)


class TestRunStore:
    """Only the last run of each model survives, structurally."""

    def test_saves_and_reloads(self, tmp_path):
        store = RunStore(tmp_path)
        store.save_run(make_record())
        runs = store.load_runs()
        assert len(runs) == 1
        assert runs[0].run_id == "klondike__dqn"

    def test_rerunning_replaces_rather_than_accumulates(self, tmp_path):
        store = RunStore(tmp_path)
        store.save_run(make_record(episodes=4))
        store.save_run(make_record(episodes=9))

        runs = store.load_runs()
        assert len(runs) == 1
        assert runs[0].episode_count == 9

    def test_reset_run_dir_clears_stale_artifacts(self, tmp_path):
        store = RunStore(tmp_path)
        target = store.reset_run_dir("klondike", "dqn")
        stale = target / "figures" / "removed_chart.png"
        stale.write_bytes(b"stale")

        store.reset_run_dir("klondike", "dqn")
        assert not stale.exists()
        assert (target / "figures").is_dir()

    def test_load_runs_is_newest_first(self, tmp_path):
        store = RunStore(tmp_path)
        older = make_record(agent="dqn")
        older.timestamps["finished_at"] = "2026-01-01T00:00:00+00:00"
        newer = make_record(agent="ppo")
        newer.timestamps["finished_at"] = "2026-06-01T00:00:00+00:00"
        store.save_run(older)
        store.save_run(newer)

        assert [r.agent for r in store.load_runs()] == ["ppo", "dqn"]

    def test_ties_break_deterministically(self, tmp_path):
        store = RunStore(tmp_path)
        for agent in ("ppo", "dqn", "q_learning"):
            record = make_record(agent=agent)
            record.timestamps["finished_at"] = "2026-01-01T00:00:00+00:00"
            store.save_run(record)

        ids = [r.run_id for r in store.load_runs()]
        assert ids == sorted(ids)

    def test_empty_store(self, tmp_path):
        assert RunStore(tmp_path / "nothing").load_runs() == []

    def test_crashed_run_dir_is_skipped_not_fatal(self, tmp_path):
        store = RunStore(tmp_path)
        store.save_run(make_record())
        store.reset_run_dir("macao", "ppo")  # no run.json written

        assert [r.run_id for r in store.load_runs()] == ["klondike__dqn"]

    def test_unreadable_record_is_skipped(self, tmp_path, capsys):
        store = RunStore(tmp_path)
        store.save_run(make_record())
        target = store.reset_run_dir("macao", "ppo")
        (target / "run.json").write_text("{not json", encoding="utf-8")

        assert len(store.load_runs()) == 1
        assert "Skipping" in capsys.readouterr().out

    def test_baselines_round_trip(self, tmp_path):
        store = RunStore(tmp_path)
        store.save_baselines(BaselineSet(
            game="klondike", measured_at="2026-01-01T00:00:00+00:00",
            protocol={"episodes": 30},
            rows=[{"agent": "Random", "cards_up": 3.1, "win_rate": 0.0}],
        ))

        assert store.has_baselines("klondike")
        loaded = store.load_baselines()["klondike"]
        assert loaded.headline_values() == {"Random": pytest.approx(3.1)}


class TestPurgeCheckpoints:
    """A shorter rerun must not leave the longer run's checkpoints behind."""

    def seed(self, tmp_path):
        target = tmp_path / "klondike_q_learning"
        target.mkdir()
        for name in (
            "checkpoint_ep200.pt", "checkpoint_ep400.pt",
            "final.pkl", "metrics.json", "notes.txt",
        ):
            (target / name).write_text("x", encoding="utf-8")
        return target

    def test_removes_run_artifacts_only(self, tmp_path):
        target = self.seed(tmp_path)
        removed = purge_checkpoints(target, game="klondike", agent="q_learning")

        assert "checkpoint_ep400.pt" in removed
        assert "final.pkl" in removed
        assert (target / "notes.txt").exists()
        assert target.is_dir()

    def test_matches_pickles_written_under_a_pt_name(self, tmp_path):
        """Trainer hardcodes .pt even for Q-learning, which pickles."""
        target = tmp_path / "klondike_q_learning"
        target.mkdir()
        (target / "checkpoint_ep100.pkl").write_text("x", encoding="utf-8")

        removed = purge_checkpoints(target, game="klondike", agent="q_learning")
        assert removed == ["checkpoint_ep100.pkl"]

    def test_refuses_a_mismatched_directory(self, tmp_path):
        target = self.seed(tmp_path)
        with pytest.raises(ValueError, match="klondike_dqn"):
            purge_checkpoints(target, game="klondike", agent="dqn")

    def test_refuses_to_escape_the_root(self, tmp_path):
        target = self.seed(tmp_path)
        with pytest.raises(ValueError, match="outside"):
            purge_checkpoints(
                target, game="klondike", agent="q_learning",
                root=tmp_path / "elsewhere",
            )

    def test_missing_directory_is_not_an_error(self, tmp_path):
        assert purge_checkpoints(
            tmp_path / "klondike_dqn", game="klondike", agent="dqn"
        ) == []


class TestLegacyMetricsImport:
    def test_reads_a_metrics_json(self, tmp_path):
        path = tmp_path / "metrics.json"
        make_metrics(5).save(str(path))

        record = RunRecord.from_metrics_json(path, game="macao", agent="ppo")
        assert record.episode_count == 5
        assert record.config is None
        assert record.series("cards_up") is None
        assert record.finished_at
        assert any("Imported" in note for note in record.notes)

    def test_reconstructs_epsilon_when_the_schedule_is_known(self, tmp_path):
        path = tmp_path / "metrics.json"
        make_metrics(5).save(str(path))

        record = RunRecord.from_metrics_json(
            path, game="klondike", agent="dqn",
            epsilon_schedule={"start": 1.0, "end": 0.05, "decay": 0.9},
        )
        assert record.episodes["epsilon_source"] == "reconstructed"
        assert len(record.series("epsilon")) == 5
