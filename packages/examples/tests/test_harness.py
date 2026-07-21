"""Tests for the importable training/benchmark harness."""

import pytest

from rl_card_lib.agents import RandomAgent
from rl_card_lib.env import CardGameEnv
from rl_card_lib.games import KlondikeSolitaire, Macao
from rl_card_lib.harness import (
    LEARNERS,
    build_learner,
    checkpoint_suffix,
    epsilon_schedule,
    evaluate_klondike,
    evaluate_macao,
    klondike_baseline_agents,
    macao_baseline_agents,
    make_episode_recorder,
    measure_baselines,
)
from rl_card_lib.trainer import Trainer


@pytest.fixture
def klondike_env():
    return CardGameEnv(KlondikeSolitaire(seed=0), max_steps=20)


class TestLearners:
    """One definition of the hyperparameters, shared by scripts and sweep."""

    @pytest.mark.parametrize("kind", LEARNERS)
    def test_builds_every_learner(self, kind, klondike_env):
        agent = build_learner(
            kind, klondike_env.observation_space.shape[0],
            klondike_env.action_space.n, seed=0,
        )
        assert agent.action_size == klondike_env.action_space.n

    def test_rejects_an_unknown_kind(self, klondike_env):
        with pytest.raises(ValueError, match="Unknown agent"):
            build_learner("not_an_agent", 10, 5, seed=0)

    def test_epsilon_schedule_matches_the_agent(self, klondike_env):
        agent = build_learner("dqn", 221, 68, seed=0)
        schedule = epsilon_schedule("dqn")
        assert schedule["start"] == agent.epsilon_start
        assert schedule["end"] == agent.epsilon_end
        assert schedule["decay"] == pytest.approx(agent.epsilon_decay)

    def test_ppo_has_no_epsilon_schedule(self):
        assert epsilon_schedule("ppo") is None

    def test_checkpoint_suffix_tracks_the_serializer(self):
        """Q-learning pickles; the rest use torch.save."""
        assert checkpoint_suffix("q_learning") == ".pkl"
        assert checkpoint_suffix("dqn") == ".pt"


class TestEvaluation:
    def test_klondike_reports_the_documented_keys(self):
        agent = RandomAgent(action_size=68, seed=0)
        result = evaluate_klondike(agent, episodes=2, max_steps=20)

        assert set(result) == {"reward", "cards_up", "win_rate"}
        assert 0 <= result["cards_up"] <= 52
        assert 0.0 <= result["win_rate"] <= 1.0

    def test_macao_reports_the_documented_keys(self):
        agent = RandomAgent(action_size=65, seed=0)
        opponent = RandomAgent(action_size=65, seed=1)
        result = evaluate_macao(agent, opponent, episodes=2, max_steps=20)

        assert set(result) == {"win_rate", "draw_rate"}
        assert 0.0 <= result["win_rate"] <= 1.0

    def test_evaluation_restores_training_mode(self):
        agent = RandomAgent(action_size=68, seed=0)
        agent.train()
        evaluate_klondike(agent, episodes=1, max_steps=10)
        assert agent.training is True


class TestEpisodeRecorder:
    """Side data the trainer does not keep, captured without touching core."""

    def cards_up_extras(self, game, agent):
        return {"cards_up": sum(len(pile) for pile in game.foundations)}

    def build(self, kind="dqn"):
        env = CardGameEnv(KlondikeSolitaire(seed=0), max_steps=15)
        agent = build_learner(
            kind, env.observation_space.shape[0], env.action_space.n, seed=0,
        )
        return env, agent, self.cards_up_extras

    def test_records_exactly_one_entry_per_episode(self):
        env, agent, extras_fn = self.build()
        trainer = Trainer(env, agent, log_interval=100, eval_interval=10_000)
        callback, extras = make_episode_recorder(env, agent, extras_fn)

        trainer.train(
            episodes=3, max_steps_per_episode=15, verbose=False, callback=callback,
        )

        for name, values in extras.items():
            assert len(values) == 3, name

    def test_custom_series_is_read_from_the_terminal_position(self):
        """The callback fires before the next reset, so the board is still there."""
        env, agent, extras_fn = self.build()
        trainer = Trainer(env, agent, log_interval=100, eval_interval=10_000)
        callback, extras = make_episode_recorder(env, agent, extras_fn)

        trainer.train(
            episodes=3, max_steps_per_episode=15, verbose=False, callback=callback,
        )

        assert all(isinstance(v, int) and 0 <= v <= 52 for v in extras["cards_up"])

    def test_no_extras_callable_records_only_the_duck_typed_series(self):
        env, agent, _ = self.build()
        trainer = Trainer(env, agent, log_interval=100, eval_interval=10_000)
        callback, extras = make_episode_recorder(env, agent)

        trainer.train(
            episodes=2, max_steps_per_episode=15, verbose=False, callback=callback,
        )

        assert set(extras) == {"epsilon", "wall_clock", "table_size"}

    def test_epsilon_is_measured_for_dqn(self):
        env, agent, extras_fn = self.build("dqn")
        trainer = Trainer(env, agent, log_interval=100, eval_interval=10_000)
        callback, extras = make_episode_recorder(env, agent, extras_fn)

        trainer.train(
            episodes=3, max_steps_per_episode=15, verbose=False, callback=callback,
        )

        assert all(0.0 <= v <= 1.0 for v in extras["epsilon"])

    def test_epsilon_is_absent_for_ppo(self):
        env, agent, extras_fn = self.build("ppo")
        trainer = Trainer(env, agent, log_interval=100, eval_interval=10_000)
        callback, extras = make_episode_recorder(env, agent, extras_fn)

        trainer.train(
            episodes=2, max_steps_per_episode=15, verbose=False, callback=callback,
        )

        assert all(v is None for v in extras["epsilon"])

    def test_table_size_is_recorded_for_tabular(self):
        env, agent, extras_fn = self.build("q_learning")
        trainer = Trainer(env, agent, log_interval=100, eval_interval=10_000)
        callback, extras = make_episode_recorder(env, agent, extras_fn)

        trainer.train(
            episodes=3, max_steps_per_episode=15, verbose=False, callback=callback,
        )

        assert extras["table_size"][-1] >= extras["table_size"][0]
        assert extras["table_size"][-1] > 0

    def test_a_game_without_the_extras_attribute_records_none(self):
        """An extras callable that finds no signal appends None, not a crash."""
        env = CardGameEnv(Macao(num_players=2, seed=0), max_steps=15)
        agent = build_learner(
            "dqn", env.observation_space.shape[0], env.action_space.n, seed=0,
        )
        trainer = Trainer(env, agent, log_interval=100, eval_interval=10_000)
        # Macao has no foundations; the Klondike-style extras returns None.
        callback, extras = make_episode_recorder(
            env, agent,
            lambda g, a: {"cards_up": getattr(g, "foundations", None) and 0},
        )

        trainer.train(
            episodes=2, max_steps_per_episode=15, verbose=False, callback=callback,
        )

        assert all(v is None for v in extras["cards_up"])

    def test_evaluation_does_not_pollute_the_series(self):
        """Trainer.evaluate() calls _run_episode directly and skips the callback."""
        env, agent, extras_fn = self.build()
        trainer = Trainer(
            env, agent, log_interval=100, eval_interval=2, eval_episodes=2,
        )
        callback, extras = make_episode_recorder(env, agent, extras_fn)

        trainer.train(
            episodes=4, max_steps_per_episode=15, verbose=False, callback=callback,
        )

        assert len(extras["cards_up"]) == 4


class TestBaselineAgents:
    def test_klondike_baselines_are_all_non_learning(self):
        agents = klondike_baseline_agents(seed=0)
        names = [name for name, _ in agents]

        assert names[0] == "Random"
        assert "Heuristic" in names
        assert any(name.startswith("MCTS") for name in names)

    def test_mcts_budget_is_a_parameter(self):
        agents = dict(klondike_baseline_agents(seed=0, mcts_simulations=7))
        assert agents["MCTS(7)"].simulations == 7

    def test_macao_baselines_use_the_macao_action_space(self):
        agents = macao_baseline_agents(seed=0)
        assert dict(agents)["Random"].action_size == 65


class TestSweepRegistry:
    """The bundled games register through the same API a custom game uses."""

    def test_bundled_games_are_registered(self):
        from rl_card_lib.harness import registered_sweep_games

        registered = registered_sweep_games()
        assert "klondike" in registered
        assert "macao" in registered

    def test_execution_and_presentation_are_both_set(self):
        from rl_card_lib.harness import sweep_game
        from rl_card_lib.report import game_spec

        k = sweep_game("klondike")
        assert k.max_steps == 300
        assert k.self_play is False
        assert k.episode_extras is not None
        # presentation forwarded to the report registry
        assert game_spec("klondike")["headline_key"] == "cards_up"

    def test_macao_declares_self_play(self):
        from rl_card_lib.harness import sweep_game

        m = sweep_game("macao")
        assert m.self_play is True
        assert m.opponent_factory is not None

    def test_env_factory_builds_a_usable_env(self):
        from rl_card_lib.harness import sweep_game

        env = sweep_game("klondike").env_factory()
        assert env.action_space.n == 68

    def test_unknown_game_lists_the_registered_ones(self):
        from rl_card_lib.harness import sweep_game

        with pytest.raises(KeyError, match="klondike"):
            sweep_game("no_such_game")

    def test_custom_game_registers_through_the_api(self):
        from rl_card_lib.harness.registry import (
            _SWEEP_GAMES, register_sweep_game, sweep_game,
        )
        from rl_card_lib.report.run_record import GAME_SPEC

        try:
            register_sweep_game(
                "toy", env_factory=lambda: None, max_steps=10,
                evaluate=lambda a, e, s: {"score": 1.0},
                label="Toy", headline_key="score", headline_label="Score",
                headline_max=100, higher_is_better=False,
            )
            assert sweep_game("toy").max_steps == 10
            # presentation reached the report registry, trainer inferred
            from rl_card_lib.report import game_spec
            assert game_spec("toy")["headline_key"] == "score"
            assert game_spec("toy")["trainer"] == "Trainer"
        finally:
            _SWEEP_GAMES.pop("toy", None)
            GAME_SPEC.pop("toy", None)


class TestGenericBaselines:
    """measure_baselines is registry-driven and degrades on unknown games."""

    def test_measures_a_registered_game(self):
        rows, protocol = measure_baselines("klondike", episodes=2, seed=0,
                                           mcts_simulations=2, verbose=False)
        names = [r["agent"] for r in rows]
        assert "Random" in names
        assert any(n.startswith("MCTS") for n in names)
        # each row carries the game's own headline metric
        assert all("cards_up" in r for r in rows)
        assert protocol["max_steps"] == 300

    def test_unregistered_game_degrades_to_no_baselines(self):
        rows, protocol = measure_baselines("no_such_game", episodes=1,
                                           verbose=False)
        assert rows == []
        assert protocol.get("unregistered") is True

    def test_search_baseline_skipped_without_copy(self):
        """A game whose copy() fails still gets the non-search baselines."""
        from rl_card_lib.harness.baselines import baseline_agents
        from rl_card_lib.harness.registry import SweepGame

        class NoCopyEnv:
            class game:
                @staticmethod
                def copy():
                    raise RuntimeError("cannot copy")
            action_space = type("A", (), {"n": 5})()

        sg = SweepGame(
            name="nc", env_factory=lambda: NoCopyEnv(), max_steps=10,
            evaluate=lambda a, e, s: {"win_rate": 0.0},
        )
        names = [name for name, _ in baseline_agents(sg)]
        assert "Random" in names
        assert not any(n.startswith("MCTS") for n in names)
