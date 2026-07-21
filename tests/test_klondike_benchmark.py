"""Tests for the solvable-pool solve-time benchmark.

The measurement logic is exercised with tiny stub games so the tests are
deterministic and fast -- a real deal's solvability depends on the ~50k-node
search and an agent's solve rate depends on its strength, neither of which
belongs in a unit test. The curation and gating tests use the real Klondike and
Macao registrations, but keep pools small.
"""

import rl_card_lib.games  # noqa: F401  (registers klondike + macao and solvers)
from rl_card_lib.games import solve_klondike
from rl_card_lib.harness import (
    curate_solvable_pool,
    load_trained_learner,
    measure_agent_on_pool,
    run_solve_benchmark,
    sweep_game,
)
from rl_card_lib.harness.registry import SweepGame
from rl_card_lib.report import RunStore, SolveBenchmarkSet
from rl_card_lib.report.html_report import HtmlReport


# -- stub game/agent so measurement is deterministic ---------------------------

class _StubGame:
    def __init__(self, solve_after):
        self.solve_after = solve_after  # moves until a win, or None to never win
        self.winner = None
        self.foundations = [[]]


class _StubEnv:
    """Minimal env matching the reset/step contract measure_agent_on_pool uses."""

    def __init__(self, solve_after):
        self.game = _StubGame(solve_after)
        self._moves = 0

    def reset(self, seed=None):
        self.game.winner = None
        self._moves = 0
        return object(), {"legal_actions": [0]}

    def step(self, action):
        self._moves += 1
        after = self.game.solve_after
        if after is not None and self._moves >= after:
            self.game.winner = 0
            return object(), 0.0, True, False, {"legal_actions": [0]}
        return object(), 0.0, False, False, {"legal_actions": [0]}


class _StubAgent:
    def __init__(self):
        self.training = True

    def eval(self):
        self.training = False

    def train(self):
        self.training = True

    def reset(self):
        pass

    def select_action(self, observation, legal_actions=None):
        return 0


def _stub_sweep(solve_after, *, max_steps=50, solver=lambda g: True):
    return SweepGame(
        name="stub",
        env_factory=lambda: _StubEnv(solve_after),
        max_steps=max_steps,
        evaluate=lambda *a, **k: {},
        episode_extras=lambda game, agent: {"cards_up": len(game.foundations[0])},
        solver=solver,
        single_player=True,
    )


class TestMeasureAgentOnPool:
    def test_always_solving_agent_reports_full_rate_and_finite_cost(self):
        sweep = _stub_sweep(solve_after=3)
        row = measure_agent_on_pool(_StubAgent(), sweep, [0, 1, 2])

        assert row["solve_rate"] == 1.0
        assert row["solve_moves"] == 3
        assert row["solve_seconds"] is not None and row["solve_seconds"] >= 0.0
        assert row["pool_size"] == 3

    def test_never_solving_agent_has_zero_rate_and_none_cost(self):
        sweep = _stub_sweep(solve_after=None)
        row = measure_agent_on_pool(_StubAgent(), sweep, [0, 1, 2])

        assert row["solve_rate"] == 0.0
        assert row["solve_moves"] is None
        assert row["solve_seconds"] is None

    def test_restores_training_mode(self):
        sweep = _stub_sweep(solve_after=2)
        agent = _StubAgent()
        assert agent.training is True
        measure_agent_on_pool(agent, sweep, [0])
        assert agent.training is True

    def test_run_solve_benchmark_labels_every_row(self):
        sweep = _stub_sweep(solve_after=2)
        rows = run_solve_benchmark(
            sweep, [("A", _StubAgent()), ("B", _StubAgent())], [0, 1], verbose=False,
        )
        assert [r["agent"] for r in rows] == ["A", "B"]
        assert all(r["solve_rate"] == 1.0 for r in rows)


class TestCurateSolvablePool:
    def test_returns_only_solvable_seeds(self):
        sweep = sweep_game("klondike")
        seeds = curate_solvable_pool(sweep, 3)
        assert len(seeds) == 3
        # Every returned deal is genuinely winnable under the full-budget solver.
        env = sweep.env_factory()
        for seed in seeds:
            env.reset(seed=seed)
            assert solve_klondike(env.game) is True

    def test_stub_solver_controls_membership(self):
        # A solver that proves every deal winnable keeps the first `size` seeds.
        always = _stub_sweep(solve_after=1, solver=lambda g: True)
        assert curate_solvable_pool(always, 3) == [0, 1, 2]
        # A solver that never returns True (only None/undecided) yields nothing.
        undecided = _stub_sweep(solve_after=1, solver=lambda g: None)
        assert curate_solvable_pool(undecided, 3, max_scan=10) == []


class TestSinglePlayerGating:
    def test_multiplayer_game_has_no_solver(self):
        assert sweep_game("macao").solver is None

    def test_curate_rejects_game_without_solver(self):
        try:
            curate_solvable_pool(sweep_game("macao"), 3)
        except ValueError as exc:
            assert "single-player" in str(exc).lower()
        else:
            raise AssertionError("expected ValueError for a game with no solver")


class TestLoadTrainedLearner:
    def test_returns_none_when_checkpoint_absent(self, tmp_path):
        env = sweep_game("klondike").env_factory()
        agent = load_trained_learner(
            "ppo", env, game="klondike", checkpoint_dir=str(tmp_path),
        )
        assert agent is None


class TestReportIntegration:
    def test_benchmark_roundtrips_and_renders(self, tmp_path):
        store = RunStore(tmp_path)
        benchmark = SolveBenchmarkSet(
            game="klondike",
            protocol={"pool_size": 2, "max_steps": 300},
            rows=[
                {"agent": "PPO (trained)", "solve_rate": 0.5, "solve_moves": 120.0,
                 "solve_seconds": 0.03, "cards_up": 40.0, "pool_size": 2},
                {"agent": "Random", "solve_rate": 0.0, "solve_moves": None,
                 "solve_seconds": None, "cards_up": 3.0, "pool_size": 2},
            ],
        )
        store.save_solve_benchmark(benchmark)

        loaded = store.load_solve_benchmarks()
        assert "klondike" in loaded
        assert loaded["klondike"].rows[0]["agent"] == "PPO (trained)"

        html = HtmlReport.build(
            store, with_figures=False, include_games=["klondike"],
        ).to_html()
        assert "Solve-time benchmark" in html
        assert "PPO (trained)" in html
