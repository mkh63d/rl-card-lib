"""The bundled games register themselves through the public sweep API.

These two calls are the reference a custom game copies. Nothing about Klondike
or Macao is special-cased in the sweep or the report: they go through the same
`register_sweep_game` door any other game would, and everything the sweep needs
to run them is declared here rather than branched on a game name.
"""

from __future__ import annotations

from rl_card_lib.agents import RandomAgent
from rl_card_lib.env import CardGameEnv
from rl_card_lib.games.heuristics import KlondikeHeuristicAgent, MacaoHeuristicAgent
from rl_card_lib.games.klondike import KlondikeSolitaire
from rl_card_lib.games.klondike_solver import solve_klondike
from rl_card_lib.games.macao import Macao
from rl_card_lib.harness.evaluation import evaluate_klondike, evaluate_macao_suite
from rl_card_lib.harness.registry import register_sweep_game

KLONDIKE_MAX_STEPS = 300
MACAO_MAX_STEPS = 200

# Node budget for curating the solvable-deal pool. Deliberately far below the
# solver's own 50k default: winnable deals resolve in a few hundred nodes, so a
# small budget keeps curation fast and lets undecided deals fail quickly (they
# are excluded from the pool anyway). See harness/solve_benchmark.py.
KLONDIKE_POOL_SOLVE_NODES = 10_000


def _klondike_extras(game, agent):
    """Cards moved to the foundations -- Klondike's progress signal."""
    foundations = getattr(game, "foundations", None)
    if foundations is None:
        return {"cards_up": None}
    return {"cards_up": sum(len(pile) for pile in foundations)}


def _evaluate_klondike(agent, episodes, seed):
    return evaluate_klondike(agent, episodes, KLONDIKE_MAX_STEPS)


def _evaluate_macao(agent, episodes, seed):
    return evaluate_macao_suite(
        agent,
        {
            "random": RandomAgent(action_size=Macao(num_players=2)
                                  .get_action_space_size(), seed=seed),
            "heuristic": MacaoHeuristicAgent(seed=seed),
        },
        episodes, MACAO_MAX_STEPS,
    )


def register_bundled_games() -> None:
    """Register Klondike and Macao. Idempotent; called on package import."""
    register_sweep_game(
        "klondike",
        env_factory=lambda: CardGameEnv(KlondikeSolitaire(), max_steps=KLONDIKE_MAX_STEPS),
        max_steps=KLONDIKE_MAX_STEPS,
        evaluate=_evaluate_klondike,
        episode_extras=_klondike_extras,
        heuristic_factory=lambda seed: KlondikeHeuristicAgent(seed=seed),
        single_player=True,
        solver=lambda game: solve_klondike(game, max_nodes=KLONDIKE_POOL_SOLVE_NODES),
        mcts_simulations=20,
        mcts_rollout_depth=15,
        # presentation
        label="Klondike Solitaire",
        headline_key="cards_up",
        headline_label="Cards to foundation",
        headline_unit="cards",
        headline_max=52.0,
        headline_format="{:.1f}",
        higher_is_better=True,
        episode_curves=["cards_up"],
        opponents=[],
        secondary=["reward", "win_rate"],
    )

    register_sweep_game(
        "macao",
        env_factory=lambda: CardGameEnv(Macao(num_players=2), max_steps=MACAO_MAX_STEPS),
        max_steps=MACAO_MAX_STEPS,
        evaluate=_evaluate_macao,
        self_play=True,
        opponent_factory=lambda seed: MacaoHeuristicAgent(seed=seed),
        heuristic_factory=lambda seed: MacaoHeuristicAgent(seed=seed),
        mcts_simulations=40,
        mcts_rollout_depth=20,
        # presentation
        label="Macao",
        headline_key="win_rate_vs_heuristic",
        headline_label="Win rate vs heuristic",
        headline_unit="",
        headline_max=1.0,
        headline_format="{:.1%}",
        higher_is_better=True,
        episode_curves=[],
        opponents=["random", "heuristic"],
        secondary=["win_rate_vs_random", "reward"],
    )
