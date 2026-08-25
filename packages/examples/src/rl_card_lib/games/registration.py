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
from rl_card_lib.harness.deals import TEST_SEEDS, TRAIN_SEEDS
from rl_card_lib.harness.evaluation import evaluate_klondike, evaluate_macao_suite
from rl_card_lib.harness.registry import register_sweep_game

KLONDIKE_MAX_STEPS = 300
MACAO_MAX_STEPS = 200

# Passes through the stock the bundled Klondike allows. Re-exported from the
# game class, which owns the number so `harness` can reach it too -- see
# KlondikeSolitaire.BUNDLED_MAX_PASSES for why a finite value is needed at all:
# under the unlimited default the draw/recycle action never stops being legal,
# so a deal can never run out of moves, so LOSS_REWARD is unreachable and the
# agent trains on a game it can neither win nor lose. See issue #18.
#
# Unlike `repeated_position_penalty` below this is not reward shaping: it
# changes what the environment *is*, so it has to be identical in training, in
# the trainer's periodic evaluation, in the final evaluation protocol, in the
# baselines and in the solver that curates the solvable-deal pool. A game that
# differs across those is not one experiment.
KLONDIKE_MAX_PASSES = KlondikeSolitaire.BUNDLED_MAX_PASSES

# Price of stepping into a position already seen this episode. Klondike's
# tableau moves are reversible and, with the default `max_passes=None`, the
# draw/recycle cycle is free -- so a *deterministic* greedy policy that ever
# revisits a position repeats its whole future from there and cycles until the
# step cap. Measured on trained checkpoints, that is what happens: 80-83 % of
# greedy steps landed in an already-seen position against 23 % for random,
# which is why the greedy policies scored below random. During training eps > 0
# hides it by breaking the cycle roughly every 20 steps; the same weights
# evaluated greedily loop.
#
# CardGameEnv has implemented this penalty (and documented it as the remedy for
# exactly this) since the env was written, but it defaults to 0.0 and nothing
# here ever passed a value -- so the safeguard was inert in every run. These
# constants are what switch it on. See issue #17.
KLONDIKE_REPEAT_PENALTY = -0.05

# Macao gets 0.0 on purpose, not by the same oversight: its positions are
# monotone -- every action moves a card out of a hand or the deck and the
# observation carries the counts -- so a repeat cannot occur. Measured over
# 5 826 random steps the repeat count is exactly 0, leaving nothing for a
# penalty to price.
MACAO_REPEAT_PENALTY = 0.0

# Node budget for curating the solvable-deal pool. Deliberately far below the
# solver's own 50k default: winnable deals resolve in a few hundred nodes, so a
# small budget keeps curation fast and lets undecided deals fail quickly (they
# are excluded from the pool anyway). See harness/solve_benchmark.py.
KLONDIKE_POOL_SOLVE_NODES = 10_000


def _klondike() -> KlondikeSolitaire:
    """The Klondike every bundled entry point plays.

    One constructor for the whole experiment, so training, evaluation, the
    baselines and the solvable-pool solver cannot end up on different rules.
    """
    return KlondikeSolitaire(max_passes=KLONDIKE_MAX_PASSES)


def _train_env(game, max_steps, repeat_penalty=0.0):
    """Env for a training run: one deal per episode out of the TRAIN pool.

    This is the env that carries the repeated-position penalty. The penalty is
    reward *shaping* -- a training signal that prices the cycle -- so it belongs
    to the env the agent learns from and to no other.
    """
    return CardGameEnv(game, max_steps=max_steps, deal_seeds=TRAIN_SEEDS,
                       repeated_position_penalty=repeat_penalty)


def _eval_env(game, max_steps):
    """Env for the trainer's periodic evaluation: held-out deals, in order.

    "cycle" plus the trainer's rewind means every evaluation point replays the
    same TEST deals, so the evaluation curve tracks the agent rather than which
    deals it happened to draw.

    Deliberately *unshaped*: no repeated-position penalty, unlike `_train_env`.
    The curve then reports the game's own return, so a rise means the agent got
    better rather than merely that it stopped paying the penalty -- and the
    number stays comparable with `harness.baselines`, which measures the random
    and heuristic policies on plain envs.
    """
    return CardGameEnv(game, max_steps=max_steps, deal_seeds=TEST_SEEDS,
                       deal_order="cycle")


def _klondike_extras(game, agent):
    """Cards moved to the foundations -- Klondike's progress signal."""
    foundations = getattr(game, "foundations", None)
    if foundations is None:
        return {"cards_up": None}
    return {"cards_up": sum(len(pile) for pile in foundations)}


def _evaluate_klondike(agent, episodes, seed):
    return evaluate_klondike(agent, episodes, KLONDIKE_MAX_STEPS,
                             max_passes=KLONDIKE_MAX_PASSES)


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
        env_factory=lambda: _train_env(_klondike(), KLONDIKE_MAX_STEPS,
                                       KLONDIKE_REPEAT_PENALTY),
        eval_env_factory=lambda: _eval_env(_klondike(), KLONDIKE_MAX_STEPS),
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
        env_factory=lambda: _train_env(Macao(num_players=2), MACAO_MAX_STEPS,
                                       MACAO_REPEAT_PENALTY),
        eval_env_factory=lambda: _eval_env(Macao(num_players=2), MACAO_MAX_STEPS),
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
