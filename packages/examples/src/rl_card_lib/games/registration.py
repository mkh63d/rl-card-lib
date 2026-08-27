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
from rl_card_lib.games.klondike import KlondikeSolitaire, bundled_klondike
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
#
# Nothing here constructs a Klondike with it: `bundled_klondike()` is the one
# constructor that applies it, and this name exists so an entry point that only
# needs the *number* -- the Gymnasium ids' `max_passes` default, the evaluation
# protocol below -- can say it without reaching into the class. See issue #38.
KLONDIKE_MAX_PASSES = KlondikeSolitaire.BUNDLED_MAX_PASSES

# Price of stepping into a position already seen this episode: 0.0, so the
# bundled Klondike trains on the game's own reward and nothing else.
#
# The diagnosis the penalty was written for still holds. Klondike's tableau
# moves are reversible and, with the default `max_passes=None`, the
# draw/recycle cycle is free -- so a *deterministic* greedy policy that ever
# revisits a position repeats its whole future from there and cycles until the
# step cap. Measured on trained checkpoints that is what happens: 73-78 % of
# greedy steps land in an already-seen position against 42 % for random. That
# is why `CardGameEnv` implements the penalty and documents it as the remedy,
# and why issue #17 switched it on here.
#
# It was then measured, and it does not work. Over 3 seeds, 5000 episodes and
# 200 held-out deals it buys 0.3-0.8 pp of that revisit fraction for the DQN
# family and moves it the *wrong* way for PPO, while costing PPO 17.09 -> 9.73
# cards to foundation -- from well above the 9.79 random baseline to below it --
# and 27.5 % -> 0.4 % of its solve rate on the proven-winnable pool. The
# penalty falls on roughly two-thirds of steps for the whole run and never
# trains away. So the bundled game ships unshaped and the mechanism stays
# opt-in: a caller who wants the cycle priced passes
# `repeated_position_penalty` to their own env. See issue #36, and
# `thesis_notes/diagnosis.md` D3 for the measurement.
KLONDIKE_REPEAT_PENALTY = 0.0

# Macao's 0.0 is a different zero from Klondike's above: not a shaping term
# that was measured and withdrawn, but one that never had anything to price.
# Its positions are monotone -- every action moves a card out of a hand or the
# deck and the observation carries the counts -- so a repeat cannot occur.
# Measured over 5 826 random steps the repeat count is exactly 0.
MACAO_REPEAT_PENALTY = 0.0

# Node budget for curating the solvable-deal pool. Deliberately far below the
# solver's own 50k default: winnable deals resolve in a few hundred nodes, so a
# small budget keeps curation fast and lets undecided deals fail quickly (they
# are excluded from the pool anyway). See harness/solve_benchmark.py.
KLONDIKE_POOL_SOLVE_NODES = 10_000


def _train_env(game, max_steps, repeat_penalty=0.0):
    """Env for a training run: one deal per episode out of the TRAIN pool.

    The only env that takes a repeated-position penalty. Both bundled games
    pass 0.0 today, for the separate reasons given on their constants, but the
    parameter belongs here and nowhere else: the penalty is reward *shaping* --
    a training signal that prices the cycle -- so a game that wants it applies
    it to the env its agent learns from and to no other.
    """
    return CardGameEnv(game, max_steps=max_steps, deal_seeds=TRAIN_SEEDS,
                       repeated_position_penalty=repeat_penalty)


def _eval_env(game, max_steps):
    """Env for the trainer's periodic evaluation: held-out deals, in order.

    "cycle" plus the trainer's rewind means every evaluation point replays the
    same TEST deals, so the evaluation curve tracks the agent rather than which
    deals it happened to draw.

    Deliberately *unshaped*, whatever `_train_env` was handed: no
    repeated-position penalty ever reaches this env. The curve then reports the
    game's own return, so a rise means the agent got better rather than merely
    that it stopped paying a shaping term -- and the number stays comparable
    with `harness.baselines`, which measures the random and heuristic policies
    on plain envs.
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
        env_factory=lambda: _train_env(bundled_klondike(), KLONDIKE_MAX_STEPS,
                                       KLONDIKE_REPEAT_PENALTY),
        eval_env_factory=lambda: _eval_env(bundled_klondike(),
                                           KLONDIKE_MAX_STEPS),
        max_steps=KLONDIKE_MAX_STEPS,
        evaluate=_evaluate_klondike,
        episode_extras=_klondike_extras,
        heuristic_factory=lambda seed: KlondikeHeuristicAgent(seed=seed),
        single_player=True,
        solver=lambda game: solve_klondike(game, max_nodes=KLONDIKE_POOL_SOLVE_NODES),
        mcts_simulations=20,
        mcts_rollout_depth=15,
        # Every Klondike episode runs to the 300-step cap, so 500 gradient
        # steps is a target refresh every 1.7 episodes. Stated explicitly
        # rather than inherited, so the pairing with the step cap is visible.
        target_update_freq=500,
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
        # A Macao episode averages 46 steps, so Klondike's 500 would refresh the
        # target only every 10.9 episodes -- 460 refreshes in a 5000-episode
        # run, for a game whose reward is a rare terminal +10. 100 gradient
        # steps is a refresh every ~2.2 episodes, the same order as Klondike's
        # 1.7, and lands in the 100-200 band the measurement pass recommended.
        target_update_freq=100,
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
