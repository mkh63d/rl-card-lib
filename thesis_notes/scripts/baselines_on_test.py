"""Measure every non-learning agent on the same TEST pool the learners use.

The learners' TEST numbers are only a comparison if the baselines play the very
same 200 deals, greedily, with no learning. That is what this does: it reuses
the pool from split.py and the evaluation functions from harness.py, so a
baseline row and a learner row differ only in the policy.

Also runs the Klondike solve-time benchmark over TEST_SOLVABLE.

Usage:
    python thesis_notes/scripts/baselines_on_test.py
    python thesis_notes/scripts/baselines_on_test.py --skip-mcts
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from harness import (  # noqa: E402
    evaluate_klondike_on_pool,
    evaluate_macao_suite_on_pool,
    frozen_exploration,
)
from split import TEST_SEEDS, klondike_test_solvable  # noqa: E402

from rl_card_lib.agents import GreedyLookaheadAgent, MCTSAgent, RandomAgent  # noqa: E402
from rl_card_lib.env import CardGameEnv  # noqa: E402
from rl_card_lib.games import KlondikeSolitaire, Macao  # noqa: E402
# The baselines have to play the same Klondike the learners train on: since #30
# the bundled game caps passes through the stock, and a baseline measured under
# the unlimited-pass default would be scored on a different game than the rows
# it sits next to. See games/registration.py, which makes the same argument.
from rl_card_lib.games.registration import KLONDIKE_MAX_PASSES  # noqa: E402
from rl_card_lib.games.heuristics import (  # noqa: E402
    KlondikeHeuristicAgent, MacaoHeuristicAgent,
)

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "raw")

KLONDIKE_MAX_STEPS = 300
MACAO_MAX_STEPS = 200
KLONDIKE_MCTS_SIMS = 20
MACAO_MCTS_SIMS = 40


def klondike_agents(seed: int, skip_mcts: bool) -> list:
    agents = [
        ("Random", RandomAgent(action_size=KlondikeSolitaire.MAX_ACTIONS, seed=seed)),
        ("Heuristic", KlondikeHeuristicAgent(seed=seed)),
        ("GreedyLookahead(1)", GreedyLookaheadAgent(depth=1, seed=seed)),
    ]
    if not skip_mcts:
        agents.append((f"MCTS({KLONDIKE_MCTS_SIMS})", MCTSAgent(
            simulations=KLONDIKE_MCTS_SIMS, rollout_depth=15, seed=seed)))
    return agents


def macao_agents(seed: int, skip_mcts: bool) -> list:
    agents = [
        ("Random", RandomAgent(action_size=Macao.MAX_ACTIONS, seed=seed)),
        ("Heuristic", MacaoHeuristicAgent(seed=seed)),
        ("GreedyLookahead(1)", GreedyLookaheadAgent(depth=1, seed=seed)),
    ]
    if not skip_mcts:
        agents.append((f"MCTS({MACAO_MCTS_SIMS})", MCTSAgent(
            simulations=MACAO_MCTS_SIMS, rollout_depth=20, seed=seed)))
    return agents


def solve_time_on_pool(agent, seeds: list[int]) -> dict:
    """Solve rate, moves and wall clock over deals proven winnable."""
    game = KlondikeSolitaire(max_passes=KLONDIKE_MAX_PASSES)
    env = CardGameEnv(game, max_steps=KLONDIKE_MAX_STEPS)
    if hasattr(agent, "bind"):
        agent.bind(env)

    solved_moves, solved_seconds, cards = [], [], []
    with frozen_exploration(agent):
        agent.eval()
        for seed in seeds:
            observation, info = env.reset(seed=seed)
            agent.reset()
            moves = 0
            started = time.perf_counter()
            for _ in range(KLONDIKE_MAX_STEPS):
                action = agent.select_action(observation, info.get("legal_actions"))
                observation, _, terminated, truncated, info = env.step(action)
                moves += 1
                if terminated or truncated:
                    break
            elapsed = time.perf_counter() - started
            if game.winner == 0:
                solved_moves.append(moves)
                solved_seconds.append(elapsed)
            cards.append(sum(len(p) for p in game.foundations))

    n = len(seeds)
    return {
        "pool_size": n,
        "solve_rate": len(solved_moves) / n if n else 0.0,
        "solve_moves": sum(solved_moves) / len(solved_moves) if solved_moves else None,
        "solve_seconds": (sum(solved_seconds) / len(solved_seconds)
                          if solved_seconds else None),
        "cards_up": sum(cards) / n if n else 0.0,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-mcts", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    torch.set_num_threads(1)
    os.makedirs(RAW, exist_ok=True)
    out: dict = {"pool": {"test": [TEST_SEEDS[0], TEST_SEEDS[-1] + 1],
                          "size": len(TEST_SEEDS)},
                 "protocol": {"klondike_max_steps": KLONDIKE_MAX_STEPS,
                              "macao_max_steps": MACAO_MAX_STEPS,
                              "mode": "greedy (agent.eval()), no learning",
                              "seed": args.seed}}

    print("== Klondike baselines on TEST ==", flush=True)
    rows = []
    for name, agent in klondike_agents(args.seed, args.skip_mcts):
        started = time.time()
        row = {"agent": name, **evaluate_klondike_on_pool(
            agent, TEST_SEEDS, KLONDIKE_MAX_STEPS,
            max_passes=KLONDIKE_MAX_PASSES)}
        rows.append(row)
        print(f"  {name:22s} cards_up={row['cards_up']:5.2f} "
              f"win={row['win_rate']:5.1%}  ({time.time() - started:.0f}s)",
              flush=True)
    out["klondike"] = rows

    print("== Macao baselines on TEST ==", flush=True)
    rows = []
    for name, agent in macao_agents(args.seed, args.skip_mcts):
        started = time.time()
        row = {"agent": name, **evaluate_macao_suite_on_pool(
            agent, TEST_SEEDS, MACAO_MAX_STEPS, opponent_seed=args.seed)}
        rows.append(row)
        print(f"  {name:22s} vs_heur={row['win_rate_vs_heuristic']:5.1%} "
              f"vs_rand={row['win_rate_vs_random']:5.1%}  "
              f"({time.time() - started:.0f}s)", flush=True)
    out["macao"] = rows

    print("== Klondike solve-time benchmark on TEST_SOLVABLE ==", flush=True)
    classified = klondike_test_solvable(verbose=False)
    solvable = classified["solvable"]
    out["test_solvable"] = {
        "size": len(solvable),
        "unsolvable": len(classified["unsolvable"]),
        "undecided": len(classified["undecided"]),
        "max_nodes": classified["max_nodes"],
        "seeds": solvable,
    }
    rows = []
    for name, agent in klondike_agents(args.seed, args.skip_mcts):
        started = time.time()
        row = {"agent": name, **solve_time_on_pool(agent, solvable)}
        rows.append(row)
        moves = "n/a" if row["solve_moves"] is None else f"{row['solve_moves']:.0f}"
        print(f"  {name:22s} solved={row['solve_rate']:5.1%} moves={moves}  "
              f"({time.time() - started:.0f}s)", flush=True)
    out["klondike_solve_time"] = rows

    path = os.path.join(RAW, "baselines_on_test.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2)
    print(f"Wrote {os.path.abspath(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
