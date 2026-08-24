"""Solve-time benchmark for the newly trained learners, on TEST_SOLVABLE.

`baselines_on_test.py` measures the non-learning agents over the pool of TEST
deals a perfect-information solver proved winnable. This adds the learners
trained under the corrected protocol, from `thesis_notes/checkpoints/`, so the
solve-rate table has both halves and every row is the same 102 deals.

PPO is measured twice: once by argmax over its policy (what `agent.eval()`
does) and once by sampling that same policy, because on Klondike the two are
very different policies -- see diagnosis.md D11.

Writes thesis_notes/raw/solve_time_learners.json.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from harness import frozen_exploration  # noqa: E402
from split import klondike_test_solvable  # noqa: E402

from rl_card_lib.env import CardGameEnv  # noqa: E402
from rl_card_lib.games import KlondikeSolitaire  # noqa: E402
from rl_card_lib.harness import build_learner  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "raw")
CHECKPOINTS = os.path.join(HERE, "..", "checkpoints")
MAX_STEPS = 300
SEEDS = (0, 1, 2)
ARMS = ("asis", "fixed", "noloop")
AGENTS = ("ppo", "double_dqn", "dqn", "q_learning")


def measure(agent, seeds: list[int], sample: bool = False) -> dict:
    game = KlondikeSolitaire()
    env = CardGameEnv(game, max_steps=MAX_STEPS)
    if hasattr(agent, "bind"):
        agent.bind(env)

    solved_moves, solved_seconds, cards = [], [], []
    with frozen_exploration(agent):
        for seed in seeds:
            # `training=True` is the branch that samples PPO's policy; the DQN
            # family would also start exploring there, so sampling is only ever
            # requested for PPO.
            agent.train() if sample else agent.eval()
            observation, info = env.reset(seed=seed)
            agent.reset()
            moves = 0
            started = time.perf_counter()
            for _ in range(MAX_STEPS):
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
        "solve_moves": (sum(solved_moves) / len(solved_moves)
                        if solved_moves else None),
        "solve_seconds": (sum(solved_seconds) / len(solved_seconds)
                          if solved_seconds else None),
        "cards_up": sum(cards) / n if n else 0.0,
    }


def path_for(agent: str, arm: str, seed: int) -> str:
    suffix = ".pkl" if agent == "q_learning" else ".pt"
    return os.path.join(CHECKPOINTS,
                        f"klondike__{agent}__{arm}__s{seed}{suffix}")


def main() -> int:
    torch.set_num_threads(1)
    pool = klondike_test_solvable(verbose=False)["solvable"]
    print(f"TEST_SOLVABLE: {len(pool)} deals", flush=True)

    out: dict = {"pool_size": len(pool), "seeds": SEEDS, "rows": []}
    for arm in ARMS:
        for agent in AGENTS:
            variants = [(False, f"{agent} ({arm})")]
            if agent == "ppo":
                variants.append((True, f"{agent} ({arm}, sampled)"))
            for sample, label in variants:
                per_seed = []
                for seed in SEEDS:
                    path = path_for(agent, arm, seed)
                    if not os.path.exists(path):
                        continue
                    learner = build_learner(agent, 221, 68, seed)
                    learner.load(path)
                    per_seed.append(measure(learner, pool, sample=sample))
                    del learner
                if not per_seed:
                    continue
                row = {
                    "agent": agent, "arm": arm, "sampled": sample,
                    "label": label, "seeds": len(per_seed),
                    "solve_rate_mean": float(np.mean(
                        [r["solve_rate"] for r in per_seed])),
                    "solve_rate_sd": float(np.std(
                        [r["solve_rate"] for r in per_seed], ddof=1))
                    if len(per_seed) > 1 else 0.0,
                    "cards_up_mean": float(np.mean(
                        [r["cards_up"] for r in per_seed])),
                    "cards_up_sd": float(np.std(
                        [r["cards_up"] for r in per_seed], ddof=1))
                    if len(per_seed) > 1 else 0.0,
                }
                moves = [r["solve_moves"] for r in per_seed
                         if r["solve_moves"] is not None]
                row["solve_moves_mean"] = float(np.mean(moves)) if moves else None
                out["rows"].append(row)
                print(f"  {label:28s} n={row['seeds']} "
                      f"solved={row['solve_rate_mean']:6.1%} "
                      f"cards={row['cards_up_mean']:5.2f}", flush=True)

    path = os.path.join(RAW, "solve_time_learners.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2)
    print(f"Wrote {os.path.abspath(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
