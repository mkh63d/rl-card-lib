"""Solve-time benchmark for the newly trained learners, on TEST_SOLVABLE.

`baselines_on_test.py` measures the non-learning agents over the pool of TEST
deals a perfect-information solver proved winnable. This adds the learners
trained under the corrected protocol, from `thesis_notes/checkpoints/`, so the
solve-rate table has both halves and every row covers the identical pool --
whatever size `split.klondike_test_solvable` currently classifies it at.

PPO is measured twice per arm, because on Klondike sampling its policy and
taking that policy's argmax are very different policies (diagnosis.md D11). The
row named plainly `ppo (<arm>)` is always the rule that arm actually uses --
argmax for `asis`, sampling for `fixed` and `noloop` since #33 -- and the second
row is the counterfactual, named for the rule it applies.

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
from rl_card_lib.games.registration import KLONDIKE_MAX_PASSES  # noqa: E402
from rl_card_lib.harness import build_learner  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "raw")
CHECKPOINTS = os.path.join(HERE, "..", "checkpoints")
MAX_STEPS = 300
SEEDS = (0, 1, 2)
AGENTS = ("ppo", "double_dqn", "dqn", "q_learning")

#: Passes through the stock each arm plays, matching what its checkpoints were
#: trained under (run_one.ARMS). The TEST_SOLVABLE pool is curated at the
#: bundled limit, which stays a valid -- merely conservative -- pool for `asis`
#: too: a deal winnable within a finite number of passes is still winnable when
#: passes are unlimited.
ARM_MAX_PASSES = {
    "asis": None,
    "fixed": KLONDIKE_MAX_PASSES,
    "noloop": KLONDIKE_MAX_PASSES,
}

#: How each arm reads a move out of PPO's policy, mirroring run_one.ARMS.
#: Only PPO has the choice; for the DQN family the greedy policy is what it
#: learned, and `measure` leaves agents without the attribute alone.
ARM_EVAL_GREEDY = {"asis": True, "fixed": False, "noloop": False}
ARMS = tuple(ARM_MAX_PASSES)


def measure(agent, seeds: list[int], sample: bool = False,
            max_passes: int | None = None) -> dict:
    game = KlondikeSolitaire(max_passes=max_passes)
    env = CardGameEnv(game, max_steps=MAX_STEPS)
    if hasattr(agent, "bind"):
        agent.bind(env)

    solved_moves, solved_seconds, cards = [], [], []
    with frozen_exploration(agent):
        for seed in seeds:
            # Both arms evaluate. `eval_greedy` picks the action rule, so the
            # sampled arm no longer has to enter training mode -- which used to
            # record every step of the measurement into PPO's rollout. Only PPO
            # has the flag; the DQN family's greedy policy is what it learned.
            if hasattr(agent, "eval_greedy"):
                agent.eval_greedy = not sample
            agent.eval()
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
            # The unlabelled row must be the arm's *own* evaluation rule, not
            # a fixed choice: `asis` reads PPO by argmax and `fixed`/`noloop`
            # by sampling (#33). Labelling the argmax measurement as
            # "ppo (fixed)" would name a configuration that arm never uses.
            arm_samples = not ARM_EVAL_GREEDY[arm]
            variants = [(arm_samples, f"{agent} ({arm})")]
            if agent == "ppo":
                other = "sampled" if not arm_samples else "argmax"
                variants.append((not arm_samples, f"{agent} ({arm}, {other})"))
            for sample, label in variants:
                per_seed = []
                for seed in SEEDS:
                    path = path_for(agent, arm, seed)
                    if not os.path.exists(path):
                        continue
                    learner = build_learner(agent, 221, 68, seed)
                    learner.load(path)
                    per_seed.append(measure(
                        learner, pool, sample=sample,
                        max_passes=ARM_MAX_PASSES[arm],
                    ))
                    del learner
                if not per_seed:
                    continue
                row = {
                    "agent": agent, "arm": arm, "sampled": sample,
                    "max_passes": ARM_MAX_PASSES[arm],
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
