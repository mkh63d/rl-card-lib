"""Does the greedy policy fail because it is greedy, or because it is bad?

Takes the already-trained checkpoints and evaluates each one twice on the same
200 TEST deals: once fully greedy (the protocol the thesis reports) and once
with a small amount of exploration left on (eps = 0.05, the value the schedule
actually floors at). Nothing is retrained.

If the loop diagnosis is right -- a deterministic policy in an environment with
reversible moves cycles, and epsilon is the only thing that ever broke the
cycle -- then the same weights should score much better with eps = 0.05 than
with eps = 0.

Writes thesis_notes/raw/greedy_vs_epsilon.json.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from harness import evaluate_klondike_on_pool, evaluate_macao_suite_on_pool  # noqa: E402
from split import TEST_SEEDS  # noqa: E402

from rl_card_lib.env import CardGameEnv  # noqa: E402
from rl_card_lib.games import KlondikeSolitaire  # noqa: E402
from rl_card_lib.harness import build_learner  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "raw")
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))

KLONDIKE_MAX_STEPS = 300


def klondike_with_epsilon(agent, epsilon: float, seeds) -> dict:
    """Play the TEST pool with a fixed exploration rate.

    `agent.eval()` is what turns exploration off in this library, so an
    epsilon-greedy evaluation cannot be expressed through the public API; the
    epsilon-greedy branch is reproduced here instead of mutating the agent.
    """
    rng = np.random.RandomState(0)
    game = KlondikeSolitaire()
    env = CardGameEnv(game, max_steps=KLONDIKE_MAX_STEPS)
    if hasattr(agent, "bind"):
        agent.bind(env)
    agent.eval()

    cards, repeats, wins = [], [], 0
    for seed in seeds:
        observation, info = env.reset(seed=seed)
        agent.reset()
        seen, repeated, steps = set(), 0, 0
        for _ in range(KLONDIKE_MAX_STEPS):
            legal = info.get("legal_actions") or []
            if legal and epsilon > 0 and rng.random_sample() < epsilon:
                action = int(rng.choice(legal))
            else:
                action = agent.select_action(observation, legal)
            observation, _, terminated, truncated, info = env.step(action)
            steps += 1
            key = observation.tobytes()
            if key in seen:
                repeated += 1
            seen.add(key)
            if terminated or truncated:
                break
        cards.append(sum(len(p) for p in game.foundations))
        repeats.append(repeated / max(1, steps))
        wins += 1 if game.winner == 0 else 0

    return {
        "epsilon": epsilon,
        "cards_up": round(float(np.mean(cards)), 3),
        "cards_up_sd": round(float(np.std(cards)), 3),
        "win_rate": wins / len(seeds),
        "repeated_position_frac": round(float(np.mean(repeats)), 4),
    }


def main() -> int:
    torch.set_num_threads(1)
    os.makedirs(RAW, exist_ok=True)

    out: dict = {"pool": "TEST (200 deals, seeds 100000-100199)",
                 "note": "same weights, only the action-selection rule differs"}

    checkpoints = {
        "dqn": os.path.join(REPO, "checkpoints", "klondike_dqn", "final.pt"),
        "double_dqn": os.path.join(REPO, "checkpoints", "klondike_double_dqn",
                                   "final.pt"),
        "ppo": os.path.join(REPO, "checkpoints", "klondike_ppo", "final.pt"),
        "q_learning": os.path.join(REPO, "checkpoints", "klondike_q_learning",
                                   "final.pkl"),
    }

    rows = {}
    for kind, path in checkpoints.items():
        if not os.path.exists(path):
            print(f"  skip {kind}: no checkpoint at {path}", flush=True)
            continue
        for epsilon in (0.0, 0.05, 0.20):
            agent = build_learner(kind, 221, 68, 0)
            agent.load(path)
            result = klondike_with_epsilon(agent, epsilon, TEST_SEEDS)
            rows.setdefault(kind, []).append(result)
            print(f"  {kind:11s} eps={epsilon:<5} cards={result['cards_up']:6.2f} "
                  f"win={result['win_rate']:5.1%} "
                  f"repeat={result['repeated_position_frac']:.1%}", flush=True)
    out["klondike"] = rows

    path = os.path.join(RAW, "greedy_vs_epsilon.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2)
    print(f"Wrote {os.path.abspath(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
