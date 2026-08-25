"""Preliminary measurement of the `noloop` arm: does pricing a repeat help?

`diagnosis.md` D3 proposes `repeated_position_penalty = -0.05` as the remedy
for greedy looping, and `run_one.py` implements the arm -- but the arm was
never actually run: `tables/ablation_fixes.csv` holds only `asis` and `fixed`.
This script produces the first numbers for it.

Deliberately *off protocol* and much cheaper than `run_one.py`: 1200 episodes
against the protocol's 5000, 2 initialisation seeds against 3, 30 TEST deals
against 200, DQN only. Its output is therefore indicative and NOT comparable
with the `asis`/`fixed` rows in the ablation table -- it is here to answer one
qualitative question (does the penalty reduce looping at all?), not to fill a
row. What it does share with `run_one.py`: the TRAIN pool, the 300-step cap,
and greedy evaluation on an *unshaped* env, so the two arms differ in exactly
one thing.

Reads nothing and writes nothing; print the output into
`raw/noloop_preliminary.json` by hand, as that file records.

    python thesis_notes/scripts/probe_noloop_preliminary.py
"""

from __future__ import annotations

import time

import numpy as np

from rl_card_lib.env import CardGameEnv
from rl_card_lib.games.klondike import KlondikeSolitaire
from rl_card_lib.harness import TRAIN_SEEDS, build_learner
from rl_card_lib.trainer import Trainer

EPISODES = 1200
INIT_SEEDS = (0, 1)
MAX_STEPS = 300
EVAL_DEALS = range(100000, 100030)

#: The two arms. `asis` is the library as published, `noloop` is D3's remedy.
ARMS = (("asis", 0.0), ("noloop", -0.05))


def train(penalty: float, init_seed: int):
    """One DQN run on Klondike, arms differing only in `penalty`."""
    env = CardGameEnv(KlondikeSolitaire(), max_steps=MAX_STEPS,
                      deal_seeds=TRAIN_SEEDS,
                      repeated_position_penalty=penalty)
    agent = build_learner("dqn", env.observation_space.shape[0],
                          env.action_space.n, init_seed)
    Trainer(env, agent).train(EPISODES, max_steps_per_episode=MAX_STEPS,
                              verbose=False)
    return agent


def probe(agent) -> tuple[float, float, float]:
    """Greedy evaluation on an unshaped env: repeat rate, draw share, cards up.

    Unshaped on purpose -- the penalty is a training signal, and measuring the
    agent on the shaped env would report how much penalty it paid rather than
    how well it plays.
    """
    game = KlondikeSolitaire()
    env = CardGameEnv(game, max_steps=MAX_STEPS)
    if hasattr(agent, "bind"):
        agent.bind(env)
    agent.eval()

    repeats = steps = draws = 0
    cards = []
    for deal in EVAL_DEALS:
        observation, info = env.reset(seed=deal)
        agent.reset()
        for _ in range(MAX_STEPS):
            action = agent.select_action(observation, info["legal_actions"])
            draws += action == 0          # "draw from stock"
            observation, _, terminated, truncated, info = env.step(action)
            steps += 1
            repeats += bool(info.get("repeated_position"))
            if terminated or truncated:
                break
        cards.append(sum(len(pile) for pile in game.foundations))

    return (repeats / max(steps, 1), draws / max(steps, 1),
            float(np.mean(cards)))


def main() -> None:
    print(f"DQN, {EPISODES} episodes, greedy eval on "
          f"{len(list(EVAL_DEALS))} TEST deals", flush=True)
    print(f"{'arm':10s} {'seed':>4s} {'repeats':>8s} {'draw':>7s} "
          f"{'cards_up':>9s}", flush=True)

    rows: dict[str, list] = {}
    for arm, penalty in ARMS:
        for init_seed in INIT_SEEDS:
            started = time.time()
            rate, draw, cards = probe(train(penalty, init_seed))
            rows.setdefault(arm, []).append((rate, draw, cards))
            print(f"{arm:10s} {init_seed:>4d} {rate:>7.1%} {draw:>6.1%} "
                  f"{cards:>9.2f}   ({time.time() - started:.0f}s)", flush=True)

    print(flush=True)
    for arm, values in rows.items():
        a = np.array(values)
        print(f"{arm:10s} mean  repeats {a[:, 0].mean():.1%}  "
              f"draw {a[:, 1].mean():.1%}  cards_up {a[:, 2].mean():.2f}",
              flush=True)


if __name__ == "__main__":
    main()
