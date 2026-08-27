"""
Train `sb3-contrib`'s MaskablePPO on Macao and measure it on the held-out deals.

This is the example that backs the claim a third-party algorithm can learn these
games, rather than merely being accepted by them. Stable-Baselines3 has
constructed `PPO("MlpPolicy", CardGameEnv(...))` since #15, but an unmasked
policy achieves nothing here: only 2--4 of Macao's 65 actions are legal in a
typical position, and a 2000-step SB3 PPO run scored -198.9 over 200 steps,
almost exactly 200 x `invalid_action_reward`. It never found a legal move.

MaskablePPO asks the environment for its mask, via a method named exactly
`action_masks()`. `CardGameEnv` supplies it, so `rl_card_lib/MacaoMasked-v0`
trains off-the-shelf with no adapter.

The model plays both seats while training, which is well posed because Macao's
observation is written from the current player's side and its rewards are paid
to whoever acted -- the same self-play arrangement the bundled agents get. It is
then evaluated seated as player 0 through `rl_card_lib.harness.evaluation`, the
same protocol and the same 200 held-out deals every bundled agent is scored on.

Reference agents are measured in the same run and written to the same file: the
existing `results/baselines/macao.json` was recorded over 30 deals, so quoting
it beside a 200-deal number would compare two different measurements.

`sb3-contrib` is an optional extra:

    pip install -e "./packages/examples[sb3]"

Usage:
    python packages/examples/scripts/train_maskable_ppo.py
    python packages/examples/scripts/train_maskable_ppo.py --timesteps 500000
    python packages/examples/scripts/train_maskable_ppo.py --episodes 50 --with-mcts
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import rl_card_lib.games  # noqa: F401  (import side effect: registration)
from rl_card_lib.agents import GreedyLookaheadAgent, MCTSAgent, RandomAgent
from rl_card_lib.games import Macao, MacaoHeuristicAgent
from rl_card_lib.harness import evaluate_macao_suite, evaluation_seeds
from rl_card_lib.harness.deals import (
    TEST_SEED_END,
    TEST_SEED_START,
    TRAIN_SEED_END,
    TRAIN_SEED_START,
)


def _require_sb3_contrib():
    """Import the optional dependency, or explain how to get it."""
    try:
        from rl_card_lib.harness.sb3_maskable import (
            MaskablePPOAgent,
            train_maskable_ppo,
        )
    except ImportError as exc:
        print(
            "error: this example needs sb3-contrib ({}).\n"
            '       pip install -e "./packages/examples[sb3]"'.format(exc),
            file=sys.stderr,
        )
        raise SystemExit(2)
    return MaskablePPOAgent, train_maskable_ppo


def _opponents(seed: int) -> dict:
    """The two named opponents every Macao row is scored against."""
    probe = Macao(num_players=2)
    return {
        "random": RandomAgent(action_size=probe.get_action_space_size(), seed=seed),
        "heuristic": MacaoHeuristicAgent(seed=seed),
    }


def _reference_agents(seed: int, *, with_mcts: bool, mcts_simulations: int) -> list:
    """Fixed-strength agents, measured on the same deals for context.

    MCTS is off by default: it costs roughly a second per deal, which dwarfs
    everything else here without changing what the example demonstrates.
    """
    probe = Macao(num_players=2)
    agents = [
        ("Random", RandomAgent(action_size=probe.get_action_space_size(), seed=seed)),
        ("Heuristic", MacaoHeuristicAgent(seed=seed)),
        ("GreedyLookahead(1)", GreedyLookaheadAgent(depth=1, seed=seed)),
    ]
    if with_mcts:
        agents.append((
            "MCTS({})".format(mcts_simulations),
            MCTSAgent(simulations=mcts_simulations, rollout_depth=20, seed=seed),
        ))
    return agents


def _measure(name: str, agent, opponents: dict, deals: list) -> dict:
    started = time.time()
    row = {"agent": name}
    row.update(evaluate_macao_suite(agent, opponents, seeds=deals))
    row["seconds"] = time.time() - started
    vs_random = row["win_rate_vs_random"]
    vs_heuristic = row["win_rate_vs_heuristic"]
    print(
        "  {:24s} vs random {:6.1%}   vs heuristic {:6.1%}   ({:.1f}s)".format(
            name, vs_random, vs_heuristic, row["seconds"],
        ),
        flush=True,
    )
    return row


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--timesteps", type=int, default=300_000,
                        help="Environment steps to train for (default: 300000)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for the policy and the opponents (default: 0)")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Deals to evaluate on (default: the whole 200-deal TEST pool)")
    parser.add_argument("--checkpoint-dir", default="./checkpoints/maskable_ppo",
                        help="Where the trained model is written")
    parser.add_argument("--results-dir", default="./results/baselines",
                        help="Where the measured rows are written")
    parser.add_argument("--with-mcts", action="store_true",
                        help="Also measure MCTS (slow: roughly a second per deal)")
    parser.add_argument("--mcts-simulations", type=int, default=40,
                        help="MCTS budget per move when --with-mcts is given")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress MaskablePPO's own progress table")
    args = parser.parse_args(argv)

    MaskablePPOAgent, train_maskable_ppo = _require_sb3_contrib()

    deals = evaluation_seeds(args.episodes)
    print(
        "Training MaskablePPO for {:,} steps (seed {})...".format(
            args.timesteps, args.seed,
        ),
        flush=True,
    )
    started = time.time()
    model = train_maskable_ppo(
        args.timesteps, seed=args.seed, verbose=0 if args.quiet else 1,
    )
    train_seconds = time.time() - started
    print("Trained in {:.1f}s".format(train_seconds), flush=True)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint = os.path.join(args.checkpoint_dir, "macao_maskable_ppo.zip")
    model.save(checkpoint)
    print("Saved {}".format(checkpoint), flush=True)

    label = "MaskablePPO({}k)".format(args.timesteps // 1000)
    print("\nEvaluating on {} held-out deals...".format(len(deals)), flush=True)
    rows = [_measure(label, MaskablePPOAgent(model, name=label),
                     _opponents(args.seed), deals)]
    for name, agent in _reference_agents(
        args.seed, with_mcts=args.with_mcts, mcts_simulations=args.mcts_simulations,
    ):
        rows.append(_measure(name, agent, _opponents(args.seed), deals))

    payload = {
        "game": "macao",
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "episodes": len(deals),
            "max_steps": 200,
            "seed": args.seed,
            "algorithm": "sb3-contrib MaskablePPO (MultiInputPolicy)",
            "env_id": "rl_card_lib/MacaoMasked-v0",
            "timesteps": args.timesteps,
            "train_seconds": train_seconds,
            "self_play": True,
            "train_pool": [TRAIN_SEED_START, TRAIN_SEED_END],
            "test_pool": [TEST_SEED_START, TEST_SEED_END],
        },
        "rows": rows,
    }
    os.makedirs(args.results_dir, exist_ok=True)
    out = os.path.join(args.results_dir, "macao_maskable_ppo.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print("\nWrote {}".format(out), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
