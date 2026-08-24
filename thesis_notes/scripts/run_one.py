"""Train one (game, agent, init-seed, arm) combination under the corrected protocol.

    TRAIN   deals drawn uniformly from seeds 0..9999, one fresh deal per episode
    TEST    all 200 deals of seeds 100000..100199, greedy, no learning, before
            and after training
    arms    `asis`  library code exactly as the thesis describes it
            `fixed` + the time-limit bootstrap fix from diagnosis.md

Writes one JSON per run to thesis_notes/raw/runs/. Re-running skips a run whose
JSON already exists unless --force is given, so the sweep is resumable.

Usage:
    python thesis_notes/scripts/run_one.py --game klondike --agent dqn \
        --init-seed 0 --episodes 5000 --arm asis
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from harness import (  # noqa: E402
    BootstrapSelfPlayTrainer,
    BootstrapTrainer,
    PooledEnv,
    block_average,
    evaluate_klondike_on_pool,
    evaluate_macao_suite_on_pool,
    make_train_recorder,
)
from split import TEST_SEEDS, TRAIN_SEEDS  # noqa: E402

from rl_card_lib.games.heuristics import MacaoHeuristicAgent  # noqa: E402
from rl_card_lib.games.klondike import KlondikeSolitaire  # noqa: E402
from rl_card_lib.games.macao import Macao  # noqa: E402
from rl_card_lib.harness import build_learner  # noqa: E402
from rl_card_lib.trainer import SelfPlayTrainer, Trainer  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(HERE, "..", "raw", "runs")

KLONDIKE_MAX_STEPS = 300
MACAO_MAX_STEPS = 200
TRAIN_BLOCK = 50           # episodes per TRAIN metric row

#: Penalty added to the reward whenever a step lands in a position already seen
#: this episode. CardGameEnv already implements it and defaults it to 0.0, so
#: the bundled games never switch it on; the "noloop" arm does. See
#: diagnosis.md D5 -- the trained greedy policies spend ~80% of their moves
#: revisiting positions.
LOOP_PENALTY = -0.05


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=HERE, text=True,
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def make_env(game_name: str, deal_rng_seed: int, arm: str = "asis"):
    penalty = LOOP_PENALTY if arm == "noloop" else 0.0
    if game_name == "klondike":
        return PooledEnv(
            KlondikeSolitaire(), pool=TRAIN_SEEDS, deal_rng_seed=deal_rng_seed,
            max_steps=KLONDIKE_MAX_STEPS, repeated_position_penalty=penalty,
        )
    if game_name == "macao":
        return PooledEnv(
            Macao(num_players=2), pool=TRAIN_SEEDS, deal_rng_seed=deal_rng_seed,
            max_steps=MACAO_MAX_STEPS, repeated_position_penalty=penalty,
        )
    raise ValueError(f"unknown game {game_name!r}")


def evaluate(game_name: str, agent, opponent_seed: int = 0) -> dict:
    if game_name == "klondike":
        return evaluate_klondike_on_pool(agent, TEST_SEEDS, KLONDIKE_MAX_STEPS)
    return evaluate_macao_suite_on_pool(
        agent, TEST_SEEDS, MACAO_MAX_STEPS, opponent_seed=opponent_seed,
    )


def run(args) -> dict:
    torch.set_num_threads(1)

    game_name = args.game
    max_steps = KLONDIKE_MAX_STEPS if game_name == "klondike" else MACAO_MAX_STEPS

    # The deal stream depends only on the init seed, so every agent trained at
    # init seed k sees the identical sequence of TRAIN deals -- a paired
    # comparison rather than four independent draws.
    env = make_env(game_name, deal_rng_seed=args.init_seed, arm=args.arm)
    agent = build_learner(
        args.agent, env.observation_space.shape[0], env.action_space.n, args.init_seed,
    )

    before = evaluate(game_name, agent, opponent_seed=args.init_seed)

    trainer_kwargs = dict(
        checkpoint_dir=None,
        log_interval=10**9,        # the sweep's own logging is not needed
        eval_interval=10**9,       # our TEST evaluation replaces it entirely
        eval_episodes=0,
        checkpoint_interval=10**9,
    )
    if game_name == "macao":
        opponent = MacaoHeuristicAgent(seed=args.init_seed)
        cls = BootstrapSelfPlayTrainer if args.arm == "fixed" else SelfPlayTrainer
        trainer = cls(env=env, agent=agent, opponent=opponent, **trainer_kwargs)
    else:
        cls = BootstrapTrainer if args.arm == "fixed" else Trainer
        trainer = cls(env=env, agent=agent, **trainer_kwargs)

    callback, series = make_train_recorder(env, agent, game_name)

    started = time.time()
    metrics = trainer.train(
        episodes=args.episodes, max_steps_per_episode=max_steps,
        verbose=False, callback=callback,
    )
    train_seconds = time.time() - started

    after = evaluate(game_name, agent, opponent_seed=args.init_seed)

    train_blocks = {
        key: block_average(values, TRAIN_BLOCK)
        for key, values in series.items()
        if key in ("reward", "steps", "win", "loss", "cards_up", "epsilon",
                   "terminated")
    }

    dealt = env.dealt_seeds
    record = {
        "schema": "thesis_notes/run/1",
        "game": game_name,
        "agent": args.agent,
        "arm": args.arm,
        "repeated_position_penalty": (
            LOOP_PENALTY if args.arm == "noloop" else 0.0),
        "init_seed": args.init_seed,
        "episodes": args.episodes,
        "max_steps_per_episode": max_steps,
        "trainer": type(trainer).__name__,
        "opponent": (
            type(getattr(trainer, "opponent", None)).__name__
            if game_name == "macao" else None
        ),
        "protocol": {
            "train_pool": [TRAIN_SEEDS.start, TRAIN_SEEDS.stop],
            "test_pool": [TEST_SEEDS[0], TEST_SEEDS[-1] + 1],
            "train_deals_drawn": len(dealt),
            "train_deals_distinct": len(set(dealt)),
            "train_test_overlap": len(set(dealt) & set(TEST_SEEDS)),
            "train_block": TRAIN_BLOCK,
        },
        "test_before": before,
        "test_after": after,
        "train_summary": {
            k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
            for k, v in metrics.summary().items()
        },
        "train_blocks": train_blocks,
        "train_series": {
            # full per-episode series, needed for the learning-curve figures
            "reward": series["reward"],
            "steps": series["steps"],
            "win": series["win"],
            "cards_up": series["cards_up"],
            "epsilon": series["epsilon"],
            "terminated": series["terminated"],
        },
        "duration": {"train_seconds": train_seconds,
                     "eval_seconds": before.get("seconds", 0)
                     + after.get("seconds", 0)},
        "host": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu": platform.processor(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "git_commit": git_commit(),
            "torch_threads": torch.get_num_threads(),
        },
    }

    if args.checkpoint_dir:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        suffix = ".pkl" if args.agent == "q_learning" else ".pt"
        path = os.path.join(
            args.checkpoint_dir,
            f"{game_name}__{args.agent}__{args.arm}__s{args.init_seed}{suffix}",
        )
        agent.save(path)
        record["checkpoint"] = path

    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", required=True, choices=["klondike", "macao"])
    parser.add_argument("--agent", required=True,
                        choices=["q_learning", "dqn", "double_dqn", "ppo"])
    parser.add_argument("--init-seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--arm", default="asis",
                        choices=["asis", "fixed", "noloop"])
    parser.add_argument("--out-dir", default=RUNS_DIR)
    parser.add_argument("--checkpoint-dir",
                        default=os.path.join(HERE, "..", "checkpoints"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    name = f"{args.game}__{args.agent}__{args.arm}__s{args.init_seed}.json"
    path = os.path.join(args.out_dir, name)
    if os.path.exists(path) and not args.force:
        print(f"skip (exists): {name}", flush=True)
        return 0

    record = run(args)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, default=str)

    head = (record["test_after"].get("cards_up")
            if args.game == "klondike"
            else record["test_after"].get("win_rate_vs_heuristic"))
    head_before = (record["test_before"].get("cards_up")
                   if args.game == "klondike"
                   else record["test_before"].get("win_rate_vs_heuristic"))
    print(f"done {name}: TEST {head_before:.4f} -> {head:.4f} "
          f"({record['duration']['train_seconds']:.0f}s train)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
