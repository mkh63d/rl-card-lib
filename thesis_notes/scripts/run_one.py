"""Train one (game, agent, init-seed, arm) combination under the corrected protocol.

    TRAIN   deals drawn uniformly from seeds 0..9999, one fresh deal per episode
    TEST    all 200 deals of seeds 100000..100199, greedy, no learning, before
            and after training

Every arm runs on the *same* library commit. That was not true before PRs
#24-#34: back then `asis` meant "stock library" and the corrections lived in
`scripts/harness.py`. The corrections are merged now, so the roles are
inverted -- `fixed` is what the library does on its own, and `asis` is
reconstructed on top of it by switching four things back:

    arms    `asis`   pre-#24 truncation handling (a time-limit cut is taught as
                     terminal), unlimited Klondike passes, target_update_freq
                     500 on both games, PPO evaluated by argmax
            `fixed`  today's library, unmodified
            `noloop` `fixed` + repeated_position_penalty = -0.05 (Klondike only)

`fixed` therefore bundles four changes rather than isolating one, and on
Klondike it is not even the same rule set as `asis` (finite vs unlimited passes).
This is a before/after-the-PRs comparison, not a single-factor ablation; see
results.md for the caveat in full.

`--no-dueling` is not a fourth arm. Arms are protocol levers, held identical
across every agent so a difference between two rows is a difference between
them; the dueling head is Double DQN's architecture and only it has one. Issue
#42 asks whether that head is why its Q barely differentiates the legal actions
(diagnosis.md D2). The flag changes the network `build_learner` returns and
nothing else -- same loss, optimiser, hidden sizes, arm and deal stream -- and
it tags this run's filenames so it can never be read as a sweep run. Driven by
thesis_notes/scripts/ablate_dueling.py; not part of the 60-run sweep.

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
    LegacySelfPlayTrainer,
    LegacyTrainer,
    block_average,
    evaluate_klondike_on_pool,
    evaluate_macao_suite_on_pool,
    make_train_recorder,
)
from split import TEST_SEEDS, TRAIN_SEEDS  # noqa: E402

# Importing the games package registers the bundled games, which is what makes
# `sweep_game` below able to answer for their per-game hyper-parameters.
import rl_card_lib.games  # noqa: E402,F401  (import side effect: registration)
from rl_card_lib.env import CardGameEnv  # noqa: E402
from rl_card_lib.games.heuristics import MacaoHeuristicAgent  # noqa: E402
from rl_card_lib.games.klondike import KlondikeSolitaire  # noqa: E402
from rl_card_lib.games.macao import Macao  # noqa: E402
from rl_card_lib.games.registration import KLONDIKE_MAX_PASSES  # noqa: E402
from rl_card_lib.harness import build_learner, sweep_game  # noqa: E402
from rl_card_lib.harness.learners import DEFAULT_TARGET_UPDATE_FREQ  # noqa: E402
from rl_card_lib.trainer import SelfPlayTrainer, Trainer  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(HERE, "..", "raw", "runs")

KLONDIKE_MAX_STEPS = 300
MACAO_MAX_STEPS = 200
TRAIN_BLOCK = 50           # episodes per TRAIN metric row

#: Penalty added to the reward whenever a step lands in a position already seen
#: this episode. `CardGameEnv` has implemented it since the env was written and
#: defaults it to 0.0; the bundled Klondike switches it on since #29, and the
#: `noloop` arm is what measured whether it helps. See diagnosis.md D3 -- the
#: trained greedy policies spend ~80% of their moves revisiting positions, and
#: the penalty alone does not stop them.
LOOP_PENALTY = -0.05

#: The four levers that separate the arms. Everything else -- the deal pools,
#: the episode count, the evaluation protocol, the library commit -- is held
#: identical, so a difference between two rows is a difference between these.
ARMS = {
    "asis": {
        "description": "library behaviour as the thesis describes it, pre-#24",
        "legacy_truncation": True,
        "klondike_max_passes": None,
        "per_game_target_update_freq": False,
        "ppo_eval_greedy": True,
        "repeated_position_penalty": 0.0,
    },
    "fixed": {
        "description": "today's library, unmodified",
        "legacy_truncation": False,
        "klondike_max_passes": KLONDIKE_MAX_PASSES,
        "per_game_target_update_freq": True,
        "ppo_eval_greedy": False,
        "repeated_position_penalty": 0.0,
    },
    "noloop": {
        "description": "fixed + repeated-position penalty",
        "legacy_truncation": False,
        "klondike_max_passes": KLONDIKE_MAX_PASSES,
        "per_game_target_update_freq": True,
        "ppo_eval_greedy": False,
        "repeated_position_penalty": LOOP_PENALTY,
    },
}


#: Filename tag for a run whose *network* differs from the sweep's, in the slot
#: that names the agent -- because the agent is still `double_dqn` and it is the
#: network that varies. Deliberately not a member of LEARNERS and deliberately
#: not an arm; `make_report.load_runs` skips any record carrying a `variant`, so
#: an architecture ablation can never be averaged into the arms it exists to be
#: compared against.
NODUELING_VARIANT = "noduel"


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=HERE, text=True,
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def arm_config(game_name: str, arm: str) -> dict:
    """Resolve `arm` into the concrete numbers this run will use.

    Returned verbatim in the run record, so the JSON says what was configured
    rather than leaving a reader to re-derive it from the arm's name.
    """
    spec = ARMS[arm]
    freq = (sweep_game(game_name).target_update_freq
            if spec["per_game_target_update_freq"]
            else DEFAULT_TARGET_UPDATE_FREQ)
    return {
        "arm": arm,
        "description": spec["description"],
        "legacy_truncation": spec["legacy_truncation"],
        "max_passes": (spec["klondike_max_passes"]
                       if game_name == "klondike" else None),
        "target_update_freq": freq,
        "ppo_eval_greedy": spec["ppo_eval_greedy"],
        "repeated_position_penalty": (
            spec["repeated_position_penalty"] if game_name == "klondike" else 0.0
        ),
    }


def variant(args) -> str | None:
    """The architecture tag for this run, or None for a stock sweep run."""
    return NODUELING_VARIANT if getattr(args, "no_dueling", False) else None


def run_stem(args) -> str:
    """The filename every artefact of this run shares.

    One definition rather than the two this file used to carry -- the JSON name
    in `main` and the checkpoint name in `run`. A plain Q-head and a dueling one
    have different parameter counts, so without the tag the #42 ablation would
    overwrite `klondike__double_dqn__fixed__s0.pt` and then fail to load it
    back; the run JSONs would collide the same way.
    """
    tag = variant(args)
    agent = f"{args.agent}_{tag}" if tag else args.agent
    return f"{args.game}__{agent}__{args.arm}__s{args.init_seed}"


def make_env(game_name: str, deal_rng_seed: int, config: dict):
    """The training env: one deal per episode, drawn from the TRAIN pool.

    `CardGameEnv` draws the deals itself since it learned about seed pools, so
    the run needs no env subclass -- `dealt_seeds` on the returned env is the
    record of what was actually played.
    """
    common = dict(deal_seeds=TRAIN_SEEDS, deal_rng_seed=deal_rng_seed,
                  repeated_position_penalty=config["repeated_position_penalty"])
    if game_name == "klondike":
        return CardGameEnv(
            KlondikeSolitaire(max_passes=config["max_passes"]),
            max_steps=KLONDIKE_MAX_STEPS, **common,
        )
    if game_name == "macao":
        return CardGameEnv(
            Macao(num_players=2), max_steps=MACAO_MAX_STEPS, **common,
        )
    raise ValueError(f"unknown game {game_name!r}")


def evaluate(game_name: str, agent, config: dict, opponent_seed: int = 0) -> dict:
    if game_name == "klondike":
        return evaluate_klondike_on_pool(
            agent, TEST_SEEDS, KLONDIKE_MAX_STEPS,
            max_passes=config["max_passes"],
            eval_greedy=config["ppo_eval_greedy"],
        )
    return evaluate_macao_suite_on_pool(
        agent, TEST_SEEDS, MACAO_MAX_STEPS, opponent_seed=opponent_seed,
        eval_greedy=config["ppo_eval_greedy"],
    )


def run(args) -> dict:
    torch.set_num_threads(1)

    game_name = args.game
    max_steps = KLONDIKE_MAX_STEPS if game_name == "klondike" else MACAO_MAX_STEPS
    config = arm_config(game_name, args.arm)

    # The deal stream depends only on the init seed, so every agent trained at
    # init seed k sees the identical sequence of TRAIN deals -- a paired
    # comparison rather than four independent draws.
    env = make_env(game_name, deal_rng_seed=args.init_seed, config=config)
    # None means "the kind's own default"; see build_learner. Only the #42
    # ablation passes False, and only for double_dqn.
    agent = build_learner(
        args.agent, env.observation_space.shape[0], env.action_space.n,
        args.init_seed, target_update_freq=config["target_update_freq"],
        dueling=False if getattr(args, "no_dueling", False) else None,
    )

    before = evaluate(game_name, agent, config, opponent_seed=args.init_seed)

    trainer_kwargs = dict(
        checkpoint_dir=None,
        log_interval=10**9,        # the sweep's own logging is not needed
        eval_interval=10**9,       # our TEST evaluation replaces it entirely
        eval_episodes=0,
        checkpoint_interval=10**9,
    )
    legacy = config["legacy_truncation"]
    if game_name == "macao":
        opponent = MacaoHeuristicAgent(seed=args.init_seed)
        cls = LegacySelfPlayTrainer if legacy else SelfPlayTrainer
        trainer = cls(env=env, agent=agent, opponent=opponent, **trainer_kwargs)
    else:
        cls = LegacyTrainer if legacy else Trainer
        trainer = cls(env=env, agent=agent, **trainer_kwargs)

    callback, series = make_train_recorder(env, agent, game_name)

    started = time.time()
    metrics = trainer.train(
        episodes=args.episodes, max_steps_per_episode=max_steps,
        verbose=False, callback=callback,
    )
    train_seconds = time.time() - started

    after = evaluate(game_name, agent, config, opponent_seed=args.init_seed)

    train_blocks = {
        key: block_average(values, TRAIN_BLOCK)
        for key, values in series.items()
        if key in ("reward", "steps", "win", "loss", "cards_up", "epsilon",
                   "terminated")
    }

    dealt = env.dealt_seeds
    record = {
        # /3 adds `variant` and `agent_config`; a /2 record simply lacks them.
        "schema": "thesis_notes/run/3",
        "game": game_name,
        "agent": args.agent,
        # The agent *is* a Double DQN; the variant is what tells the two
        # architectures apart. Kept out of `arm_config`, which describes the
        # protocol and must stay identical across the agents in an arm.
        "variant": variant(args),
        "agent_config": {"dueling": getattr(agent, "dueling", None)},
        "arm": args.arm,
        "arm_config": config,
        "repeated_position_penalty": config["repeated_position_penalty"],
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
        path = os.path.join(args.checkpoint_dir, run_stem(args) + suffix)
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
    parser.add_argument("--arm", default="asis", choices=sorted(ARMS))
    parser.add_argument("--out-dir", default=RUNS_DIR)
    parser.add_argument("--checkpoint-dir",
                        default=os.path.join(HERE, "..", "checkpoints"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--no-dueling", action="store_true",
        help="double_dqn only: a plain Q-head instead of the dueling one (#42)",
    )
    args = parser.parse_args(argv)

    if args.no_dueling and args.agent != "double_dqn":
        parser.error("--no-dueling only applies to double_dqn")

    os.makedirs(args.out_dir, exist_ok=True)
    name = run_stem(args) + ".json"
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
