"""Does Double DQN's dueling head earn its place? Issue #42.

`build_learner` gave Double DQN a dueling head and plain DQN none, and nothing
had ever measured the alternative. `DuelingQNetwork` computes
`Q = V + (A - mean A)`, so the part that depends on the action is a *centred*
advantage and a large V squeezes the legal actions together by construction.
That squeeze is measured: over the 200 TEST deals the legal-action Q spread is
5.4% of mean Q for Double DQN against plain DQN's 57.0% on Klondike, while on
Macao -- short episodes, modest V -- the two sit together at ~20%
(diagnosis.md D2). The asymmetry across the two games is what makes the
architectural reading credible rather than a seed artefact.

This script runs the same agent with `dueling=False` and compares both halves
of the question: the score, and the Q spread.

Same protocol as the sweep, because it *is* the sweep -- every run here is a
`run_one.py` subprocess with `--arm fixed`, 5000 episodes, TRAIN deals drawn by
the same init-seeded stream and greedy evaluation on all 200 TEST deals. The
control is not retrained: it is the three `double_dqn`/`fixed` runs already in
`raw/runs/`, so the comparison is against the numbers the thesis reports.

    paired          the TRAIN deal stream (a function of the init seed alone),
                    the arm, the episode count, the TEST pool, the evaluation
                    code path, the library commit
    NOT pairable    the initial weights. A plain head and a dueling head hold
                    different parameter counts, so the same torch seed draws a
                    different network. Three seeds, and the spread across them
                    is reported, for exactly this reason.

Score comes from the run records; Q spread comes from a greedy replay of the
checkpoints. They are kept in separate key groups because they are not the same
measurement -- `play_macao` fixes the opponent at seed 0 while `run_one`
evaluates against `opponent_seed=init_seed`, so the two Macao win rates
legitimately differ.

Writes:
    raw/ablation_dueling/runs/*.json    one per run, run_one's own schema
    checkpoints/{game}__double_dqn_noduel__fixed__s{seed}.pt
    raw/ablation_dueling.json           the comparison (this is the artefact)

Klondike measured 4.1 h per run in the control, so the seeds must overlap; the
default six workers puts every job in flight at once.

Usage:
    python thesis_notes/scripts/ablate_dueling.py --workers 6
    python thesis_notes/scripts/ablate_dueling.py --dry-run
    python thesis_notes/scripts/ablate_dueling.py --skip-train   # measure only
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_one  # noqa: E402
from probe_policy_diagnostics import (  # noqa: E402
    ARM_MAX_PASSES,
    aggregate,
    load_checkpoint,
    play_klondike,
    play_macao,
)
from run_sweep_all import run_pool  # noqa: E402
from split import TEST_SEEDS  # noqa: E402

from rl_card_lib.harness import build_learner  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "raw")
RUNS = os.path.join(RAW, "runs")
CHECKPOINTS = os.path.join(HERE, "..", "checkpoints")
ABLATION_RUNS = os.path.join(RAW, "ablation_dueling", "runs")
OUT = os.path.join(RAW, "ablation_dueling.json")

#: The two configurations under comparison. `label` is the slot run_one puts in
#: the filename, and None means the stock sweep run -- which is the control and
#: is never retrained here.
VARIANTS = {
    "dueling": {"label": None, "dueling": None,
                "description": "the shipped default, Q = V + (A - mean A)"},
    "noduel": {"label": "double_dqn_noduel", "dueling": False,
               "description": "the same agent with a plain Q-head (#42)"},
}

#: Which `test_after` key is the headline for each game -- the same choice
#: make_report.TEST_SPEC makes, kept here so this script does not import the
#: report generator for one string.
HEADLINE = {"klondike": "cards_up", "macao": "win_rate_vs_heuristic"}


def check_the_tree_is_this_one() -> None:
    """Refuse to spend half a day of CPU measuring the wrong working tree.

    `run_one.py` imports `rl_card_lib` from wherever the interpreter finds it.
    Run from a git worktree with no PYTHONPATH that is the *installed* package,
    which points at whichever tree was pip-installed -- so the runs would
    measure code this checkout does not contain, and would not say so.
    """
    if "dueling" not in inspect.signature(build_learner).parameters:
        import rl_card_lib.harness.learners as learners
        raise SystemExit(
            "build_learner has no `dueling` parameter, so this interpreter is "
            "not importing this checkout:\n"
            f"    {learners.__file__}\n"
            "Set PYTHONPATH to this tree's packages/*/src directories.")


def control_is_present(games, seeds, arm) -> None:
    """The control is read, never retrained; say so early if it is missing."""
    missing = [
        os.path.relpath(path, os.path.join(HERE, ".."))
        for game in games for seed in seeds
        for path in (
            os.path.join(RUNS, f"{game}__double_dqn__{arm}__s{seed}.json"),
            os.path.join(CHECKPOINTS, f"{game}__double_dqn__{arm}__s{seed}.pt"),
        )
        if not os.path.exists(path)
    ]
    if missing:
        raise SystemExit(
            "the control this compares against is not on disk:\n    "
            + "\n    ".join(missing)
            + "\nRun thesis_notes/scripts/run_sweep_all.py first.")


def read_json(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def stem(game: str, seed: int, arm: str) -> str:
    """run_one's own naming, asked of run_one rather than reimplemented."""
    return run_one.run_stem(argparse.Namespace(
        game=game, agent="double_dqn", arm=arm, init_seed=seed,
        no_dueling=True))


def control_seconds(game: str, seed: int, arm: str) -> float:
    """What the dueling control spent training, as this run's cost estimate."""
    record = read_json(
        os.path.join(RUNS, f"{game}__double_dqn__{arm}__s{seed}.json"))
    return float(record["duration"]["train_seconds"]) if record else 3600.0


def jobs(games, seeds, episodes, arm, force) -> list[dict]:
    """One `run_one.py --no-dueling` subprocess per (game, seed).

    Cost is taken from what the control actually took rather than from
    run_sweep_all.COST, whose 1.32 s/episode for Klondike underestimates the
    measured 4.1 h by more than a factor of two. It only orders the queue.
    """
    out = []
    for game in games:
        for seed in seeds:
            name = stem(game, seed, arm)
            done = os.path.exists(os.path.join(ABLATION_RUNS, name + ".json"))
            if done and not force:
                continue
            out.append({
                "game": game, "agent": "double_dqn", "arm": arm,
                "init_seed": seed, "episodes": episodes, "stem": name,
                "cost": control_seconds(game, seed, arm),
                "extra": ["--no-dueling", "--out-dir", ABLATION_RUNS]
                + (["--force"] if force else []),
            })
    out.sort(key=lambda job: -job["cost"])
    return out


def score_rows(game: str, seeds, arm: str) -> dict:
    """Mean and sample sd of the headline TEST metric, over the seeds."""
    key = HEADLINE[game]
    out = {}
    for name, spec in VARIANTS.items():
        directory = RUNS if spec["label"] is None else ABLATION_RUNS
        agent = spec["label"] or "double_dqn"
        records = [
            read_json(os.path.join(directory, f"{game}__{agent}__{arm}__s{seed}.json"))
            for seed in seeds
        ]
        records = [record for record in records if record]
        if not records:
            continue
        after = [record["test_after"][key] for record in records]
        before = [record["test_before"][key] for record in records]
        out[name] = {
            "seeds": len(records),
            "test_after_mean": round(float(np.mean(after)), 4),
            "test_after_sd": (round(float(np.std(after, ddof=1)), 4)
                              if len(after) > 1 else None),
            "test_before_mean": round(float(np.mean(before)), 4),
            "train_seconds_mean": round(float(np.mean(
                [record["duration"]["train_seconds"] for record in records])), 1),
            "per_seed": [
                {"init_seed": record["init_seed"],
                 "test_after": round(record["test_after"][key], 4)}
                for record in records
            ],
        }
    return out


def q_rows(game: str, seeds, arm: str, deals) -> dict:
    """Replay both variants greedily and aggregate the legal-action Q spread.

    Both sides go through the same `play_*` code path in the same process, so a
    difference cannot be an artefact of how each side was measured; the
    control's number is cross-checked against policy_diagnostics.json.
    """
    out = {}
    for name, spec in VARIANTS.items():
        rows = []
        for seed in seeds:
            agent = load_checkpoint(game, "double_dqn", arm, seed,
                                    label=spec["label"], dueling=spec["dueling"])
            if agent is None:
                continue
            rows.append(play_klondike(agent, deals, ARM_MAX_PASSES[arm])
                        if game == "klondike" else play_macao(agent, deals))
            print(f"    {game} {name} s{seed}: spread "
                  f"{rows[-1].get('spread_as_pct_of_mean_Q')}% of mean Q",
                  flush=True)
        if rows:
            out[name] = aggregate(rows)
    return out


def delta(block: dict, key: str):
    """noduel minus dueling, or None when either side is missing."""
    a = block.get("noduel", {}).get(key)
    b = block.get("dueling", {}).get(key)
    return round(a - b, 4) if a is not None and b is not None else None


def cross_check(report: dict, games, arm: str) -> dict:
    """Our control against the one already recorded, so drift is visible."""
    recorded = read_json(os.path.join(RAW, "policy_diagnostics.json")) or {}
    out = {}
    for game in games:
        here = (report.get(game, {}).get("q", {})
                .get("dueling", {}).get("spread_as_pct_of_mean_Q"))
        there = ((recorded.get(game) or {}).get(f"double_dqn__{arm}") or {}
                 ).get("spread_as_pct_of_mean_Q")
        out[game] = {"remeasured_here": here, "policy_diagnostics": there}
        if here is not None and there is not None and abs(here - there) > 0.5:
            print(f"  WARNING: {game} control spread {here}% here against "
                  f"{there}% in policy_diagnostics.json", flush=True)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--games", nargs="+", default=["klondike", "macao"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--arm", default="fixed", choices=sorted(run_one.ARMS))
    parser.add_argument("--deals", type=int, default=len(TEST_SEEDS),
                        help="first N TEST deals for the Q replay")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    check_the_tree_is_this_one()
    control_is_present(args.games, args.seeds, args.arm)
    os.makedirs(ABLATION_RUNS, exist_ok=True)

    started = time.time()
    pending = [] if args.skip_train else jobs(
        args.games, args.seeds, args.episodes, args.arm, args.force)

    total = sum(job["cost"] for job in pending)
    longest = max((job["cost"] for job in pending), default=0)
    print(f"{len(pending)} run(s) to train, {total / 3600:.1f} CPU-hours "
          f"estimated from what the control took, ~{longest / 3600:.1f} h wall "
          f"at {args.workers} workers", flush=True)
    for job in pending:
        print(f"  {job['stem']:48s} ~{job['cost'] / 60:6.1f} min", flush=True)
    if args.dry_run:
        return 0

    if pending:
        done, failed = run_pool(pending, args.workers)
        print(f"trained {len(done)} run(s)", flush=True)
        if failed:
            print(f"FAILED: {', '.join(failed)}", flush=True)
            return 1

    deals = TEST_SEEDS[:args.deals]
    report = {
        "schema": "thesis_notes/ablation/dueling/1",
        "question": ("Does Double DQN's dueling head cost score or Q "
                     "differentiation? (issue #42)"),
        "protocol": {
            "arm": args.arm,
            "episodes": args.episodes,
            "init_seeds": list(args.seeds),
            "deals": len(deals),
            "test_pool": [TEST_SEEDS[0], TEST_SEEDS[-1] + 1],
            "variants": {k: v["description"] for k, v in VARIANTS.items()},
            "paired": ("TRAIN deal stream, arm, episode count, TEST pool and "
                       "evaluation code path; NOT the initial weights, whose "
                       "parameter counts differ between the two heads"),
            "score_from": "the run records' test_after",
            "q_from": ("a greedy replay of the checkpoints; play_macao fixes "
                       "the opponent at seed 0, so its win_rate is not the "
                       "protocol's win_rate_vs_heuristic"),
            "control_runs": "raw/runs/{game}__double_dqn__{arm}__s{seed}.json",
            "ablation_runs": ("raw/ablation_dueling/runs/"
                              "{game}__double_dqn_noduel__{arm}__s{seed}.json"),
        },
    }

    for game in args.games:
        print(f"  replaying {game} on {len(deals)} deals", flush=True)
        block = {
            "headline_key": HEADLINE[game],
            "score": score_rows(game, args.seeds, args.arm),
            "q": q_rows(game, args.seeds, args.arm, deals),
        }
        block["delta"] = {
            "test_after": delta(block["score"], "test_after_mean"),
            "mean_legal_Q": delta(block["q"], "mean_legal_Q"),
            "mean_legal_Q_spread": delta(block["q"], "mean_legal_Q_spread"),
            "spread_as_pct_of_mean_Q": delta(block["q"],
                                             "spread_as_pct_of_mean_Q"),
        }
        report[game] = block

    report["cross_check"] = cross_check(report, args.games, args.arm)
    report["seconds"] = round(time.time() - started, 1)

    with open(OUT, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(f"  wrote raw/{os.path.basename(OUT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
