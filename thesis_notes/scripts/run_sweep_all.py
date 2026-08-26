"""Run the whole corrected sweep across a pool of single-threaded processes.

Each job is one (game, agent, init-seed, arm) run of run_one.py in its own
process with torch pinned to a single thread. Measured on this machine that is
*faster* per run than letting one process use six threads on 221x256 matmuls,
and it turns the sweep into an embarrassingly parallel job.

Jobs are scheduled longest-estimate-first so the critical path is short, and
run_one.py skips a run whose JSON already exists, so an interrupted sweep is
resumed by re-running this script.

Usage:
    python thesis_notes/scripts/run_sweep_all.py --workers 7
    python thesis_notes/scripts/run_sweep_all.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
RUNS_DIR = os.path.join(HERE, "..", "raw", "runs")

SEEDS = (0, 1, 2)
AGENTS = ("double_dqn", "dqn", "ppo", "q_learning")

ASIS_AGENTS = AGENTS
# Every agent, not just the DQN family. When `fixed` meant "+ the time-limit
# bootstrap fix" it only made sense for the value-based learners, because that
# fix changes a TD target and PPO and Q-learning were unaffected. Since the
# corrections merged, `fixed` means "the library as shipped" -- so leaving PPO
# and Q-learning out would report them *only* in the pre-fix configuration, and
# for PPO that is the whole result (diagnosis.md D11: argmax vs sampling).
FIXED_AGENTS = AGENTS
# The repeated-position penalty only matters where the greedy policy was
# measured looping, which is Klondike; Macao episodes are too short for it.
# All four agents there: PPO loops hardest of any of them under argmax, so
# whether the penalty helps it is exactly the question D3 asks.
NOLOOP_AGENTS = {"klondike": AGENTS, "macao": ()}

EPISODES = {"klondike": 5000, "macao": 5000}

# Rough seconds per episode, measured on this machine at one torch thread.
# Only used to order the queue; nothing depends on it being accurate.
COST = {
    ("klondike", "double_dqn"): 1.32,
    ("klondike", "dqn"): 0.92,
    ("klondike", "ppo"): 0.05,
    ("klondike", "q_learning"): 0.02,
    ("macao", "double_dqn"): 0.08,
    ("macao", "dqn"): 0.07,
    ("macao", "ppo"): 0.02,
    ("macao", "q_learning"): 0.005,
}


def jobs() -> list[dict]:
    out = []
    for game in ("klondike", "macao"):
        for arm, agents in (("asis", ASIS_AGENTS), ("fixed", FIXED_AGENTS),
                            ("noloop", NOLOOP_AGENTS[game])):
            for agent in agents:
                for seed in SEEDS:
                    out.append({
                        "game": game, "agent": agent, "arm": arm,
                        "init_seed": seed, "episodes": EPISODES[game],
                        "cost": COST[(game, agent)] * EPISODES[game],
                    })
    out.sort(key=lambda j: -j["cost"])
    return out


def command(job: dict) -> list[str]:
    return [
        sys.executable, os.path.join(HERE, "run_one.py"),
        "--game", job["game"], "--agent", job["agent"],
        "--init-seed", str(job["init_seed"]), "--episodes", str(job["episodes"]),
        "--arm", job["arm"],
    ]


def name(job: dict) -> str:
    return f"{job['game']}__{job['agent']}__{job['arm']}__s{job['init_seed']}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    todo = jobs()
    pending = [j for j in todo
               if not os.path.exists(os.path.join(RUNS_DIR, name(j) + ".json"))]

    total_cost = sum(j["cost"] for j in pending)
    print(f"{len(todo)} jobs, {len(pending)} pending, "
          f"{total_cost / 3600:.1f} CPU-hours estimated, "
          f"~{total_cost / max(1, args.workers) / 3600:.1f} h wall at "
          f"{args.workers} workers", flush=True)
    for job in pending:
        print(f"  {name(job):48s} ~{job['cost'] / 60:6.1f} min", flush=True)
    if args.dry_run:
        return 0

    os.makedirs(RUNS_DIR, exist_ok=True)
    work: queue.Queue = queue.Queue()
    for job in pending:
        work.put(job)

    env = dict(os.environ)
    env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"})

    started = time.time()
    done, failed = [], []
    lock = threading.Lock()

    def worker() -> None:
        while True:
            try:
                job = work.get_nowait()
            except queue.Empty:
                return
            tick = time.time()
            proc = subprocess.run(
                command(job), cwd=REPO, env=env,
                capture_output=True, text=True,
            )
            elapsed = time.time() - tick
            with lock:
                if proc.returncode == 0:
                    done.append(name(job))
                    tail = (proc.stdout or "").strip().splitlines()
                    print(f"[{len(done) + len(failed)}/{len(pending)}] "
                          f"{elapsed / 60:5.1f} min  {tail[-1] if tail else name(job)}",
                          flush=True)
                else:
                    failed.append(name(job))
                    print(f"[FAIL] {name(job)}\n{(proc.stderr or '')[-2000:]}",
                          flush=True)
            work.task_done()

    threads = [threading.Thread(target=worker, daemon=True)
               for _ in range(args.workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    print(f"\nfinished {len(done)} run(s) in {(time.time() - started) / 60:.1f} min",
          flush=True)
    if failed:
        print(f"FAILED: {', '.join(failed)}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
