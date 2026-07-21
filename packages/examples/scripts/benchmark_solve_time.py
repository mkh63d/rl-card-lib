"""
Benchmark how quickly each agent solves *winnable* deals.

Unlike `benchmark_agents.py` (which prints a before-snapshot over all deals),
this curates a pool of deals a perfect-information solver proves are winnable,
then plays every agent -- the non-learning baselines *and* the trained learners
loaded from disk -- over that same pool, recording:

    solve rate       share of winnable deals the agent actually won
    moves to solve   mean move count, over solved deals only
    time to solve    mean wall-clock per deal, over solved deals only

Results are written under `results/solve_benchmark/<game>.json` and picked up by
the HTML report's "Solve-time benchmark" section on its next build.

Only single-player games qualify: a solvable-deal pool needs a solver, which an
adversarial game (e.g. Macao) cannot provide. Such games are skipped with a note.

Usage:
    python benchmark_solve_time.py --game klondike --pool-size 50
    python benchmark_solve_time.py --game klondike --pool-size 20 --skip-trained
    python benchmark_solve_time.py --game all --refresh-pool
"""

import argparse
import json
import os

# Import side effect: registers the bundled games (and their solvers).
import rl_card_lib.games  # noqa: F401
from rl_card_lib.harness import (
    LEARNERS,
    agent_class_name,
    curate_solvable_pool,
    load_trained_learner,
    registered_sweep_games,
    run_solve_benchmark,
    sweep_game,
)
from rl_card_lib.harness.baselines import baseline_agents
from rl_card_lib.report import RunStore, SolveBenchmarkSet
from rl_card_lib.report.run_record import utc_now


def _pool_path(store: RunStore, game: str) -> str:
    return os.path.join(store.solve_benchmark_dir, f"{game}_pool.json")


def solvable_pool(store: RunStore, sweep, size: int, start_seed: int, refresh: bool):
    """The game's solvable-deal pool, curated once and cached on disk.

    Curation runs a solver per candidate deal, so it is worth caching: the same
    pool is reused across agents and across runs unless --refresh-pool or a
    larger --pool-size is asked for.
    """
    path = _pool_path(store, sweep.name)
    if not refresh and os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as handle:
            cached = json.load(handle)
        if cached.get("start_seed") == start_seed and len(cached.get("seeds", [])) >= size:
            print(f"  reusing cached pool of {len(cached['seeds'])} deals "
                  f"({path})", flush=True)
            return cached["seeds"][:size]

    print(f"  curating {size} solvable deals (this runs the solver)...", flush=True)
    seeds = curate_solvable_pool(sweep, size, start_seed=start_seed, verbose=True)
    store.solve_benchmark_dir.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"start_seed": start_seed, "seeds": seeds}, handle, indent=2)
    return seeds


def trained_learners(sweep, store: RunStore, checkpoint_dir: str, seed: int) -> list:
    """(label, agent) pairs for every learner with a loadable checkpoint.

    A learner still training (no checkpoint yet) is skipped with a note rather
    than crashing the benchmark.
    """
    pairs = []
    for kind in LEARNERS:
        env = sweep.env_factory()
        agent = load_trained_learner(
            kind, env, game=sweep.name, run_store=store,
            checkpoint_dir=checkpoint_dir, seed=seed,
        )
        if agent is None:
            print(f"  skipping {kind}: no trained checkpoint found", flush=True)
            continue
        pairs.append((f"{agent_class_name(kind)} (trained)", agent))
    return pairs


def benchmark_game(game: str, args, store: RunStore) -> None:
    sweep = sweep_game(game)
    if sweep.solver is None:
        print(f"\n=== {game}: skipped (no solver -- single-player games only) ===")
        return

    print(f"\n=== {game}: solve-time benchmark ===")
    seeds = solvable_pool(store, sweep, args.pool_size, args.start_seed, args.refresh_pool)
    if not seeds:
        print("  no solvable deals found; nothing to benchmark.", flush=True)
        return

    agents = baseline_agents(sweep, seed=args.seed)
    if not args.skip_trained:
        agents += trained_learners(sweep, store, args.checkpoint_dir, args.seed)

    print(f"  playing {len(agents)} agents over {len(seeds)} winnable deals:", flush=True)
    rows = run_solve_benchmark(sweep, agents, seeds)

    protocol = {
        "pool_size": len(seeds),
        "max_steps": sweep.max_steps,
        "start_seed": args.start_seed,
        "solver": "perfect-information",
    }
    path = store.save_solve_benchmark(SolveBenchmarkSet(
        game=game, measured_at=utc_now(), protocol=protocol, rows=rows,
    ))
    print(f"  wrote {path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--game", default="klondike",
        choices=[*registered_sweep_games(), "all"],
        help="Game to benchmark, or 'all' registered single-player games",
    )
    parser.add_argument("--pool-size", type=int, default=50,
                        help="Number of winnable deals to benchmark over")
    parser.add_argument("--start-seed", type=int, default=0,
                        help="First deal seed to scan when curating the pool")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for constructed agents")
    parser.add_argument("--results-dir", default="./results")
    parser.add_argument("--checkpoint-dir", default="./checkpoints")
    parser.add_argument("--refresh-pool", action="store_true",
                        help="Re-curate the solvable pool instead of reusing the cache")
    parser.add_argument("--skip-trained", action="store_true",
                        help="Benchmark only the non-learning baselines")
    args = parser.parse_args()

    store = RunStore(args.results_dir)
    games = registered_sweep_games() if args.game == "all" else [args.game]
    for game in games:
        benchmark_game(game, args, store)


if __name__ == "__main__":
    main()
