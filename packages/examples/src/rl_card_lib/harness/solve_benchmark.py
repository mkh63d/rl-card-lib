"""Solve-time benchmark over a pool of confirmed-solvable deals.

Win rate answers "how often does this agent win"; this module answers the two
questions win rate cannot: over deals that are *known to be winnable*, how many
does the agent actually solve, and -- for the ones it solves -- how many moves
and how much wall-clock time did it take. Measuring over a solvable-only pool
removes the ~20% of deals no policy can win, so a low solve rate means the agent
failed a winnable deal rather than drew an impossible one.

Everything here is generic over a registered `SweepGame`: it needs only the
game's `solver` (to curate the pool) and its `env_factory`. A game without a
solver -- any adversarial/multiplayer game, e.g. Macao -- cannot be benchmarked
this way, and the functions say so rather than guess.
"""

from __future__ import annotations

import time
from typing import Optional


def curate_solvable_pool(
    sweep_game,
    size: int,
    *,
    start_seed: int = 0,
    max_scan: Optional[int] = None,
    verbose: bool = False,
) -> list[int]:
    """Collect `size` deal seeds the game's solver proves are winnable.

    Scans seeds from `start_seed` upward, keeping only those the solver returns
    True for; deals it proves unwinnable (False) or cannot decide within its
    budget (None) are skipped, because a benchmark of solve time is meaningless
    on a deal that may have no solution. The solver carries its own node budget
    (set where the game is registered), so this stays generic.

    Args:
        sweep_game: A registered game that declares a `solver`
        size: How many solvable seeds to collect
        start_seed: First seed to try
        max_scan: Give up after scanning this many seeds (default: 50x size, a
            margin over the ~80% winnable rate plus undecided deals)
        verbose: Print progress as the pool fills

    Returns:
        The solvable seeds, in ascending order

    Raises:
        ValueError: if the game declares no solver (e.g. a multiplayer game)
    """
    if sweep_game.solver is None:
        raise ValueError(
            f"Game {sweep_game.name!r} has no solver, so a solvable-deal pool "
            "cannot be curated. The solve-time benchmark is single-player only."
        )
    if max_scan is None:
        max_scan = 50 * max(size, 1)

    env = sweep_game.env_factory()
    seeds: list[int] = []
    seed = start_seed
    scanned = 0
    while len(seeds) < size and scanned < max_scan:
        env.reset(seed=seed)
        if sweep_game.solver(env.game) is True:
            seeds.append(seed)
            if verbose:
                print(f"  pool {len(seeds)}/{size}: seed {seed}", flush=True)
        seed += 1
        scanned += 1

    if len(seeds) < size:
        print(
            f"  warning: only found {len(seeds)} solvable deals in {scanned} "
            f"seeds (wanted {size}); raise max_scan to look further.", flush=True,
        )
    return seeds


def measure_agent_on_pool(agent, sweep_game, seeds: list[int]) -> dict:
    """Play one agent over the solvable pool and report solve rate/moves/time.

    For each seeded deal the agent plays until the game ends or the game's step
    cap is hit, counting moves and timing only the play loop. A deal counts as
    solved when the game reports a win. Move and time means are taken over
    solved deals only -- averaging in the move cap of a deal the agent never
    solved would make a worse agent look faster.

    Args:
        agent: Any agent following the select_action/eval/reset contract
        sweep_game: The registered game, for its env and step cap
        seeds: Solvable deal seeds from `curate_solvable_pool`

    Returns:
        A row dict: solve_rate, solve_moves, solve_seconds (means over solved
        deals, None if it solved none), plus any per-episode extras the game
        reports (e.g. cards_up) averaged over all deals, and pool_size.
    """
    was_training = getattr(agent, "training", False)
    if hasattr(agent, "eval"):
        agent.eval()

    env = sweep_game.env_factory()
    max_steps = sweep_game.max_steps
    solved_moves: list[int] = []
    solved_seconds: list[float] = []
    extra_totals: dict[str, float] = {}
    extra_counts: dict[str, int] = {}

    for seed in seeds:
        observation, info = env.reset(seed=seed)
        agent.reset()
        if hasattr(agent, "bind"):
            agent.bind(env)

        moves = 0
        started = time.perf_counter()
        for _ in range(max_steps):
            action = agent.select_action(observation, info.get("legal_actions"))
            observation, _, terminated, truncated, info = env.step(action)
            moves += 1
            if terminated or truncated:
                break
        elapsed = time.perf_counter() - started

        if env.game.winner == 0:
            solved_moves.append(moves)
            solved_seconds.append(elapsed)

        if sweep_game.episode_extras is not None:
            for key, value in sweep_game.episode_extras(env.game, agent).items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    extra_totals[key] = extra_totals.get(key, 0.0) + float(value)
                    extra_counts[key] = extra_counts.get(key, 0) + 1

    if was_training and hasattr(agent, "train"):
        agent.train()

    pool_size = len(seeds)
    row: dict = {
        "solve_rate": (len(solved_moves) / pool_size) if pool_size else 0.0,
        "solve_moves": (sum(solved_moves) / len(solved_moves)) if solved_moves else None,
        "solve_seconds": (
            sum(solved_seconds) / len(solved_seconds)) if solved_seconds else None,
        "pool_size": pool_size,
    }
    for key, total in extra_totals.items():
        row[key] = total / extra_counts[key]
    return row


def run_solve_benchmark(
    sweep_game, agents: list, seeds: list[int], *, verbose: bool = True,
) -> list[dict]:
    """Measure every (name, agent) pair over the same solvable pool.

    Args:
        sweep_game: The registered game
        agents: (name, agent) pairs -- baselines and loaded learners alike
        seeds: The shared solvable-deal pool
        verbose: Print a row per agent as it finishes

    Returns:
        One result row per agent, each carrying its `agent` name
    """
    rows: list[dict] = []
    for name, agent in agents:
        row = {"agent": name, **measure_agent_on_pool(agent, sweep_game, seeds)}
        rows.append(row)
        if verbose:
            moves = "  n/a" if row["solve_moves"] is None else f"{row['solve_moves']:5.1f}"
            secs = "   n/a" if row["solve_seconds"] is None else f"{row['solve_seconds']:6.3f}s"
            print(f"  {name:24s} solved={row['solve_rate']:5.1%}  "
                  f"moves={moves}  time/solve={secs}", flush=True)
    return rows


__all__ = [
    "curate_solvable_pool",
    "measure_agent_on_pool",
    "run_solve_benchmark",
]
