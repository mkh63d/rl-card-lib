"""
Watch one agent play one deal, move by move.

`benchmark_solve_time.py` answers "how often does this agent win?"; this answers
"what does it actually do?". It replays a single seeded deal, printing the board
after every move together with the action, its reward and the flags the env
raised -- an illegal action, a position already seen this episode, the step cap.

The deal is chosen so the question is worth asking: with no --seed, the seed
comes from the cached pool of deals the perfect-information solver has *proved*
winnable, so a loss is the agent's and not the shuffle's. The solver is also
consulted for an explicit --seed, and says whether it is winnable, proven
unwinnable, or undecided within its node budget. Games with no solver (Macao,
and any adversarial game -- there is no perfect-information solve) skip the
check and replay anyway.

Agents are either a baseline, which needs nothing on disk, or a trained learner
loaded from its checkpoint.

Usage:
    python watch_episode.py --game klondike --agent Random --seed 0
    python watch_episode.py --agent double_dqn --pool-index 3 --step
    python watch_episode.py --agent Heuristic --delay 0.4 --html episode.html
    python watch_episode.py --agent ppo --seed 12345 --summary-only
"""

import argparse
import json
import os
import sys
from typing import Optional

# Import side effect: registers the bundled games (and their solvers).
import rl_card_lib.games  # noqa: F401
from rl_card_lib.harness import (
    LEARNERS,
    curate_solvable_pool,
    load_trained_learner,
    registered_sweep_games,
    sweep_game,
)
from rl_card_lib.harness.baselines import baseline_agents
from rl_card_lib.harness.deals import TEST_SEED_START
from rl_card_lib.report import RunStore
from rl_card_lib.utils.console import console_safe
from rl_card_lib.visualizer import (
    episode_summary,
    play_episode,
    step_printer,
    write_episode_html,
)


def resolve_agent(sweep, name: str, args):
    """The named agent: a baseline built on the spot, or a loaded checkpoint.

    Baselines are matched case-insensitively because their registered names --
    "GreedyLookahead(1)", "MCTS(20)" -- carry capitals and parentheses that are
    a nuisance to type at a shell prompt.

    Returns:
        (agent, display_name), or exits with status 2 when the name is unknown
        or the learner has no checkpoint yet.
    """
    baselines = baseline_agents(sweep, seed=args.agent_seed)
    for label, agent in baselines:
        if label.lower() == name.lower():
            return agent, label

    if name.lower() in LEARNERS:
        kind = name.lower()
        env = sweep.env_factory()
        agent = load_trained_learner(
            kind, env, game=sweep.name,
            run_store=RunStore(args.results_dir),
            checkpoint_dir=args.checkpoint_dir,
            seed=args.agent_seed,
        )
        if agent is None:
            print(f"no trained checkpoint for {kind} on {sweep.name}; "
                  f"train it first or pick a baseline.", file=sys.stderr)
            raise SystemExit(2)
        return agent, kind

    known = [label for label, _ in baselines] + list(LEARNERS)
    print(f"unknown agent {name!r}; try one of: {', '.join(known)}", file=sys.stderr)
    raise SystemExit(2)


def resolve_seed(sweep, args) -> Optional[int]:
    """The deal to replay: the one asked for, or one the solver proved winnable.

    The cached pool is reused rather than re-curated -- curation runs the solver
    once per candidate deal, which is far too slow to sit in front of a replay.
    --curate is the way to ask for it anyway.

    Returns:
        A deal seed, or None to leave the choice to the env -- which is all a
        game with no solver can offer.
    """
    if args.seed is not None:
        return args.seed

    path = os.path.join(args.results_dir, "solve_benchmark", f"{sweep.name}_pool.json")
    if not args.curate and os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as handle:
            seeds = json.load(handle).get("seeds") or []
        if seeds:
            return seeds[args.pool_index % len(seeds)]

    if sweep.solver is None:
        # No solver means no winnable pool to draw from -- an adversarial game
        # has no perfect-information solve. Let the env deal from its own pool
        # instead of refusing: the deal is still reported, and still replayable
        # by passing the seed the summary prints back as --seed.
        return None

    print("  curating a winnable deal (this runs the solver)...", flush=True)
    seeds = curate_solvable_pool(
        sweep, args.pool_index + 1, start_seed=TEST_SEED_START, verbose=False,
    )
    if not seeds:
        print("no winnable deal found; pass --seed.", file=sys.stderr)
        raise SystemExit(2)
    return seeds[args.pool_index % len(seeds)]


def report_solvability(sweep, seed: Optional[int]) -> None:
    """Say whether the deal can be won at all, on a throwaway env.

    Throwaway because the solver searches from the game it is handed and is
    under no obligation to leave it where it found it -- the env the agent
    plays must be dealt fresh.
    """
    if sweep.solver is None:
        print(f"  solvability  not checked ({sweep.name} has no solver)")
        return

    env = sweep.env_factory()
    env.reset(seed=seed)
    verdict = sweep.solver(env.game)
    print("  solvability  " + {
        True: "winnable (the solver found a win)",
        False: "proven unwinnable",
        None: "undecided within the solver's node budget",
    }[verdict])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--game", default="klondike", choices=registered_sweep_games(),
                        help="Registered game to replay")
    parser.add_argument("--agent", default="Random",
                        help="Baseline name (Random, Heuristic, GreedyLookahead(1), "
                             f"MCTS(n)) or a learner: {', '.join(LEARNERS)}")
    parser.add_argument("--seed", type=int, default=None,
                        help="Deal seed. Omitted, a seed comes from the cached "
                             "pool of proven-winnable deals.")
    parser.add_argument("--pool-index", type=int, default=0,
                        help="Which deal of the winnable pool to replay")
    parser.add_argument("--curate", action="store_true",
                        help="Curate a winnable deal with the solver instead of "
                             "reading the cached pool")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Step cap; the game's own cap by default")
    parser.add_argument("--step", action="store_true",
                        help="Wait for Enter after each move")
    parser.add_argument("--delay", type=float, default=0.0,
                        help="Seconds to pause after each move, for auto-play")
    parser.add_argument("--no-board", action="store_true",
                        help="Print only the move lines, without the board")
    parser.add_argument("--summary-only", action="store_true",
                        help="Print nothing until the episode is over")
    parser.add_argument("--html", default=None, metavar="PATH",
                        help="Also write the episode as a self-contained HTML page")
    parser.add_argument("--agent-seed", type=int, default=0,
                        help="Seed for the constructed agent")
    parser.add_argument("--results-dir", default="./results")
    parser.add_argument("--checkpoint-dir", default="./checkpoints")
    args = parser.parse_args()

    sweep = sweep_game(args.game)
    agent, agent_name = resolve_agent(sweep, args.agent, args)
    seed = resolve_seed(sweep, args)

    deal = f"deal {seed}" if seed is not None else "a deal of the env's own choosing"
    print(f"\n=== {sweep.name}: {agent_name} on {deal} ===")
    report_solvability(sweep, seed)
    print()

    board = not args.no_board and not args.summary_only
    # The HTML page is built from the boards, so they are still captured when
    # the terminal is not being given them -- and not captured at all when
    # nothing will read them, since render() runs once per step.
    capture_board = board or args.html is not None

    def show_opening(text, _info):
        """Print the deal itself, before anyone has moved.

        Every board after this one is only readable as a change from it.
        """
        print(console_safe(text))

    printer = None
    on_reset = None
    if not args.summary_only:
        printer = step_printer(
            board=board,
            delay=args.delay,
            # `input()` lives here rather than in the library: an entry point
            # knows it has a console, a library function does not.
            pause=(lambda: input("")) if args.step else None,
        )
        if board:
            on_reset = show_opening

    env = sweep.env_factory()
    record = play_episode(
        env, agent,
        seed=seed,
        max_steps=args.max_steps,
        game=sweep.name,
        agent_name=agent_name,
        capture_board=capture_board,
        on_step=printer,
        on_reset=on_reset,
    )

    print(episode_summary(record))

    if args.html:
        path = write_episode_html(record, args.html)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
