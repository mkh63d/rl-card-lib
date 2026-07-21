"""
Train every learner on every game, record the results, and render the report.

Produces `results/index.html`: one self-contained page with an overview table
sorted newest-run-first, comparison charts per game, and a detailed section per
model. Only the last run of each model is kept -- a run is keyed by
{game}__{agent}, so re-running a pair replaces it rather than piling up.

The expensive parts are separable:

    --skip-baselines   reuse cached baseline measurements (MCTS is the slow one)
    --html-only        re-render the page from stored JSON, training nothing

Usage:
    python run_sweep.py --episodes 200
    python run_sweep.py --games klondike --agents dqn,double_dqn --episodes 5000
    python run_sweep.py --html-only
"""

import argparse
import os
import sys
import time
import traceback

# Importing the games package registers the bundled games (and any the user
# imported first). Nothing here branches on a game name; it all comes from the
# registry, so a third game is swept by registering it, not editing this file.
import rl_card_lib.games  # noqa: F401  (import side effect: registration)
from rl_card_lib.harness import (
    LEARNERS,
    agent_class_name,
    build_learner,
    checkpoint_suffix,
    make_episode_recorder,
    measure_baselines,
    registered_sweep_games,
    sweep_game,
)
from rl_card_lib.report import BaselineSet, RunRecord, RunStore, purge_checkpoints
from rl_card_lib.report.html_report import HtmlReport
from rl_card_lib.report.run_record import host_info, utc_now
from rl_card_lib.report.training_report import TrainingReport
from rl_card_lib.trainer import SelfPlayTrainer, Trainer


def train_one(game: str, kind: str, args, store: RunStore) -> RunRecord:
    """Train one learner on one game and return its record."""
    started_at = utc_now()
    started = time.time()

    checkpoint_dir = os.path.join(args.checkpoint_dir, f"{game}_{kind}")
    store.reset_run_dir(game, kind)
    removed = purge_checkpoints(
        checkpoint_dir, game=game, agent=kind, root=args.checkpoint_dir,
    )
    if removed:
        print(f"  cleared {len(removed)} stale checkpoint file(s)", flush=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    spec = sweep_game(game)
    max_steps = spec.max_steps
    env = spec.env_factory()
    agent = build_learner(
        kind, env.observation_space.shape[0], env.action_space.n, args.seed,
    )

    episodes = args.episodes_for(kind)
    eval_seconds = 0.0

    tick = time.time()
    before = spec.evaluate(agent, args.eval_episodes, args.seed)
    eval_seconds += time.time() - tick
    print(f"  before: {_format_metrics(before)}", flush=True)

    # Ten evaluation points rather than two, so the evaluation chart is a
    # curve. Q-learning writes only a final checkpoint: its table reaches
    # hundreds of MB and intermediate copies buy nothing.
    checkpoint_interval = (
        episodes + 1 if kind == "q_learning" else max(1, episodes // 4)
    )
    trainer_kwargs = dict(
        checkpoint_dir=checkpoint_dir,
        log_interval=max(1, episodes // 5),
        eval_interval=max(1, episodes // 10),
        eval_episodes=min(20, args.eval_episodes),
        checkpoint_interval=checkpoint_interval,
    )
    if spec.self_play:
        # --self-play forces the zero-lag mirror; otherwise the game's declared
        # opponent (a fixed heuristic) gives an absolute number to read.
        opponent = None if args.self_play else (
            spec.opponent_factory(args.seed) if spec.opponent_factory else None
        )
        trainer = SelfPlayTrainer(env=env, agent=agent, opponent=opponent,
                                  **trainer_kwargs)
    else:
        trainer = Trainer(env=env, agent=agent, **trainer_kwargs)

    # Captured before training so the recorded epsilon is the start value and
    # the table size is zero -- the configuration, not the outcome.
    config = TrainingReport.from_trainer(
        trainer, episodes=episodes, max_steps_per_episode=max_steps,
    ).as_dict()

    callback, extras = make_episode_recorder(env, agent, spec.episode_extras)
    tick = time.time()
    metrics = trainer.train(
        episodes=episodes, max_steps_per_episode=max_steps,
        verbose=args.verbose, callback=callback,
    )
    train_seconds = time.time() - tick

    tick = time.time()
    after = spec.evaluate(agent, args.eval_episodes, args.seed)
    eval_seconds += time.time() - tick
    print(f"  after:  {_format_metrics(after)}  ({train_seconds:.0f}s train)",
          flush=True)

    suffix = checkpoint_suffix(kind)
    final_path = os.path.join(checkpoint_dir, f"final{suffix}")
    agent.save(final_path)
    metrics.save(os.path.join(checkpoint_dir, "metrics.json"))
    run_dir = store.run_dir(game, kind)
    metrics.save(str(run_dir / "metrics.json"))

    record = RunRecord.from_training(
        game=game, agent=kind, agent_class=agent_class_name(kind),
        metrics=metrics, config=config,
        started_at=started_at, finished_at=utc_now(),
        train_seconds=train_seconds, eval_seconds=eval_seconds,
        episode_extras=extras,
        baseline_before=before, baseline_after=after,
        host=host_info(seed=args.seed, device="cpu"),
        env_max_steps=max_steps,
        artifacts={
            "checkpoint": final_path,
            "checkpoint_bytes": os.path.getsize(final_path),
            "checkpoint_format": "pickle" if suffix == ".pkl" else "torch",
            "metrics_json": "metrics.json",
            "total_seconds": round(time.time() - started, 1),
        },
    )
    store.save_run(record)
    return record


def _format_metrics(metrics: dict) -> str:
    return "  ".join(
        f"{key}={value:.3f}" if isinstance(value, float) else f"{key}={value}"
        for key, value in metrics.items()
    )


def run_baselines(game: str, args, store: RunStore) -> None:
    """Measure the non-learning agents once and cache them.

    They do not learn, so there is nothing to re-measure between sweeps; MCTS
    on Klondike pays its full simulation budget on every one of hundreds of
    moves and is the main reason a sweep feels slow.
    """
    if args.skip_baselines and store.has_baselines(game):
        print(f"\n=== {game} baselines: cached, skipping ===", flush=True)
        return

    print(f"\n=== {game} baselines ({args.baseline_episodes} episodes) ===",
          flush=True)
    rows, protocol = measure_baselines(
        game, args.baseline_episodes, seed=args.seed,
        mcts_simulations=args.mcts_simulations,
    )
    store.save_baselines(BaselineSet(
        game=game, measured_at=utc_now(), protocol=protocol, rows=rows,
    ))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--games", default="all",
                        help="all (every registered game) | comma-separated names")
    parser.add_argument("--agents", default="all",
                        help=f"all | comma-separated from {','.join(LEARNERS)}")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--episodes-q-learning", type=int, default=None,
                        help="Override the episode count for Q-learning alone; "
                             "its table grows without bound and can exhaust RAM")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=30,
                        help="Deals per before/after evaluation")
    parser.add_argument("--baseline-episodes", type=int, default=30)
    parser.add_argument("--mcts-simulations", type=int, default=None)
    parser.add_argument("--self-play", action="store_true",
                        help="Macao: mirror match instead of a fixed heuristic")
    parser.add_argument("--results-dir", default="./results")
    parser.add_argument("--checkpoint-dir", default="./checkpoints")
    parser.add_argument("--skip-baselines", action="store_true",
                        help="Reuse cached baselines when they already exist")
    parser.add_argument("--html-only", action="store_true",
                        help="Re-render the report from stored records only")
    parser.add_argument("--no-embed", action="store_true",
                        help="Reference figures on disk instead of embedding")
    parser.add_argument("--no-figures", action="store_true")
    # Report-side filters, with no default: the page covers everything in the
    # store unless asked otherwise. For reporting on your own game alone.
    parser.add_argument("--report-games", default=None,
                        help="Only report these games, comma-separated")
    parser.add_argument("--report-exclude-games", default=None,
                        help="Report every game except these, comma-separated")
    parser.add_argument("--report-exclude-builtin-games", action="store_true",
                        help="Shorthand for --report-exclude-games klondike,macao")
    parser.add_argument("--keep-going", action="store_true",
                        help="Continue the sweep when one run fails")
    parser.add_argument("--quiet", dest="verbose", action="store_false",
                        default=True)
    return parser


def resolve(args) -> tuple:
    registered = registered_sweep_games()
    if args.games in ("both", "all"):
        games = tuple(registered)
    else:
        games = tuple(g.strip() for g in args.games.split(",") if g.strip())
    agents = LEARNERS if args.agents == "all" else tuple(
        a.strip() for a in args.agents.split(",") if a.strip()
    )
    for game in games:
        if game not in registered:
            known = ", ".join(registered) or "none"
            raise SystemExit(
                f"Unknown game {game!r}. Registered: {known}. Register a custom "
                "game with rl_card_lib.harness.register_sweep_game before "
                "sweeping it."
            )
    for agent in agents:
        if agent not in LEARNERS:
            raise SystemExit(f"Unknown agent {agent!r}, expected one of {LEARNERS}")

    # Q-learning last: its table can exhaust memory, and a crash there should
    # not cost the runs that would otherwise have completed after it.
    agents = tuple(sorted(agents, key=lambda a: a == "q_learning"))
    return games, agents


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    args.episodes_for = lambda kind: (
        args.episodes_q_learning
        if kind == "q_learning" and args.episodes_q_learning
        else args.episodes
    )

    store = RunStore(args.results_dir)
    failures = []

    if not args.html_only:
        games, agents = resolve(args)

        if args.episodes >= 2000 and "q_learning" in agents:
            print(
                "Warning: Q-learning stores one entry per distinct observation "
                f"and never prunes. At {args.episodes} episodes its table can "
                "reach multiple GB and the pickle write copies it. Use "
                "--episodes-q-learning to cap it if memory is tight.\n",
                file=sys.stderr,
            )

        for game in games:
            run_baselines(game, args, store)

        for game in games:
            for kind in agents:
                episodes = args.episodes_for(kind)
                print(f"\n--- {game} / {kind} / {episodes} episodes ---",
                      flush=True)
                try:
                    train_one(game, kind, args, store)
                except Exception:
                    failures.append(f"{game}/{kind}")
                    traceback.print_exc()
                    if not args.keep_going:
                        print("\nAborting. Pass --keep-going to continue past "
                              "a failed run.", file=sys.stderr)
                        return 1
                    print(f"  FAILED, continuing: {game}/{kind}",
                          file=sys.stderr, flush=True)

    runs = store.load_runs()
    if not runs:
        print("No run records to report on.", file=sys.stderr)
        return 1

    command = "python packages/examples/scripts/run_sweep.py " + " ".join(
        sys.argv[1:]
    )

    def split(value):
        return [v.strip() for v in value.split(",") if v.strip()] if value else None

    exclude = split(args.report_exclude_games) or []
    if args.report_exclude_builtin_games:
        exclude = sorted(set(exclude) | {"klondike", "macao"})

    report = HtmlReport.build(
        store, embed=not args.no_embed, with_figures=not args.no_figures,
        include_games=split(args.report_games), exclude_games=exclude or None,
        command=command.strip(),
    )
    out = report.write(os.path.join(args.results_dir, "index.html"))
    size_mb = os.path.getsize(out) / 1_048_576

    print(f"\nWrote {out} ({size_mb:.1f} MB) covering {len(runs)} run(s)")
    if failures:
        print(f"Failed: {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
