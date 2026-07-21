"""
Train the learning agents on either game and report against the baselines.

Trains one or more of the learning agents, then evaluates the result against the
non-learning baselines so the numbers mean something. Defaults are sized for a
smoke run that finishes in minutes on CPU; pass --episodes for a real one.

On Macao the agent trains against the heuristic rather than itself by default.
Self-play win rate sits near 50% by construction and cannot tell you whether the
agent got better or the opponent got worse; a fixed opponent gives an absolute
number. Pass --self-play for the mirror match.

Klondike reports cards moved to the foundations, not just reward. The shaped
reward used to contain a repeatable tableau-shuffle loop that made reward
anti-correlated with winning; the loop no longer pays, but cards-up stays the
headline metric because it cannot be gamed by any reward change.

Usage:
    python train_agents.py --game klondike --agent double_dqn --episodes 500
    python train_agents.py --game macao --agent ppo --episodes 2000
    python train_agents.py --game both --agent all --episodes 500
"""

import argparse
import os
import time

from rl_card_lib.agents import RandomAgent
from rl_card_lib.env import CardGameEnv
from rl_card_lib.games import (
    KlondikeHeuristicAgent,
    KlondikeSolitaire,
    Macao,
    MacaoHeuristicAgent,
)
from rl_card_lib.harness import (
    LEARNERS,
    build_learner,
    evaluate_klondike,
    evaluate_macao,
)
from rl_card_lib.trainer import SelfPlayTrainer, Trainer


def train_klondike(kind: str, episodes: int, seed: int, checkpoint_dir: str) -> dict:
    """Train one learner on Klondike and compare it to the baselines."""
    game = KlondikeSolitaire()
    env = CardGameEnv(game, max_steps=300)
    agent = build_learner(
        kind, env.observation_space.shape[0], env.action_space.n, seed
    )

    print(f"\n--- Klondike / {kind} / {episodes} episodes ---", flush=True)
    before = evaluate_klondike(agent, 30)
    print(f"  before: reward={before['reward']:7.2f} "
          f"cards_up={before['cards_up']:5.1f} wins={before['win_rate']:5.1%}",
          flush=True)

    trainer = Trainer(
        env=env, agent=agent,
        checkpoint_dir=os.path.join(checkpoint_dir, f"klondike_{kind}"),
        log_interval=max(1, episodes // 5),
        eval_interval=max(1, episodes // 2),
        eval_episodes=20,
        checkpoint_interval=max(1, episodes // 2),
    )
    started = time.time()
    metrics = trainer.train(episodes=episodes, max_steps_per_episode=300, verbose=True)
    elapsed = time.time() - started

    after = evaluate_klondike(agent, 30)
    print(f"  after:  reward={after['reward']:7.2f} "
          f"cards_up={after['cards_up']:5.1f} wins={after['win_rate']:5.1%}"
          f"  ({elapsed:.0f}s)", flush=True)

    agent.save(os.path.join(
        checkpoint_dir, f"klondike_{kind}",
        "final.pkl" if kind == "q_learning" else "final.pt",
    ))
    metrics.save(os.path.join(checkpoint_dir, f"klondike_{kind}", "metrics.json"))

    return {"agent": kind, "before": before, "after": after, "seconds": elapsed}


def train_macao(
    kind: str, episodes: int, seed: int, checkpoint_dir: str, self_play: bool
) -> dict:
    """Train one learner on Macao and compare it to the baselines."""
    game = Macao(num_players=2)
    env = CardGameEnv(game, max_steps=200)
    agent = build_learner(
        kind, env.observation_space.shape[0], env.action_space.n, seed
    )
    opponent = None if self_play else MacaoHeuristicAgent(seed=seed)
    yardstick = MacaoHeuristicAgent(seed=seed)
    random_opponent = RandomAgent(action_size=env.action_space.n, seed=seed)

    label = "self-play" if self_play else "vs heuristic"
    print(f"\n--- Macao / {kind} / {episodes} episodes / {label} ---", flush=True)
    before_random = evaluate_macao(agent, random_opponent, 40)
    before_heuristic = evaluate_macao(agent, yardstick, 40)
    print(f"  before: vs_random={before_random['win_rate']:5.1%} "
          f"vs_heuristic={before_heuristic['win_rate']:5.1%}", flush=True)

    trainer = SelfPlayTrainer(
        env=env, agent=agent, opponent=opponent,
        checkpoint_dir=os.path.join(checkpoint_dir, f"macao_{kind}"),
        log_interval=max(1, episodes // 5),
        eval_interval=max(1, episodes // 2),
        eval_episodes=20,
        checkpoint_interval=max(1, episodes // 2),
    )
    started = time.time()
    metrics = trainer.train(episodes=episodes, max_steps_per_episode=200, verbose=True)
    elapsed = time.time() - started

    after_random = evaluate_macao(agent, random_opponent, 40)
    after_heuristic = evaluate_macao(agent, yardstick, 40)
    print(f"  after:  vs_random={after_random['win_rate']:5.1%} "
          f"vs_heuristic={after_heuristic['win_rate']:5.1%}  ({elapsed:.0f}s)",
          flush=True)

    agent.save(os.path.join(
        checkpoint_dir, f"macao_{kind}",
        "final.pkl" if kind == "q_learning" else "final.pt",
    ))
    metrics.save(os.path.join(checkpoint_dir, f"macao_{kind}", "metrics.json"))

    return {
        "agent": kind,
        "before": {"vs_random": before_random["win_rate"],
                   "vs_heuristic": before_heuristic["win_rate"]},
        "after": {"vs_random": after_random["win_rate"],
                  "vs_heuristic": after_heuristic["win_rate"]},
        "seconds": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", choices=["klondike", "macao", "both"], default="both")
    parser.add_argument("--agent", choices=(*LEARNERS, "all"), default="all")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--self-play", action="store_true")
    parser.add_argument("--checkpoint-dir", default="./checkpoints")
    args = parser.parse_args()

    kinds = LEARNERS if args.agent == "all" else (args.agent,)

    if args.game in ("klondike", "both"):
        print("\n=== Klondike baselines ===")
        for name, baseline in (
            ("Random", RandomAgent(action_size=200, seed=args.seed)),
            ("Heuristic", KlondikeHeuristicAgent(seed=args.seed)),
        ):
            result = evaluate_klondike(baseline, 30)
            print(f"  {name:12s} reward={result['reward']:7.2f} "
                  f"cards_up={result['cards_up']:5.1f} wins={result['win_rate']:5.1%}",
                  flush=True)

        for kind in kinds:
            train_klondike(kind, args.episodes, args.seed, args.checkpoint_dir)

    if args.game in ("macao", "both"):
        print("\n=== Macao baselines (vs random opponent) ===")
        reference = RandomAgent(action_size=60, seed=args.seed)
        for name, baseline in (
            ("Random", RandomAgent(action_size=60, seed=args.seed + 1)),
            ("Heuristic", MacaoHeuristicAgent(seed=args.seed)),
        ):
            result = evaluate_macao(baseline, reference, 40)
            print(f"  {name:12s} wins={result['win_rate']:5.1%}", flush=True)

        for kind in kinds:
            train_macao(
                kind, args.episodes, args.seed, args.checkpoint_dir, args.self_play
            )


if __name__ == "__main__":
    main()
