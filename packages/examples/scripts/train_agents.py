"""
Train the learning agents on either game and report against the baselines.

Trains one or more of the learning agents, then evaluates the result against the
non-learning baselines so the numbers mean something. Defaults are sized for a
smoke run that finishes in minutes on CPU; pass --episodes for a real one.

On Macao the agent trains against the heuristic rather than itself by default.
Self-play win rate sits near 50% by construction and cannot tell you whether the
agent got better or the opponent got worse; a fixed opponent gives an absolute
number. Pass --self-play for the mirror match.

Klondike reports cards moved to the foundations, not just reward, because the
shaped reward contains a repeatable tableau-shuffle loop (see TODO.md). Reward
going up on Klondike is not evidence of solitaire skill; cards-up is.

Usage:
    python train_agents.py --game klondike --agent double_dqn --episodes 500
    python train_agents.py --game macao --agent ppo --episodes 2000
    python train_agents.py --game both --agent all --episodes 500
"""

import argparse
import os
import random
import time

import numpy as np

from rl_card_lib.agents import (
    DoubleDQNAgent,
    DQNAgent,
    PPOAgent,
    QLearningAgent,
    RandomAgent,
)
from rl_card_lib.env import CardGameEnv
from rl_card_lib.games import (
    KlondikeHeuristicAgent,
    KlondikeSolitaire,
    Macao,
    MacaoHeuristicAgent,
)
from rl_card_lib.trainer import SelfPlayTrainer, Trainer

LEARNERS = ("q_learning", "dqn", "double_dqn", "ppo")


def build_learner(kind: str, state_size: int, action_size: int, seed: int):
    """
    Construct one learning agent by name.

    Args:
        kind: One of LEARNERS
        state_size: Observation width
        action_size: Number of actions
        seed: Random seed

    Returns:
        The constructed agent
    """
    if kind == "q_learning":
        return QLearningAgent(
            action_size=action_size,
            learning_rate=0.1,
            gamma=0.95,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay=0.99995,
            seed=seed,
        )
    if kind == "dqn":
        return DQNAgent(
            state_size=state_size, action_size=action_size,
            hidden_sizes=[256, 128], learning_rate=5e-4, gamma=0.95,
            epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.9995,
            buffer_size=50_000, batch_size=64, target_update_freq=500,
            device="cpu", seed=seed,
        )
    if kind == "double_dqn":
        return DoubleDQNAgent(
            state_size=state_size, action_size=action_size,
            hidden_sizes=[256, 128], learning_rate=5e-4, gamma=0.95,
            epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.9995,
            buffer_size=50_000, batch_size=64, target_update_freq=500,
            dueling=True, device="cpu", seed=seed,
        )
    if kind == "ppo":
        return PPOAgent(
            state_size=state_size, action_size=action_size,
            hidden_sizes=[256, 128], learning_rate=3e-4, gamma=0.95,
            gae_lambda=0.95, clip_epsilon=0.2, epochs=4, minibatch_size=64,
            rollout_steps=1024, entropy_coef=0.01, device="cpu", seed=seed,
        )
    raise ValueError(f"Unknown agent {kind!r}, expected one of {LEARNERS}")


def evaluate_klondike(agent, episodes: int, max_steps: int = 300) -> dict:
    """
    Play fixed deals and report reward, foundation progress and wins.

    Args:
        agent: Agent to evaluate (switched to eval mode)
        episodes: Deals to play
        max_steps: Move cap per deal

    Returns:
        Dict of averaged metrics
    """
    was_training = agent.training
    agent.eval()

    rewards, cards_up, wins = [], [], 0
    for seed in range(episodes):
        random.seed(10_000 + seed)
        np.random.seed(10_000 + seed)

        game = KlondikeSolitaire()
        env = CardGameEnv(game, max_steps=max_steps)
        if hasattr(agent, "bind"):
            agent.bind(env)

        observation, info = env.reset()
        agent.reset()
        total = 0.0

        for _ in range(max_steps):
            action = agent.select_action(observation, info.get("legal_actions"))
            observation, reward, terminated, truncated, info = env.step(action)
            total += reward
            if terminated or truncated:
                break

        rewards.append(total)
        cards_up.append(sum(len(pile) for pile in game.foundations))
        wins += 1 if game.winner == 0 else 0

    if was_training:
        agent.train()

    return {
        "reward": float(np.mean(rewards)),
        "cards_up": float(np.mean(cards_up)),
        "win_rate": wins / episodes,
    }


def evaluate_macao(agent, opponent, episodes: int, max_steps: int = 200) -> dict:
    """
    Play fixed games against an opponent and report the win rate.

    Args:
        agent: Agent to evaluate (switched to eval mode)
        opponent: Policy to play against
        episodes: Games to play
        max_steps: Move cap per game

    Returns:
        Dict of averaged metrics
    """
    was_training = agent.training
    agent.eval()

    wins, draws = 0, 0
    for seed in range(episodes):
        random.seed(10_000 + seed)
        np.random.seed(10_000 + seed)

        game = Macao(num_players=2)
        env = CardGameEnv(game, max_steps=max_steps)
        for participant in (agent, opponent):
            if hasattr(participant, "bind"):
                participant.bind(env)

        observation, info = env.reset()
        agent.reset()

        for _ in range(max_steps):
            actor = game.current_player_idx
            chooser = agent if actor == 0 else opponent
            action = chooser.select_action(observation, info.get("legal_actions"))
            observation, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        if game.winner == 0:
            wins += 1
        elif game.winner is None:
            draws += 1

    if was_training:
        agent.train()

    return {"win_rate": wins / episodes, "draw_rate": draws / episodes}


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
