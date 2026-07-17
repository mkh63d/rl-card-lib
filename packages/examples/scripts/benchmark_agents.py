"""
Benchmark every agent on both games, without training any of them.

This establishes the baselines a learner has to beat. The non-learning agents
here (heuristic, lookahead, MCTS) play at full strength immediately, so their
numbers are the bar; the learning agents are untrained at this point and should
score near random, which is what makes this a useful "before" snapshot.

Klondike reports cards moved to the foundations alongside reward on purpose: the
two disagree, because the shaped reward contains a repeatable tableau-shuffle
loop worth +0.04 a move (see TODO.md). Cards-up is the honest metric.

Usage:
    python benchmark_agents.py [--episodes N] [--game klondike|macao|both]
"""

import argparse
import random
import time

import numpy as np

from rl_card_lib.agents import (
    DoubleDQNAgent,
    DQNAgent,
    GreedyLookaheadAgent,
    MCTSAgent,
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


def benchmark_klondike(episodes: int, max_steps: int = 300) -> list[dict]:
    """
    Play every agent over the same deals and report how far each one gets.

    Args:
        episodes: Deals to play per agent
        max_steps: Move cap per deal

    Returns:
        One result row per agent
    """
    probe = KlondikeSolitaire()
    state_size = probe.get_observation_shape()[0]
    action_size = probe.get_action_space_size()

    def build_agents():
        return [
            ("Random", RandomAgent(action_size=action_size, seed=0)),
            ("Heuristic", KlondikeHeuristicAgent(seed=0)),
            ("GreedyLookahead(1)", GreedyLookaheadAgent(depth=1, seed=0)),
            ("MCTS(60)", MCTSAgent(simulations=60, rollout_depth=30, seed=0)),
            ("MCTS(60)+heur", MCTSAgent(
                simulations=60, rollout_depth=30,
                rollout_policy=KlondikeHeuristicAgent(seed=0), seed=0,
            )),
            ("QLearning (untrained)", QLearningAgent(action_size=action_size, seed=0)),
            ("DQN (untrained)", DQNAgent(
                state_size=state_size, action_size=action_size,
                hidden_sizes=[128], device="cpu", seed=0,
            )),
            ("DoubleDQN (untrained)", DoubleDQNAgent(
                state_size=state_size, action_size=action_size,
                hidden_sizes=[128], device="cpu", seed=0,
            )),
            ("PPO (untrained)", PPOAgent(
                state_size=state_size, action_size=action_size,
                hidden_sizes=[128], device="cpu", seed=0,
            )),
        ]

    results = []
    for name, agent in build_agents():
        rewards, cards_up, wins = [], [], 0
        started = time.time()

        for seed in range(episodes):
            random.seed(seed)
            np.random.seed(seed)

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

        results.append({
            "agent": name,
            "reward": float(np.mean(rewards)),
            "cards_up": float(np.mean(cards_up)),
            "win_rate": wins / episodes,
            "seconds": time.time() - started,
        })
        print(f"  {name:24s} reward={results[-1]['reward']:7.2f}  "
              f"cards_up={results[-1]['cards_up']:5.1f}  "
              f"wins={results[-1]['win_rate']:5.1%}  "
              f"({results[-1]['seconds']:.1f}s)", flush=True)

    return results


def benchmark_macao(episodes: int, max_steps: int = 200) -> list[dict]:
    """
    Play every agent against a random opponent and report the win rate.

    Args:
        episodes: Games to play per agent
        max_steps: Move cap per game

    Returns:
        One result row per agent
    """
    probe = Macao(num_players=2)
    state_size = probe.get_observation_shape()[0]
    action_size = probe.get_action_space_size()

    def build_agents():
        return [
            ("Random", RandomAgent(action_size=action_size, seed=0)),
            ("Heuristic", MacaoHeuristicAgent(seed=0)),
            ("GreedyLookahead(1)", GreedyLookaheadAgent(depth=1, seed=0)),
            ("MCTS(60)", MCTSAgent(simulations=60, rollout_depth=30, seed=0)),
            ("MCTS(60)x4det", MCTSAgent(
                simulations=60, determinizations=4, rollout_depth=30, seed=0
            )),
            ("QLearning (untrained)", QLearningAgent(action_size=action_size, seed=0)),
            ("DQN (untrained)", DQNAgent(
                state_size=state_size, action_size=action_size,
                hidden_sizes=[128], device="cpu", seed=0,
            )),
            ("DoubleDQN (untrained)", DoubleDQNAgent(
                state_size=state_size, action_size=action_size,
                hidden_sizes=[128], device="cpu", seed=0,
            )),
            ("PPO (untrained)", PPOAgent(
                state_size=state_size, action_size=action_size,
                hidden_sizes=[128], device="cpu", seed=0,
            )),
        ]

    results = []
    for name, agent in build_agents():
        wins, draws = 0, 0
        started = time.time()

        for seed in range(episodes):
            random.seed(seed)
            np.random.seed(seed)

            game = Macao(num_players=2)
            env = CardGameEnv(game, max_steps=max_steps)
            if hasattr(agent, "bind"):
                agent.bind(env)
            opponent = RandomAgent(action_size=action_size, seed=seed)

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

        results.append({
            "agent": name,
            "win_rate": wins / episodes,
            "draw_rate": draws / episodes,
            "seconds": time.time() - started,
        })
        print(f"  {name:24s} wins={results[-1]['win_rate']:5.1%}  "
              f"draws={results[-1]['draw_rate']:5.1%}  "
              f"({results[-1]['seconds']:.1f}s)", flush=True)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument(
        "--game", choices=["klondike", "macao", "both"], default="both"
    )
    args = parser.parse_args()

    if args.game in ("klondike", "both"):
        print(f"\n=== Klondike ({args.episodes} deals, same deals for every agent) ===")
        print("    reward and cards_up disagree by design; see TODO.md reward loop")
        benchmark_klondike(args.episodes)

    if args.game in ("macao", "both"):
        print(f"\n=== Macao vs random opponent ({args.episodes} games) ===")
        benchmark_macao(args.episodes)


if __name__ == "__main__":
    main()
