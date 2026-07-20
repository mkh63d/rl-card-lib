"""
Benchmark every agent on both games, without training any of them.

This establishes the baselines a learner has to beat. The non-learning agents
here (heuristic, lookahead, MCTS) play at full strength immediately, so their
numbers are the bar; the learning agents are untrained at this point and should
score near random, which is what makes this a useful "before" snapshot.

Klondike reports cards moved to the foundations alongside reward. The two used
to disagree wildly (the shaped reward once paid +0.04 a move for shuffling two
tableau piles back and forth, and agents that optimized it farmed the loop);
that payment is gone, so reward and cards-up should now agree in ranking.
Cards-up remains the metric to trust if they ever diverge again.

Usage:
    python benchmark_agents.py [--episodes N] [--game klondike|macao|both]
"""

import argparse

from rl_card_lib.agents import (
    DoubleDQNAgent,
    DQNAgent,
    MCTSAgent,
    PPOAgent,
    QLearningAgent,
)
from rl_card_lib.games import (
    KlondikeHeuristicAgent,
    KlondikeSolitaire,
    Macao,
)
from rl_card_lib.harness import (
    klondike_baseline_agents,
    macao_baseline_agents,
    run_klondike_baselines,
    run_macao_baselines,
)


def benchmark_klondike(episodes: int, max_steps: int = 300) -> list[dict]:
    """
    Play every agent over the same deals and report how far each one gets.

    The non-learning agents come from the shared harness; the untrained
    learners are added here because they only make sense as a "before"
    snapshot, not as a reusable baseline.

    Args:
        episodes: Deals to play per agent
        max_steps: Move cap per deal

    Returns:
        One result row per agent
    """
    probe = KlondikeSolitaire()
    state_size = probe.get_observation_shape()[0]
    action_size = probe.get_action_space_size()

    agents = klondike_baseline_agents(seed=0)
    agents.insert(4, ("MCTS(20)+heur", MCTSAgent(
        simulations=20, rollout_depth=15,
        rollout_policy=KlondikeHeuristicAgent(seed=0), seed=0,
    )))
    agents += [
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
    return run_klondike_baselines(agents, episodes, max_steps)


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

    agents = macao_baseline_agents(seed=0)
    agents.insert(4, ("MCTS(40)x4det", MCTSAgent(
        simulations=40, determinizations=4, rollout_depth=20, seed=0,
    )))
    agents += [
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
    return run_macao_baselines(agents, episodes, max_steps)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument(
        "--game", choices=["klondike", "macao", "both"], default="both"
    )
    args = parser.parse_args()

    if args.game in ("klondike", "both"):
        print(f"\n=== Klondike ({args.episodes} deals, same deals for every agent) ===")
        benchmark_klondike(args.episodes)

    if args.game in ("macao", "both"):
        print(f"\n=== Macao vs random opponent ({args.episodes} games) ===")
        benchmark_macao(args.episodes)


if __name__ == "__main__":
    main()
