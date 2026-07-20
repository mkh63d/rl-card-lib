"""The non-learning agents a learner has to beat.

Split out of `scripts/benchmark_agents.py` so the sweep can ask for only the
fixed-strength agents and skip the untrained-learner rows, and so the MCTS
budget is a parameter rather than a literal buried in a closure.

These agents do not learn, so they are measured once per game and cached.
MCTS on Klondike is the expensive one: deals run to the step cap and it pays
its whole simulation budget on every move.
"""

from __future__ import annotations

import random
import time
from typing import Optional

import numpy as np

from rl_card_lib.agents import GreedyLookaheadAgent, MCTSAgent, RandomAgent
from rl_card_lib.env import CardGameEnv
from rl_card_lib.games import (
    KlondikeHeuristicAgent,
    KlondikeSolitaire,
    Macao,
    MacaoHeuristicAgent,
)


def klondike_baseline_agents(seed: int = 0, *, mcts_simulations: int = 20) -> list:
    """(name, agent) pairs for Klondike.

    The MCTS budget defaults low on purpose: Klondike deals run to hundreds of
    moves and MCTS pays the full budget on each one, so this is what finishes
    in minutes rather than what plays best.
    """
    probe = KlondikeSolitaire()
    action_size = probe.get_action_space_size()
    return [
        ("Random", RandomAgent(action_size=action_size, seed=seed)),
        ("Heuristic", KlondikeHeuristicAgent(seed=seed)),
        ("GreedyLookahead(1)", GreedyLookaheadAgent(depth=1, seed=seed)),
        (f"MCTS({mcts_simulations})", MCTSAgent(
            simulations=mcts_simulations, rollout_depth=15, seed=seed,
        )),
    ]


def macao_baseline_agents(seed: int = 0, *, mcts_simulations: int = 40) -> list:
    """(name, agent) pairs for Macao."""
    probe = Macao(num_players=2)
    action_size = probe.get_action_space_size()
    return [
        ("Random", RandomAgent(action_size=action_size, seed=seed)),
        ("Heuristic", MacaoHeuristicAgent(seed=seed)),
        ("GreedyLookahead(1)", GreedyLookaheadAgent(depth=1, seed=seed)),
        (f"MCTS({mcts_simulations})", MCTSAgent(
            simulations=mcts_simulations, rollout_depth=20, seed=seed,
        )),
    ]


def run_klondike_baselines(
    agents: list, episodes: int, max_steps: int = 300, verbose: bool = True,
) -> list[dict]:
    """Play every agent over the same deals and report how far each one gets."""
    results = []
    for name, agent in agents:
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
        if verbose:
            row = results[-1]
            print(f"  {name:24s} reward={row['reward']:7.2f}  "
                  f"cards_up={row['cards_up']:5.1f}  "
                  f"wins={row['win_rate']:5.1%}  ({row['seconds']:.1f}s)", flush=True)

    return results


def run_macao_baselines(
    agents: list, episodes: int, max_steps: int = 200,
    opponent_seed: Optional[int] = None, verbose: bool = True,
) -> list[dict]:
    """Play every agent against a random opponent and report the win rate."""
    probe = Macao(num_players=2)
    action_size = probe.get_action_space_size()

    results = []
    for name, agent in agents:
        wins, draws = 0, 0
        started = time.time()

        for seed in range(episodes):
            random.seed(seed)
            np.random.seed(seed)

            game = Macao(num_players=2)
            env = CardGameEnv(game, max_steps=max_steps)
            if hasattr(agent, "bind"):
                agent.bind(env)
            opponent = RandomAgent(
                action_size=action_size,
                seed=seed if opponent_seed is None else opponent_seed,
            )

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
        if verbose:
            row = results[-1]
            print(f"  {name:24s} wins={row['win_rate']:5.1%}  "
                  f"draws={row['draw_rate']:5.1%}  ({row['seconds']:.1f}s)", flush=True)

    return results


def measure_baselines(
    game: str, episodes: int, *, seed: int = 0, mcts_simulations: Optional[int] = None,
    verbose: bool = True,
) -> tuple[list[dict], dict]:
    """Measure one game's baselines. Returns (rows, protocol)."""
    if game == "klondike":
        simulations = 20 if mcts_simulations is None else mcts_simulations
        agents = klondike_baseline_agents(seed, mcts_simulations=simulations)
        rows = run_klondike_baselines(agents, episodes, verbose=verbose)
        max_steps = 300
    elif game == "macao":
        simulations = 40 if mcts_simulations is None else mcts_simulations
        agents = macao_baseline_agents(seed, mcts_simulations=simulations)
        rows = run_macao_baselines(agents, episodes, verbose=verbose)
        max_steps = 200
    else:
        raise ValueError(f"Unknown game {game!r}")

    protocol = {
        "episodes": episodes,
        "max_steps": max_steps,
        "seed": seed,
        "mcts_simulations": simulations,
        "opponent": "random" if game == "macao" else None,
    }
    return rows, protocol
