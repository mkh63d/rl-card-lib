"""Fixed-deal evaluation protocols for both games.

Moved verbatim from `scripts/train_agents.py` so the sweep records exactly the
numbers the scripts print, rather than a second implementation that might
drift from them.

Note that both functions reseed the global RNGs once per episode. That is the
practice the library deliberately removed from its own components, and it means
an evaluation perturbs the training RNG stream. It is kept because changing it
would change every historical number; the report states it as a caveat.
"""

from __future__ import annotations

import random

import numpy as np

from rl_card_lib.env import CardGameEnv
from rl_card_lib.games import KlondikeSolitaire, Macao


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


def evaluate_macao_suite(agent, opponents: dict, episodes: int,
                         max_steps: int = 200) -> dict:
    """Evaluate against several named opponents in one pass.

    Flattened into `win_rate_vs_<name>` keys so the report's headline metric
    (win rate against the heuristic) can be looked up without knowing how the
    sweep happened to structure its results.
    """
    out: dict = {}
    for name, opponent in opponents.items():
        result = evaluate_macao(agent, opponent, episodes, max_steps)
        out[f"win_rate_vs_{name}"] = result["win_rate"]
        out[f"draw_rate_vs_{name}"] = result["draw_rate"]
    return out
