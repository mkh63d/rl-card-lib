"""Fixed-deal evaluation protocols for both games.

Moved verbatim from `scripts/train_agents.py` so the sweep records exactly the
numbers the scripts print, rather than a second implementation that might
drift from them.

The deals really are fixed: each one comes from `env.reset(seed=...)`, which
reseeds the game's own RNG, so two runs of the same evaluation return the same
numbers and two agents are compared on the same boards. The seeds are the
held-out TEST pool declared in `rl_card_lib.harness.deals`, disjoint from the
deals a training run draws. Nothing global is reseeded.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from rl_card_lib.agents import Agent
from rl_card_lib.env import CardGameEnv
from rl_card_lib.games import KlondikeSolitaire, Macao
from rl_card_lib.harness.deals import evaluation_seeds


def _deals(
    episodes: Optional[int], seeds: Optional[Sequence[int]],
) -> list[int]:
    """Resolve the (episodes, seeds) pair into one list of deal seeds."""
    return list(seeds) if seeds is not None else evaluation_seeds(episodes)


def evaluate_klondike(
    agent: Agent, episodes: Optional[int] = None, max_steps: int = 300, *,
    seeds: Optional[Sequence[int]] = None,
) -> dict:
    """
    Play fixed deals and report reward, foundation progress and wins.

    Args:
        agent: Agent to evaluate (switched to eval mode)
        episodes: Deals to play, taken from the front of the TEST pool
        max_steps: Move cap per deal
        seeds: Explicit deal seeds, overriding `episodes`

    Returns:
        Dict of averaged metrics
    """
    deals = _deals(episodes, seeds)

    was_training = agent.training
    agent.eval()

    # One game and one env for the whole evaluation: the deal is chosen by the
    # seed handed to reset(), not by building a fresh KlondikeSolitaire (whose
    # RNG nothing seeded) each time round.
    game = KlondikeSolitaire()
    env = CardGameEnv(game, max_steps=max_steps)
    if hasattr(agent, "bind"):
        agent.bind(env)

    rewards, cards_up, wins = [], [], 0
    for deal in deals:
        observation, info = env.reset(seed=deal)
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

    if not deals:
        return {"reward": 0.0, "cards_up": 0.0, "win_rate": 0.0}
    return {
        "reward": float(np.mean(rewards)),
        "cards_up": float(np.mean(cards_up)),
        "win_rate": wins / len(deals),
    }


def evaluate_macao(
    agent: Agent, opponent: Agent, episodes: Optional[int] = None,
    max_steps: int = 200, *, seeds: Optional[Sequence[int]] = None,
) -> dict:
    """
    Play fixed games against an opponent and report the win rate.

    Args:
        agent: Agent to evaluate (switched to eval mode)
        opponent: Policy to play against
        episodes: Games to play, dealt from the front of the TEST pool
        max_steps: Move cap per game
        seeds: Explicit deal seeds, overriding `episodes`

    Returns:
        Dict of averaged metrics
    """
    deals = _deals(episodes, seeds)

    was_training = agent.training
    agent.eval()

    game = Macao(num_players=2)
    env = CardGameEnv(game, max_steps=max_steps)
    for participant in (agent, opponent):
        if hasattr(participant, "bind"):
            participant.bind(env)

    wins, draws = 0, 0
    for deal in deals:
        observation, info = env.reset(seed=deal)
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

    if not deals:
        return {"win_rate": 0.0, "draw_rate": 0.0}
    return {"win_rate": wins / len(deals), "draw_rate": draws / len(deals)}


def evaluate_macao_suite(agent: Agent, opponents: dict,
                         episodes: Optional[int] = None, max_steps: int = 200,
                         *, seeds: Optional[Sequence[int]] = None) -> dict:
    """Evaluate against several named opponents in one pass.

    Flattened into `win_rate_vs_<name>` keys so the report's headline metric
    (win rate against the heuristic) can be looked up without knowing how the
    sweep happened to structure its results. Every opponent is faced on the
    same deals, so the rows are comparable to each other as well as across runs.
    """
    deals = _deals(episodes, seeds)
    out: dict = {}
    for name, opponent in opponents.items():
        result = evaluate_macao(agent, opponent, max_steps=max_steps, seeds=deals)
        out[f"win_rate_vs_{name}"] = result["win_rate"]
        out[f"draw_rate_vs_{name}"] = result["draw_rate"]
    return out
