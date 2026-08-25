"""How the trained policies actually behave: looping, action mix, Q spread.

Four numbers in `diagnosis.md` used to come from one-off scripts that were never
committed -- `raw/greedy_loop_probe.json`, `raw/action_mix_probe.json`,
`raw/q_spread_probe.json` and `raw/ppo_argmax_vs_sampled.json` had no producer
in the tree, so after PRs #24-#34 there was no way to bring them up to date.
This script is that producer. It answers, over the held-out TEST deals:

    D2   how strongly does the learned Q differentiate the *legal* actions
    D3   what fraction of greedy steps land in a position already seen
    D11  argmax vs sampling for PPO -- on Klondike, two different policies

Everything is measured from the sweep's own checkpoints in
`thesis_notes/checkpoints/`, so a row is an average over the same three
initialisation seeds as the results tables, and every arm is reported
separately. The previous versions of these files averaged a single checkpoint
from the library sweep and could not say which arm they described.

Writes raw/policy_diagnostics.json (everything, per arm and seed) plus the four
legacy-named files above, rebuilt from the `fixed` arm so the figures and prose
that already read them keep working.

    python thesis_notes/scripts/probe_policy_diagnostics.py
    python thesis_notes/scripts/probe_policy_diagnostics.py --deals 50
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from harness import frozen_exploration  # noqa: E402
from split import TEST_SEEDS  # noqa: E402

from rl_card_lib.agents import RandomAgent  # noqa: E402
from rl_card_lib.env import CardGameEnv  # noqa: E402
from rl_card_lib.games import KlondikeSolitaire, Macao  # noqa: E402
from rl_card_lib.games.heuristics import (  # noqa: E402
    KlondikeHeuristicAgent,
    MacaoHeuristicAgent,
)
from rl_card_lib.games.registration import KLONDIKE_MAX_PASSES  # noqa: E402
from rl_card_lib.harness import build_learner  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "raw")
CHECKPOINTS = os.path.join(HERE, "..", "checkpoints")

KLONDIKE_MAX_STEPS = 300
MACAO_MAX_STEPS = 200
SEEDS = (0, 1, 2)
LEARNERS = ("dqn", "double_dqn", "ppo", "q_learning")

#: Passes through the stock each arm plays -- mirrors run_one.ARMS, so a
#: checkpoint is always replayed on the rules it was trained under.
ARM_MAX_PASSES = {"asis": None, "fixed": KLONDIKE_MAX_PASSES,
                  "noloop": KLONDIKE_MAX_PASSES}
ARMS = tuple(ARM_MAX_PASSES)

#: Klondike's action encoding, from the comment above `KlondikeSolitaire`:
#: 0 draw, 1-7 waste->tableau, 8-11 waste->foundation, 12-18 tableau->
#: foundation, 19-67 tableau<->tableau. Grouped the way D3 reads them -- what
#: the policy spends its moves on.


def action_group(action: int) -> str:
    if action == 0:
        return "draw/recycle"
    if 1 <= action <= 7:
        return "waste->tableau"
    if 8 <= action <= 18:
        return "-> foundation"
    return "tableau<->tableau"


def q_spread(agent, observation, legal) -> tuple[float, float] | None:
    """Mean legal Q and the max-min spread across the legal actions.

    Spread is deliberately max-min rather than a standard deviation: D2 asks how
    far apart the best and worst legal moves look to the network, which is what
    an argmax actually decides on. PPO has no Q at all, so it returns None.
    """
    if not hasattr(agent, "get_q_values") or not legal:
        return None
    values = np.asarray(agent.get_q_values(observation), dtype=np.float64)
    legal_values = values[list(legal)]
    return float(np.mean(legal_values)), float(legal_values.max() - legal_values.min())


def play_klondike(agent, seeds, max_passes, sample_ppo=None) -> dict:
    """Replay every deal greedily, recording behaviour rather than score alone."""
    game = KlondikeSolitaire(max_passes=max_passes)
    env = CardGameEnv(game, max_steps=KLONDIKE_MAX_STEPS)
    if hasattr(agent, "bind"):
        agent.bind(env)

    repeats = steps = wins = 0
    groups: dict[str, int] = {}
    distinct_per_episode, cards, q_means, q_spreads = [], [], [], []

    with frozen_exploration(agent):
        if sample_ppo is not None and hasattr(agent, "eval_greedy"):
            agent.eval_greedy = not sample_ppo
        agent.eval()
        for seed in seeds:
            observation, info = env.reset(seed=seed)
            agent.reset()
            seen_actions = set()
            for _ in range(KLONDIKE_MAX_STEPS):
                legal = info.get("legal_actions")
                measured = q_spread(agent, observation, legal)
                if measured is not None:
                    q_means.append(measured[0])
                    q_spreads.append(measured[1])
                action = agent.select_action(observation, legal)
                observation, _, terminated, truncated, info = env.step(action)
                groups[action_group(action)] = groups.get(action_group(action), 0) + 1
                seen_actions.add(action)
                steps += 1
                repeats += int(bool(info.get("repeated_position")))
                if terminated or truncated:
                    break
            distinct_per_episode.append(len(seen_actions))
            cards.append(sum(len(pile) for pile in game.foundations))
            wins += 1 if game.winner == 0 else 0

    n = len(seeds)
    out = {
        "episodes": n,
        "steps": steps,
        "repeated_position_frac": round(repeats / steps, 4) if steps else 0.0,
        "distinct_actions_per_episode": round(float(np.mean(distinct_per_episode)), 2),
        "cards_up": round(float(np.mean(cards)), 3),
        "win_rate": round(wins / n, 4) if n else 0.0,
        "action_mix": {k: round(v / steps, 3) for k, v in sorted(groups.items())}
        if steps else {},
    }
    if q_means:
        mean_q = float(np.mean(q_means))
        mean_spread = float(np.mean(q_spreads))
        out["positions"] = len(q_means)
        out["mean_legal_Q"] = round(mean_q, 3)
        out["mean_legal_Q_spread"] = round(mean_spread, 4)
        out["spread_as_pct_of_mean_Q"] = (
            round(100 * mean_spread / abs(mean_q), 1) if mean_q else None
        )
    return out


def play_macao(agent, seeds, sample_ppo=None) -> dict:
    """The control condition: short episodes, no reversible cycle to fall into."""
    game = Macao(num_players=2)
    env = CardGameEnv(game, max_steps=MACAO_MAX_STEPS)
    opponent = MacaoHeuristicAgent(seed=0)
    for participant in (agent, opponent):
        if hasattr(participant, "bind"):
            participant.bind(env)

    repeats = steps = wins = 0
    q_means, q_spreads = [], []

    with frozen_exploration(agent, opponent):
        if sample_ppo is not None and hasattr(agent, "eval_greedy"):
            agent.eval_greedy = not sample_ppo
        agent.eval()
        opponent.eval()
        for seed in seeds:
            observation, info = env.reset(seed=seed)
            agent.reset()
            opponent.reset()
            for _ in range(MACAO_MAX_STEPS):
                actor = game.current_player_idx
                legal = info.get("legal_actions")
                chooser = agent if actor == 0 else opponent
                if actor == 0:
                    measured = q_spread(agent, observation, legal)
                    if measured is not None:
                        q_means.append(measured[0])
                        q_spreads.append(measured[1])
                action = chooser.select_action(observation, legal)
                observation, _, terminated, truncated, info = env.step(action)
                if actor == 0:
                    steps += 1
                    repeats += int(bool(info.get("repeated_position")))
                if terminated or truncated:
                    break
            wins += 1 if game.winner == 0 else 0

    n = len(seeds)
    out = {
        "episodes": n,
        "steps": steps,
        "repeated_position_frac": round(repeats / steps, 4) if steps else 0.0,
        "win_rate": round(wins / n, 4) if n else 0.0,
    }
    if q_means:
        mean_q = float(np.mean(q_means))
        mean_spread = float(np.mean(q_spreads))
        out["positions"] = len(q_means)
        out["mean_legal_Q"] = round(mean_q, 3)
        out["mean_legal_Q_spread"] = round(mean_spread, 4)
        out["spread_as_pct_of_mean_Q"] = (
            round(100 * mean_spread / abs(mean_q), 1) if mean_q else None
        )
    return out


def load_checkpoint(game: str, kind: str, arm: str, seed: int):
    """Rebuild the agent and load its weights, or None when the run is missing."""
    suffix = ".pkl" if kind == "q_learning" else ".pt"
    path = os.path.join(CHECKPOINTS, f"{game}__{kind}__{arm}__s{seed}{suffix}")
    if not os.path.exists(path):
        return None
    action_size = (KlondikeSolitaire.MAX_ACTIONS if game == "klondike"
                   else Macao.MAX_ACTIONS)
    state_size = 221 if game == "klondike" else 126
    agent = build_learner(kind, state_size, action_size, seed)
    agent.load(path)
    return agent


def mean_of(rows: list[dict], key: str):
    values = [r[key] for r in rows if r.get(key) is not None]
    return round(float(np.mean(values)), 4) if values else None


def aggregate(rows: list[dict]) -> dict:
    """Average the per-seed measurements into the row the tables quote."""
    out = {"seeds": len(rows)}
    for key in ("repeated_position_frac", "distinct_actions_per_episode",
                "cards_up", "win_rate", "mean_legal_Q", "mean_legal_Q_spread",
                "spread_as_pct_of_mean_Q", "positions"):
        value = mean_of(rows, key)
        if value is not None:
            out[key] = value
    mixes = [r["action_mix"] for r in rows if r.get("action_mix")]
    if mixes:
        out["action_mix"] = {
            group: round(float(np.mean([m.get(group, 0.0) for m in mixes])), 3)
            for group in sorted({g for m in mixes for g in m})
        }
    return out


def legacy_files(report: dict) -> dict:
    """Rebuild the four historical JSONs from the `fixed` arm.

    `figure_action_rule.py` and the diagnosis prose read these names. Keeping
    them means the rest of the pipeline needs no change; the full per-arm data
    stays in policy_diagnostics.json.
    """
    klondike = report["klondike"]
    macao = report["macao"]

    def fixed(agent, key="fixed"):
        return klondike.get(f"{agent}__{key}", {})

    loop = {}
    for agent in ("dqn", "double_dqn", "ppo", "random"):
        row = fixed(agent)
        if row:
            loop[agent] = {
                "repeated_position_frac": row.get("repeated_position_frac"),
                "distinct_actions_per_episode": row.get(
                    "distinct_actions_per_episode"),
                "cards_up": row.get("cards_up"),
            }

    mix = {}
    for agent in ("dqn", "double_dqn", "ppo", "random", "heuristic"):
        row = fixed(agent)
        if row.get("action_mix"):
            entry = dict(row["action_mix"])
            if row.get("mean_legal_Q_spread") is not None:
                entry["mean_legal_Q_spread"] = row["mean_legal_Q_spread"]
            mix[agent] = entry

    spread = {}
    for game, table in (("klondike", klondike), ("macao", macao)):
        for agent in ("dqn", "double_dqn"):
            row = table.get(f"{agent}__fixed", {})
            if row.get("mean_legal_Q") is not None:
                spread[f"{game}_{agent}"] = {
                    "positions": row.get("positions"),
                    "mean_legal_Q": row.get("mean_legal_Q"),
                    "mean_legal_Q_spread": row.get("mean_legal_Q_spread"),
                    "spread_as_pct_of_mean_Q": row.get("spread_as_pct_of_mean_Q"),
                }

    ppo = {}
    for game, table in (("klondike", klondike), ("macao", macao)):
        for rule in ("argmax", "sampled"):
            row = table.get(f"ppo__fixed__{rule}", {})
            if row:
                entry = {"win_rate": row.get("win_rate"),
                         "repeated_position_frac": row.get(
                             "repeated_position_frac")}
                if row.get("cards_up") is not None:
                    entry["cards_up"] = row["cards_up"]
                ppo[f"{game}_ppo_{rule}"] = entry

    return {"greedy_loop_probe.json": loop, "action_mix_probe.json": mix,
            "q_spread_probe.json": spread, "ppo_argmax_vs_sampled.json": ppo}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deals", type=int, default=None,
                        help="Use only the first N TEST deals (default: all)")
    args = parser.parse_args(argv)
    torch.set_num_threads(1)

    seeds = TEST_SEEDS[:args.deals] if args.deals else list(TEST_SEEDS)
    print(f"TEST deals: {len(seeds)}", flush=True)
    report: dict = {"deals": len(seeds), "init_seeds": list(SEEDS),
                    "klondike": {}, "macao": {}}
    started = time.time()

    for game in ("klondike", "macao"):
        for arm in ARMS:
            for kind in LEARNERS:
                variants = [(None, f"{kind}__{arm}")]
                if kind == "ppo":
                    variants = [(False, f"ppo__{arm}__argmax"),
                                (True, f"ppo__{arm}__sampled")]
                for sample_ppo, label in variants:
                    rows = []
                    for seed in SEEDS:
                        agent = load_checkpoint(game, kind, arm, seed)
                        if agent is None:
                            continue
                        rows.append(
                            play_klondike(agent, seeds, ARM_MAX_PASSES[arm],
                                          sample_ppo)
                            if game == "klondike"
                            else play_macao(agent, seeds, sample_ppo)
                        )
                        del agent
                    if not rows:
                        continue
                    report[game][label] = aggregate(rows)
                    row = report[game][label]
                    print(f"  {game:9s} {label:24s} n={row['seeds']} "
                          f"repeat={row.get('repeated_position_frac', 0):6.1%} "
                          f"win={row.get('win_rate', 0):6.1%}", flush=True)

    # Reference policies, measured on the bundled rules the `fixed` arm plays.
    for label, agent in (("random", None), ("heuristic", None)):
        for game in ("klondike", "macao"):
            if game == "klondike":
                built = (RandomAgent(action_size=KlondikeSolitaire.MAX_ACTIONS,
                                     seed=0) if label == "random"
                         else KlondikeHeuristicAgent(seed=0))
                measured = play_klondike(built, seeds, KLONDIKE_MAX_PASSES)
            else:
                built = (RandomAgent(action_size=Macao.MAX_ACTIONS, seed=0)
                         if label == "random" else MacaoHeuristicAgent(seed=0))
                measured = play_macao(built, seeds)
            measured["seeds"] = 1
            report[game][f"{label}__fixed"] = measured
            print(f"  {game:9s} {label + '__fixed':24s} n=1 "
                  f"repeat={measured.get('repeated_position_frac', 0):6.1%} "
                  f"win={measured.get('win_rate', 0):6.1%}", flush=True)

    report["seconds"] = round(time.time() - started, 1)
    os.makedirs(RAW, exist_ok=True)
    path = os.path.join(RAW, "policy_diagnostics.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"Wrote {os.path.abspath(path)}")

    for name, payload in legacy_files(report).items():
        if not payload:
            continue
        with open(os.path.join(RAW, name), "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"Wrote {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
