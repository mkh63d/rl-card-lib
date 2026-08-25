"""Measure the facts protocol.md and diagnosis.md assert about the stock code.

Everything here runs against `packages/` unmodified. Output goes to
thesis_notes/raw/protocol_probe.json.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rl_card_lib.agents import RandomAgent  # noqa: E402
from rl_card_lib.env import CardGameEnv  # noqa: E402
from rl_card_lib.games import KlondikeSolitaire, Macao  # noqa: E402
from rl_card_lib.games.heuristics import (  # noqa: E402
    KlondikeHeuristicAgent, MacaoHeuristicAgent,
)
from rl_card_lib.games.registration import KLONDIKE_MAX_PASSES  # noqa: E402
from rl_card_lib.harness import build_learner  # noqa: E402
from rl_card_lib.harness.evaluation import evaluate_klondike  # noqa: E402
from rl_card_lib.trainer import SelfPlayTrainer, Trainer  # noqa: E402

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "raw")


# ---------------------------------------------------------------------------
# (a) what deal does the training loop actually play?
# ---------------------------------------------------------------------------

def deal_stream() -> dict:
    """Are consecutive training deals identical, and are they reproducible?"""
    def tableau_signature(game) -> tuple:
        return tuple(c.to_index() for pile in game.tableaux for c in pile)

    def hand_signature(game) -> tuple:
        return tuple(c.to_index() for p in game.players for c in p.hand)

    out = {}

    # Klondike: ten consecutive env.reset() calls, the way Trainer does it.
    game = KlondikeSolitaire()
    env = CardGameEnv(game, max_steps=300)
    sigs = []
    for _ in range(10):
        env.reset()
        sigs.append(tableau_signature(game))
    out["klondike_consecutive_resets_distinct"] = len(set(sigs))
    out["klondike_consecutive_resets_n"] = len(sigs)

    # Same process, restarted from scratch: is the sequence reproducible?
    def first_three(global_seed: int | None) -> list:
        if global_seed is not None:
            random.seed(global_seed)
            np.random.seed(global_seed)
        g = KlondikeSolitaire()
        e = CardGameEnv(g, max_steps=300)
        return [(e.reset(), tableau_signature(g))[1] for _ in range(3)]

    out["klondike_reproducible_under_global_seed"] = (
        first_three(1234) == first_three(1234)
    )
    out["klondike_reproducible_with_reset_seed"] = (
        [env.reset(seed=s) is not None and tableau_signature(game) for s in (7, 8, 9)]
        == [env.reset(seed=s) is not None and tableau_signature(game) for s in (7, 8, 9)]
    )

    # Macao, same questions.
    mgame = Macao(num_players=2)
    menv = CardGameEnv(mgame, max_steps=200)
    msigs = []
    for _ in range(10):
        menv.reset()
        msigs.append(hand_signature(mgame))
    out["macao_consecutive_resets_distinct"] = len(set(msigs))
    out["macao_consecutive_resets_n"] = len(msigs)

    # The published evaluation protocol claims fixed deals.
    def eval_protocol_deal(seed_offset: int) -> tuple:
        random.seed(10_000 + seed_offset)
        np.random.seed(10_000 + seed_offset)
        g = KlondikeSolitaire()
        return tableau_signature(g)

    repeats = [eval_protocol_deal(0) for _ in range(5)]
    out["evaluate_klondike_deal_0_distinct_over_5_calls"] = len(set(repeats))

    # And the whole published evaluator, run twice on the same agent.
    agent = RandomAgent(action_size=68, seed=0)
    a = evaluate_klondike(agent, episodes=30, max_steps=300)
    agent = RandomAgent(action_size=68, seed=0)
    b = evaluate_klondike(agent, episodes=30, max_steps=300)
    out["evaluate_klondike_repeat_identical"] = a == b
    out["evaluate_klondike_repeat_a"] = a
    out["evaluate_klondike_repeat_b"] = b
    return out


# ---------------------------------------------------------------------------
# (b) what is one episode?
# ---------------------------------------------------------------------------

def episode_shape(episodes: int = 200) -> dict:
    """Episode length and how episodes end, per game and per policy."""
    out: dict = {}

    def klondike_policy_stats(agent, label, max_passes=None):
        game = KlondikeSolitaire(max_passes=max_passes)
        env = CardGameEnv(game, max_steps=300)
        if hasattr(agent, "bind"):
            agent.bind(env)
        lengths, term, trunc, cards, wins, dead = [], 0, 0, [], 0, 0
        for seed in range(100_000, 100_000 + episodes):
            obs, info = env.reset(seed=seed)
            agent.reset()
            n = 0
            terminated = truncated = False
            for _ in range(300):
                action = agent.select_action(obs, info.get("legal_actions"))
                obs, r, terminated, truncated, info = env.step(action)
                n += 1
                if terminated or truncated:
                    break
            lengths.append(n)
            term += int(terminated)
            trunc += int(truncated)
            cards.append(sum(len(p) for p in game.foundations))
            wins += int(game.winner == 0)
            dead += int(terminated and game.winner != 0)
        out[label] = {
            "episodes": episodes,
            "mean_steps": float(np.mean(lengths)),
            "median_steps": float(np.median(lengths)),
            "min_steps": int(np.min(lengths)),
            "max_steps": int(np.max(lengths)),
            "terminated_rate": term / episodes,
            "truncated_rate": trunc / episodes,
            "win_rate": wins / episodes,
            "dead_deal_rate": dead / episodes,
            "mean_cards_up": float(np.mean(cards)),
            "max_passes": max_passes,
        }

    klondike_policy_stats(RandomAgent(action_size=68, seed=0), "klondike_random")
    klondike_policy_stats(KlondikeHeuristicAgent(seed=0), "klondike_heuristic")
    klondike_policy_stats(RandomAgent(action_size=68, seed=0),
                          "klondike_random_max_passes_3", max_passes=3)
    klondike_policy_stats(KlondikeHeuristicAgent(seed=0),
                          "klondike_heuristic_max_passes_3", max_passes=3)

    def macao_policy_stats(agent, opponent, label):
        game = Macao(num_players=2)
        env = CardGameEnv(game, max_steps=200)
        for p in (agent, opponent):
            if hasattr(p, "bind"):
                p.bind(env)
        lengths, term, trunc, wins, draws = [], 0, 0, 0, 0
        for seed in range(100_000, 100_000 + episodes):
            obs, info = env.reset(seed=seed)
            agent.reset()
            opponent.reset()
            n = 0
            terminated = truncated = False
            for _ in range(200):
                actor = game.current_player_idx
                chooser = agent if actor == 0 else opponent
                action = chooser.select_action(obs, info.get("legal_actions"))
                obs, r, terminated, truncated, info = env.step(action)
                n += 1
                if terminated or truncated:
                    break
            lengths.append(n)
            term += int(terminated)
            trunc += int(truncated)
            wins += int(game.winner == 0)
            draws += int(game.winner is None)
        out[label] = {
            "episodes": episodes,
            "mean_steps": float(np.mean(lengths)),
            "median_steps": float(np.median(lengths)),
            "min_steps": int(np.min(lengths)),
            "max_steps": int(np.max(lengths)),
            "terminated_rate": term / episodes,
            "truncated_rate": trunc / episodes,
            "win_rate_player0": wins / episodes,
            "draw_rate": draws / episodes,
        }

    macao_policy_stats(RandomAgent(action_size=65, seed=0),
                       RandomAgent(action_size=65, seed=1), "macao_random_vs_random")
    macao_policy_stats(MacaoHeuristicAgent(seed=0), MacaoHeuristicAgent(seed=1),
                       "macao_heuristic_vs_heuristic")
    macao_policy_stats(RandomAgent(action_size=65, seed=0),
                       MacaoHeuristicAgent(seed=1), "macao_random_vs_heuristic")
    return out


# ---------------------------------------------------------------------------
# (c) who does the Macao agent play against, in training and in evaluation?
# ---------------------------------------------------------------------------

def macao_opponents() -> dict:
    """Read the opponent configuration straight off a constructed trainer."""
    from rl_card_lib.harness.registry import sweep_game
    import rl_card_lib.games  # noqa: F401  registration side effect

    spec = sweep_game("macao")
    env = spec.env_factory()
    agent = build_learner("dqn", env.observation_space.shape[0], env.action_space.n, 0)

    default = SelfPlayTrainer(env=env, agent=agent,
                              opponent=spec.opponent_factory(0))
    mirror = SelfPlayTrainer(env=spec.env_factory(), agent=agent, opponent=None)
    return {
        "registered_self_play": spec.self_play,
        "registered_opponent_factory": (
            spec.opponent_factory(0).__class__.__name__
            if spec.opponent_factory else None
        ),
        "num_players": env.game.num_players,
        "opponents_per_episode": env.game.num_players - 1,
        "default_sweep_trainer": type(default).__name__,
        "default_sweep_opponent": type(default.opponent).__name__,
        "default_sweep_self_play_flag": default.self_play,
        "with_self_play_flag_opponent": type(mirror.opponent).__name__,
        "with_self_play_flag_is_snapshot": mirror.opponent is not agent,
        "opponent_update_interval": mirror.opponent_update_interval,
        "evaluation_opponents": ["RandomAgent", "MacaoHeuristicAgent"],
    }


# ---------------------------------------------------------------------------
# (Task 4) hyper-parameters as actually constructed
# ---------------------------------------------------------------------------

def hyperparameters() -> dict:
    from rl_card_lib.harness import registered_sweep_games, sweep_game

    out = {}
    for kind in ("q_learning", "dqn", "double_dqn", "ppo"):
        agent = build_learner(kind, 221, 68, 0)
        row = {"class": type(agent).__name__}
        for field in ("learning_rate", "gamma", "epsilon_start", "epsilon_end",
                      "epsilon_decay", "buffer_size", "batch_size",
                      "target_update_freq", "hidden_sizes", "gae_lambda",
                      "clip_epsilon", "epochs", "minibatch_size", "rollout_steps",
                      "entropy_coef", "value_coef", "max_grad_norm", "precision",
                      # Since #33 PPO's evaluation action rule is a setting, not
                      # an assumption: sampling the policy and taking its argmax
                      # are different policies on Klondike (diagnosis.md D11),
                      # so the table has to say which one produced the numbers.
                      "optimistic_init", "dueling", "eval_greedy"):
            if hasattr(agent, field):
                value = getattr(agent, field)
                row[field] = list(value) if isinstance(value, list) else value
        # Since #19 this one is declared per game, so the single number
        # build_learner returns describes only the default. Record what each
        # game actually trains with -- the table has one column per agent and
        # cannot otherwise say that DQN on Macao refreshes 5x more often.
        if "target_update_freq" in row:
            row["target_update_freq"] = {
                game: sweep_game(game).target_update_freq
                for game in registered_sweep_games()
            }
        buf = getattr(agent, "replay_buffer", None)
        if buf is not None:
            row["replay_buffer_class"] = type(buf).__name__
            row["replay_buffer_maxlen"] = getattr(buf.buffer, "maxlen", None)
        out[kind] = row
    return out


def epsilon_schedule() -> dict:
    """When does epsilon actually reach its floor, and what perturbs it?"""
    start, end, decay = 1.0, 0.05, 0.995

    # The literal loop from DQNAgent.reset(), which is where the decay lives.
    eps, episode, first_at_floor = start, 0, None
    trace = {}
    for episode in range(1, 6001):
        if episode > 1 and eps > end:
            eps *= decay
        if first_at_floor is None and eps <= end:
            first_at_floor = episode
        if episode in (1, 100, 300, 500, 600, 1000, 2500, 5000):
            trace[episode] = eps

    # The number of extra reset() calls the sweep's evaluations add.
    episodes = 5000
    eval_interval = max(1, episodes // 10)
    eval_episodes = min(20, 30)
    in_training_extra = (episodes // eval_interval) * eval_episodes
    klondike_before_after = 2 * 30
    macao_before_after = 2 * 2 * 30

    return {
        "start": start, "end": end, "decay_per_episode": decay,
        "episodes_to_floor_pure": first_at_floor,
        "analytic_episodes_to_floor": (np.log(end / start) / np.log(decay)) + 1,
        "epsilon_at_episode": trace,
        "final_epsilon_after_floor": end * decay,
        "eval_resets_during_training": in_training_extra,
        "eval_resets_before_after_klondike": klondike_before_after,
        "eval_resets_before_after_macao": macao_before_after,
        "recorded_epsilon_in_run_json_klondike": 0.8647077305675338,
        "recorded_epsilon_in_run_json_macao": 0.7439808620067382,
        "explains_klondike": float(start * decay ** 29),
        "explains_macao": float(start * decay ** 59),
        "fraction_of_training_at_floor_5000ep": 1 - (first_at_floor / episodes),
    }


def target_update_cadence() -> dict:
    """Each game's declared cadence, expressed in episodes of that game.

    The frequency is read from the registry rather than hard-coded: since the
    fix for issue #19 it is a per-game value, so a single number here would
    describe neither game.
    """
    from rl_card_lib.harness import sweep_game

    klondike_steps = 300.0     # every Klondike episode hits the cap
    macao_steps = 46.0         # measured mean, see episode_shape
    klondike_freq = sweep_game("klondike").target_update_freq
    macao_freq = sweep_game("macao").target_update_freq
    return {
        "gradient_steps_per_env_step": 1,
        "note": "DQNAgent.learn() takes one gradient step per env step once the "
                "buffer holds batch_size transitions, so train_steps ~ env steps. "
                "target_update_freq is declared per game in register_sweep_game.",
        "klondike_target_update_freq_gradient_steps": klondike_freq,
        "klondike_mean_steps_per_episode": klondike_steps,
        "klondike_target_updates_per_episode": klondike_steps / klondike_freq,
        "klondike_episodes_per_target_update": klondike_freq / klondike_steps,
        "macao_target_update_freq_gradient_steps": macao_freq,
        "macao_mean_steps_per_episode": macao_steps,
        "macao_target_updates_per_episode": macao_steps / macao_freq,
        "macao_episodes_per_target_update": macao_freq / macao_steps,
        "klondike_total_updates_5000_ep": 5000 * klondike_steps / klondike_freq,
        "macao_total_updates_5000_ep": 5000 * macao_steps / macao_freq,
    }


def truncation_and_terminal_reward() -> dict:
    """Is the terminal reward, or the bootstrap, lost at the step cap?

    Rewritten for the post-#24 contract. `Trainer._learn` now receives
    `terminated` as `done` and `truncated` as its own argument, so the old
    version of this probe -- which inferred truncation from `done` being set
    while the game was not over -- would report zero truncations on today's
    library and look like the problem had vanished. It measures both flags
    directly instead, and on both pass rules: `max_passes=None` is the
    configuration D4 calls pathological, and the bundled finite limit is what
    the agents actually train on since #30.
    """
    torch.set_num_threads(1)
    out: dict = {
        "klondike_class_default_max_passes": KlondikeSolitaire().max_passes,
        "klondike_bundled_max_passes": KLONDIKE_MAX_PASSES,
        "klondike_loss_reward_constant": KlondikeSolitaire.LOSS_REWARD,
        "klondike_draw_always_legal_with_unlimited_passes": True,
        "macao_pays_terminal_on_truncation": True,
        "macao_truncation_reward_rule":
            "0.1 * (mean opponent hand size - actor hand size), Macao._finish_step",
        "by_max_passes": {},
    }

    for label, passes in (("unlimited", None), ("bundled", KLONDIKE_MAX_PASSES)):
        captured: list[dict] = []

        class Spy(Trainer):
            def _learn(self, agent, obs, action, reward, next_obs, done, info,
                       truncated: bool = False):
                captured.append({
                    # `done` is `terminated` since #24; keep the game's own view
                    # alongside it so the two can be shown to agree.
                    "terminated": bool(done),
                    "game_over": bool(getattr(self.env.game, "done", False)),
                    "truncated": bool(truncated),
                    "reward": float(reward),
                })
                return super()._learn(agent, obs, action, reward, next_obs,
                                      done, info, truncated=truncated)

        env = CardGameEnv(KlondikeSolitaire(max_passes=passes), max_steps=300)
        agent = build_learner("dqn", 221, 68, 0)
        trainer = Spy(env=env, agent=agent, log_interval=10**9,
                      eval_interval=10**9, checkpoint_interval=10**9)
        trainer.train(episodes=5, max_steps_per_episode=300, verbose=False)

        ended = [c for c in captured if c["terminated"] or c["truncated"]]
        out["by_max_passes"][label] = {
            "max_passes": passes,
            "transitions_seen": len(captured),
            "transitions_ending_an_episode": len(ended),
            "terminated": sum(1 for c in ended if c["terminated"]),
            "truncated_only": sum(1 for c in ended
                                  if c["truncated"] and not c["terminated"]),
            "terminated_flag_matches_game_over": all(
                c["terminated"] == c["game_over"] for c in captured),
            "rewards_at_episode_end": [round(c["reward"], 4) for c in ended],
        }
    return out


def invalid_action_livelock() -> dict:
    """Does an episode of nothing but illegal actions end? Since #25, yes.

    D10 reported this as a livelock: `CardGameEnv.step()` returned on the
    illegal-action branch before touching `_step_count`, so `max_steps` was
    never reached and the episode ran forever, paying `invalid_action_reward`
    each time. PR #25 counts the step and applies the cap on that branch too.
    The probe still runs the same experiment -- it just no longer hardcodes the
    old conclusion, and reports what it observes.
    """
    env = CardGameEnv(Macao(num_players=2), max_steps=200)
    obs, info = env.reset(seed=0)
    legal = set(info["legal_actions"])
    illegal = next(a for a in range(65) if a not in legal)
    steps = 0
    ended = False
    for _ in range(5000):
        obs, r, term, trunc, info = env.step(illegal)
        steps += 1
        if term or trunc:
            ended = True
            break
    return {
        "illegal_steps_taken": steps,
        "episode_ended": ended,
        "env_internal_step_count": env._step_count,
        "env_max_steps": env.max_steps,
        "reward_per_illegal_step": env.invalid_action_reward,
        "probe_step_budget": 5000,
        "livelocks": not ended,
        "cause": (
            "CardGameEnv.step() returns before incrementing _step_count when "
            "the action is illegal, so max_steps is never reached"
            if not ended else
            "fixed in #25: the illegal-action branch increments _step_count and "
            "applies max_steps, so the episode truncates like any other"
        ),
    }


def main() -> int:
    os.makedirs(RAW, exist_ok=True)
    report = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    for name, fn in (
        ("deal_stream", deal_stream),
        ("episode_shape", episode_shape),
        ("macao_opponents", macao_opponents),
        ("hyperparameters", hyperparameters),
        ("epsilon_schedule", epsilon_schedule),
        ("target_update_cadence", target_update_cadence),
        ("truncation_and_terminal_reward", truncation_and_terminal_reward),
        ("invalid_action_livelock", invalid_action_livelock),
    ):
        print(f"-- {name}", flush=True)
        report[name] = fn()

    path = os.path.join(RAW, "protocol_probe.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)
    print(f"Wrote {os.path.abspath(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
