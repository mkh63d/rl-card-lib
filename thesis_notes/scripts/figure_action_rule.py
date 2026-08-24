"""The action-selection rule, not the weights: one figure.

Reads raw/greedy_vs_epsilon.json and raw/ppo_argmax_vs_sampled.json — both
measured on the same 200 TEST deals with the same trained checkpoints — and
draws what changing only the action rule does to the score and to how often the
policy revisits a position.
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_report import (  # noqa: E402
    BASELINE, FIGURES, LABEL, RAW, SERIES, SURFACE, TEXT_MUTED,
    TEXT_PRIMARY, TEXT_SECONDARY, caption, emit, load_json, write_csv,
)

AGENTS = ["ppo", "double_dqn", "dqn", "q_learning"]
RANDOM_BASELINE = 11.59      # raw/baselines_on_test.json, Klondike, TEST pool


def main() -> int:
    data = load_json("greedy_vs_epsilon.json")
    if not data:
        print("no greedy_vs_epsilon.json; run probe_greedy_vs_epsilon.py first")
        return 1
    rows = data["klondike"]
    epsilons = [0.0, 0.05, 0.20]

    fig, (left, right) = plt.subplots(1, 2, figsize=(11.0, 4.4))

    endpoints = []
    for agent in AGENTS:
        if agent not in rows:
            continue
        series = {r["epsilon"]: r for r in rows[agent]}
        ys = [series[e]["cards_up"] for e in epsilons]
        rs = [series[e]["repeated_position_frac"] * 100 for e in epsilons]
        colour = SERIES[agent]
        left.plot(epsilons, ys, color=colour, marker="o", markersize=8,
                  markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3,
                  label=LABEL[agent])
        right.plot(epsilons, rs, color=colour, marker="o", markersize=8,
                   markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3,
                   label=LABEL[agent])
        endpoints.append((LABEL[agent], ys[-1], rs[-1], colour))

    left.axhline(RANDOM_BASELINE, color=BASELINE, linestyle="--", linewidth=1.2)
    left.annotate("random baseline", xy=(0.155, RANDOM_BASELINE), xytext=(0, -14),
                  textcoords="offset points", fontsize=9.5, color=BASELINE,
                  ha="center")
    right.axhline(23.0, color=BASELINE, linestyle="--", linewidth=1.2)
    right.annotate("random policy", xy=(0.10, 23.0), xytext=(0, 6),
                   textcoords="offset points", fontsize=9.5, color=BASELINE,
                   ha="center")

    for ax, ylabel, title in (
        (left, "Cards to foundation (0–52)",
         "What the policy scores"),
        (right, "Steps revisiting a seen position (%)",
         "How often it cycles"),
    ):
        ax.set_xlabel("Exploration rate ε used at evaluation time")
        ax.set_ylabel(ylabel)
        ax.set_title(title, color=TEXT_PRIMARY, loc="left", fontsize=11.5)
        ax.set_xticks(epsilons)
        ax.set_xticklabels(["0\n(greedy)", "0.05", "0.20"])
        ax.set_xlim(-0.012, 0.212)
    left.set_ylim(0, 24)
    right.set_ylim(0, 92)

    # No right-edge direct labels here: the panels sit side by side, so a
    # label at x = 0.20 would collide with the second panel's axis. The legend
    # carries identity instead.
    left.legend(loc="upper left", ncol=2)

    fig.suptitle("Klondike: the same trained weights, three action-selection "
                 "rules", fontsize=13, weight="bold", color=TEXT_PRIMARY,
                 x=0.045, ha="left", y=1.04)
    caption(left,
            "Nothing is retrained. The identical checkpoints are replayed over "
            "the identical 200 TEST deals; only the rule that turns the network "
            "output into an action changes.\n"
            "The greedy column (ε = 0) is the protocol the thesis reports.", 66)
    fig.subplots_adjust(wspace=0.30)
    emit(fig, "action_rule_klondike")

    csv_rows = []
    for agent in AGENTS:
        for r in rows.get(agent, []):
            csv_rows.append([LABEL[agent], r["epsilon"], r["cards_up"],
                             r["cards_up_sd"], r["win_rate"],
                             r["repeated_position_frac"]])
    ppo = load_json("ppo_argmax_vs_sampled.json") or {}
    for key, label in (("klondike_ppo_argmax", "PPO (argmax over its policy)"),
                       ("klondike_ppo_sampled", "PPO (sampling its policy)")):
        if key in ppo:
            row = ppo[key]
            csv_rows.append([label, "n/a", row.get("cards_up"), "",
                             row["win_rate"], row["repeated_position_frac"]])
    write_csv("action_rule_klondike.csv",
              ["agent", "eval_epsilon", "cards_up", "cards_up_sd", "win_rate",
               "repeated_position_frac"], csv_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
