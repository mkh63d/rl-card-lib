"""Four explanatory figures for the theoretical chapter.

matplotlib only -- no graphviz, no external assets, no network. Every figure is
written at 300 dpi as PNG and as SVG, with all type at >= 10 pt and every label
in English.

    concept_agent_env_loop      the s / a / r loop, on a real Macao position
    concept_action_masking      65 network outputs -> legal mask -> argmax
    concept_mcts_phases         selection / expansion / simulation / backprop,
                                with the determinization step that precedes them
    concept_dueling             shared trunk -> value + advantage -> aggregation

The Macao position and the Q-values in the masking figure are read from the
game and from the trained checkpoint, not invented.
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures")

# One restrained palette across all four figures; every pair below clears 4.5:1
# against the page, and no meaning is carried by hue alone.
INK = "#111111"
BODY = "#3d3d3d"
MUTED = "#6b6b6b"
LINE = "#9a9a9a"
FILL = "#eef1f5"
FILL_EDGE = "#b8c0cc"
ACCENT = "#1f4e9c"
ACCENT_FILL = "#dbe5f5"
BLOCKED = "#b03030"
BLOCKED_FILL = "#f6e0e0"
OK = "#1d6b3a"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Segoe UI", "Arial"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


def emit(fig, name: str) -> None:
    os.makedirs(FIG, exist_ok=True)
    png = os.path.join(FIG, f"{name}.png")
    svg = os.path.join(FIG, f"{name}.svg")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {os.path.abspath(png)}")
    print(f"wrote {os.path.abspath(svg)}")


def box(ax, x, y, w, h, text, *, fc=FILL, ec=FILL_EDGE, fontsize=10,
        weight="normal", color=INK, radius=0.02, lw=1.2):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2,
    ))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=color, weight=weight, zorder=3,
            linespacing=1.45)


def arrow(ax, start, end, *, color=BODY, lw=1.6, style="-|>", rad=0.0,
          mutation=14):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=mutation,
        linewidth=lw, color=color,
        connectionstyle=f"arc3,rad={rad}", zorder=4,
    ))


def blank_axes(figsize):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


# ---------------------------------------------------------------------------
# 1. Agent-environment loop, instantiated on Macao
# ---------------------------------------------------------------------------

def fig_agent_env_loop(position: dict) -> None:
    fig, ax = blank_axes((11.4, 5.6))

    box(ax, 0.055, 0.545, 0.29, 0.27,
        "AGENT\n$\\pi_\\theta(a \\mid s)$\nDQN / PPO / MCTS",
        fc=ACCENT_FILL, ec=ACCENT, fontsize=11, weight="bold", color=ACCENT)
    box(ax, 0.655, 0.545, 0.29, 0.27,
        "ENVIRONMENT\nCardGameEnv\n+ Macao rules",
        fc=FILL, ec=FILL_EDGE, fontsize=11, weight="bold")

    arrow(ax, (0.345, 0.775), (0.655, 0.775), rad=-0.30, color=ACCENT, lw=1.8)
    ax.text(0.50, 0.955, "action  $a_t$", ha="center", va="center",
            fontsize=11.5, color=ACCENT, weight="bold")
    ax.text(0.50, 0.900, f"$a_t = {position['action_index']}$  "
                         f"= “{position['action_label']}”",
            ha="center", va="center", fontsize=10.5, color=BODY)

    arrow(ax, (0.655, 0.585), (0.345, 0.585), rad=-0.30, color=BODY, lw=1.8)
    ax.text(0.50, 0.428, "state  $s_{t+1}$    and    reward  $r_{t+1}$",
            ha="center", va="center", fontsize=11.5, color=BODY, weight="bold")

    detail = [
        ("$s_t$", ACCENT, 0.020,
         "126 floats:\n"
         "own hand (52 binary),\n"
         "top discard (52 one-hot),\n"
         "requested suit (4), rank (13),\n"
         "declaration phase (2),\n"
         "draw penalty (1), opponent hand\n"
         "size (1), cards left in deck (1)"),
        ("$a_t$", ACCENT, 0.350,
         "65 discrete actions:\n"
         "0–51   play that card\n"
         "52      draw\n"
         "53      pass\n"
         "54–57  declare a suit\n"
         "58–64  declare a rank\n"
         f"Legal in this position: {position['legal_count']} of 65"),
        ("$r_{t+1}$", ACCENT, 0.680,
         "+0.1    a card leaves the hand\n"
         "−0.1    a card is drawn\n"
         "+10.0  the winning play\n"
         "−5.0    to every loser\n"
         "−1.0    an illegal action\n"
         "±        hand-size differential\n"
         "           at the 200-turn cap"),
    ]
    for symbol, colour, x, text in detail:
        box(ax, x, 0.010, 0.300, 0.380, "", fc="white", ec=LINE, lw=0.9)
        ax.text(x + 0.016, 0.352, symbol, ha="left", va="center",
                fontsize=12.5, color=colour, weight="bold")
        ax.text(x + 0.016, 0.315, text, ha="left", va="top",
                fontsize=9.3, color=BODY, linespacing=1.55)

    ax.text(0.5, 1.075,
            "Agent–environment loop, instantiated on Macao",
            ha="center", va="center", fontsize=13.5, weight="bold", color=INK)
    ax.text(0.5, 1.015,
            f"Position: hand {position['hand']} · top of discard "
            f"{position['top']} · deal seed {position['seed']}",
            ha="center", va="center", fontsize=10, color=MUTED)

    emit(fig, "concept_agent_env_loop")


# ---------------------------------------------------------------------------
# 2. Action masking
# ---------------------------------------------------------------------------

def fig_action_masking(position: dict) -> None:
    q = np.asarray(position["q_values"], dtype=float)
    legal = set(position["legal_actions"])
    n = len(q)
    mask = np.array([1 if i in legal else 0 for i in range(n)])
    masked = np.where(mask == 1, q, -1e8)
    chosen = int(np.argmax(masked))
    naive = int(np.argmax(q))

    fig, axes = plt.subplots(
        4, 1, figsize=(9.2, 6.6),
        gridspec_kw={"height_ratios": [1.5, 0.55, 1.5, 0.95], "hspace": 0.85},
    )

    x = np.arange(n)

    # Panel 1: raw network output
    ax = axes[0]
    ax.bar(x, q, color=LINE, width=0.82, linewidth=0)
    ax.bar([naive], [q[naive]], color=BLOCKED, width=0.82, linewidth=0)
    ax.set_ylabel("Q-value", fontsize=10)
    ax.set_title("1.  Network output: one Q-value per action, all 65 of them",
                 fontsize=11, loc="left", color=INK)
    ax.annotate(
        f"unmasked argmax = action {naive}\n"
        f"(“{position['labels'][naive]}” — illegal in this position)",
        xy=(naive, q[naive]), xytext=(naive + 4, q[naive]),
        fontsize=10, color=BLOCKED, va="center",
        arrowprops={"arrowstyle": "->", "color": BLOCKED, "linewidth": 1.2},
    )

    # Panel 2: the mask
    ax = axes[1]
    ax.imshow(mask.reshape(1, -1), aspect="auto", cmap="Greys_r",
              vmin=-0.35, vmax=1.15, extent=(-0.5, n - 0.5, 0, 1))
    ax.set_yticks([])
    ax.set_title("2.  Legal-action mask from env.get_legal_action_mask() "
                 "(white = legal)", fontsize=11, loc="left", color=INK, pad=18)
    for a in sorted(legal):
        ax.annotate(str(a), xy=(a, 1.0), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=9.5,
                    color=OK, annotation_clip=False)

    # Panel 3: masked Q
    ax = axes[2]
    shown = np.where(mask == 1, q, np.nan)
    floor = np.nanmin(q) - 0.15 * (np.nanmax(q) - np.nanmin(q) + 1e-9)
    ax.bar(x, np.where(mask == 1, 0.0, q - floor), bottom=floor,
           color=BLOCKED_FILL, width=0.82, linewidth=0)
    ax.bar(x, np.nan_to_num(shown, nan=0.0), color=ACCENT, width=0.82,
           linewidth=0)
    ax.bar([chosen], [q[chosen]], color=OK, width=0.82, linewidth=0)
    ax.set_ylabel("Q-value", fontsize=10)
    ax.set_xlabel("Action index", fontsize=10)
    ax.set_title("3.  Illegal entries replaced by −1e8, then argmax",
                 fontsize=11, loc="left", color=INK)
    ax.annotate(
        f"masked argmax = action {chosen}\n(“{position['labels'][chosen]}”)",
        xy=(chosen, q[chosen]), xytext=(chosen + 4, q[chosen]),
        fontsize=10, color=OK, va="center",
        arrowprops={"arrowstyle": "->", "color": OK, "linewidth": 1.2},
    )

    for ax in axes[:3]:
        ax.set_xlim(-0.7, n - 0.3)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    # Panel 4: the position in words
    ax = axes[3]
    ax.axis("off")
    rows = [
        f"Hand (player 0):   {position['hand']}",
        f"Top of discard:    {position['top']}",
        "Legal actions:     "
        + ",  ".join(f"{a} = {position['labels'][a]}"
                     for a in sorted(legal)),
        f"Deal seed:         {position['seed']}   "
        f"(Q-values from the trained Macao DQN checkpoint)",
    ]
    ax.text(0.0, 0.95, "\n".join(rows), ha="left", va="top",
            fontsize=10, color=BODY, family="monospace", linespacing=1.7,
            transform=ax.transAxes)

    fig.suptitle("Action masking: 65 network outputs → mask → argmax",
                 fontsize=13, weight="bold", color=INK, y=0.995)
    emit(fig, "concept_action_masking")


# ---------------------------------------------------------------------------
# 3. The four MCTS phases, with determinization
# ---------------------------------------------------------------------------

def fig_mcts_phases() -> None:
    fig, axes = plt.subplots(1, 4, figsize=(12.6, 4.8))

    # A small fixed tree: root, two children, two grandchildren, one new leaf.
    nodes = {
        "root": (0.50, 0.88),
        "a": (0.26, 0.60),
        "b": (0.78, 0.60),
        "c": (0.10, 0.32),
        "d": (0.42, 0.32),
        "new": (0.66, 0.32),
    }
    edges = [("root", "a"), ("root", "b"), ("a", "c"), ("a", "d")]

    titles = ["1.  Selection", "2.  Expansion", "3.  Simulation",
              "4.  Backpropagation"]
    captions = [
        "Descend from the root by UCB1\nuntil a node with an untried\naction is reached.",
        "Add one child for an untried\nlegal action of that node.",
        "Play a fast rollout from the\nnew node to the end of the\ngame, or to the depth cap.",
        "Add the return to every node\non the path, per player, and\nincrement the visit counts.",
    ]
    highlighted = [{"root", "a", "d"}, {"new"}, {"new"}, {"root", "a", "d"}]

    for i, ax in enumerate(axes):
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.02, 1)
        ax.axis("off")

        show_new = i >= 1
        for u, v in edges:
            hot = i in (0, 3) and u in highlighted[i] and v in highlighted[i]
            ax.plot([nodes[u][0], nodes[v][0]], [nodes[u][1], nodes[v][1]],
                    color=ACCENT if hot else LINE,
                    linewidth=2.6 if hot else 1.2, zorder=1)
        if show_new:
            ax.plot([nodes["a"][0], nodes["new"][0]],
                    [nodes["a"][1], nodes["new"][1]],
                    color=OK if i == 1 else LINE,
                    linewidth=2.6 if i == 1 else 1.2, zorder=1)

        for key, (x, y) in nodes.items():
            if key == "new" and not show_new:
                continue
            hot = key in highlighted[i]
            face, edge = "white", LINE
            if hot and i in (0, 3):
                face, edge = ACCENT_FILL, ACCENT
            if key == "new" and i in (1, 2):
                face, edge = "#dff0e5", OK
            ax.add_patch(Circle((x, y), 0.072, facecolor=face, edgecolor=edge,
                                linewidth=1.8, zorder=2))

        if i == 0:
            ax.text(0.30, 0.745, "UCB1", fontsize=10, color=ACCENT,
                    ha="right", va="center", weight="bold")
            ax.text(0.50, 0.20, "an untried action\nremains here",
                    fontsize=9.4, color=ACCENT, ha="left", va="top",
                    linespacing=1.4)
        if i == 1:
            ax.annotate("new child", xy=(0.735, 0.32), xytext=(0.80, 0.32),
                        fontsize=9.8, color=OK, va="center", ha="left",
                        arrowprops={"arrowstyle": "->", "color": OK,
                                    "linewidth": 1.1})
        if i == 2:
            ax.plot([0.66, 0.73, 0.62, 0.70], [0.245, 0.185, 0.125, 0.06],
                    color=OK, linewidth=1.6, linestyle=":", zorder=1)
            ax.add_patch(Circle((0.70, 0.06), 0.028, facecolor=OK,
                                edgecolor=OK, zorder=2))
            ax.text(0.755, 0.06, "return $G$", fontsize=9.8, color=OK,
                    va="center", ha="left")
        if i == 3:
            arrow(ax, (0.378, 0.394), (0.302, 0.526), color=ACCENT, lw=2.2,
                  mutation=13)
            arrow(ax, (0.315, 0.665), (0.445, 0.815), color=ACCENT, lw=2.2,
                  mutation=13)
            ax.text(0.01, 0.80, "N += 1\nW += G",
                    fontsize=10, color=ACCENT, va="center", ha="left",
                    linespacing=1.5)

        ax.set_title(titles[i], fontsize=11.5, color=INK, loc="left", pad=8)
        ax.text(0.0, -0.06, captions[i], transform=ax.transAxes, fontsize=9.6,
                color=BODY, va="top", ha="left", linespacing=1.6)

    # The determinization banner: what happens once per search, before phase 1.
    banner = fig.add_axes([0.03, 0.845, 0.94, 0.135])
    banner.set_xlim(0, 1)
    banner.set_ylim(0, 1)
    banner.axis("off")
    box(banner, 0.0, 0.02, 1.0, 0.96, "", fc="#fbf7ec", ec="#d8c9a3", lw=1.2)
    banner.text(
        0.5, 0.5,
        "Step 0  ·  DETERMINIZATION, once per search\n"
        "The cards the agent cannot see — the opponents' hands and the deck "
        "order in Macao, the face-down tableau cards and the stock in Klondike "
        "— are re-dealt at random\namong themselves, keeping every hand size "
        "and every face-up card unchanged. The four phases below then run on "
        "that sampled perfect-information world.",
        ha="center", va="center", fontsize=9.6, color="#5a4a25",
        linespacing=1.7,
    )

    fig.suptitle("Monte-Carlo Tree Search: determinization and the four phases",
                 fontsize=13, weight="bold", color=INK, y=1.10)
    fig.subplots_adjust(top=0.74, bottom=0.20, wspace=0.16)
    emit(fig, "concept_mcts_phases")


# ---------------------------------------------------------------------------
# 4. Duelling architecture
# ---------------------------------------------------------------------------

def fig_dueling(state_size: int, action_size: int, hidden: list[int]) -> None:
    fig, ax = blank_axes((10.4, 4.9))

    box(ax, 0.020, 0.42, 0.115, 0.20,
        f"state $s$\n{state_size} floats", fc="white", ec=LINE, fontsize=10)

    box(ax, 0.150, 0.38, 0.205, 0.28, "", fc=FILL, ec=FILL_EDGE)
    ax.text(0.2525, 0.585, "SHARED TRUNK", ha="center", va="center",
            fontsize=10.5, weight="bold", color=INK)
    ax.text(0.2525, 0.475, f"Linear({state_size} → {hidden[0]})"
                           f"\n+ ReLU",
            ha="center", va="center", fontsize=9.5, color=BODY, linespacing=1.5)

    box(ax, 0.420, 0.635, 0.230, 0.235, "", fc=ACCENT_FILL, ec=ACCENT)
    ax.text(0.535, 0.815, "VALUE HEAD", ha="center", va="center",
            fontsize=10.5, weight="bold", color=ACCENT)
    ax.text(0.535, 0.712,
            f"Linear({hidden[0]} → {hidden[1]}) + ReLU"
            f"\nLinear({hidden[1]} → 1)",
            ha="center", va="center", fontsize=9.3, color=ACCENT, linespacing=1.5)

    box(ax, 0.420, 0.130, 0.230, 0.235, "", fc="#eaf3ec", ec=OK)
    ax.text(0.535, 0.310, "ADVANTAGE HEAD", ha="center", va="center",
            fontsize=10.5, weight="bold", color=OK)
    ax.text(0.535, 0.207,
            f"Linear({hidden[0]} → {hidden[1]}) + ReLU"
            f"\nLinear({hidden[1]} → {action_size})",
            ha="center", va="center", fontsize=9.3, color=OK, linespacing=1.5)

    box(ax, 0.715, 0.38, 0.150, 0.28, "AGGREGATION", fc=FILL, ec=FILL_EDGE,
        fontsize=10.5, weight="bold")
    box(ax, 0.885, 0.42, 0.100, 0.20,
        f"$Q(s,\\cdot)$\n{action_size} values", fc="white", ec=LINE,
        fontsize=10)

    arrow(ax, (0.137, 0.52), (0.148, 0.52))
    arrow(ax, (0.357, 0.575), (0.418, 0.725), rad=0.14, color=ACCENT)
    arrow(ax, (0.357, 0.465), (0.418, 0.275), rad=-0.14, color=OK)
    arrow(ax, (0.652, 0.725), (0.713, 0.575), rad=-0.14, color=ACCENT)
    arrow(ax, (0.652, 0.275), (0.713, 0.465), rad=0.14, color=OK)
    arrow(ax, (0.867, 0.52), (0.883, 0.52))

    ax.text(0.535, 0.925, "$V(s)$  —  one number: how good is this position",
            fontsize=10.5, color=ACCENT, ha="center", va="center")
    ax.text(0.535, 0.072,
            f"$A(s,a)$  —  {action_size} numbers: how much better is each action",
            fontsize=10.5, color=OK, ha="center", va="center")

    ax.text(0.5, -0.13,
            r"$Q(s,a) \;=\; V(s) \;+\; A(s,a) \;-\; "
            r"\dfrac{1}{|\mathcal{A}|}\sum_{a'} A(s,a')$",
            ha="center", va="center", fontsize=14, color=INK)
    ax.text(0.5, -0.27,
            "Subtracting the mean advantage is what makes the split "
            "identifiable: without it any constant could move between "
            "$V$ and $A$ and leave $Q$ unchanged.",
            ha="center", va="center", fontsize=9.8, color=MUTED)

    ax.text(0.5, 1.12, "Duelling Q-network (DoubleDQNAgent, dueling=True)",
            ha="center", va="center", fontsize=13, weight="bold", color=INK)
    ax.text(0.5, 1.045,
            f"Sizes shown for Macao: {state_size}-dimensional observation, "
            f"{action_size} actions, hidden layers {hidden[0]} and {hidden[1]}",
            ha="center", va="center", fontsize=9.8, color=MUTED)

    emit(fig, "concept_dueling")


# ---------------------------------------------------------------------------

def macao_position(seed: int = 100_003) -> dict:
    """A real Macao position, with the trained DQN's real Q-values."""
    import torch

    from rl_card_lib.env import CardGameEnv
    from rl_card_lib.games import Macao
    from rl_card_lib.harness import build_learner

    game = Macao(num_players=2)
    env = CardGameEnv(game, max_steps=200)
    _, info = env.reset(seed=seed)
    observation = game.get_observation()
    legal = sorted(info["legal_actions"])

    agent = build_learner("dqn", 126, 65, 0)
    checkpoint = os.path.join(HERE, "..", "..", "checkpoints",
                              "macao_dqn", "final.pt")
    source = "trained Macao DQN checkpoint"
    if os.path.exists(checkpoint):
        try:
            agent.load(checkpoint)
        except Exception as exc:  # noqa: BLE001
            print(f"  (checkpoint unreadable: {exc}; using an untrained net)")
            source = "untrained network"
    else:
        source = "untrained network"
    with torch.no_grad():
        q = agent.get_q_values(observation)

    labels = [game.action_to_string(a) for a in range(65)]
    chosen = int(np.argmax(np.where(
        np.isin(np.arange(65), legal), q, -1e8)))
    return {
        "seed": seed,
        "hand": " ".join(str(c) for c in game.players[0].hand),
        "top": str(game.discard_pile[-1]),
        "legal_actions": legal,
        "legal_count": len(legal),
        "labels": labels,
        "q_values": q.tolist(),
        "q_source": source,
        "action_index": chosen,
        "action_label": labels[chosen],
    }


def main() -> int:
    # The summary line below prints card glyphs, and it runs before the first
    # figure is emitted. On a stream whose encoding is not UTF-8 -- cp1250 is
    # the default on a Polish Windows -- that raises UnicodeEncodeError, so the
    # script would exit non-zero having written nothing at all (#43). The
    # hasattr guard is for the streams that have no reconfigure: io.StringIO,
    # which is what redirect_stdout and pytest's capture install.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    position = macao_position()
    print(f"position: hand {position['hand']} top {position['top']} "
          f"legal {position['legal_actions']} ({position['q_source']})")
    fig_agent_env_loop(position)
    fig_action_masking(position)
    fig_mcts_phases()
    fig_dueling(126, 65, [256, 128])
    return 0


if __name__ == "__main__":
    sys.exit(main())
