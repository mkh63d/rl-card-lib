"""How to *run* a game in the sweep, declared once per game.

The report's `register_game` (in rl-card-lib-report) says how a game is
*presented* -- its label, headline metric, bounds. That lives in `report`
because `report` depends only on `core`. Running a game additionally needs a
gymnasium env, a trainer choice and an evaluation protocol, which pull in the
harness -- so the *execution* spec lives here, in the examples package.

`register_sweep_game` records the execution spec and forwards the presentation
fields to `report.register_game`, so a game author makes one call and gets both.
The two bundled games register themselves through exactly this API (see
`rl_card_lib.games.registration`) -- they are the worked examples, not special
cases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from rl_card_lib.harness.learners import DEFAULT_TARGET_UPDATE_FREQ
from rl_card_lib.report import register_game

# Execution fields belong to SweepGame; everything else is forwarded to the
# report's presentation registry.
_PRESENTATION_KEYS = (
    "label", "headline_key", "headline_label", "headline_unit",
    "headline_max", "headline_format", "higher_is_better",
    "episode_curves", "opponents", "secondary",
)


@dataclass
class SweepGame:
    """Everything the sweep needs to train and evaluate one game."""

    name: str
    env_factory: Callable[[], Any]
    max_steps: int
    evaluate: Callable[[Any, int, int], dict]
    # Environment for the trainer's periodic in-training evaluation. Declaring
    # one that deals from the held-out pool is what keeps the report's
    # evaluation curve a test-set curve; without it the trainer evaluates on
    # `env_factory()`, i.e. on training deals.
    eval_env_factory: Optional[Callable[[], Any]] = None
    self_play: bool = False
    opponent_factory: Optional[Callable[[int], Any]] = None
    heuristic_factory: Optional[Callable[[int], Any]] = None
    mcts_simulations: int = 20
    mcts_rollout_depth: int = 15
    # Gradient steps between target-network refreshes for the DQN family. It is
    # per-game because it is effectively counted in environment steps, so the
    # same number buys a different number of refreshes per episode in a game
    # with shorter episodes: 500 is a refresh every 1.7 episodes on 300-step
    # Klondike but every 10.9 on 46-step Macao. Declare a smaller value for a
    # short-episode game to keep the per-episode cadence comparable.
    target_update_freq: int = DEFAULT_TARGET_UPDATE_FREQ
    # Per-episode custom series, read after each episode while the game still
    # holds its terminal state. Klondike returns {"cards_up": ...}.
    episode_extras: Optional[Callable[[Any, Any], dict]] = None
    # Single-player solvability oracle: given a freshly dealt game it returns
    # True (winnable), False (proven unwinnable) or None (undecided within its
    # budget), mirroring solve_klondike. Only single-player games can supply one
    # -- an adversarial game has no perfect-information solve -- so the
    # solve-time benchmark runs only for games that declare both. Left None on
    # multiplayer games (e.g. Macao), which the benchmark then skips.
    solver: Optional[Callable[[Any], Optional[bool]]] = None
    single_player: bool = False
    presentation: dict = field(default_factory=dict)


_SWEEP_GAMES: dict[str, SweepGame] = {}


def register_sweep_game(
    name: str,
    *,
    env_factory: Callable[[], Any],
    max_steps: int,
    evaluate: Callable[[Any, int, int], dict],
    eval_env_factory: Optional[Callable[[], Any]] = None,
    self_play: bool = False,
    opponent_factory: Optional[Callable[[int], Any]] = None,
    heuristic_factory: Optional[Callable[[int], Any]] = None,
    mcts_simulations: int = 20,
    mcts_rollout_depth: int = 15,
    target_update_freq: int = DEFAULT_TARGET_UPDATE_FREQ,
    episode_extras: Optional[Callable[[Any, Any], dict]] = None,
    solver: Optional[Callable[[Any], Optional[bool]]] = None,
    single_player: bool = False,
    **presentation: Any,
) -> SweepGame:
    """Register a game for the training sweep and the report.

    Execution arguments are keyword-only and documented on `SweepGame`. Any
    other keyword is a presentation field forwarded to `report.register_game`
    (label, headline_key, headline_label, headline_max, higher_is_better,
    episode_curves, ...). The trainer class is derived from `self_play` unless
    given explicitly.

    Example::

        register_sweep_game(
            "hearts",
            env_factory=lambda: CardGameEnv(Hearts(), max_steps=200),
            max_steps=200,
            evaluate=evaluate_hearts,
            heuristic_factory=lambda seed: HeartsHeuristicAgent(seed=seed),
            label="Hearts",
            headline_key="penalty_points",
            headline_label="Penalty points",
            headline_max=26,
            higher_is_better=False,
            episode_curves=["penalty_points"],
        )
    """
    game = SweepGame(
        name=name, env_factory=env_factory, max_steps=max_steps,
        evaluate=evaluate, eval_env_factory=eval_env_factory,
        self_play=self_play,
        opponent_factory=opponent_factory, heuristic_factory=heuristic_factory,
        mcts_simulations=mcts_simulations, mcts_rollout_depth=mcts_rollout_depth,
        target_update_freq=target_update_freq,
        episode_extras=episode_extras,
        solver=solver, single_player=single_player,
        presentation={k: presentation[k] for k in _PRESENTATION_KEYS
                      if k in presentation},
    )
    _SWEEP_GAMES[name] = game

    forwarded = dict(game.presentation)
    forwarded.setdefault("trainer", "SelfPlayTrainer" if self_play else "Trainer")
    register_game(name, **forwarded)
    return game


def sweep_game(name: str) -> SweepGame:
    """Look up a registered sweep game, with a message listing the known ones."""
    if name not in _SWEEP_GAMES:
        known = ", ".join(sorted(_SWEEP_GAMES)) or "none registered"
        raise KeyError(f"No sweep game {name!r}. Registered: {known}")
    return _SWEEP_GAMES[name]


def registered_sweep_games() -> list[str]:
    """Names of every registered sweep game, in registration order."""
    return list(_SWEEP_GAMES)


def is_registered(name: str) -> bool:
    return name in _SWEEP_GAMES


__all__ = [
    "SweepGame",
    "register_sweep_game",
    "sweep_game",
    "registered_sweep_games",
    "is_registered",
]
