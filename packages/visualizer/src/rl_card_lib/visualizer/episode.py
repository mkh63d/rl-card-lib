"""Watching one episode: an agent, a game, a seed, step by step.

The library could already *measure* an episode -- `measure_agent_on_pool` plays
a whole pool of deals and reports solve rate and mean moves -- but nothing could
show one. The individual moves were thrown away, so a question as ordinary as
"what does the trained DQN actually do on a deal it fails?" had no answer short
of writing a throwaway loop.

Everything here is duck-typed against the env/agent contract rather than
imported from it. That is not squeamishness about typing: `examples` depends on
`visualizer`, so importing `rl_card_lib.games` or `rl_card_lib.harness` here
would close a dependency cycle. Working from the contract alone is also what
makes the replay generic -- a custom game that satisfies what `CardGameEnv`
expects of it replays without a line here knowing its name.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional

#: Step cap for an env that declares none. A replay has to stop somewhere, and
#: a game with reversible moves (Klondike's tableau, its draw/recycle cycle)
#: will otherwise cycle forever. Envs built through the sweep all carry their
#: own `max_steps`, so this is the floor for a hand-rolled one.
DEFAULT_MAX_STEPS = 1000


@dataclass(frozen=True)
class EpisodeStep:
    """One move: what the agent did, what it got, and what the board looked
    like afterwards.

    Attributes:
        index: 0-based position in the episode.
        action: The action index the agent chose.
        action_label: `env.action_to_string(action)` -- "Move waste to
            foundation 2" rather than "17".
        reward: Reward for this step alone.
        total_reward: Episode return through this step, so a renderer need not
            accumulate it itself.
        terminated: The game reached a terminal state.
        truncated: The step cap fired.
        invalid: The action was illegal, so the game was *not* stepped -- the
            env charges a step and the invalid-action reward anyway.
        repeated: The step landed in a position already seen this episode.
            Worth showing: it is the signature of a greedy policy cycling.
        winner: `info["winner"]`; 0 is the sole player of a single-player game,
            None when no one has won yet.
        board: `env.render()` after the step, held as raw Unicode. Empty when
            board capture was switched off or the game cannot render.
    """

    index: int
    action: int
    action_label: str
    reward: float
    total_reward: float
    terminated: bool
    truncated: bool
    invalid: bool
    repeated: bool
    winner: Optional[int]
    board: str


@dataclass
class EpisodeRecord:
    """A whole episode, replayable without the agent or the env.

    This is what makes the export backends possible: once the record exists,
    rendering it to a terminal or to an HTML page needs neither torch nor the
    game, so an episode can be written out and looked at later.
    """

    game: str
    seed: Optional[int]
    agent: str
    opening_board: str
    steps: list[EpisodeStep] = field(default_factory=list)
    winner: Optional[int] = None
    total_reward: float = 0.0
    terminated: bool = False
    truncated: bool = False

    @property
    def step_count(self) -> int:
        return len(self.steps)

    @property
    def solved(self) -> bool:
        """Whether the player won.

        Single-player games report their sole player as winner 0, which is what
        `measure_agent_on_pool` counts a solved deal by. A game still in play,
        or one nobody won, reports None.
        """
        return self.winner == 0

    @property
    def outcome(self) -> str:
        """One word for the ending, for a header line or a summary."""
        if self.solved:
            return "solved"
        if self.truncated and not self.terminated:
            return "truncated"
        if self.terminated:
            return "lost"
        return "unfinished"


@contextmanager
def _ansi_render(env) -> Iterator[None]:
    """Borrow the env's renderer without letting it print on its own.

    `CardGameEnv.step()` calls `self.render()` itself when `render_mode` is
    "human", so leaving that mode alone would print every board twice -- once
    unlabelled from inside the step, once from the renderer here. With
    `render_mode` None the other way, `render()` returns None and we would
    capture nothing at all. "ansi" is the one mode that hands the string back
    and does nothing else.

    The env is left exactly as it was found: the mode is restored when there
    was one, and the attribute removed again when there was not. Setting it
    back to None in that second case would look like a restore while actually
    leaving a `render_mode` on an object that never had one.
    """
    had_mode = hasattr(env, "render_mode")
    original = getattr(env, "render_mode", None)
    try:
        env.render_mode = "ansi"
    except AttributeError:
        # An env that will not take the attribute -- a read-only property, or
        # __slots__ without the field. Boards come back empty and the rest of
        # the replay still works, which beats refusing to run.
        yield
        return
    try:
        yield
    finally:
        try:
            if had_mode:
                env.render_mode = original
            else:
                delattr(env, "render_mode")
        except AttributeError:
            # Cleanup is best-effort by design: the episode is over and its
            # record is in the caller's hands, so raising here would throw a
            # finished result away over tidying up.
            pass


def _board(env) -> str:
    """The current board, or "" for an env that cannot draw one."""
    try:
        return env.render() or ""
    except Exception:
        # A game with no render() at all, or one that raises on a state it did
        # not expect. A missing board is a smaller loss than a lost replay.
        return ""


def _action_label(env, action: int) -> str:
    """A readable name for an action.

    `CardGameEnv.action_to_string` already falls back on its own, so this
    catches only an env that lacks the method entirely.
    """
    try:
        return env.action_to_string(action)
    except Exception:
        return f"Action {action}"


def _game_name(env) -> str:
    return type(getattr(env, "game", env)).__name__


def iter_episode(
    env,
    agent,
    *,
    seed: Optional[int] = None,
    max_steps: Optional[int] = None,
    capture_board: bool = True,
    on_reset: Optional[Callable[[str, dict], None]] = None,
) -> Iterator[EpisodeStep]:
    """Play one episode, yielding each move as it happens.

    A generator rather than a function returning a list, because the point of
    watching an agent play is that the moves arrive while it plays. MCTS
    spends seconds per move; a caller that had to wait for the whole episode
    before seeing anything would be staring at a blank terminal for a minute.
    `play_episode` drains this into a record for callers that want the
    artifact instead.

    The agent is put in eval mode for the duration and restored afterwards --
    measuring or watching an agent must not change it, the same bracket
    `measure_agent_on_pool` uses.

    Args:
        env: A Gymnasium-style env following the `CardGameEnv` contract.
        agent: Anything with `select_action(observation, legal_actions)`.
        seed: Deal seed. None leaves the env to pick, from its deal pool when
            it has one.
        max_steps: Step cap; the env's own `max_steps` when omitted.
        capture_board: Set False to skip `render()` per step -- a summary-only
            run has no use for the boards and they dominate the record's size.
        on_reset: Called once after the deal with `(opening_board, info)`, for
            a caller that wants to show the starting position.

    Yields:
        One `EpisodeStep` per move, ending with the step that terminated or
        truncated the episode.
    """
    was_training = getattr(agent, "training", False)
    if hasattr(agent, "eval"):
        agent.eval()

    try:
        with _ansi_render(env):
            observation, info = env.reset(seed=seed)
            if hasattr(agent, "reset"):
                agent.reset()
            if hasattr(agent, "bind"):
                # MCTS plans by copying the env it was handed; nothing else
                # in the agent zoo needs the binding.
                agent.bind(env)
            if on_reset is not None:
                on_reset(_board(env) if capture_board else "", info)

            cap = max_steps if max_steps is not None else getattr(env, "max_steps", None)
            if cap is None:
                cap = DEFAULT_MAX_STEPS

            total = 0.0
            for index in range(int(cap)):
                # The observation is handed over exactly as the env produced
                # it -- never coerced to an array. `MaskedCardGameEnv` yields a
                # dict of {observation, action_mask}, and an agent trained on
                # that shape needs it whole.
                # Normalised once, then used for every one of stepping,
                # labelling and recording. An external policy hands back what
                # its framework produced -- np.int64 out of an argmax, a 0-d
                # array off a tensor -- and casting only on the way into the
                # record is how the action that was played and the action that
                # was written down come to differ.
                action = int(agent.select_action(observation, info.get("legal_actions")))
                observation, reward, terminated, truncated, info = env.step(action)
                total += float(reward)

                yield EpisodeStep(
                    index=index,
                    action=action,
                    action_label=_action_label(env, action),
                    reward=float(reward),
                    total_reward=total,
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                    invalid=bool(info.get("invalid_action", False)),
                    repeated=bool(info.get("repeated_position", False)),
                    winner=info.get("winner"),
                    board=_board(env) if capture_board else "",
                )

                if terminated or truncated:
                    break
    finally:
        if was_training and hasattr(agent, "train"):
            agent.train()


def play_episode(
    env,
    agent,
    *,
    seed: Optional[int] = None,
    max_steps: Optional[int] = None,
    game: Optional[str] = None,
    agent_name: Optional[str] = None,
    capture_board: bool = True,
    on_step: Optional[Callable[[EpisodeStep], None]] = None,
    on_reset: Optional[Callable[[str, dict], None]] = None,
) -> EpisodeRecord:
    """Play one episode and return it as a record.

    `on_step` fires as each move is made, so a caller gets live output *and*
    the finished record from a single call rather than choosing between them.

    Args:
        env: A Gymnasium-style env following the `CardGameEnv` contract.
        agent: Anything with `select_action(observation, legal_actions)`.
        seed: Deal seed; see `iter_episode`.
        max_steps: Step cap; the env's own `max_steps` when omitted.
        game: Name for the record. Defaults to the wrapped game's class name.
        agent_name: Name for the record. Defaults to `agent.name`.
        capture_board: Set False to record no boards.
        on_step: Called with each `EpisodeStep` as it is produced.
        on_reset: Called once after the deal with `(opening_board, info)`.

    Returns:
        The finished `EpisodeRecord`.
    """
    opening: dict[str, Any] = {"board": ""}

    def _capture(board: str, info: dict) -> None:
        opening["board"] = board
        if on_reset is not None:
            on_reset(board, info)

    steps: list[EpisodeStep] = []
    for step in iter_episode(
        env,
        agent,
        seed=seed,
        max_steps=max_steps,
        capture_board=capture_board,
        on_reset=_capture,
    ):
        steps.append(step)
        if on_step is not None:
            on_step(step)

    last = steps[-1] if steps else None
    return EpisodeRecord(
        game=game or _game_name(env),
        # An unseeded reset on an env with a deal pool still dealt a specific
        # seed, and `last_deal_seed` is the only record of which -- without it
        # the replay could not be reproduced.
        seed=seed if seed is not None else getattr(env, "last_deal_seed", None),
        agent=agent_name or getattr(agent, "name", type(agent).__name__),
        opening_board=opening["board"],
        steps=steps,
        winner=last.winner if last is not None else None,
        total_reward=last.total_reward if last is not None else 0.0,
        terminated=last.terminated if last is not None else False,
        truncated=last.truncated if last is not None else False,
    )
