"""Printing an episode to a terminal, as a log or one move at a time.

The formatting functions are pure and take a step or a record, so the same
output can be written to a file, captured in a test, or paged. Only
`print_episode` and `step_printer` touch a stream, and both go through
`console_safe`: the suit glyphs are U+2660..U+2666 and a Polish Windows console
runs cp1250, where an unguarded `print` loses the whole frame to a
`UnicodeEncodeError` (#47).

Boards are held in the record as raw Unicode and downgraded only here, at the
moment of writing. That keeps the record faithful for the HTML export, which is
UTF-8 and should show the real glyphs, while the terminal still gets something
it can encode.
"""

from __future__ import annotations

import sys
import time
from typing import Callable, Iterable, Optional, TextIO, Union

from rl_card_lib.utils.console import console_safe
from rl_card_lib.visualizer.episode import EpisodeRecord, EpisodeStep

_RULE = "-" * 60


def format_step(step: EpisodeStep, *, board: bool = True) -> str:
    """One move as a header line, optionally followed by the board.

    Args:
        step: The move to format.
        board: Include the board `step.board` holds. False leaves the header,
            which is what a scrollable log of a 300-step Klondike wants.

    Returns:
        The formatted text, without a trailing newline.
    """
    flags = []
    if step.invalid:
        # Worth calling out rather than hiding: the env charged a step and the
        # invalid-action reward without stepping the game, so the board below
        # is unchanged and the move did nothing.
        flags.append("illegal")
    if step.repeated:
        flags.append("repeat")
    if step.terminated:
        flags.append("terminal")
    if step.truncated:
        flags.append("step cap")
    suffix = f"   [{', '.join(flags)}]" if flags else ""

    head = (
        f"[{step.index:>4}] {step.action_label}"
        f"   (action {step.action})"
        f"   reward {step.reward:+.3f}"
        f"   total {step.total_reward:+.3f}{suffix}"
    )
    if board and step.board:
        return f"{head}\n{step.board}"
    return head


def episode_summary(record: EpisodeRecord) -> str:
    """The header block: who played what, on which deal, and how it ended."""
    return "\n".join([
        _RULE,
        f"game     {record.game}",
        f"seed     {record.seed}",
        f"agent    {record.agent}",
        f"steps    {record.step_count}",
        f"return   {record.total_reward:+.3f}",
        f"outcome  {record.outcome}",
        _RULE,
    ])


def format_episode(record: EpisodeRecord, *, board: bool = True) -> str:
    """A whole episode as one string: opening deal, every move, summary."""
    blocks = []
    if board and record.opening_board:
        blocks.append(record.opening_board)
    blocks.extend(format_step(step, board=board) for step in record.steps)
    blocks.append(episode_summary(record))
    return "\n".join(blocks)


def _write(stream: TextIO, text: str) -> None:
    stream.write(console_safe(text, stream) + "\n")
    # Flushed per block because the whole point of watching is seeing moves as
    # they are made; a block-buffered pipe would hold them all to the end.
    stream.flush()


def step_printer(
    *,
    stream: Optional[TextIO] = None,
    board: bool = True,
    delay: float = 0.0,
    pause: Optional[Callable[[], None]] = None,
) -> Callable[[EpisodeStep], None]:
    """A printer to hand to `play_episode`'s `on_step`.

    This is how a caller gets both halves at once -- moves printed as they are
    made, and the finished record for an export -- rather than choosing
    between a live stream and an artifact.

    Args:
        stream: Where to write; `sys.stdout` when omitted, read at call time
            so a redirect after construction is still honoured.
        board: Include the board under each move.
        delay: Seconds to wait after each move, for auto-play.
        pause: Called after each move. The step-by-step mode passes
            `lambda: input("")` -- `input()` stays in the entry point so that
            nothing in the library blocks on a console that may not have one.

    Returns:
        A callable taking one `EpisodeStep`.
    """
    def _print(step: EpisodeStep) -> None:
        _write(stream if stream is not None else sys.stdout, format_step(step, board=board))
        if delay > 0:
            time.sleep(delay)
        if pause is not None:
            pause()

    return _print


def print_episode(
    episode: Union[EpisodeRecord, Iterable[EpisodeStep]],
    *,
    stream: Optional[TextIO] = None,
    board: bool = True,
    delay: float = 0.0,
    pause: Optional[Callable[[], None]] = None,
) -> None:
    """Print an episode, finished or still being played.

    Handed an `EpisodeRecord` it prints the opening deal, the moves and the
    summary. Handed an iterator -- `iter_episode(env, agent, seed=...)` -- it
    prints each move as the agent makes it, which is the same code path and
    the reason the engine yields steps rather than returning a list.

    Args:
        episode: A finished record, or any iterable of `EpisodeStep`.
        stream: Where to write; `sys.stdout` when omitted.
        board: Include the board under each move.
        delay: Seconds to wait after each move, for auto-play.
        pause: Called after each move; see `step_printer`.
    """
    out = stream if stream is not None else sys.stdout
    record = episode if isinstance(episode, EpisodeRecord) else None
    steps: Iterable[EpisodeStep] = record.steps if record is not None else episode

    if record is not None and board and record.opening_board:
        _write(out, record.opening_board)

    printer = step_printer(stream=out, board=board, delay=delay, pause=pause)
    for step in steps:
        printer(step)

    if record is not None:
        _write(out, episode_summary(record))
