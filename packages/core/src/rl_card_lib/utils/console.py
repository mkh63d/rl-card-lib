"""Printing card text to a console that may not be able to encode it.

`Suit.symbol` returns U+2660..U+2666, so every card string carries glyphs a
legacy code page cannot represent -- cp1250, the default on a Polish Windows,
is the case that motivated this (#47). `print` encodes the whole string before
writing any of it, so one glyph loses the entire frame, and `CardGameEnv.render`
is called from inside `step()`, so the failure takes down a running episode
rather than a finished script's last line.

This sits a level below the fix for #43: the scripts there reconfigure their own
stdout, which is an entry-point action a library cannot take on its consumer's
behalf.
"""

import sys
from typing import Optional, TextIO

from rl_card_lib.cardgames.card import Suit

# Built from Suit itself so the fallback cannot drift from the glyphs it
# replaces. Rank.symbol is already ASCII, so only the suits need a mapping.
_ASCII_SUITS = {ord(suit.symbol): suit.ascii_symbol for suit in Suit}


def console_safe(text: str, stream: Optional[TextIO] = None) -> str:
    """Return `text` in a form `stream`'s encoding can represent.

    Unchanged on a UTF-8 stream -- the glyphs still print as glyphs -- and
    unchanged on a stream reporting no encoding: `io.StringIO`, what
    `contextlib.redirect_stdout` and pytest's capture install, holds `str` and
    reports `encoding` as `None`, so it needs no downgrade. Where the encoding
    cannot take the suit glyphs they become `C`/`D`/`H`/`S`, which stays
    readable where `?` would not.

    Args:
        text: Rendered text, typically from `game.render()`.
        stream: Stream the text is headed for; `sys.stdout` when omitted, and
            read at call time because stdout can be redirected long after the
            env was constructed.

    Returns:
        The text, transliterated only as far as the stream requires.
    """
    if stream is None:
        stream = sys.stdout
    encoding = getattr(stream, "encoding", None)
    if not encoding:
        return text
    try:
        text.encode(encoding)
        return text
    except UnicodeEncodeError:
        pass
    except LookupError:
        # An encoding name Python does not know: nothing to reason about, and
        # guessing would be worse than leaving the text alone.
        return text
    downgraded = text.translate(_ASCII_SUITS)
    try:
        downgraded.encode(encoding)
        return downgraded
    except UnicodeEncodeError:
        # The floor, not the mechanism: only a game rendering non-ASCII of its
        # own reaches here, and a replacement char beats raising mid-episode.
        return downgraded.encode(encoding, errors="replace").decode(
            encoding, errors="replace"
        )
