"""Examples package entry for rl_card_lib games and scripts."""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

from rl_card_lib.games import KlondikeSolitaire, Macao

__all__ = [
    "KlondikeSolitaire",
    "Macao",
]
