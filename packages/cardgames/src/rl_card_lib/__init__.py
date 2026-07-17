"""Package entry for `rl_card_lib` when installed from `packages/cardgames`."""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

from rl_card_lib.cardgames import Card, Suit, Rank, Deck, Player, CardGame

__all__ = [
    "Card",
    "Suit",
    "Rank",
    "Deck",
    "Player",
    "CardGame",
]
