"""Reusable rule helpers for card games."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence

from rl_card_lib.cardgames.card import Card, Suit, Rank


def is_same_suit(a: Card, b: Card) -> bool:
    return a.suit == b.suit


def is_same_color(a: Card, b: Card) -> bool:
    return a.color == b.color


def is_same_rank(a: Card, b: Card) -> bool:
    return a.rank == b.rank


def is_higher_than(a: Card, b: Card) -> bool:
    return a.rank > b.rank


def is_lower_than(a: Card, b: Card) -> bool:
    return a.rank < b.rank


def is_adjacent_rank(a: Card, b: Card) -> bool:
    return abs(int(a.rank) - int(b.rank)) == 1


def is_next_higher(a: Card, b: Card) -> bool:
    return int(a.rank) == int(b.rank) + 1


def is_next_lower(a: Card, b: Card) -> bool:
    return int(a.rank) + 1 == int(b.rank)


def is_alternating_color(a: Card, b: Card) -> bool:
    return a.color != b.color


def is_strictly_increasing(cards: Sequence[Card]) -> bool:
    if len(cards) < 2:
        return True
    return all(int(cards[i].rank) < int(cards[i + 1].rank) for i in range(len(cards) - 1))


def is_strictly_decreasing(cards: Sequence[Card]) -> bool:
    if len(cards) < 2:
        return True
    return all(int(cards[i].rank) > int(cards[i + 1].rank) for i in range(len(cards) - 1))


def is_consecutive_increasing(cards: Sequence[Card]) -> bool:
    if len(cards) < 2:
        return True
    return all(int(cards[i].rank) + 1 == int(cards[i + 1].rank) for i in range(len(cards) - 1))


def is_consecutive_decreasing(cards: Sequence[Card]) -> bool:
    if len(cards) < 2:
        return True
    return all(int(cards[i].rank) == int(cards[i + 1].rank) + 1 for i in range(len(cards) - 1))


def is_same_suit_sequence(cards: Sequence[Card]) -> bool:
    if len(cards) < 2:
        return True
    suit = cards[0].suit
    return all(card.suit == suit for card in cards)


def is_alternating_colors_sequence(cards: Sequence[Card]) -> bool:
    if len(cards) < 2:
        return True
    return all(cards[i].color != cards[i + 1].color for i in range(len(cards) - 1))


def count_by_color(cards: Iterable[Card]) -> dict[str, int]:
    counts: dict[str, int] = {"red": 0, "black": 0}
    for card in cards:
        counts[card.color] += 1
    return counts


def count_by_suit(cards: Iterable[Card]) -> dict[Suit, int]:
    counts: dict[Suit, int] = {suit: 0 for suit in Suit}
    for card in cards:
        counts[card.suit] += 1
    return counts


def count_by_rank(cards: Iterable[Card]) -> dict[Rank, int]:
    counts: dict[Rank, int] = {rank: 0 for rank in Rank}
    for card in cards:
        counts[card.rank] += 1
    return counts


def count_face_up(cards: Iterable[Card]) -> int:
    return sum(1 for card in cards if card.face_up)


def count_face_down(cards: Iterable[Card]) -> int:
    return sum(1 for card in cards if not card.face_up)


def filter_by_suit(cards: Iterable[Card], suit: Suit) -> list[Card]:
    return [card for card in cards if card.suit == suit]


def filter_by_color(cards: Iterable[Card], color: str) -> list[Card]:
    return [card for card in cards if card.color == color]


def has_card(cards: Iterable[Card], card: Card) -> bool:
    return any(card == current for current in cards)


def top_card(cards: Sequence[Card]) -> Card | None:
    return cards[-1] if cards else None


def bottom_card(cards: Sequence[Card]) -> Card | None:
    return cards[0] if cards else None


def is_standard_deck(cards: Iterable[Card]) -> bool:
    unique = {(card.suit, card.rank) for card in cards}
    return len(unique) == 52 and all((suit, rank) in unique for suit in Suit for rank in Rank)


def can_stack_alternating_descending(moving: Card, target: Card) -> bool:
    return is_alternating_color(moving, target) and is_next_lower(moving, target)


def can_stack_same_suit_ascending(moving: Card, target: Card) -> bool:
    return is_same_suit(moving, target) and is_next_higher(moving, target)


def multiset_counts(cards: Iterable[Card]) -> Counter[tuple[Suit, Rank]]:
    return Counter((card.suit, card.rank) for card in cards)
