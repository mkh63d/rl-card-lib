"""Player representation."""

from typing import Optional
from rl_card_lib.cardgames.card import Card


class Player:
    def __init__(
        self,
        player_id: int,
        name: Optional[str] = None,
        is_agent: bool = False
    ):
        self.player_id = player_id
        self.name = name or f"Player {player_id}"
        self.is_agent = is_agent
        self.hand: list[Card] = []
        self.score: int = 0

    def __str__(self) -> str:
        return f"{self.name} (hand: {len(self.hand)} cards, score: {self.score})"

    def __repr__(self) -> str:
        return f"Player(id={self.player_id}, name={self.name}, hand_size={len(self.hand)})"

    def add_cards(self, cards: list[Card]) -> "Player":
        self.hand.extend(cards)
        return self

    def add_card(self, card: Card) -> "Player":
        self.hand.append(card)
        return self

    def remove_card(self, card: Card) -> Card:
        if card not in self.hand:
            raise ValueError(f"Card {card} not in hand")
        self.hand.remove(card)
        return card

    def play_card(self, index: int) -> Card:
        if index < 0 or index >= len(self.hand):
            raise IndexError(f"Card index {index} out of range")
        return self.hand.pop(index)

    def has_card(self, card: Card) -> bool:
        return card in self.hand

    def hand_size(self) -> int:
        return len(self.hand)

    def clear_hand(self) -> list[Card]:
        cards = self.hand.copy()
        self.hand.clear()
        return cards

    def sort_hand(self, by_suit: bool = True) -> "Player":
        if by_suit:
            self.hand.sort(key=lambda c: (c.suit, c.rank))
        else:
            self.hand.sort(key=lambda c: (c.rank, c.suit))
        return self

    def add_score(self, points: int) -> "Player":
        self.score += points
        return self

    def reset_score(self) -> "Player":
        self.score = 0
        return self

    def encode_hand(self) -> list[float]:
        encoding = []
        for card in self.hand:
            encoding.extend(card.encode())
        return encoding

    def encode_hand_binary(self) -> list[float]:
        encoding = [0.0] * 52
        for card in self.hand:
            encoding[card.to_index()] = 1.0
        return encoding
