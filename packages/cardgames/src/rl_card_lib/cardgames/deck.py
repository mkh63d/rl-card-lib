"""Deck of cards management."""

import random
from typing import Optional
from rl_card_lib.cardgames.card import Card, Suit, Rank


class Deck:
    def __init__(self, cards: Optional[list[Card]] = None):
        if cards is not None:
            self.cards = cards.copy()
        else:
            self.cards = self._create_standard_deck()

    def _create_standard_deck(self) -> list[Card]:
        return [
            Card(suit=suit, rank=rank, face_up=False)
            for suit in Suit
            for rank in Rank
        ]

    def __len__(self) -> int:
        return len(self.cards)

    def __iter__(self):
        return iter(self.cards)

    def __getitem__(self, index: int) -> Card:
        return self.cards[index]

    def __str__(self) -> str:
        return f"Deck({len(self.cards)} cards)"

    def shuffle(
        self,
        seed: Optional[int] = None,
        rng: Optional[random.Random] = None,
    ) -> "Deck":
        """
        Shuffle the deck in place.

        Args:
            seed: Seed for a one-off private `random.Random`; never touches the
                process-wide RNG, so other components' randomness is unaffected
            rng: Existing `random.Random` to draw from; wins over `seed`

        Returns:
            self, so calls can be chained
        """
        if rng is None:
            rng = random.Random(seed) if seed is not None else random
        rng.shuffle(self.cards)
        return self

    def draw(self, count: int = 1, face_up: bool = True) -> list[Card]:
        if count > len(self.cards):
            raise ValueError(f"Cannot draw {count} cards, only {len(self.cards)} available")
        drawn = []
        for _ in range(count):
            card = self.cards.pop()
            card.face_up = face_up
            drawn.append(card)
        return drawn

    def draw_one(self, face_up: bool = True) -> Card:
        return self.draw(1, face_up)[0]

    def add_to_bottom(self, cards: list[Card]) -> "Deck":
        self.cards = cards + self.cards
        return self

    def add_to_top(self, cards: list[Card]) -> "Deck":
        self.cards.extend(cards)
        return self

    def peek(self, count: int = 1) -> list[Card]:
        return self.cards[-count:] if count <= len(self.cards) else self.cards.copy()

    def is_empty(self) -> bool:
        return len(self.cards) == 0

    def reset(self) -> "Deck":
        self.cards = self._create_standard_deck()
        return self

    def copy(self) -> "Deck":
        return Deck(cards=[Card(c.suit, c.rank, c.face_up) for c in self.cards])

    def encode(self) -> list[float]:
        encoding = [0.0] * 52
        for card in self.cards:
            encoding[card.to_index()] = 1.0
        return encoding
