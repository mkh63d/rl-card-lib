"""Deck of cards management."""

import random
from typing import Optional
from rl_card_lib.core.card import Card, Suit, Rank


class Deck:
    """
    Represents a deck of playing cards.
    
    Supports standard 52-card deck and custom configurations.
    
    Attributes:
        cards: List of cards currently in the deck
    """
    
    def __init__(self, cards: Optional[list[Card]] = None):
        """
        Initialize a deck.
        
        Args:
            cards: Optional list of cards. If None, creates a standard 52-card deck.
        """
        if cards is not None:
            self.cards = cards.copy()
        else:
            self.cards = self._create_standard_deck()
    
    def _create_standard_deck(self) -> list[Card]:
        """Create a standard 52-card deck."""
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
    
    def shuffle(self, seed: Optional[int] = None) -> "Deck":
        """
        Shuffle the deck in place.
        
        Args:
            seed: Optional random seed for reproducibility
            
        Returns:
            Self for method chaining
        """
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.cards)
        return self
    
    def draw(self, count: int = 1, face_up: bool = True) -> list[Card]:
        """
        Draw cards from the top of the deck.
        
        Args:
            count: Number of cards to draw
            face_up: Whether drawn cards should be face up
            
        Returns:
            List of drawn cards
            
        Raises:
            ValueError: If trying to draw more cards than available
        """
        if count > len(self.cards):
            raise ValueError(f"Cannot draw {count} cards, only {len(self.cards)} available")
        
        drawn = []
        for _ in range(count):
            card = self.cards.pop()
            card.face_up = face_up
            drawn.append(card)
        return drawn
    
    def draw_one(self, face_up: bool = True) -> Card:
        """
        Draw a single card from the top of the deck.
        
        Args:
            face_up: Whether the card should be face up
            
        Returns:
            The drawn card
        """
        return self.draw(1, face_up)[0]
    
    def add_to_bottom(self, cards: list[Card]) -> "Deck":
        """
        Add cards to the bottom of the deck.
        
        Args:
            cards: Cards to add
            
        Returns:
            Self for method chaining
        """
        self.cards = cards + self.cards
        return self
    
    def add_to_top(self, cards: list[Card]) -> "Deck":
        """
        Add cards to the top of the deck.
        
        Args:
            cards: Cards to add
            
        Returns:
            Self for method chaining
        """
        self.cards.extend(cards)
        return self
    
    def peek(self, count: int = 1) -> list[Card]:
        """
        Look at the top cards without removing them.
        
        Args:
            count: Number of cards to peek at
            
        Returns:
            List of cards from the top
        """
        return self.cards[-count:] if count <= len(self.cards) else self.cards.copy()
    
    def is_empty(self) -> bool:
        return len(self.cards) == 0
    
    def reset(self) -> "Deck":
        """Reset deck to a fresh 52-card deck (unshuffled)."""
        self.cards = self._create_standard_deck()
        return self
    
    def copy(self) -> "Deck":
        """Create a copy of the deck."""
        return Deck(cards=[Card(c.suit, c.rank, c.face_up) for c in self.cards])
    
    def encode(self) -> list[float]:
        """
        Encode the deck state for neural network input.
        
        Returns:
            List of floats representing remaining cards as a binary vector
        """
        # 52-element vector: 1 if card is in deck, 0 otherwise
        encoding = [0.0] * 52
        for card in self.cards:
            encoding[card.to_index()] = 1.0
        return encoding
