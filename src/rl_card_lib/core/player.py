"""Player representation."""

from typing import Optional
from rl_card_lib.core.card import Card


class Player:
    """
    Represents a player in a card game.
    
    Attributes:
        player_id: Unique identifier for the player
        name: Display name of the player
        hand: Cards currently held by the player
        score: Current score/points
    """
    
    def __init__(
        self,
        player_id: int,
        name: Optional[str] = None,
        is_agent: bool = False
    ):
        """
        Initialize a player.
        
        Args:
            player_id: Unique identifier
            name: Optional display name (defaults to "Player {id}")
            is_agent: Whether this player is controlled by an RL agent
        """
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
        """
        Add cards to the player's hand.
        
        Args:
            cards: Cards to add
            
        Returns:
            Self for method chaining
        """
        self.hand.extend(cards)
        return self
    
    def add_card(self, card: Card) -> "Player":
        """
        Add a single card to the player's hand.
        
        Args:
            card: Card to add
            
        Returns:
            Self for method chaining
        """
        self.hand.append(card)
        return self
    
    def remove_card(self, card: Card) -> Card:
        """
        Remove a specific card from the player's hand.
        
        Args:
            card: Card to remove
            
        Returns:
            The removed card
            
        Raises:
            ValueError: If card is not in hand
        """
        if card not in self.hand:
            raise ValueError(f"Card {card} not in hand")
        self.hand.remove(card)
        return card
    
    def play_card(self, index: int) -> Card:
        """
        Play a card from a specific position in hand.
        
        Args:
            index: Index of the card to play
            
        Returns:
            The played card
            
        Raises:
            IndexError: If index is out of range
        """
        if index < 0 or index >= len(self.hand):
            raise IndexError(f"Card index {index} out of range")
        return self.hand.pop(index)
    
    def has_card(self, card: Card) -> bool:
        return card in self.hand
    
    def hand_size(self) -> int:
        return len(self.hand)
    
    def clear_hand(self) -> list[Card]:
        """
        Remove all cards from hand.
        
        Returns:
            List of removed cards
        """
        cards = self.hand.copy()
        self.hand.clear()
        return cards
    
    def sort_hand(self, by_suit: bool = True) -> "Player":
        """
        Sort the hand.
        
        Args:
            by_suit: If True, sort by suit then rank. If False, sort by rank then suit.
            
        Returns:
            Self for method chaining
        """
        if by_suit:
            self.hand.sort(key=lambda c: (c.suit, c.rank))
        else:
            self.hand.sort(key=lambda c: (c.rank, c.suit))
        return self
    
    def add_score(self, points: int) -> "Player":
        """
        Add points to the player's score.
        
        Args:
            points: Points to add (can be negative)
            
        Returns:
            Self for method chaining
        """
        self.score += points
        return self
    
    def reset_score(self) -> "Player":
        """Reset score to zero."""
        self.score = 0
        return self
    
    def encode_hand(self) -> list[float]:
        """
        Encode the hand for neural network input.
        
        Returns:
            Flattened list of card encodings
        """
        # Fixed size encoding: max 52 cards, each with 18 features
        # For variable hand sizes, pad with zeros
        encoding = []
        for card in self.hand:
            encoding.extend(card.encode())
        return encoding
    
    def encode_hand_binary(self) -> list[float]:
        """
        Encode hand as a binary vector (which cards are present).
        
        Returns:
            52-element vector where 1 indicates card is in hand
        """
        encoding = [0.0] * 52
        for card in self.hand:
            encoding[card.to_index()] = 1.0
        return encoding
