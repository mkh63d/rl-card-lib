"""Card representation with suits and ranks."""

from enum import IntEnum
from dataclasses import dataclass


class Suit(IntEnum):
    """Card suits enumeration."""
    CLUBS = 0
    DIAMONDS = 1
    HEARTS = 2
    SPADES = 3
    
    @property
    def symbol(self) -> str:
        """Return Unicode symbol for the suit."""
        symbols = {
            Suit.CLUBS: "♣",
            Suit.DIAMONDS: "♦",
            Suit.HEARTS: "♥",
            Suit.SPADES: "♠",
        }
        return symbols[self]
    
    @property
    def color(self) -> str:
        """Return color of the suit (red or black)."""
        if self in (Suit.DIAMONDS, Suit.HEARTS):
            return "red"
        return "black"


class Rank(IntEnum):
    """Card ranks enumeration."""
    ACE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    
    @property
    def symbol(self) -> str:
        """Return string representation of the rank."""
        symbols = {
            Rank.ACE: "A",
            Rank.TWO: "2",
            Rank.THREE: "3",
            Rank.FOUR: "4",
            Rank.FIVE: "5",
            Rank.SIX: "6",
            Rank.SEVEN: "7",
            Rank.EIGHT: "8",
            Rank.NINE: "9",
            Rank.TEN: "10",
            Rank.JACK: "J",
            Rank.QUEEN: "Q",
            Rank.KING: "K",
        }
        return symbols[self]


@dataclass
class Card:
    """
    Represents a playing card.
    
    Attributes:
        suit: The suit of the card (Clubs, Diamonds, Hearts, Spades)
        rank: The rank of the card (Ace through King)
        face_up: Whether the card is face up (visible) or face down
    """
    suit: Suit
    rank: Rank
    face_up: bool = True
    
    def __str__(self) -> str:
        if not self.face_up:
            return "[??]"
        return f"[{self.rank.symbol}{self.suit.symbol}]"
    
    def __repr__(self) -> str:
        return f"Card({self.suit.name}, {self.rank.name}, face_up={self.face_up})"
    
    def __hash__(self) -> int:
        return hash((self.suit, self.rank))
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Card):
            return NotImplemented
        return self.suit == other.suit and self.rank == other.rank
    
    @property
    def color(self) -> str:
        """Return the color of the card based on its suit."""
        return self.suit.color
    
    def flip(self) -> "Card":
        """Flip the card (toggle face_up)."""
        self.face_up = not self.face_up
        return self
    
    def to_index(self) -> int:
        """
        Convert card to a unique index (0-51).
        
        Returns:
            Integer index representing the card
        """
        return int(self.suit) * 13 + (int(self.rank) - 1)
    
    @classmethod
    def from_index(cls, index: int, face_up: bool = True) -> "Card":
        """
        Create a card from its index.
        
        Args:
            index: Integer index (0-51)
            face_up: Whether the card should be face up
            
        Returns:
            Card instance
        """
        suit = Suit(index // 13)
        rank = Rank((index % 13) + 1)
        return cls(suit=suit, rank=rank, face_up=face_up)
    
    def encode(self) -> list[float]:
        """
        Encode card as a feature vector for neural network input.
        
        Returns:
            List of floats representing the card:
            - 4 values for suit (one-hot)
            - 13 values for rank (one-hot)
            - 1 value for face_up status
        """
        encoding = [0.0] * 18  # 4 suits + 13 ranks + 1 visibility
        encoding[int(self.suit)] = 1.0
        encoding[4 + int(self.rank) - 1] = 1.0
        encoding[17] = 1.0 if self.face_up else 0.0
        return encoding
