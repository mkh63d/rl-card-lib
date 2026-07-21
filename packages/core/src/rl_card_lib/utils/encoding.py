"""Encoding utilities for neural network inputs."""

import numpy as np

from rl_card_lib.cardgames.card import Card


def one_hot_encode(value: int, size: int) -> np.ndarray:
    """
    Create a one-hot encoded vector.
    
    Args:
        value: Index to encode
        size: Size of the output vector
        
    Returns:
        One-hot encoded numpy array
    """
    encoding = np.zeros(size, dtype=np.float32)
    if 0 <= value < size:
        encoding[value] = 1.0
    return encoding


def binary_encode_cards(cards: list[Card]) -> np.ndarray:
    """
    Encode a list of cards as a binary vector.
    
    Args:
        cards: List of cards to encode
        
    Returns:
        52-element binary vector
    """
    encoding = np.zeros(52, dtype=np.float32)
    for card in cards:
        encoding[card.to_index()] = 1.0
    return encoding


def encode_action_mask(
    legal_actions: list[int],
    action_size: int
) -> np.ndarray:
    """
    Create a binary mask for legal actions.
    
    Args:
        legal_actions: List of legal action indices
        action_size: Total number of possible actions
        
    Returns:
        Binary mask array
    """
    mask = np.zeros(action_size, dtype=np.float32)
    for action in legal_actions:
        if 0 <= action < action_size:
            mask[action] = 1.0
    return mask


def encode_card_features(card: Card) -> np.ndarray:
    """
    Encode a card as a feature vector.
    
    Returns:
        17-element feature vector:
        - Suit (4 one-hot)
        - Rank (13 one-hot)
    """
    features = np.zeros(17, dtype=np.float32)
    features[int(card.suit)] = 1.0
    features[4 + int(card.rank) - 1] = 1.0
    return features


def encode_hand_sorted(
    hand: list[Card],
    max_hand_size: int = 13
) -> np.ndarray:
    """
    Encode a hand with cards sorted and padded.
    
    Args:
        hand: List of cards in hand
        max_hand_size: Maximum hand size for padding
        
    Returns:
        Flattened feature array
    """
    # Sort by suit then rank
    sorted_hand = sorted(hand, key=lambda c: (c.suit, c.rank))
    
    features = []
    for i in range(max_hand_size):
        if i < len(sorted_hand):
            features.extend(encode_card_features(sorted_hand[i]))
        else:
            features.extend(np.zeros(17, dtype=np.float32))
    
    return np.array(features, dtype=np.float32)


def normalize_value(
    value: float,
    min_val: float,
    max_val: float
) -> float:
    """
    Normalize a value to [0, 1] range.
    
    Args:
        value: Value to normalize
        min_val: Minimum expected value
        max_val: Maximum expected value
        
    Returns:
        Normalized value
    """
    if max_val == min_val:
        return 0.5
    return (value - min_val) / (max_val - min_val)
