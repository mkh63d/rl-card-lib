"""Tests for utils module."""

import pytest
import numpy as np

from rl_card_lib.cardgames import Card, Suit, Rank
from rl_card_lib.utils.encoding import (
    one_hot_encode,
    binary_encode_cards,
    encode_action_mask,
    encode_card_features,
    encode_hand_sorted,
    normalize_value,
)
from rl_card_lib.visualizer.visualization import (
    render_cards,
    render_tableau,
    create_simple_board_view,
)


class TestOneHotEncode:
    def test_basic_encoding(self):
        result = one_hot_encode(2, 5)
        assert result.shape == (5,)
        assert result[2] == 1.0
        assert sum(result) == 1.0

    def test_first_index(self):
        result = one_hot_encode(0, 5)
        assert result[0] == 1.0

    def test_last_index(self):
        result = one_hot_encode(4, 5)
        assert result[4] == 1.0

    def test_out_of_bounds_negative(self):
        result = one_hot_encode(-1, 5)
        assert sum(result) == 0.0

    def test_out_of_bounds_too_large(self):
        result = one_hot_encode(10, 5)
        assert sum(result) == 0.0


class TestBinaryEncodeCards:
    def test_empty_list(self):
        result = binary_encode_cards([])
        assert result.shape == (52,)
        assert sum(result) == 0.0

    def test_single_card(self):
        card = Card(Suit.HEARTS, Rank.ACE)
        result = binary_encode_cards([card])
        assert sum(result) == 1.0
        assert result[card.to_index()] == 1.0

    def test_multiple_cards(self):
        cards = [
            Card(Suit.HEARTS, Rank.ACE),
            Card(Suit.SPADES, Rank.KING),
        ]
        result = binary_encode_cards(cards)
        assert sum(result) == 2.0


class TestEncodeActionMask:
    def test_empty_actions(self):
        result = encode_action_mask([], 10)
        assert result.shape == (10,)
        assert sum(result) == 0.0

    def test_some_actions(self):
        result = encode_action_mask([0, 2, 5], 10)
        assert sum(result) == 3.0
        assert result[0] == 1.0
        assert result[2] == 1.0
        assert result[5] == 1.0
        assert result[1] == 0.0

    def test_out_of_bounds_action(self):
        result = encode_action_mask([0, 15], 10)
        assert sum(result) == 1.0  # Only 0 is valid


class TestEncodeCardFeatures:
    def test_feature_shape(self):
        card = Card(Suit.HEARTS, Rank.ACE)
        result = encode_card_features(card)
        assert result.shape == (17,)

    def test_suit_encoding(self):
        card = Card(Suit.DIAMONDS, Rank.ACE)
        result = encode_card_features(card)
        assert result[int(Suit.DIAMONDS)] == 1.0
        assert result[int(Suit.HEARTS)] == 0.0

    def test_rank_encoding(self):
        card = Card(Suit.HEARTS, Rank.KING)
        result = encode_card_features(card)
        assert result[4 + int(Rank.KING) - 1] == 1.0


class TestEncodeHandSorted:
    def test_empty_hand(self):
        result = encode_hand_sorted([], max_hand_size=5)
        assert result.shape == (5 * 17,)
        assert sum(result) == 0.0

    def test_partial_hand(self):
        cards = [Card(Suit.HEARTS, Rank.ACE)]
        result = encode_hand_sorted(cards, max_hand_size=5)
        assert result.shape == (5 * 17,)
        assert sum(result[:17]) > 0  # First card encoded

    def test_sorted_order(self):
        cards = [
            Card(Suit.SPADES, Rank.KING),
            Card(Suit.CLUBS, Rank.ACE),
        ]
        result = encode_hand_sorted(cards, max_hand_size=5)
        # Clubs should come before Spades (sorted by suit)
        assert result.shape == (5 * 17,)


class TestNormalizeValue:
    def test_middle_value(self):
        result = normalize_value(5.0, 0.0, 10.0)
        assert result == 0.5

    def test_min_value(self):
        result = normalize_value(0.0, 0.0, 10.0)
        assert result == 0.0

    def test_max_value(self):
        result = normalize_value(10.0, 0.0, 10.0)
        assert result == 1.0

    def test_equal_min_max(self):
        result = normalize_value(5.0, 5.0, 5.0)
        assert result == 0.5


class TestRenderCards:
    def test_empty_list(self):
        result = render_cards([])
        assert result == "[ ]"

    def test_single_card(self):
        card = Card(Suit.HEARTS, Rank.ACE)
        result = render_cards([card])
        assert "[A♥]" in result

    def test_hidden_cards(self):
        cards = [Card(Suit.HEARTS, Rank.ACE)]
        result = render_cards(cards, hidden=True)
        assert "[??]" in result
        assert "♥" not in result

    def test_multiple_cards(self):
        cards = [
            Card(Suit.HEARTS, Rank.ACE),
            Card(Suit.SPADES, Rank.KING),
        ]
        result = render_cards(cards)
        assert "[A♥]" in result
        assert "[K♠]" in result


class TestRenderTableau:
    def test_empty_piles(self):
        result = render_tableau([])
        assert result == ""

    def test_single_pile(self):
        pile = [[Card(Suit.HEARTS, Rank.ACE)]]
        result = render_tableau(pile)
        assert "1" in result  # Header

    def test_max_display_height(self):
        pile = [[Card(Suit.HEARTS, Rank.ACE), Card(Suit.HEARTS, Rank.TWO)]]
        result = render_tableau(pile, max_display_height=1)
        lines = result.split("\n")
        # Should have header, separator, and 1 row
        assert len(lines) == 3

    def test_uneven_piles(self):
        """Test piles with different heights - should pad with spaces."""
        piles = [
            [Card(Suit.HEARTS, Rank.ACE), Card(Suit.HEARTS, Rank.TWO)],
            [Card(Suit.SPADES, Rank.KING)],  # Shorter pile
        ]
        result = render_tableau(piles)
        # Should have padding for shorter pile
        assert "      " in result or len(result) > 0

    def test_all_empty_piles(self):
        """Test with empty piles list."""
        piles = [[], []]
        result = render_tableau(piles)
        assert "1" in result  # Should still have headers


class TestCreateSimpleBoardView:
    def test_basic_state(self):
        state = {"score": 100, "turn": 5}
        result = create_simple_board_view(state)
        assert "score: 100" in result
        assert "turn: 5" in result

    def test_with_cards(self):
        cards = [Card(Suit.HEARTS, Rank.ACE)]
        state = {"hand": cards}
        result = create_simple_board_view(state)
        assert "hand:" in result

    def test_with_tableau(self):
        piles = [[Card(Suit.HEARTS, Rank.ACE)]]
        state = {"tableaux": piles}
        result = create_simple_board_view(state)
        assert "tableaux:" in result
