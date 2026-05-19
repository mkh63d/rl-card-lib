"""Tests for core module."""

import pytest
from rl_card_lib.cardgames import Card, Deck, Player, Suit, Rank


class TestCard:
    """Tests for Card class."""
    
    def test_card_creation(self):
        """Test basic card creation."""
        card = Card(Suit.HEARTS, Rank.ACE)
        assert card.suit == Suit.HEARTS
        assert card.rank == Rank.ACE
        assert card.face_up == True
    
    def test_card_face_down(self):
        """Test face-down card."""
        card = Card(Suit.SPADES, Rank.KING, face_up=False)
        assert card.face_up == False
        assert str(card) == "[??]"
    
    def test_card_flip(self):
        """Test flipping a card."""
        card = Card(Suit.DIAMONDS, Rank.QUEEN, face_up=False)
        card.flip()
        assert card.face_up == True
    
    def test_card_color(self):
        """Test card colors."""
        hearts = Card(Suit.HEARTS, Rank.ACE)
        diamonds = Card(Suit.DIAMONDS, Rank.ACE)
        clubs = Card(Suit.CLUBS, Rank.ACE)
        spades = Card(Suit.SPADES, Rank.ACE)
        
        assert hearts.color == "red"
        assert diamonds.color == "red"
        assert clubs.color == "black"
        assert spades.color == "black"
    
    def test_card_index_conversion(self):
        """Test card to/from index conversion."""
        for suit in Suit:
            for rank in Rank:
                card = Card(suit, rank)
                index = card.to_index()
                recovered = Card.from_index(index)
                assert recovered.suit == card.suit
                assert recovered.rank == card.rank
    
    def test_card_encoding(self):
        """Test card feature encoding."""
        card = Card(Suit.HEARTS, Rank.KING)
        encoding = card.encode()
        
        assert len(encoding) == 18
        assert encoding[int(Suit.HEARTS)] == 1.0  # Suit one-hot
        assert encoding[4 + int(Rank.KING) - 1] == 1.0  # Rank one-hot
        assert encoding[17] == 1.0  # Face up


class TestDeck:
    """Tests for Deck class."""
    
    def test_deck_creation(self):
        """Test standard deck creation."""
        deck = Deck()
        assert len(deck) == 52
    
    def test_deck_shuffle(self):
        """Test deck shuffling."""
        deck1 = Deck()
        deck2 = Deck()
        
        deck1.shuffle(seed=42)
        deck2.shuffle(seed=42)
        
        # Same seed should give same order
        for c1, c2 in zip(deck1, deck2):
            assert c1 == c2
    
    def test_deck_draw(self):
        """Test drawing cards."""
        deck = Deck()
        cards = deck.draw(5)
        
        assert len(cards) == 5
        assert len(deck) == 47
    
    def test_deck_draw_one(self):
        """Test drawing single card."""
        deck = Deck()
        card = deck.draw_one()
        
        assert isinstance(card, Card)
        assert len(deck) == 51
    
    def test_deck_draw_too_many(self):
        """Test drawing more cards than available."""
        deck = Deck()
        deck.draw(50)
        
        with pytest.raises(ValueError):
            deck.draw(5)
    
    def test_deck_reset(self):
        """Test deck reset."""
        deck = Deck()
        deck.draw(20)
        deck.reset()
        
        assert len(deck) == 52


class TestPlayer:
    """Tests for Player class."""
    
    def test_player_creation(self):
        """Test player creation."""
        player = Player(player_id=1, name="Alice")
        assert player.player_id == 1
        assert player.name == "Alice"
        assert player.hand_size() == 0
    
    def test_player_add_cards(self):
        """Test adding cards to hand."""
        player = Player(0)
        cards = [Card(Suit.HEARTS, Rank.ACE), Card(Suit.SPADES, Rank.KING)]
        player.add_cards(cards)
        
        assert player.hand_size() == 2
    
    def test_player_play_card(self):
        """Test playing a card."""
        player = Player(0)
        card = Card(Suit.HEARTS, Rank.ACE)
        player.add_card(card)
        
        played = player.play_card(0)
        assert played == card
        assert player.hand_size() == 0
    
    def test_player_score(self):
        """Test score management."""
        player = Player(0)
        player.add_score(10)
        player.add_score(5)
        
        assert player.score == 15
        
        player.reset_score()
        assert player.score == 0


class TestCardExtended:
    """Extended Card tests."""

    def test_card_suit_symbol(self):
        """Test suit symbols."""
        for suit in Suit:
            assert suit.symbol in "♣♦♥♠"

    def test_card_rank_symbol(self):
        """Test rank symbols."""
        for rank in Rank:
            assert rank.symbol in "A23456789JQK" or rank.symbol == "10"

    def test_card_hash(self):
        """Test card hashing."""
        card1 = Card(Suit.HEARTS, Rank.ACE)
        card2 = Card(Suit.HEARTS, Rank.ACE)
        assert hash(card1) == hash(card2)


class TestDeckExtended:
    """Extended Deck tests."""

    def test_deck_copy(self):
        """Test deck copy."""
        deck = Deck()
        copied = deck.copy()
        assert len(copied) == len(deck)
        
        deck.draw(10)
        assert len(copied) == 52  # Copy unchanged

    def test_deck_peek(self):
        """Test peeking at cards."""
        deck = Deck()
        cards = deck.peek(3)
        assert len(cards) == 3
        assert len(deck) == 52  # No cards removed

    def test_deck_peek_too_many(self):
        """Test peeking more than available."""
        deck = Deck()
        deck.draw(50)
        cards = deck.peek(10)
        assert len(cards) == 2

    def test_deck_is_empty(self):
        """Test empty check."""
        deck = Deck()
        assert not deck.is_empty()
        deck.draw(52)
        assert deck.is_empty()

    def test_deck_add_to_top(self):
        """Test adding to top."""
        deck = Deck()
        card = deck.draw_one()
        initial_len = len(deck)
        deck.add_to_top([card])
        assert len(deck) == initial_len + 1

    def test_deck_add_to_bottom(self):
        """Test adding to bottom."""
        deck = Deck()
        card = deck.draw_one()
        deck.add_to_bottom([card])
        assert len(deck) == 52

    def test_deck_encode(self):
        """Test deck encoding."""
        deck = Deck()
        encoding = deck.encode()
        assert len(encoding) == 52
        assert sum(encoding) == 52

    def test_deck_custom_cards(self):
        """Test deck with custom cards."""
        cards = [Card(Suit.HEARTS, Rank.ACE), Card(Suit.SPADES, Rank.KING)]
        deck = Deck(cards=cards)
        assert len(deck) == 2


class TestPlayerExtended:
    """Extended Player tests."""

    def test_player_str_repr(self):
        """Test string representations."""
        player = Player(1, name="Test")
        assert "Test" in str(player)
        assert "Test" in repr(player)

    def test_player_remove_card(self):
        """Test removing specific card."""
        player = Player(0)
        card = Card(Suit.HEARTS, Rank.ACE)
        player.add_card(card)
        
        removed = player.remove_card(card)
        assert removed == card
        assert player.hand_size() == 0

    def test_player_remove_card_not_found(self):
        """Test removing card not in hand."""
        player = Player(0)
        card = Card(Suit.HEARTS, Rank.ACE)
        
        with pytest.raises(ValueError):
            player.remove_card(card)

    def test_player_play_card_invalid_index(self):
        """Test playing card with invalid index."""
        player = Player(0)
        
        with pytest.raises(IndexError):
            player.play_card(5)

    def test_player_clear_hand(self):
        """Test clearing hand."""
        player = Player(0)
        player.add_cards([
            Card(Suit.HEARTS, Rank.ACE),
            Card(Suit.SPADES, Rank.KING),
        ])
        
        cards = player.clear_hand()
        assert len(cards) == 2
        assert player.hand_size() == 0

    def test_player_sort_hand_by_suit(self):
        """Test sorting hand by suit."""
        player = Player(0)
        player.add_cards([
            Card(Suit.SPADES, Rank.ACE),
            Card(Suit.CLUBS, Rank.KING),
        ])
        player.sort_hand(by_suit=True)
        assert player.hand[0].suit == Suit.CLUBS

    def test_player_sort_hand_by_rank(self):
        """Test sorting hand by rank."""
        player = Player(0)
        player.add_cards([
            Card(Suit.HEARTS, Rank.KING),
            Card(Suit.HEARTS, Rank.ACE),
        ])
        player.sort_hand(by_suit=False)
        assert player.hand[0].rank == Rank.ACE

    def test_player_encode_hand(self):
        """Test hand encoding."""
        player = Player(0)
        player.add_card(Card(Suit.HEARTS, Rank.ACE))
        encoding = player.encode_hand()
        assert len(encoding) == 18  # Single card encoding

    def test_player_encode_hand_binary(self):
        """Test binary hand encoding."""
        player = Player(0)
        card = Card(Suit.HEARTS, Rank.ACE)
        player.add_card(card)
        encoding = player.encode_hand_binary()
        assert len(encoding) == 52
        assert encoding[card.to_index()] == 1.0


class TestCardGame:
    """Tests for CardGame base class (via concrete implementations)."""

    def test_game_basic_methods(self):
        """Test basic CardGame methods."""
        from rl_card_lib.games import Macao
        
        game = Macao(num_players=2)
        game.reset()
        
        # Test next_player
        current = game.current_player_idx
        game.next_player()
        assert game.current_player_idx != current  # Should have switched

    def test_game_get_current_player(self):
        """Test get_current_player method."""
        from rl_card_lib.games import Macao
        
        game = Macao(num_players=2)
        game.reset()
        
        player = game.get_current_player()
        assert player is not None
        assert player.player_id == 0

    def test_game_action_to_string_default(self):
        """Test default action_to_string."""
        from rl_card_lib.games import KlondikeSolitaire
        
        game = KlondikeSolitaire()
        game.reset()
        s = game.action_to_string(999)  # Out of range
        assert "tableau" in s or "Action" in s

    def test_game_render(self):
        """Test render method."""
        from rl_card_lib.games import KlondikeSolitaire
        
        game = KlondikeSolitaire()
        game.reset()
        rendered = game.render()
        assert isinstance(rendered, str)

    def test_game_get_reward(self):
        """Test get_reward method."""
        from rl_card_lib.games import KlondikeSolitaire
        
        game = KlondikeSolitaire()
        game.reset()
        reward = game.get_reward(0)
        assert isinstance(reward, (int, float))

    def test_game_get_winner(self):
        """Test get_winner method."""
        from rl_card_lib.games import KlondikeSolitaire
        
        game = KlondikeSolitaire()
        game.reset()
        winner = game.get_winner()
        assert winner is None  # Game not over

    def test_game_log_action(self):
        """Test logging actions."""
        from rl_card_lib.games import KlondikeSolitaire
        
        game = KlondikeSolitaire()
        game.reset()
        game.log_action(0, 0, 0.5)
        
        history = game.get_history()
        assert len(history) == 1
        assert history[0]["action"] == 0

    def test_game_legal_action_mask(self):
        """Test legal action mask."""
        from rl_card_lib.games import KlondikeSolitaire
        
        game = KlondikeSolitaire()
        game.reset()
        mask = game.get_legal_action_mask()
        assert mask.dtype == bool
        assert mask[0] == True  # Draw should be legal

    def test_game_copy_not_implemented(self):
        """Test copy raises NotImplementedError."""
        from rl_card_lib.games import KlondikeSolitaire
        
        game = KlondikeSolitaire()
        game.reset()
        
        with pytest.raises(NotImplementedError):
            game.copy()

    def test_game_default_render(self):
        """Test default render method from base class."""
        from rl_card_lib.cardgames import CardGame
        from rl_card_lib.games import Macao
        
        game = Macao()
        # Access base class render via explicit call before full initialization
        # This tests the default implementation
        base_render = CardGame.render(game)
        assert "Macao" in base_render
        assert "Turn" in base_render

    def test_game_default_action_to_string(self):
        """Test default action_to_string method."""
        from rl_card_lib.cardgames import CardGame
        from rl_card_lib.games import Macao
        
        game = Macao()
        # Access base class method
        base_str = CardGame.action_to_string(game, 999)
        assert "Action 999" in base_str


class TestCardReprAndCompare:
    """Additional tests for Card special methods."""
    
    def test_card_repr(self):
        """Test card __repr__ method."""
        card = Card(Suit.HEARTS, Rank.ACE)
        repr_str = repr(card)
        assert "Card(" in repr_str
        assert "HEARTS" in repr_str
        assert "ACE" in repr_str
        assert "face_up=True" in repr_str
    
    def test_card_eq_with_non_card(self):
        """Test card equality with non-Card object returns NotImplemented."""
        card = Card(Suit.HEARTS, Rank.ACE)
        result = card.__eq__("not a card")
        assert result is NotImplemented
        
        result = card.__eq__(42)
        assert result is NotImplemented


class TestDeckIndexing:
    """Tests for Deck indexing."""
    
    def test_deck_getitem(self):
        """Test Deck __getitem__ method."""
        deck = Deck()
        card = deck[0]
        assert isinstance(card, Card)
        
        card_last = deck[-1]
        assert isinstance(card_last, Card)
        
        card_middle = deck[25]
        assert isinstance(card_middle, Card)
    
    def test_deck_str(self):
        """Test Deck __str__ method."""
        deck = Deck()
        s = str(deck)
        assert "Deck" in s
        assert "52" in s


class TestPlayerHasCard:
    """Tests for Player has_card method."""
    
    def test_player_has_card_true(self):
        """Test has_card returns True when card exists."""
        player = Player("Test")
        card = Card(Suit.HEARTS, Rank.ACE)
        player.add_card(card)
        
        assert player.has_card(card) == True
    
    def test_player_has_card_false(self):
        """Test has_card returns False when card doesn't exist."""
        player = Player("Test")
        card1 = Card(Suit.HEARTS, Rank.ACE)
        card2 = Card(Suit.SPADES, Rank.KING)
        player.add_card(card1)
        
        assert player.has_card(card2) == False