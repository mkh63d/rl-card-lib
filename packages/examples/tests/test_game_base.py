"""Tests for CardGame base class behavior using example games."""

import numpy as np
import pytest

from rl_card_lib.cardgames import CardGame
from rl_card_lib.games import KlondikeSolitaire, Macao


class TestCardGame:
    """Tests for CardGame base class (via concrete implementations)."""

    def test_game_basic_methods(self):
        """Test basic CardGame methods."""
        game = Macao(num_players=2)
        game.reset()

        # Test next_player
        current = game.current_player_idx
        game.next_player()
        assert game.current_player_idx != current  # Should have switched

    def test_game_get_current_player(self):
        """Test get_current_player method."""
        game = Macao(num_players=2)
        game.reset()

        player = game.get_current_player()
        assert player is not None
        assert player.player_id == 0

    def test_game_action_to_string_default(self):
        """Test default action_to_string."""
        game = KlondikeSolitaire()
        game.reset()
        s = game.action_to_string(999)  # Out of range
        assert "tableau" in s or "Action" in s

    def test_game_render(self):
        """Test render method."""
        game = KlondikeSolitaire()
        game.reset()
        rendered = game.render()
        assert isinstance(rendered, str)

    def test_game_get_reward(self):
        """Test get_reward method."""
        game = KlondikeSolitaire()
        game.reset()
        reward = game.get_reward(0)
        assert isinstance(reward, (int, float))

    def test_game_get_winner(self):
        """Test get_winner method."""
        game = KlondikeSolitaire()
        game.reset()
        winner = game.get_winner()
        assert winner is None  # Game not over

    def test_game_log_action(self):
        """Test logging actions."""
        game = KlondikeSolitaire()
        game.reset()
        game.log_action(0, 0, 0.5)

        history = game.get_history()
        assert len(history) == 1
        assert history[0]["action"] == 0

    def test_game_legal_action_mask(self):
        """Test legal action mask."""
        game = KlondikeSolitaire()
        game.reset()
        mask = game.get_legal_action_mask()
        assert mask.dtype == bool
        assert bool(mask[0]) is True  # Draw should be legal

    def test_game_copy_not_implemented_by_default(self):
        """Test copy raises NotImplementedError for games that don't override it."""
        class UncopyableGame(CardGame):
            def reset(self): return np.zeros(1, dtype=np.float32)
            def step(self, action): return np.zeros(1, dtype=np.float32), 0.0, False, False, {}
            def get_legal_actions(self): return [0]
            def get_observation(self): return np.zeros(1, dtype=np.float32)
            def get_action_space_size(self): return 1
            def get_observation_shape(self): return (1,)
            def is_game_over(self): return False

        with pytest.raises(NotImplementedError):
            UncopyableGame().copy()

    def test_game_copy_is_independent(self):
        """Test a copied game can be played without disturbing the original."""
        game = KlondikeSolitaire()
        game.reset()
        before = game.get_observation().copy()

        clone = game.copy()
        for action in clone.get_legal_actions()[:5]:
            clone.step(action)

        assert np.array_equal(game.get_observation(), before)

    def test_game_default_render(self):
        """Test default render method from base class."""
        game = Macao()
        # Access base class render via explicit call before full initialization
        # This tests the default implementation
        base_render = CardGame.render(game)
        assert "Macao" in base_render
        assert "Turn" in base_render

    def test_game_default_action_to_string(self):
        """Test default action_to_string method."""
        game = Macao()
        # Access base class method
        base_str = CardGame.action_to_string(game, 999)
        assert "Action 999" in base_str
