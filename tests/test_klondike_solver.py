"""Tests for the perfect-information Klondike solvability search."""

import numpy as np

from rl_card_lib.cardgames import Card, Rank, Suit
from rl_card_lib.games import KlondikeSolitaire, solve_klondike


def nearly_won_game():
    """Everything on the foundations except the king of clubs."""
    game = KlondikeSolitaire()
    game.foundations = [
        [Card(Suit(s), Rank(r), True) for r in range(1, 14)] for s in range(4)
    ]
    game.foundations[0] = [Card(Suit.CLUBS, Rank(r), True) for r in range(1, 13)]
    game.tableaux = [[] for _ in range(7)]
    game.tableaux[0] = [Card(Suit.CLUBS, Rank.KING, True)]
    game.stock = []
    game.waste = []
    return game


def dead_game():
    """A face-down card trapped under a card no pile accepts."""
    game = KlondikeSolitaire()
    game.foundations = [[] for _ in range(4)]
    game.tableaux = [[] for _ in range(7)]
    game.tableaux[0] = [
        Card(Suit.CLUBS, Rank.FIVE, False),
        Card(Suit.HEARTS, Rank.SEVEN, True),
    ]
    game.stock = []
    game.waste = []
    return game


class TestSolveKlondike:
    """Tests for solve_klondike()."""

    def test_winnable_position_is_true(self):
        assert solve_klondike(nearly_won_game()) is True

    def test_already_won_position_is_true(self):
        game = nearly_won_game()
        game.step(12)  # send the last king up
        assert game.winner == 0
        assert solve_klondike(game) is True

    def test_dead_position_is_false(self):
        assert solve_klondike(dead_game()) is False

    def test_budget_exhaustion_is_none(self):
        game = KlondikeSolitaire(seed=1)
        game.reset(seed=1)
        # One node cannot decide a fresh deal either way.
        assert solve_klondike(game, max_nodes=1) is None

    def test_input_game_is_not_mutated(self):
        game = KlondikeSolitaire(seed=2)
        game.reset(seed=2)
        before = np.asarray(game.get_observation()).copy()
        solve_klondike(game, max_nodes=200)
        assert np.array_equal(np.asarray(game.get_observation()), before)

    def test_finds_a_win_through_the_stock(self):
        """The winning card is buried in the stock, so the draw must be used."""
        game = nearly_won_game()
        game.tableaux[0] = []
        game.stock = [Card(Suit.CLUBS, Rank.KING, False)]
        assert solve_klondike(game) is True

    def test_respects_max_passes_in_the_key(self):
        """With passes exhausted and nothing playable, the deal is dead."""
        game = dead_game()
        game.max_passes = 1
        game.passes = 0
        game.stock = []
        game.waste = [Card(Suit.DIAMONDS, Rank.NINE, True)]
        assert solve_klondike(game) is False
