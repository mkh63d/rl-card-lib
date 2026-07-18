"""Hand-written expert agents for the example games.

These encode how a competent human plays, and exist to be beaten: a learner that
cannot match them has not learned much, and one that beats them has found
something the rules above do not capture. They also serve as rollout policies for
MCTSAgent and as non-trivial opponents for Macao self-play.
"""

from collections import deque
from typing import Any, Optional

from rl_card_lib.agents import HeuristicAgent
from rl_card_lib.cardgames import Card, Rank, Suit

RED_SUITS = (Suit.DIAMONDS, Suit.HEARTS)
BLACK_SUITS = (Suit.CLUBS, Suit.SPADES)


class KlondikeHeuristicAgent(HeuristicAgent):
    """
    Klondike expert following the standard priority order.

    Roughly, in descending order: send a card to the foundation when it also
    reveals a face-down card, then aces and twos, then any move that turns a
    face-down card face up, then safe foundation moves, then playing off the
    waste, and only draw when nothing better exists.

    Two rules do most of the work:

    Foundation moves are not automatically good. A card sent up early is gone
    from the tableau, where it might have been the only thing able to hold an
    opposite-color card. A move is only "safe" once both opposite-color
    foundations are within one rank of it, which guarantees nothing still needs
    it; unsafe ones score well below safe ones and below playing off the waste,
    so they happen only when little else is available. Demoting them further, to
    below even drawing, measures worse (38.7% vs 43.3% win rate over 150 deals):
    stalling on a card that is merely *probably* needed wastes more games than
    it saves.

    Tableau-to-tableau moves that reveal nothing are scored *negative*. They
    are legal, reversible and infinitely repeatable — the same loop the
    environment's reward once paid for (it no longer does): a scorer that liked
    them even slightly would shuffle two piles back and forth forever. A short
    memory of recent moves discourages the remaining cycles.

    Wins roughly 43% of draw-1 deals, in the range a thoughtful human reaches.

    Args:
        game: Game or environment to read state from (can be bound later)
        seed: Random seed for tie-breaking
        memory: Number of recent moves that count as repeats
    """

    def __init__(
        self,
        game: Optional[Any] = None,
        seed: Optional[int] = None,
        memory: int = 8,
    ):
        super().__init__(game=game, name="KlondikeHeuristicAgent", seed=seed)
        self._recent: deque = deque(maxlen=memory)

    def reset(self) -> None:
        """Forget the previous episode's move history."""
        self._recent.clear()

    def select_action(self, observation, legal_actions=None) -> int:
        """Select a move and remember it, so repeats can be discouraged."""
        action = super().select_action(observation, legal_actions)
        self._recent.append(action)
        return action

    def _foundation_rank(self, game: Any, suit: Suit) -> int:
        """Rank on top of a suit's foundation, 0 if the pile is empty."""
        pile = game.foundations[int(suit)]
        return int(pile[-1].rank) if pile else 0

    def _is_safe_to_foundation(self, game: Any, card: Card) -> bool:
        """
        Whether sending `card` up can never strand a tableau build.

        Safe if no card that might still need it as a resting place is left in
        play: both opposite-color foundations must have reached at least one rank
        below it, and the other same-color foundation at least two below.

        Args:
            game: Game to read the foundations from
            card: Card being considered for the foundation

        Returns:
            True if the move cannot cost a later tableau placement
        """
        rank = int(card.rank)
        if rank <= 2:
            # Nothing is ever built on an ace or a two.
            return True

        if card.color == "red":
            opposite, same = BLACK_SUITS, RED_SUITS
        else:
            opposite, same = RED_SUITS, BLACK_SUITS

        if min(self._foundation_rank(game, suit) for suit in opposite) < rank - 1:
            return False

        other_same = next(suit for suit in same if suit != card.suit)
        return self._foundation_rank(game, other_same) >= rank - 2

    def _reveals_by_removing_top(self, game: Any, pile_idx: int) -> bool:
        """Whether taking the top card off a pile turns a face-down card up."""
        pile = game.tableaux[pile_idx]
        return len(pile) >= 2 and not pile[-2].face_up

    def _tableau_move_index(self, game: Any, from_pile: int, to_pile: int) -> int:
        """
        Index of the card the game would move, mirroring _move_tableau_to_tableau.

        The action encoding names only the two piles, so the game picks the card:
        the first face-up one that fits. Scoring has to make the same choice.

        Args:
            game: Game to read the tableaux from
            from_pile: Source pile index
            to_pile: Destination pile index

        Returns:
            Index into the source pile, or -1 if no card fits
        """
        for idx, card in enumerate(game.tableaux[from_pile]):
            if card.face_up and game._can_place_on_tableau(card, to_pile):
                return idx
        return -1

    def score_action(self, game: Any, action: int) -> float:
        """
        Rate a legal Klondike move.

        Args:
            game: Game to read the position from
            action: Legal action index

        Returns:
            Score on a roughly -100 to 115 scale
        """
        if action == 0:
            # Drawing is cheap and reveals a card; recycling the waste costs a
            # pass and shows nothing new, so it is the last resort.
            return 10.0 if game.stock else -40.0

        if 12 <= action <= 18:
            pile_idx = action - 12
            card = game.tableaux[pile_idx][-1]
            if int(card.rank) <= 2:
                base = 90.0
            elif self._is_safe_to_foundation(game, card):
                base = 70.0
            else:
                base = 15.0
            return base + (25.0 if self._reveals_by_removing_top(game, pile_idx) else 0.0)

        if 8 <= action <= 11:
            card = game.waste[-1]
            if int(card.rank) <= 2:
                return 90.0
            return 60.0 if self._is_safe_to_foundation(game, card) else 15.0

        if 1 <= action <= 7:
            # Emptying the waste is good on its own: it exposes the card beneath
            # and keeps the pile from clogging.
            return 50.0

        relative = action - 19
        from_pile, to_pile = relative // 7, relative % 7
        move_idx = self._tableau_move_index(game, from_pile, to_pile)
        if move_idx == -1:
            return -100.0

        reveals = move_idx > 0 and not game.tableaux[from_pile][move_idx - 1].face_up
        if reveals:
            return 80.0

        empties_column = move_idx == 0
        if empties_column:
            # Shuffling a king between bare columns achieves nothing.
            if not game.tableaux[to_pile]:
                return -60.0
            return 45.0

        score = -10.0
        if action in self._recent:
            score -= 30.0
        return score


class MacaoHeuristicAgent(HeuristicAgent):
    """
    Macao expert: punish the opponent, hold the wild cards, go out fast.

    The ordering reflects what actually decides the game:

    Counter a pending penalty whenever possible. Eating 5 cards from a king is
    close to losing a race that is decided by hand size, so any legal counter
    beats drawing.

    Hold jacks and aces. They are playable on anything, which makes them the
    cards that rescue an otherwise dead hand. Spending one early trades that
    insurance for a turn that an ordinary matching card would have covered.

    Attack when the opponent is nearly out. A skip or a draw-penalty card is
    worth much more against a two-card hand than against a seven-card hand, so
    their score rises as the opponent's hand shrinks.

    Args:
        game: Game or environment to read state from (can be bound later)
        seed: Random seed for tie-breaking
    """

    def __init__(self, game: Optional[Any] = None, seed: Optional[int] = None):
        super().__init__(game=game, name="MacaoHeuristicAgent", seed=seed)

    def score_action(self, game: Any, action: int) -> float:
        """
        Rate a legal Macao move.

        Args:
            game: Game to read the position from
            action: Legal action index

        Returns:
            Score, with 1000 reserved for the winning move
        """
        if action >= game.SUIT_DECLARATION_OFFSET:
            return self._score_declaration(game, action)

        if action == 53:
            return -100.0

        if action == 52:
            # Only ever chosen when nothing else is legal; eating a stacked
            # penalty is worse than drawing one card by choice.
            return -60.0 if game.draw_penalty > 0 else -20.0

        player = game.get_current_player()
        card = Card.from_index(action)

        if len(player.hand) == 1:
            return 1000.0

        if game.draw_penalty > 0:
            return 200.0 + game._get_draw_penalty(card)

        opponent_cards = min(
            len(other.hand)
            for idx, other in enumerate(game.players)
            if idx != game.current_player_idx
        )

        if game._is_draw_card(card):
            score = 60.0 + 2.0 * game._get_draw_penalty(card)
        elif card.rank == Rank.FOUR:
            score = 55.0
        elif card.rank == Rank.JACK:
            score = 12.0
        elif card.rank == Rank.ACE:
            score = 18.0
        else:
            score = 35.0

        if opponent_cards <= 2 and (
            game._is_draw_card(card) or card.rank == Rank.FOUR
        ):
            score += 40.0

        # Prefer playing out of a suit we hold plenty of: the ones left behind
        # are likelier to match the next discard.
        suit_count = sum(1 for held in player.hand if held.suit == card.suit)
        return score + 2.0 * suit_count

    def _score_declaration(self, game: Any, action: int) -> float:
        """
        Rate a suit/rank declaration after an Ace or Jack.

        Declares whatever the hand holds most of, which is what the game itself
        hardcoded before declaring became an action: it maximizes the chance of
        having a legal reply on the next turn.

        Args:
            game: Game to read the position from
            action: Declaration action index (54-64)

        Returns:
            Count of matching cards in hand, as the score
        """
        hand = game.get_current_player().hand
        if action < game.RANK_DECLARATION_OFFSET:
            suit = Suit(action - game.SUIT_DECLARATION_OFFSET)
            return float(sum(1 for card in hand if card.suit == suit))
        rank = game.DECLARABLE_RANKS[action - game.RANK_DECLARATION_OFFSET]
        return float(sum(1 for card in hand if card.rank == rank))
