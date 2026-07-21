"""Macao (Crazy Eights variant) game implementation."""

import random
from typing import Any, Optional
import numpy as np

from rl_card_lib.cardgames import Card, Suit, Rank, Deck, Player, CardGame


def _clone_cards(cards: list[Card]) -> list[Card]:
    """Copy a pile so the clone's cards can be flipped without touching the original."""
    return [Card(c.suit, c.rank, c.face_up) for c in cards]


class Macao(CardGame):
    """
    Macao card game (Polish variant of Crazy Eights / Uno).

    Rules:
    - 2-4 players, each starts with 5 cards
    - Play a card matching suit or rank of top discard pile card
    - Special cards:
        - 2, 3: Next player draws 2/3 cards (can be stacked)
        - 4: Next player skips turn
        - Jack: Request a rank (except special cards)
        - Ace: Change suit
        - King of Spades/Hearts: Draw 5 cards for next player
    - First player to empty hand wins

    For RL training, we implement a simplified 2-player version.

    Playing an Ace or a Jack (with cards still in hand) moves the game into a
    declaration phase: the same player's next action names the requested suit
    (actions 54-57) or rank (actions 58-64). Declaring is part of the action
    space on purpose; it used to be a hardcoded most-common-in-hand rule, which
    made the game's two most strategic decisions unlearnable.

    Rewards are actor-relative: every reward `step()` returns is paid to the
    player who took the action, including the terminal one (the winner is
    always the actor on the winning play). Use `get_reward()` for any other
    player's terminal payoff. Shaped per-move rewards are potential-based on
    hand size (see CARD_POTENTIAL), so they cannot be farmed by any loop of
    moves; sparse mode drops them entirely and pays +1 to the winner.
    """

    # Action encoding:
    # 0-51: Play the card with that index
    # 52: Draw (one card, or the accumulated penalty)
    # 53: Pass (only when skipped, or when stuck with an empty deck)
    # 54-57: Declare a suit after playing an Ace (Suit order: ♣ ♦ ♥ ♠)
    # 58-64: Declare a rank after playing a Jack (DECLARABLE_RANKS order)

    ACTION_DRAW = 52
    ACTION_PASS = 53
    SUIT_DECLARATION_OFFSET = 54
    RANK_DECLARATION_OFFSET = 58

    #: Ranks a Jack may request: everything without a special effect.
    DECLARABLE_RANKS = (
        Rank.FIVE, Rank.SIX, Rank.SEVEN, Rank.EIGHT,
        Rank.NINE, Rank.TEN, Rank.QUEEN,
    )

    MAX_ACTIONS = RANK_DECLARATION_OFFSET + len(DECLARABLE_RANKS)  # 65, all used
    HAND_SIZE = 5

    #: Terminal payoffs (shaped mode). The winner's reward arrives on the
    #: winning step; the losers' share is only visible through get_reward(),
    #: since a step() reward always belongs to the actor.
    WIN_REWARD = 10.0
    LOSS_REWARD = -5.0

    #: Shaped per-move rewards are potential-based on hand size: a card leaving
    #: the hand pays this, a card entering costs it. Any move sequence's shaped
    #: total then telescopes to (cards shed - cards gained) * this constant, so
    #: no loop of moves can mint reward and only actually going out pays.
    #: Flat per-play bonuses are not safe here: they made hoarding profitable
    #: (draw cards to build a big hand, then harvest a bonus per play), which
    #: search agents found and exploited by never finishing the game.
    CARD_POTENTIAL = 0.1

    def __init__(
        self,
        num_players: int = 2,
        max_turns: int = 200,
        reward_mode: str = "shaped",
        seed: Optional[int] = None,
    ):
        """
        Initialize Macao game.

        Args:
            num_players: Number of players (2-4)
            max_turns: Maximum turns before draw
            reward_mode: "shaped" pays small per-move bonuses plus the terminal
                payoff; "sparse" pays only +1 to the winner on the winning move
            seed: Seed for this game's private RNG, used for shuffling. The
                process-wide RNG is never touched.
        """
        super().__init__(num_players=num_players)

        if reward_mode not in ("shaped", "sparse"):
            raise ValueError(
                f"reward_mode must be 'shaped' or 'sparse', got {reward_mode!r}"
            )

        self.max_turns = max_turns
        self.reward_mode = reward_mode
        self._rng = random.Random(seed)

        # Game state
        self.discard_pile: list[Card] = []
        self.requested_suit: Optional[Suit] = None
        self.requested_rank: Optional[Rank] = None
        self.draw_penalty: int = 0  # Accumulated draw penalty
        self.skip_next: bool = False
        #: "suit" after an Ace, "rank" after a Jack, None otherwise. While set,
        #: the player who played the card must declare before play continues.
        self.pending_declaration: Optional[str] = None

        self.reset()

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """
        Reset game to initial state.

        Args:
            seed: Reseeds this game's private RNG first, making the deal (and
                every later shuffle) reproducible without touching global state

        Returns:
            The initial observation
        """
        if seed is not None:
            self._rng = random.Random(seed)
        self.deck = Deck()
        self.deck.shuffle(rng=self._rng)

        # Create players
        self.players = [
            Player(player_id=i, is_agent=(i == 0))
            for i in range(self.num_players)
        ]

        # Deal cards
        for player in self.players:
            cards = self.deck.draw(self.HAND_SIZE, face_up=True)
            player.add_cards(cards)

        # Start discard pile
        while True:
            card = self.deck.draw_one(face_up=True)
            # Don't start with special card
            if card.rank not in (Rank.TWO, Rank.THREE, Rank.FOUR, Rank.JACK, Rank.ACE):
                if not (card.rank == Rank.KING and card.suit in (Suit.SPADES, Suit.HEARTS)):
                    self.discard_pile = [card]
                    break
            # Put back and try again
            self.deck.add_to_bottom([card])

        # Reset state
        self.current_player_idx = 0
        self.requested_suit = None
        self.requested_rank = None
        self.draw_penalty = 0
        self.skip_next = False
        self.pending_declaration = None
        self.done = False
        self.winner = None
        self._turn_count = 0

        return self.get_observation()

    def get_observation(self) -> np.ndarray:
        """
        Get current state observation.

        Encodes:
        - Current player's hand (binary 52)
        - Top discard card (one-hot 52)
        - Requested suit (one-hot 4)
        - Requested rank (one-hot 13)
        - Pending declaration phase (2 binary: declaring suit, declaring rank)
        - Draw penalty (normalized)
        - Opponent hand sizes (normalized)
        - Cards remaining in deck (normalized)
        """
        observation = []

        # Current player's hand (52 binary)
        current_player = self.get_current_player()
        hand_encoding = [0.0] * 52
        for card in current_player.hand:
            hand_encoding[card.to_index()] = 1.0
        observation.extend(hand_encoding)

        # Top discard card (52 one-hot)
        top_encoding = [0.0] * 52
        if self.discard_pile:
            top_encoding[self.discard_pile[-1].to_index()] = 1.0
        observation.extend(top_encoding)

        # Requested suit (4 one-hot)
        suit_encoding = [0.0] * 4
        if self.requested_suit is not None:
            suit_encoding[int(self.requested_suit)] = 1.0
        observation.extend(suit_encoding)

        # Requested rank (13 one-hot)
        rank_encoding = [0.0] * 13
        if self.requested_rank is not None:
            rank_encoding[int(self.requested_rank) - 1] = 1.0
        observation.extend(rank_encoding)

        # Declaration phase flags: the legal actions change completely while a
        # declaration is pending, so the network needs to see the phase.
        observation.append(1.0 if self.pending_declaration == "suit" else 0.0)
        observation.append(1.0 if self.pending_declaration == "rank" else 0.0)

        # Draw penalty (normalized 0-15)
        observation.append(min(self.draw_penalty / 15.0, 1.0))

        # Opponent hand sizes (normalized)
        for i, player in enumerate(self.players):
            if i != self.current_player_idx:
                observation.append(len(player.hand) / 15.0)

        # Cards in deck (normalized)
        observation.append(len(self.deck) / 52.0)

        return np.array(observation, dtype=np.float32)

    def get_observation_shape(self) -> tuple[int, ...]:
        """Get observation shape."""
        # 52 (hand) + 52 (top card) + 4 (suit) + 13 (rank) + 2 (declaration
        # phase) + 1 (penalty) + (num_players-1) (opponent hands) + 1 (deck)
        return (52 + 52 + 4 + 13 + 2 + 1 + (self.num_players - 1) + 1,)

    def get_action_space_size(self) -> int:
        """Get action space size."""
        return self.MAX_ACTIONS

    def get_legal_actions(self) -> list[int]:
        """Get list of legal action indices."""
        legal = []
        current_player = self.get_current_player()
        top_card = self.discard_pile[-1] if self.discard_pile else None

        # A pending declaration preempts everything: the player who just played
        # the Ace/Jack must name a suit/rank before play continues.
        if self.pending_declaration == "suit":
            return [self.SUIT_DECLARATION_OFFSET + i for i in range(4)]
        if self.pending_declaration == "rank":
            return [
                self.RANK_DECLARATION_OFFSET + i
                for i in range(len(self.DECLARABLE_RANKS))
            ]

        # If there's a draw penalty, must draw or counter
        if self.draw_penalty > 0:
            for card in current_player.hand:
                if self._is_draw_card(card):
                    # Can counter with same rank
                    if card.rank == top_card.rank:
                        legal.append(card.to_index())
                    # Can counter King with King
                    elif (card.rank == Rank.KING and
                          card.suit in (Suit.SPADES, Suit.HEARTS)):
                        legal.append(card.to_index())

            # Action 52: Draw penalty cards
            legal.append(52)
            return legal

        # If skipped, only action is to pass (draw one)
        if self.skip_next:
            legal.append(53)  # Pass action
            return legal

        # Regular play: find matching cards
        for card in current_player.hand:
            if self._can_play_card(card, top_card):
                legal.append(card.to_index())

        # Action 52: Draw a card (always legal if not forced)
        if len(self.deck) > 0:
            legal.append(52)

        # If no playable cards and can't draw, must pass
        if not legal:
            legal.append(53)

        return legal

    def _can_play_card(self, card: Card, top_card: Optional[Card]) -> bool:
        """Check if a card can be played on the discard pile."""
        if top_card is None:
            return True

        # If there's a requested suit (from Ace)
        if self.requested_suit is not None:
            if card.suit == self.requested_suit:
                return True
            # Ace can always be played
            if card.rank == Rank.ACE:
                return True
            return False

        # If there's a requested rank (from Jack)
        if self.requested_rank is not None:
            if card.rank == self.requested_rank:
                return True
            # Jack can always be played
            if card.rank == Rank.JACK:
                return True
            return False

        # Normal matching rules
        # Jack is wild - can be played on anything
        if card.rank == Rank.JACK:
            return True

        # Ace is wild for suit
        if card.rank == Rank.ACE:
            return True

        # Match suit or rank
        return card.suit == top_card.suit or card.rank == top_card.rank

    def _is_draw_card(self, card: Card) -> bool:
        """Check if card is a draw penalty card."""
        if card.rank in (Rank.TWO, Rank.THREE):
            return True
        if card.rank == Rank.KING and card.suit in (Suit.SPADES, Suit.HEARTS):
            return True
        return False

    def _get_draw_penalty(self, card: Card) -> int:
        """Get draw penalty for a card."""
        if card.rank == Rank.TWO:
            return 2
        if card.rank == Rank.THREE:
            return 3
        if card.rank == Rank.KING and card.suit in (Suit.SPADES, Suit.HEARTS):
            return 5
        return 0

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute an action for the current player.

        The returned reward always belongs to the acting player. Actions must be
        legal; an action `get_legal_actions()` would not offer raises
        ValueError rather than being absorbed into a penalty, so a buggy caller
        fails loudly. (`CardGameEnv` filters illegal actions before they reach
        this method.)

        Args:
            action: Action index to execute

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        self._turn_count += 1
        reward = 0.0
        actor_idx = self.current_player_idx
        current_player = self.get_current_player()

        # Declaration phase: the Ace/Jack player names the suit/rank.
        if self.pending_declaration is not None:
            if self.pending_declaration == "suit" and (
                self.SUIT_DECLARATION_OFFSET
                <= action
                < self.SUIT_DECLARATION_OFFSET + 4
            ):
                self.requested_suit = Suit(action - self.SUIT_DECLARATION_OFFSET)
            elif self.pending_declaration == "rank" and (
                self.RANK_DECLARATION_OFFSET
                <= action
                < self.RANK_DECLARATION_OFFSET + len(self.DECLARABLE_RANKS)
            ):
                self.requested_rank = self.DECLARABLE_RANKS[
                    action - self.RANK_DECLARATION_OFFSET
                ]
            else:
                raise ValueError(
                    f"Action {action} is not a legal {self.pending_declaration} "
                    f"declaration"
                )
            self.pending_declaration = None
            self.next_player()
            return self._finish_step(actor_idx, reward)

        if action >= self.SUIT_DECLARATION_OFFSET:
            raise ValueError(
                f"Action {action} is a declaration, but no declaration is pending"
            )

        # Action 52: Draw card(s)
        if action == 52:
            if self.draw_penalty > 0:
                # Draw penalty cards
                draw_count = min(self.draw_penalty, len(self.deck))
                if draw_count > 0:
                    cards = self.deck.draw(draw_count, face_up=True)
                    current_player.add_cards(cards)
                self.draw_penalty = 0
                reward = self.CARD_POTENTIAL * -draw_count
            else:
                # Draw one card
                if len(self.deck) > 0:
                    card = self.deck.draw_one(face_up=True)
                    current_player.add_card(card)
                    reward = -self.CARD_POTENTIAL

            self.next_player()

        # Action 53: Pass
        elif action == 53:
            self.skip_next = False
            self.next_player()
            reward = 0.0

        # Play a card
        elif action < 52:
            card = Card.from_index(action)

            # Find and play the card
            card_to_play = None
            for c in current_player.hand:
                if c.suit == card.suit and c.rank == card.rank:
                    card_to_play = c
                    break

            if card_to_play is None:
                raise ValueError(
                    f"Player {self.current_player_idx} does not hold {card}"
                )

            current_player.remove_card(card_to_play)
            self.discard_pile.append(card_to_play)

            # Clear requests
            self.requested_suit = None
            self.requested_rank = None

            # Every play pays the same CARD_POTENTIAL: one card left the hand.
            # Special cards get no extra bonus; their value has to show up in
            # the game outcome, not in a side payment the agent could chase
            # for its own sake.
            reward = self.CARD_POTENTIAL

            # Apply card effects
            if card_to_play.rank == Rank.FOUR:
                self.skip_next = True

            elif self._is_draw_card(card_to_play):
                self.draw_penalty += self._get_draw_penalty(card_to_play)

            elif card_to_play.rank in (Rank.ACE, Rank.JACK):
                # The declaration is the player's own next action (54-64), not
                # a hardcoded rule. A winning Ace/Jack skips it: there is no
                # hand left to strand an opponent against.
                if current_player.hand:
                    self.pending_declaration = (
                        "suit" if card_to_play.rank == Rank.ACE else "rank"
                    )

            # Check for win
            if len(current_player.hand) == 0:
                self.done = True
                self.winner = self.current_player_idx
                # Paid to the actor, who is always the winner here. Other
                # players' terminal payoffs are exposed via get_reward().
                reward = self.WIN_REWARD
            elif self.pending_declaration is None:
                self.next_player()

        return self._finish_step(actor_idx, reward)

    def _finish_step(
        self, actor_idx: int, reward: float
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Shared tail of step(): terminal bookkeeping, reshuffling and the info dict.

        Args:
            actor_idx: Player who took the action this step is finishing
            reward: Shaped reward accumulated for that player so far

        Returns:
            The (observation, reward, terminated, truncated, info) tuple
        """
        terminated = self.done
        truncated = self._turn_count >= self.max_turns and not terminated

        if self.reward_mode == "sparse":
            reward = 1.0 if terminated and self.winner == actor_idx else 0.0
        elif truncated:
            # Ran out of turns: score the actor's position by hand-size
            # differential, from the actor's own seat.
            actor_hand = len(self.players[actor_idx].hand)
            others = [
                len(p.hand)
                for idx, p in enumerate(self.players)
                if idx != actor_idx
            ]
            reward = 0.1 * (sum(others) / len(others) - actor_hand)

        # Reshuffle discard if deck empty
        if len(self.deck) == 0 and len(self.discard_pile) > 1:
            top_card = self.discard_pile.pop()
            self.deck.cards = self.discard_pile
            self.deck.shuffle(rng=self._rng)
            for card in self.deck.cards:
                card.face_up = False
            self.discard_pile = [top_card]

        info = {
            "current_player": self.current_player_idx,
            "hand_sizes": [len(p.hand) for p in self.players],
            "deck_size": len(self.deck),
        }

        return self.get_observation(), reward, terminated, truncated, info

    def get_reward(self, player_idx: int) -> float:
        """
        Terminal payoff for any player, win or lose.

        `step()` can only pay the acting player, so the losers' terminal reward
        never appears in a step return. Trainers and search agents that want it
        can query it here once the game is done.

        Args:
            player_idx: Player to score

        Returns:
            Winner/loser payoff once the game has a winner, else 0.0
        """
        if not self.done or self.winner is None:
            return 0.0
        if self.reward_mode == "sparse":
            return 1.0 if player_idx == self.winner else -1.0
        return self.WIN_REWARD if player_idx == self.winner else self.LOSS_REWARD

    def is_game_over(self) -> bool:
        """Check if game is over."""
        return self.done or self._turn_count >= self.max_turns

    def copy(self) -> "Macao":
        """
        Return an independent copy of the current position.

        Bypasses __init__ so no cards are dealt and no shuffling happens.

        Returns:
            A game whose state matches this one but shares no mutable objects
        """
        clone = object.__new__(type(self))

        # Game / CardGame state
        clone.num_players = self.num_players
        clone.current_player_idx = self.current_player_idx
        clone.done = self.done
        clone.winner = self.winner
        clone._turn_count = self._turn_count
        clone._history = list(self._history)
        clone.deck = self.deck.copy()

        clone.players = []
        for player in self.players:
            clone_player = Player(
                player_id=player.player_id,
                name=player.name,
                is_agent=player.is_agent,
            )
            clone_player.hand = _clone_cards(player.hand)
            clone_player.score = player.score
            clone.players.append(clone_player)

        # Macao state
        clone.max_turns = self.max_turns
        clone.reward_mode = self.reward_mode
        # Same RNG state, own RNG object: the clone's future shuffles match the
        # original's without either being able to advance the other.
        clone._rng = random.Random()
        clone._rng.setstate(self._rng.getstate())
        clone.discard_pile = _clone_cards(self.discard_pile)
        clone.requested_suit = self.requested_suit
        clone.requested_rank = self.requested_rank
        clone.draw_penalty = self.draw_penalty
        clone.skip_next = self.skip_next
        clone.pending_declaration = self.pending_declaration

        return clone

    def determinize(
        self,
        observer_idx: int = 0,
        rng: Optional[Any] = None,
    ) -> "Macao":
        """
        Return a copy with the opponents' hands and the deck re-dealt at random.

        The observer sees its own hand, the discard pile and everyone's hand
        *sizes*, but not which cards the opponents hold nor the deck order. This
        re-deals everything it cannot see, keeping hand sizes intact, so search
        agents plan over a state they could have deduced rather than the truth.

        Note this ignores what could be inferred from the play so far, e.g. an
        opponent who drew rather than played probably has no matching suit.

        Args:
            observer_idx: Player whose knowledge the sample must stay consistent with
            rng: `random.Random` instance to draw from (None for the global one)

        Returns:
            A game the observer cannot distinguish from this one
        """
        rng = rng or random
        clone = self.copy()

        unseen: list[Card] = []
        for idx, player in enumerate(clone.players):
            if idx != observer_idx:
                unseen.extend(player.hand)
        unseen.extend(clone.deck.cards)

        rng.shuffle(unseen)

        pos = 0
        for idx, player in enumerate(clone.players):
            if idx != observer_idx:
                hand_size = len(player.hand)
                player.hand = unseen[pos:pos + hand_size]
                for card in player.hand:
                    card.face_up = True
                pos += hand_size

        clone.deck.cards = unseen[pos:]
        for card in clone.deck.cards:
            card.face_up = False

        return clone

    def render(self) -> str:
        """Render game state as string."""
        lines = [f"=== Macao ({self.num_players} players) ===", ""]

        # Discard pile
        top_card = self.discard_pile[-1] if self.discard_pile else None
        lines.append(f"Discard: {top_card} (pile: {len(self.discard_pile)} cards)")
        lines.append(f"Deck: {len(self.deck)} cards")

        if self.requested_suit:
            lines.append(f"Requested suit: {self.requested_suit.symbol}")
        if self.requested_rank:
            lines.append(f"Requested rank: {self.requested_rank.symbol}")
        if self.pending_declaration:
            lines.append(f"Awaiting {self.pending_declaration} declaration")
        if self.draw_penalty > 0:
            lines.append(f"Draw penalty: {self.draw_penalty}")

        lines.append("")

        # Players
        for i, player in enumerate(self.players):
            marker = ">> " if i == self.current_player_idx else "   "
            hand_str = " ".join(str(c) for c in player.hand) if player.is_agent else f"[{len(player.hand)} cards]"
            lines.append(f"{marker}Player {i}: {hand_str}")

        lines.append("")
        lines.append(f"Turn: {self._turn_count}")

        return "\n".join(lines)

    def action_to_string(self, action: int) -> str:
        """Convert action to readable string."""
        if action == 52:
            return "Draw card(s)"
        elif action == 53:
            return "Pass"
        elif action < 52:
            card = Card.from_index(action)
            return f"Play {card}"
        elif self.SUIT_DECLARATION_OFFSET <= action < self.RANK_DECLARATION_OFFSET:
            suit = Suit(action - self.SUIT_DECLARATION_OFFSET)
            return f"Declare suit {suit.symbol}"
        elif action < self.RANK_DECLARATION_OFFSET + len(self.DECLARABLE_RANKS):
            rank = self.DECLARABLE_RANKS[action - self.RANK_DECLARATION_OFFSET]
            return f"Declare rank {rank.symbol}"
        return f"Action {action}"
