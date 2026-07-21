"""Klondike Solitaire game implementation."""

import random
from typing import Any, Optional

import numpy as np

from rl_card_lib.cardgames import Card, Suit, Rank, Deck, CardGame


def _clone_cards(cards: list[Card]) -> list[Card]:
    """Copy a pile so the clone's cards can be flipped without touching the original."""
    return [Card(c.suit, c.rank, c.face_up) for c in cards]


class KlondikeSolitaire(CardGame):
    """
    Klondike Solitaire card game.

    The classic single-player patience card game. The goal is to move all 52 cards
    to the four foundation piles (one per suit, built Ace to King).

    Game Layout:
    - 7 tableau piles (columns of cards, top card face-up)
    - 4 foundation piles (one per suit, built A-K)
    - Stock pile (remaining cards to draw from)
    - Waste pile (cards drawn from stock)

    Actions:
    - Draw from stock to waste (recycles the waste when the stock is empty)
    - Move card(s) between tableaux
    - Move card from tableau/waste to foundation

    Rewards (shaped mode): +1.0 per card to a foundation, +0.2 per face-down
    card revealed (+0.1 for the reveal on a foundation move), +0.1 for playing
    off the waste, -0.1 for recycling the waste, -0.01 per move, and
    LOSS_REWARD when the deal dies. Non-revealing tableau moves pay nothing on
    purpose: they are reversible, so any positive payment is farmable. Sparse
    mode pays only the win/loss terminals.
    """

    # Action encoding:
    # 0: Draw from stock
    # 1-7: Move waste to tableau
    # 8-11: Move waste to foundation
    # 12-18: Move tableau top to foundation
    # 19-67: Move between tableaux (19 + from_pile * 7 + to_pile)
    #
    # The tableau block names only the two piles, never the card. That is not a
    # restriction: the face-up section of a tableau pile is always one strictly
    # descending, alternating-color run, so for a given destination at most one
    # card in the run can legally move there (its rank and color are forced by
    # the destination's top card, and empty piles accept only the run's single
    # king). The card the game picks is therefore the only card it could pick.
    #
    # 68 = 19 + 6 * 7 + 6 + 1 is the tightest bound the encoding allows; the
    # seven from == to diagonal entries are wasted but keeping the formula
    # simple is worth seven dead outputs (it used to be 132 dead ones).

    MAX_ACTIONS = 68

    #: Reward paid on the terminal step of a lost deal, in both reward modes.
    #: Without it a stuck deal is indistinguishable from running out of time.
    LOSS_REWARD = -1.0

    def __init__(
        self,
        draw_count: int = 1,
        max_passes: Optional[int] = None,
        reward_mode: str = "shaped",
        seed: Optional[int] = None,
    ):
        """
        Initialize Klondike Solitaire.

        Args:
            draw_count: Number of cards to draw from stock (1 or 3)
            max_passes: Maximum passes through the deck (None for unlimited).
                Note that with unlimited passes a dead deal never runs out of
                legal moves, so it can only end by truncation, never as a loss.
            reward_mode: "shaped" pays per-move progress rewards (foundations,
                reveals) plus a small step cost; "sparse" pays only +1 for a won
                deal and LOSS_REWARD for a dead one. Sparse cannot be farmed and
                needs no tuning, at the cost of a much weaker learning signal.
            seed: Seed for this game's private RNG, used for shuffling. The
                process-wide RNG is never touched.
        """
        super().__init__(num_players=1)

        if reward_mode not in ("shaped", "sparse"):
            raise ValueError(
                f"reward_mode must be 'shaped' or 'sparse', got {reward_mode!r}"
            )

        self.draw_count = draw_count
        self.max_passes = max_passes
        self.reward_mode = reward_mode
        self._rng = random.Random(seed)

        # Game state
        self.tableaux: list[list[Card]] = [[] for _ in range(7)]
        self.foundations: list[list[Card]] = [[] for _ in range(4)]  # One per suit
        self.stock: list[Card] = []
        self.waste: list[Card] = []
        self.passes: int = 0

        self.reset()

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """
        Reset the game to a freshly dealt state.

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

        # Clear all piles
        self.tableaux = [[] for _ in range(7)]
        self.foundations = [[] for _ in range(4)]
        self.stock = []
        self.waste = []
        self.passes = 0
        self.done = False
        self.winner = None
        self._turn_count = 0

        # Deal to tableaux
        for i in range(7):
            for j in range(i, 7):
                card = self.deck.draw_one(face_up=False)
                self.tableaux[j].append(card)
            # Flip top card face up
            self.tableaux[i][-1].face_up = True

        # Remaining cards go to stock
        self.stock = list(self.deck.cards)
        for card in self.stock:
            card.face_up = False
        self.deck.cards.clear()

        return self.get_observation()

    def get_observation(self) -> np.ndarray:
        """
        Get current state as observation vector.

        Returns:
            Numpy array representing the game state
        """
        # Encode card locations
        # For each card: [in_tableau, in_foundation, in_stock_or_waste, face_up]
        # Plus: foundation top ranks (4), tableau pile sizes (7)

        observation = []

        # Encode all 52 cards' locations and visibility
        card_info = {}

        # Stock cards
        for card in self.stock:
            card_info[card.to_index()] = [0, 0, 1, 0]  # In stock, face down

        # Waste cards
        for card in self.waste:
            card_info[card.to_index()] = [0, 0, 1, 1]  # In waste, face up

        # Tableau cards
        for pile in self.tableaux:
            for card in pile:
                card_info[card.to_index()] = [1, 0, 0, 1 if card.face_up else 0]

        # Foundation cards
        for pile in self.foundations:
            for card in pile:
                card_info[card.to_index()] = [0, 1, 0, 1]  # In foundation, face up

        # Add card info in order
        for i in range(52):
            if i in card_info:
                observation.extend(card_info[i])
            else:
                observation.extend([0, 0, 0, 0])

        # Foundation top ranks (normalized 0-1)
        for pile in self.foundations:
            if pile:
                observation.append(pile[-1].rank / 13.0)
            else:
                observation.append(0.0)

        # Tableau pile sizes (normalized)
        for pile in self.tableaux:
            observation.append(len(pile) / 19.0)  # Max possible pile size ~19

        # Waste visible count
        observation.append(len(self.waste) / 24.0)

        # Stock count
        observation.append(len(self.stock) / 24.0)

        return np.array(observation, dtype=np.float32)

    def get_observation_shape(self) -> tuple[int, ...]:
        """Get observation shape."""
        return (52 * 4 + 4 + 7 + 2,)  # 221 features

    def get_action_space_size(self) -> int:
        """Get total number of actions."""
        return self.MAX_ACTIONS

    def get_legal_actions(self) -> list[int]:
        """Get list of legal action indices."""
        legal = []

        # Action 0: Draw from stock, or flip the waste back to the stock. The
        # recycle is only legal while passes remain; without this check a deal
        # that ran out of passes could still "draw" forever as a no-op.
        if self.stock or (self.waste and self._can_recycle()):
            legal.append(0)

        # Actions 1-7: Move waste top to tableau 1-7
        if self.waste:
            waste_card = self.waste[-1]
            for i in range(7):
                if self._can_place_on_tableau(waste_card, i):
                    legal.append(1 + i)

        # Actions 8-11: Move waste top to foundation 1-4
        if self.waste:
            waste_card = self.waste[-1]
            for i in range(4):
                if self._can_place_on_foundation(waste_card, i):
                    legal.append(8 + i)

        # Actions 12-18: Move tableau top to foundation
        for i in range(7):
            if self.tableaux[i]:
                top_card = self.tableaux[i][-1]
                if top_card.face_up:
                    for j in range(4):
                        if self._can_place_on_foundation(top_card, j):
                            legal.append(12 + i)
                            break

        # Actions 19+: Move between tableaux
        # Encode as: 19 + from_pile * 7 * 13 + to_pile * 13 + num_cards
        # Simplified: 19 + from_pile * 7 + to_pile (move entire face-up sequence)
        action_offset = 19
        for from_pile in range(7):
            if not self.tableaux[from_pile]:
                continue

            # Find first face-up card
            face_up_start = -1
            for idx, card in enumerate(self.tableaux[from_pile]):
                if card.face_up:
                    face_up_start = idx
                    break

            if face_up_start == -1:
                continue

            # Try moving from each face-up position
            for card_idx in range(face_up_start, len(self.tableaux[from_pile])):
                moving_card = self.tableaux[from_pile][card_idx]
                for to_pile in range(7):
                    if from_pile == to_pile:
                        continue
                    if self._can_place_on_tableau(moving_card, to_pile):
                        # Encode action
                        action = action_offset + from_pile * 7 + to_pile
                        if action not in legal:
                            legal.append(action)

        return legal

    def _can_place_on_tableau(self, card: Card, pile_idx: int) -> bool:
        """Check if card can be placed on tableau pile."""
        pile = self.tableaux[pile_idx]

        if not pile:
            # Only King can go on empty pile
            return card.rank == Rank.KING

        top_card = pile[-1]
        if not top_card.face_up:
            return False

        # Must be opposite color and one rank lower
        if card.color == top_card.color:
            return False
        if card.rank != top_card.rank - 1:
            return False

        return True

    def _can_place_on_foundation(self, card: Card, foundation_idx: int) -> bool:
        """Check if card can be placed on foundation."""
        pile = self.foundations[foundation_idx]

        if not pile:
            # Only Ace can start foundation
            return card.rank == Rank.ACE

        top_card = pile[-1]

        # Must be same suit and one rank higher
        if card.suit != top_card.suit:
            return False
        if card.rank != top_card.rank + 1:
            return False

        return True

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute an action."""
        self._turn_count += 1
        reward = 0.0

        if action == 0:
            # Draw from stock or reset
            reward = self._draw_from_stock()
        elif 1 <= action <= 7:
            # Move waste to tableau
            reward = self._move_waste_to_tableau(action - 1)
        elif 8 <= action <= 11:
            # Move waste to foundation
            reward = self._move_waste_to_foundation(action - 8)
        elif 12 <= action <= 18:
            # Move tableau top to foundation
            reward = self._move_tableau_to_foundation(action - 12)
        elif action >= 19:
            # Move between tableaux
            relative_action = action - 19
            from_pile = relative_action // 7
            to_pile = relative_action % 7
            reward = self._move_tableau_to_tableau(from_pile, to_pile)

        # Small penalty for each move to encourage efficiency
        reward -= 0.01

        if self.reward_mode == "sparse":
            reward = 0.0

        won = self._check_win()
        if won:
            if self.reward_mode == "sparse":
                reward = 1.0
        elif not self.get_legal_actions():
            # No legal moves left: the deal is dead. Ending it here (instead of
            # only ever truncating) gives the agent an actual loss signal and
            # keeps hundreds of pointless post-mortem moves out of the replay
            # buffer. Reachable only with a finite max_passes, since otherwise
            # draw/recycle stays legal forever.
            self.done = True
            reward += self.LOSS_REWARD

        terminated = self.done
        truncated = False

        info = {
            "foundations": [len(f) for f in self.foundations],
            "cards_in_foundation": sum(len(f) for f in self.foundations),
        }

        return self.get_observation(), reward, terminated, truncated, info

    def _can_recycle(self) -> bool:
        """Whether flipping the waste back into the stock is still allowed."""
        return self.max_passes is None or self.passes + 1 < self.max_passes

    def _draw_from_stock(self) -> float:
        """Draw card(s) from stock to waste."""
        if self.stock:
            for _ in range(min(self.draw_count, len(self.stock))):
                card = self.stock.pop()
                card.face_up = True
                self.waste.append(card)
            return 0.0
        elif self.waste:
            if not self._can_recycle():
                # Unreachable through legal play: get_legal_actions() withholds
                # action 0 once the passes run out.
                return -0.5
            # Reset: move waste back to stock
            self.passes += 1
            while self.waste:
                card = self.waste.pop()
                card.face_up = False
                self.stock.append(card)
            return -0.1  # Small penalty for recycling
        return 0.0

    def _move_waste_to_tableau(self, pile_idx: int) -> float:
        """Move top waste card to tableau."""
        if not self.waste:
            return -0.5

        card = self.waste[-1]
        if not self._can_place_on_tableau(card, pile_idx):
            return -0.5

        self.waste.pop()
        self.tableaux[pile_idx].append(card)
        return 0.1  # Small reward for building tableaux

    def _move_waste_to_foundation(self, foundation_idx: int) -> float:
        """Move top waste card to foundation."""
        if not self.waste:
            return -0.5

        card = self.waste[-1]
        # Find correct foundation for suit
        target_foundation = int(card.suit)

        if not self._can_place_on_foundation(card, target_foundation):
            return -0.5

        self.waste.pop()
        self.foundations[target_foundation].append(card)
        return 1.0  # Good reward for foundation progress

    def _move_tableau_to_foundation(self, pile_idx: int) -> float:
        """Move top tableau card to foundation."""
        pile = self.tableaux[pile_idx]
        if not pile or not pile[-1].face_up:
            return -0.5

        card = pile[-1]
        target_foundation = int(card.suit)

        if not self._can_place_on_foundation(card, target_foundation):
            return -0.5

        pile.pop()
        self.foundations[target_foundation].append(card)

        # Flip newly exposed card
        if pile and not pile[-1].face_up:
            pile[-1].face_up = True
            return 1.1  # Bonus for revealing card

        return 1.0

    def _move_tableau_to_tableau(self, from_pile: int, to_pile: int) -> float:
        """Move cards between tableaux."""
        if from_pile == to_pile:
            return -0.5

        source = self.tableaux[from_pile]
        if not source:
            return -0.5

        # Find first face-up card that can move
        move_from_idx = -1
        for idx, card in enumerate(source):
            if card.face_up and self._can_place_on_tableau(card, to_pile):
                move_from_idx = idx
                break

        if move_from_idx == -1:
            return -0.5

        # Move cards
        cards_to_move = source[move_from_idx:]
        self.tableaux[from_pile] = source[:move_from_idx]
        self.tableaux[to_pile].extend(cards_to_move)

        # Only a reveal pays. Tableau shuffling used to earn 0.05 per card
        # against a 0.01 step cost, and a non-revealing move is reversible, so
        # moving a card back and forth was unbounded free reward: agents that
        # optimized the reward farmed that loop instead of playing solitaire
        # (measured: 139 of 150 moves, 5x fewer cards up than random play).
        # At 0.0 the step cost makes every pointless shuffle a small net loss.
        reward = 0.0
        if self.tableaux[from_pile] and not self.tableaux[from_pile][-1].face_up:
            self.tableaux[from_pile][-1].face_up = True
            reward += 0.2  # Bonus for revealing card

        return reward

    def _check_win(self) -> bool:
        """Check if all cards are in foundations."""
        total = sum(len(f) for f in self.foundations)
        if total == 52:
            self.done = True
            self.winner = 0
            return True
        return False

    def is_game_over(self) -> bool:
        """Check if game is over."""
        if self.done:
            return True

        # Game is stuck if no legal moves
        return len(self.get_legal_actions()) == 0

    def copy(self) -> "KlondikeSolitaire":
        """
        Return an independent copy of the current position.

        Bypasses __init__ so no cards are dealt and no shuffling happens.

        Returns:
            A game whose state matches this one but shares no mutable objects
        """
        clone = object.__new__(type(self))

        # Game / CardGame state
        clone.num_players = self.num_players
        clone.players = []  # Klondike is single-player and never fills this
        clone.current_player_idx = self.current_player_idx
        clone.done = self.done
        clone.winner = self.winner
        clone._turn_count = self._turn_count
        clone._history = list(self._history)
        clone.deck = self.deck.copy()

        # Klondike state
        clone.draw_count = self.draw_count
        clone.max_passes = self.max_passes
        clone.reward_mode = self.reward_mode
        # Same RNG state, own RNG object: the clone's future shuffles match the
        # original's without either being able to advance the other.
        clone._rng = random.Random()
        clone._rng.setstate(self._rng.getstate())
        clone.tableaux = [_clone_cards(pile) for pile in self.tableaux]
        clone.foundations = [_clone_cards(pile) for pile in self.foundations]
        clone.stock = _clone_cards(self.stock)
        clone.waste = _clone_cards(self.waste)
        clone.passes = self.passes

        return clone

    def determinize(
        self,
        observer_idx: int = 0,
        rng: Optional[Any] = None,
    ) -> "KlondikeSolitaire":
        """
        Return a copy with the face-down cards shuffled among themselves.

        The player cannot see the face-down tableau cards or the stock, so every
        assignment of those cards to those slots is consistent with what has been
        observed. Search agents sample one instead of reading the real cards.

        Note this discards one thing a careful human would remember: stock cards
        seen on an earlier pass are treated as unknown again once recycled.

        Args:
            observer_idx: Unused, Klondike has a single player
            rng: `random.Random` instance to draw from (None for the global one)

        Returns:
            A game whose face-up cards match this one and whose hidden cards are re-dealt
        """
        rng = rng or random
        clone = self.copy()

        slots: list[tuple[list[Card], int]] = []
        hidden: list[Card] = []

        for pile in clone.tableaux:
            for idx, card in enumerate(pile):
                if not card.face_up:
                    slots.append((pile, idx))
                    hidden.append(card)

        for idx, card in enumerate(clone.stock):
            slots.append((clone.stock, idx))
            hidden.append(card)

        rng.shuffle(hidden)
        for (pile, idx), card in zip(slots, hidden):
            card.face_up = False
            pile[idx] = card

        return clone

    def render(self) -> str:
        """Render game state as string."""
        lines = ["=== Klondike Solitaire ===", ""]

        # Stock and Waste
        stock_str = f"[{len(self.stock)}]" if self.stock else "[ ]"
        waste_str = str(self.waste[-1]) if self.waste else "[ ]"
        lines.append(f"Stock: {stock_str}  Waste: {waste_str}")
        lines.append("")

        # Foundations
        found_str = "Foundations: "
        for i, pile in enumerate(self.foundations):
            if pile:
                found_str += f"{pile[-1]} "
            else:
                found_str += f"[{Suit(i).symbol}] "
        lines.append(found_str)
        lines.append("")

        # Tableaux
        lines.append("Tableaux:")
        max_height = max(len(pile) for pile in self.tableaux) if any(self.tableaux) else 0
        for row in range(max_height):
            row_str = ""
            for pile in self.tableaux:
                if row < len(pile):
                    row_str += f"{pile[row]} "
                else:
                    row_str += "     "
            lines.append(row_str)

        lines.append("")
        lines.append(f"Turn: {self._turn_count}")

        return "\n".join(lines)

    def action_to_string(self, action: int) -> str:
        """Convert action to readable string."""
        if action == 0:
            return "Draw from stock"
        elif 1 <= action <= 7:
            return f"Move waste to tableau {action}"
        elif 8 <= action <= 11:
            return f"Move waste to foundation {action - 7}"
        elif 12 <= action <= 18:
            return f"Move tableau {action - 11} top to foundation"
        else:
            relative_action = action - 19
            from_pile = relative_action // 7 + 1
            to_pile = relative_action % 7 + 1
            return f"Move from tableau {from_pile} to tableau {to_pile}"
