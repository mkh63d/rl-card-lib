"""Macao (Crazy Eights variant) game implementation."""

from typing import Optional
import numpy as np

from rl_card_lib.core.card import Card, Suit, Rank
from rl_card_lib.core.deck import Deck
from rl_card_lib.core.player import Player
from rl_card_lib.core.game import CardGame


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
    """
    
    MAX_ACTIONS = 60  # 52 cards + special actions (draw, pass, suit choices)
    HAND_SIZE = 5
    
    def __init__(self, num_players: int = 2, max_turns: int = 200):
        """
        Initialize Macao game.
        
        Args:
            num_players: Number of players (2-4)
            max_turns: Maximum turns before draw
        """
        super().__init__(num_players=num_players)
        
        self.max_turns = max_turns
        
        # Game state
        self.discard_pile: list[Card] = []
        self.requested_suit: Optional[Suit] = None
        self.requested_rank: Optional[Rank] = None
        self.draw_penalty: int = 0  # Accumulated draw penalty
        self.skip_next: bool = False
        
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset game to initial state."""
        self.deck = Deck()
        self.deck.shuffle()
        
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
        # 52 (hand) + 52 (top card) + 4 (suit) + 13 (rank) + 1 (penalty) + 
        # (num_players-1) (opponent hands) + 1 (deck)
        return (52 + 52 + 4 + 13 + 1 + (self.num_players - 1) + 1,)
    
    def get_action_space_size(self) -> int:
        """Get action space size."""
        return self.MAX_ACTIONS
    
    def get_legal_actions(self) -> list[int]:
        """Get list of legal action indices."""
        legal = []
        current_player = self.get_current_player()
        top_card = self.discard_pile[-1] if self.discard_pile else None
        
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
        """Execute an action."""
        self._turn_count += 1
        reward = 0.0
        current_player = self.get_current_player()
        
        # Action 52: Draw card(s)
        if action == 52:
            if self.draw_penalty > 0:
                # Draw penalty cards
                draw_count = min(self.draw_penalty, len(self.deck))
                if draw_count > 0:
                    cards = self.deck.draw(draw_count, face_up=True)
                    current_player.add_cards(cards)
                self.draw_penalty = 0
                reward = -0.1 * draw_count  # Penalty for drawing
            else:
                # Draw one card
                if len(self.deck) > 0:
                    card = self.deck.draw_one(face_up=True)
                    current_player.add_card(card)
                    reward = -0.05  # Small penalty for drawing
            
            self.next_player()
        
        # Action 53: Pass
        elif action == 53:
            self.skip_next = False
            self.next_player()
            reward = -0.1
        
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
                # Invalid action
                reward = -1.0
                self.next_player()
            else:
                current_player.remove_card(card_to_play)
                self.discard_pile.append(card_to_play)
                
                # Clear requests
                self.requested_suit = None
                self.requested_rank = None
                
                # Apply card effects
                if card_to_play.rank == Rank.FOUR:
                    self.skip_next = True
                    reward = 0.2
                
                elif self._is_draw_card(card_to_play):
                    self.draw_penalty += self._get_draw_penalty(card_to_play)
                    reward = 0.3
                
                elif card_to_play.rank == Rank.ACE:
                    # Choose most common suit in hand
                    suit_counts = {}
                    for c in current_player.hand:
                        suit_counts[c.suit] = suit_counts.get(c.suit, 0) + 1
                    if suit_counts:
                        self.requested_suit = max(suit_counts, key=suit_counts.get)
                    reward = 0.2
                
                elif card_to_play.rank == Rank.JACK:
                    # Choose most common rank in hand (non-special)
                    valid_ranks = [r for r in Rank if r not in 
                                   (Rank.TWO, Rank.THREE, Rank.FOUR, Rank.JACK, Rank.ACE, Rank.KING)]
                    rank_counts = {}
                    for c in current_player.hand:
                        if c.rank in valid_ranks:
                            rank_counts[c.rank] = rank_counts.get(c.rank, 0) + 1
                    if rank_counts:
                        self.requested_rank = max(rank_counts, key=rank_counts.get)
                    else:
                        self.requested_rank = Rank.SEVEN  # Default
                    reward = 0.2
                
                else:
                    reward = 0.1  # Normal card play
                
                # Check for win
                if len(current_player.hand) == 0:
                    self.done = True
                    self.winner = self.current_player_idx
                    if current_player.is_agent:
                        reward = 10.0  # Big reward for winning
                    else:
                        reward = -5.0  # Penalty if opponent wins
                else:
                    self.next_player()
        
        # Check for game end conditions
        terminated = self.done
        truncated = self._turn_count >= self.max_turns
        
        if truncated and not terminated:
            # Draw - penalize based on hand sizes
            agent_hand = len(self.players[0].hand)
            opponent_hand = len(self.players[1].hand) if self.num_players > 1 else 0
            reward = 0.1 * (opponent_hand - agent_hand)  # Reward having fewer cards
        
        # Reshuffle discard if deck empty
        if len(self.deck) == 0 and len(self.discard_pile) > 1:
            top_card = self.discard_pile.pop()
            self.deck.cards = self.discard_pile
            self.deck.shuffle()
            for card in self.deck.cards:
                card.face_up = False
            self.discard_pile = [top_card]
        
        info = {
            "current_player": self.current_player_idx,
            "hand_sizes": [len(p.hand) for p in self.players],
            "deck_size": len(self.deck),
        }
        
        return self.get_observation(), reward, terminated, truncated, info
    
    def is_game_over(self) -> bool:
        """Check if game is over."""
        return self.done or self._turn_count >= self.max_turns
    
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
        return f"Action {action}"
