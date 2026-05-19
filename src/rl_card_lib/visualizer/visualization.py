"""Visualization utilities for card games."""

from typing import Optional
from rl_card_lib.cardgames.card import Card


def render_cards(cards: list[Card], hidden: bool = False) -> str:
    """
    Render a list of cards as a string.

    Args:
        cards: List of cards to render
        hidden: If True, show all cards as hidden

    Returns:
        String representation of cards
    """
    if not cards:
        return "[ ]"

    if hidden:
        return " ".join(["[??]" for _ in cards])

    return " ".join(str(card) for card in cards)


def render_tableau(
    piles: list[list[Card]],
    max_display_height: Optional[int] = None
) -> str:
    """
    Render tableau piles vertically.

    Args:
        piles: List of card piles
        max_display_height: Maximum rows to display

    Returns:
        String representation of tableau
    """
    if not piles:
        return ""

    # Find max height
    max_height = max(len(pile) for pile in piles) if any(piles) else 0

    if max_display_height:
        max_height = min(max_height, max_display_height)

    lines = []

    # Header
    header = " ".join(f"  {i+1}  " for i in range(len(piles)))
    lines.append(header)
    lines.append("-" * len(header))

    # Cards
    for row in range(max_height):
        row_str = ""
        for pile in piles:
            if row < len(pile):
                row_str += f"{pile[row]} "
            else:
                row_str += "      "
        lines.append(row_str)

    return "\n".join(lines)


def create_simple_board_view(game_state: dict) -> str:
    """
    Create a simple text board view from game state dict.

    Args:
        game_state: Dictionary containing game state

    Returns:
        Formatted board string
    """
    lines = []

    for key, value in game_state.items():
        if isinstance(value, list) and value and isinstance(value[0], Card):
            lines.append(f"{key}: {render_cards(value)}")
        elif isinstance(value, list) and value and isinstance(value[0], list):
            lines.append(f"{key}:")
            lines.append(render_tableau(value))
        else:
            lines.append(f"{key}: {value}")

    return "\n".join(lines)
