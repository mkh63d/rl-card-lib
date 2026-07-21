"""Budgeted perfect-information solvability search for Klondike deals.

Win rates on Klondike are only comparable when the denominator is known:
roughly 80% of deals are winnable with perfect play, so "43% of all deals" and
"43% of solvable deals" are very different results. This module classifies a
deal so win rates can be reported over solvable deals only.

The search sees the face-down cards (it works on full copies of the game), so
what it decides is *perfect-information* solvability: could a player who knew
every card win this deal. That is the standard denominator in the solitaire
literature, and an upper bound on what any honest player can achieve.
"""

from typing import Optional

from rl_card_lib.games.klondike import KlondikeSolitaire

#: Move-ordering buckets, tried in this order: foundation moves first (they are
#: pure progress), then everything else, then the draw, which never progresses
#: the board by itself.
_FOUNDATION_ACTIONS = range(8, 19)


def _position_key(game: KlondikeSolitaire) -> tuple:
    """
    Hashable identity of a position, for the transposition table.

    Deliberately excludes the pass counter and turn count: two positions with
    identical cards are identical for solvability when passes are unlimited,
    and folding them together is what keeps draw-recycle cycles finite. With a
    finite max_passes the remaining passes do change what is reachable, so
    then they are part of the key.

    Args:
        game: Position to fingerprint

    Returns:
        A tuple usable as a dict/set key
    """
    piles: list[tuple] = []
    for pile in game.tableaux:
        piles.append(tuple((int(c.suit), int(c.rank), c.face_up) for c in pile))
    for pile in game.foundations:
        piles.append(tuple((int(c.suit), int(c.rank)) for c in pile))
    piles.append(tuple((int(c.suit), int(c.rank)) for c in game.stock))
    piles.append(tuple((int(c.suit), int(c.rank)) for c in game.waste))
    if game.max_passes is not None:
        piles.append(("passes", game.passes))
    return tuple(piles)


def _ordered_actions(game: KlondikeSolitaire) -> list[int]:
    """
    Legal actions, most promising first.

    Foundation moves are always tried first and the draw last; between them,
    tableau moves that reveal a face-down card beat ones that do not. Good
    ordering is what lets a depth-first search find wins inside a small budget.

    Args:
        game: Position to move in

    Returns:
        Legal actions sorted for the search
    """
    def priority(action: int) -> int:
        if action in _FOUNDATION_ACTIONS:
            return 0
        if action == 0:
            return 3
        if action >= 19:
            source = game.tableaux[(action - 19) // 7]
            for idx, card in enumerate(source):
                if card.face_up:
                    return 1 if idx > 0 and not source[idx - 1].face_up else 2
        return 2

    return sorted(game.get_legal_actions(), key=priority)


def solve_klondike(
    game: KlondikeSolitaire,
    max_nodes: int = 50_000,
) -> Optional[bool]:
    """
    Decide whether the deal is winnable with perfect information.

    Runs a depth-first search over full game states, pruning positions already
    seen, until it wins, exhausts the reachable state space, or spends the node
    budget. The input game is never mutated; the search runs on copies.

    Winnable deals usually resolve within a few hundred nodes thanks to the
    move ordering; the budget is mostly spent on deals that end up None. Expect
    roughly a millisecond per node, so the default budget can take tens of
    seconds per undecided deal.

    Args:
        game: Position to solve, typically freshly reset
        max_nodes: Positions to expand before giving up

    Returns:
        True if a winning line exists from this position, False if the search
        space was exhausted without one (proof of unsolvability), None if the
        budget ran out first (unknown)
    """
    root = game.copy()
    if root._check_win():
        return True

    visited = {_position_key(root)}
    stack = [root]
    expanded = 0

    while stack:
        if expanded >= max_nodes:
            return None
        position = stack.pop()
        expanded += 1

        for action in reversed(_ordered_actions(position)):
            child = position.copy()
            _, _, terminated, _, _ = child.step(action)
            if child.winner == 0:
                return True
            if terminated:
                continue
            key = _position_key(child)
            if key not in visited:
                visited.add(key)
                stack.append(child)

    return False
