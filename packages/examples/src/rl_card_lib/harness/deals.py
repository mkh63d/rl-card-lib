"""The train/test deal split, declared once for every experiment.

A deal is identified by the integer seed handed to `game.reset(seed=...)`, which
reseeds the game's private `random.Random` and therefore fixes the shuffle
exactly (see `KlondikeSolitaire.reset` / `Macao.reset`). Two disjoint integer
ranges therefore give two disjoint, reproducible pools of deals:

    TRAIN  seeds       0 ..   9_999   (10 000 deals)
    TEST   seeds 100_000 .. 100_199   (   200 deals)

The gap between them is deliberate. Disjointness is a property of the ranges
rather than of a sampling procedure, so it holds no matter how many training
episodes are run and needs no bookkeeping to stay true.

Evaluation always takes the *first* n TEST seeds (`evaluation_seeds(n)`), never
a random sample, so every agent is scored on identical deals and raising the
episode count yields a superset of the deals already measured -- the comparison
is paired, and one agent's number can be checked against another's deal by deal.
"""

from __future__ import annotations

from typing import Optional

TRAIN_SEED_START = 0
TRAIN_SEED_END = 10_000          # exclusive
TEST_SEED_START = 100_000
TEST_SEED_END = 100_200          # exclusive

#: Deals a training run may draw from. A range, so the bounds stay readable at
#: a glance and set operations against TEST_SEEDS are cheap.
TRAIN_SEEDS = range(TRAIN_SEED_START, TRAIN_SEED_END)

#: The held-out deals every evaluation and baseline measurement is scored on.
TEST_SEEDS = list(range(TEST_SEED_START, TEST_SEED_END))


def pools_are_disjoint() -> bool:
    """Whether the two ranges can never touch."""
    return TRAIN_SEED_END <= TEST_SEED_START


def evaluation_seeds(episodes: Optional[int] = None) -> list[int]:
    """The first `episodes` held-out deals (the whole pool when None).

    Args:
        episodes: How many deals to evaluate on

    Returns:
        A prefix of TEST_SEEDS

    Raises:
        ValueError: if more deals are asked for than the pool holds. Silently
            wrapping would score an agent on some deals twice and quietly break
            the "every agent sees the same deals" guarantee; raising says to
            widen TEST_SEED_END instead.
    """
    if episodes is None:
        return list(TEST_SEEDS)
    if episodes < 0:
        raise ValueError(f"episodes must not be negative, got {episodes}")
    if episodes > len(TEST_SEEDS):
        raise ValueError(
            f"asked for {episodes} evaluation deals but the TEST pool holds "
            f"{len(TEST_SEEDS)} (seeds {TEST_SEED_START}..{TEST_SEED_END - 1}). "
            "Raise TEST_SEED_END in rl_card_lib.harness.deals to widen it."
        )
    return TEST_SEEDS[:episodes]


__all__ = [
    "TEST_SEEDS",
    "TEST_SEED_END",
    "TEST_SEED_START",
    "TRAIN_SEEDS",
    "TRAIN_SEED_END",
    "TRAIN_SEED_START",
    "evaluation_seeds",
    "pools_are_disjoint",
]
