"""The train/test deal split used by every experiment in thesis_notes/.

A deal is identified by the integer seed handed to `game.reset(seed=...)`, which
reseeds the game's private `random.Random` and therefore fixes the shuffle
exactly (see `KlondikeSolitaire.reset` / `Macao.reset`). Two disjoint integer
ranges therefore give two disjoint, reproducible pools of deals.

    TRAIN          seeds      0 ..   9_999   (10 000 deals)
    TEST           seeds 100_000 .. 100_199  (   200 deals)
    TEST_SOLVABLE  the subset of TEST that the perfect-information solver
                   proves winnable (Klondike only)

The two ranges are no longer declared here: the library owns them in
`rl_card_lib.harness.deals` since the deal-pool support landed in `CardGameEnv`,
and two copies of the numbers would let the sweep and the library drift onto
different pools. This module re-exports them under the names the thesis text
uses and adds the one thing the library has no equivalent for -- TEST_SOLVABLE.

TEST_SOLVABLE is a *subset of TEST*, not a third range: that keeps the
solvable-deal benchmark on the same held-out deals as everything else, so
"solve rate over winnable deals" and "cards to foundation over all deals" are
two views of one evaluation set rather than two experiments. This is why it is
not `harness.solve_benchmark.curate_solvable_pool`, which scans upward for the
first N winnable seeds and so answers a different question.
"""

from __future__ import annotations

import json
import os

from rl_card_lib.harness.deals import (
    TEST_SEED_END,
    TEST_SEED_START,
    TEST_SEEDS,
    TRAIN_SEED_END,
    TRAIN_SEED_START,
    TRAIN_SEEDS,
    pools_are_disjoint,
)

RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "raw")
SOLVABLE_CACHE = os.path.join(RAW_DIR, "klondike_test_solvable.json")

#: Node budget per deal when classifying TEST deals. Winnable deals resolve in
#: a few hundred nodes; this budget is spent almost entirely on deals that end
#: up unwinnable or undecided. Deliberately above the library's own
#: KLONDIKE_POOL_SOLVE_NODES (10k): curating a pool can afford to skip an
#: undecided deal, but classifying a fixed pool cannot, so a larger budget here
#: buys a smaller `undecided` bucket.
SOLVE_NODES = 20_000

__all__ = [
    "SOLVE_NODES",
    "TEST_SEED_END",
    "TEST_SEED_START",
    "TEST_SEEDS",
    "TRAIN_SEED_END",
    "TRAIN_SEED_START",
    "TRAIN_SEEDS",
    "klondike_test_solvable",
    "train_seeds_disjoint_from_test",
]


def train_seeds_disjoint_from_test() -> bool:
    """Sanity check the two ranges never touch."""
    return pools_are_disjoint()


def klondike_test_solvable(force: bool = False, verbose: bool = True) -> dict:
    """Classify every TEST deal with the perfect-information solver.

    Returns a dict with `solvable`, `unsolvable` and `undecided` seed lists.
    Cached to raw/klondike_test_solvable.json, since the scan costs minutes.

    The game is built with the bundled `max_passes`, not the class default: the
    solver's transposition key includes the pass count whenever the limit is
    finite, so "winnable" is a different property under the two rules. A pool
    curated under unlimited passes would call deals winnable that the agents,
    who play the bundled game, can never win.
    """
    if not force and os.path.exists(SOLVABLE_CACHE):
        with open(SOLVABLE_CACHE, "r", encoding="utf-8") as handle:
            return json.load(handle)

    from rl_card_lib.games.klondike import KlondikeSolitaire
    from rl_card_lib.games.klondike_solver import solve_klondike
    from rl_card_lib.games.registration import KLONDIKE_MAX_PASSES

    game = KlondikeSolitaire(max_passes=KLONDIKE_MAX_PASSES)
    solvable, unsolvable, undecided = [], [], []
    for seed in TEST_SEEDS:
        game.reset(seed=seed)
        verdict = solve_klondike(game, max_nodes=SOLVE_NODES)
        if verdict is True:
            solvable.append(seed)
        elif verdict is False:
            unsolvable.append(seed)
        else:
            undecided.append(seed)
        if verbose and (seed - TEST_SEED_START + 1) % 25 == 0:
            print(f"  classified {seed - TEST_SEED_START + 1}/{len(TEST_SEEDS)}: "
                  f"{len(solvable)} solvable, {len(unsolvable)} unsolvable, "
                  f"{len(undecided)} undecided", flush=True)

    out = {
        "pool": "TEST",
        "seed_range": [TEST_SEED_START, TEST_SEED_END],
        "max_nodes": SOLVE_NODES,
        "max_passes": KLONDIKE_MAX_PASSES,
        "solvable": solvable,
        "unsolvable": unsolvable,
        "undecided": undecided,
    }
    os.makedirs(RAW_DIR, exist_ok=True)
    with open(SOLVABLE_CACHE, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2)
    return out


if __name__ == "__main__":
    print(f"TRAIN: {TRAIN_SEED_START}..{TRAIN_SEED_END - 1} ({len(TRAIN_SEEDS)} deals)")
    print(f"TEST:  {TEST_SEED_START}..{TEST_SEED_END - 1} ({len(TEST_SEEDS)} deals)")
    print(f"disjoint: {train_seeds_disjoint_from_test()}")
    data = klondike_test_solvable()
    print(f"TEST_SOLVABLE: {len(data['solvable'])} solvable, "
          f"{len(data['unsolvable'])} proven unsolvable, "
          f"{len(data['undecided'])} undecided at {data['max_nodes']} nodes")
