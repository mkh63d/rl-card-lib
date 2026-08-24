"""The train/test deal split used by every experiment in thesis_notes/.

A deal is identified by the integer seed handed to `game.reset(seed=...)`, which
reseeds the game's private `random.Random` and therefore fixes the shuffle
exactly (see `KlondikeSolitaire.reset` / `Macao.reset`). Two disjoint integer
ranges therefore give two disjoint, reproducible pools of deals.

    TRAIN          seeds      0 ..   9_999   (10 000 deals)
    TEST           seeds 100_000 .. 100_199  (   200 deals)
    TEST_SOLVABLE  the subset of TEST that the perfect-information solver
                   proves winnable (Klondike only)

TEST_SOLVABLE is a *subset of TEST*, not a third range: that keeps the
solvable-deal benchmark on the same held-out deals as everything else, so
"solve rate over winnable deals" and "cards to foundation over all deals" are
two views of one evaluation set rather than two experiments.
"""

from __future__ import annotations

import json
import os

TRAIN_SEED_START = 0
TRAIN_SEED_END = 10_000          # exclusive
TEST_SEED_START = 100_000
TEST_SEED_END = 100_200          # exclusive

TRAIN_SEEDS = range(TRAIN_SEED_START, TRAIN_SEED_END)
TEST_SEEDS = list(range(TEST_SEED_START, TEST_SEED_END))

RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "raw")
SOLVABLE_CACHE = os.path.join(RAW_DIR, "klondike_test_solvable.json")

#: Node budget per deal when classifying TEST deals. Winnable deals resolve in
#: a few hundred nodes; this budget is spent almost entirely on deals that end
#: up unwinnable or undecided.
SOLVE_NODES = 20_000


def train_seeds_disjoint_from_test() -> bool:
    """Sanity check the two ranges never touch."""
    return TRAIN_SEED_END <= TEST_SEED_START


def klondike_test_solvable(force: bool = False, verbose: bool = True) -> dict:
    """Classify every TEST deal with the perfect-information solver.

    Returns a dict with `solvable`, `unsolvable` and `undecided` seed lists.
    Cached to raw/klondike_test_solvable.json, since the scan costs minutes.
    """
    if not force and os.path.exists(SOLVABLE_CACHE):
        with open(SOLVABLE_CACHE, "r", encoding="utf-8") as handle:
            return json.load(handle)

    from rl_card_lib.games.klondike import KlondikeSolitaire
    from rl_card_lib.games.klondike_solver import solve_klondike

    game = KlondikeSolitaire()
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
