## Summary

Neither the training loop nor the "fixed-deal" evaluation protocol controls
which deal is played. Training deals come from OS entropy, and the evaluation
functions reseed the *global* RNGs, which the games never read. The consequence
is that **no reported number in `results/` is reproducible**, and every agent
and baseline was scored on a **different** random sample of deals.

## Evidence

`Trainer._run_episode` and `SelfPlayTrainer._run_episode` call `reset()` with no
seed:

```python
# packages/core/src/rl_card_lib/trainer/trainer.py:195  and  :423
observation, info = self.env.reset()
```

`CardGameEnv.reset()` only forwards a seed when it is given one
(`card_game_env.py:93-97`), and `Game.reset()` without a seed keeps drawing from
the private generator (`klondike.py:120-123`). That generator is created as
`random.Random(None)` — seeded from OS entropy — in both games
(`klondike.py:98`, `macao.py:109`).

The evaluation protocol tries to pin the deal with the global RNG, which has no
effect on the game's private one:

```python
# packages/examples/src/rl_card_lib/harness/evaluation.py:40-45
for seed in range(episodes):
    random.seed(10_000 + seed)
    np.random.seed(10_000 + seed)

    game = KlondikeSolitaire()          # <- random.Random(None) inside
```

Same construction in `evaluate_macao` (`evaluation.py:93-97`) and in
`run_klondike_baselines` / `run_macao_baselines` (`baselines.py:72-77`, `:125-130`).

## Measured

| Check | Result |
|---|---|
| 10 consecutive `env.reset()` on Klondike — distinct deals | **10 / 10** |
| 10 consecutive `env.reset()` on Macao — distinct deals | **10 / 10** |
| "deal #0" of the evaluation protocol, requested 5 times — distinct deals | **5 / 5** |
| `evaluate_klondike(RandomAgent(seed=0), 30)` run twice | `cards_up` = **12.57** and **12.83** — not identical |
| the same sequence via `env.reset(seed=7)` | reproducible ✔ |

Reproduce: `python thesis_notes/scripts/probe_protocol.py` → `deal_stream` in
`thesis_notes/raw/protocol_probe.json`.

## Why it matters

1. `evaluation.py`'s own module docstring calls these "Fixed-deal evaluation
   protocols". They are not fixed.
2. With `--eval-episodes 30` and a per-deal standard deviation of 5–8 cards, the
   standard error of the mean is 1–1.5 cards. Differences of the size reported
   between agents are inside that noise, and the agents were not even measured
   on the same deals, so the comparison is unpaired on top of being noisy.
3. There is no train/test split at all — not because the sets overlap (they
   almost certainly do not, out of 52! deals), but because neither set is
   defined.

## Suggested fix

The mechanism already exists and is already used correctly in one place —
`solve_benchmark.py:65` and `:112` reset with `env.reset(seed=seed)`. Thread the
same thing through the main path:

1. `Trainer`/`SelfPlayTrainer`: accept a deal-seed source (a pool or an
   iterator) and pass it to `env.reset(seed=...)`, or accept an env whose
   `reset()` supplies the seed itself.
2. `evaluate_klondike` / `evaluate_macao` / `run_*_baselines`: replace the
   `random.seed(10_000 + i)` lines with `env.reset(seed=pool[i])` and drop the
   global reseeding entirely.
3. Declare the pools once (e.g. TRAIN = seeds `0..9999`, TEST = seeds
   `100000..100199`) so the split is disjoint by construction and every agent is
   scored on the identical deals.

A worked implementation that does exactly this, without touching `packages/`,
is in `thesis_notes/scripts/harness.py` (`PooledEnv`,
`evaluate_klondike_on_pool`, `evaluate_macao_on_pool`) and
`thesis_notes/scripts/split.py`.

## Follow-up

Once fixed, `evaluation.py`'s docstring paragraph about reseeding global RNGs
("it is kept because changing it would change every historical number") can go
away with it.
