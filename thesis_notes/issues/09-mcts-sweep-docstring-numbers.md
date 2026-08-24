## Summary

`sweep_mcts_budget.py` documents anchor numbers that its own committed output
contradicts. The docstring says ~77 % at 40 simulations and ~90 % at 60; the CSV
in `results/mcts_budget_sweep/` says 73 % and 83 %.

## Evidence

```python
# packages/examples/scripts/sweep_mcts_budget.py:20-25
The defaults are plain MCTS (determinizations=1, rollout_depth=20, seed=0),
which reproduces the headline anchors: ~77% win rate at 40 simulations and ~90%
at 60.
```

`results/mcts_budget_sweep/macao_mcts_budget_sweep.csv`, produced by that script
with those exact defaults (100 games per point, seed 0):

| simulations | win rate vs random |
|---:|---:|
| 1 | 0.02 |
| 2 | 0.18 |
| 5 | 0.34 |
| 10 | 0.47 |
| 20 | 0.64 |
| **40** | **0.73** |
| **60** | **0.83** |
| 80 | 0.76 |
| **120** | **0.88** |

So 40 → 73 % (not 77 %) and 60 → 83 % (not 90 %). The number closest to "about
87 %" is 88 %, which occurs at **120** simulations.

## Suggested fix

Update the docstring to the values the committed CSV actually holds, e.g.:

> The defaults are plain MCTS (determinizations=1, rollout_depth=20, seed=0),
> which reproduce the committed curve: 64 % at 20 simulations, 73 % at 40, 83 %
> at 60 and 88 % at 120.

Worth adding: 100 games per point gives a standard error of about 4–5
percentage points, which is enough to explain the non-monotonic dip at 80
simulations (76 %). Saying so in the docstring would stop the next reader
treating that dip as a finding.

## Note

The same numbers appear in the thesis text ("about 87 % at 60 simulations per
move", twice). That is outside this repository, but it is the same drift and is
tracked in `thesis_notes/chapter6_obsolete.md` (F1, F3, I1).
