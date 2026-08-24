## Summary

`KlondikeSolitaire.LOSS_REWARD` is dead code under the default configuration.
It fires only when the position has no legal action, but with the default
`max_passes=None` the draw/recycle action is always legal, so a Klondike deal
can never die. The practical consequence is that a learning agent gets **no
terminal signal at all**: it never wins (0.0 % greedy) and never loses.

## Evidence

```python
# packages/examples/src/rl_card_lib/games/klondike.py:353-364
won = self._check_win()
if won:
    if self.reward_mode == "sparse":
        reward = 1.0
elif not self.get_legal_actions():
    # No legal moves left: the deal is dead. ...
    # Reachable only with a finite max_passes, since otherwise
    # draw/recycle stays legal forever.
    self.done = True
    reward += self.LOSS_REWARD
```

The comment already states the condition. The default is the unreachable one:

```python
# packages/examples/src/rl_card_lib/games/klondike.py:68
max_passes: Optional[int] = None,

# packages/examples/src/rl_card_lib/games/klondike.py:226
if self.stock or (self.waste and self._can_recycle()):
    legal.append(0)

# packages/examples/src/rl_card_lib/games/klondike.py:376-378
def _can_recycle(self) -> bool:
    return self.max_passes is None or self.passes + 1 < self.max_passes
```

and `registration.py:58` constructs `KlondikeSolitaire()` with no `max_passes`.

## Measured

200 TEST deals (seeds 100000–100199), 300-step cap:

| Configuration | `terminated` | `truncated` | dead deals (loss) | win rate | cards up |
|---|---:|---:|---:|---:|---:|
| random, `max_passes=None` (default) | **0.0 %** | **100.0 %** | **0.0 %** | 0.0 % | 11.59 |
| heuristic, `max_passes=None` | 45.5 % | 54.5 % | **0.0 %** | 45.5 % | 28.74 |
| random, `max_passes=3` | 4.5 % | 95.5 % | **4.5 %** | 0.0 % | 9.80 |
| heuristic, `max_passes=3` | 40.0 % | 60.0 % | 1.5 % | 38.5 % | 25.84 |

So under the default, `LOSS_REWARD = -1.0` is never paid to anyone, ever. With
`max_passes=3` it becomes reachable, at a cost of about 3 cards of heuristic
performance (28.74 → 25.84).

Reproduce: `python thesis_notes/scripts/probe_protocol.py` →
`episode_shape` in `thesis_notes/raw/protocol_probe.json`.

## Why it matters

The reward docstring advertises `LOSS_REWARD` as the thing that makes a stuck
deal distinguishable from running out of time:

> Reward paid on the terminal step of a lost deal, in both reward modes.
> Without it a stuck deal is indistinguishable from running out of time.

Under the shipped default the two are exactly indistinguishable, because the
first case cannot occur.

## Suggested fix

Pick one and make it explicit:

- **A.** Give the registered Klondike a finite `max_passes` (3 is the common
  draw-1 house rule) so the loss branch is live.
- **B.** Keep unlimited passes, delete `LOSS_REWARD` and the branch, and say in
  the docstring that a Klondike episode ends either as a win or at the step cap.

Either is fine; the current state — a documented terminal reward that cannot
occur — is the one to avoid. Note that A also changes what the environment *is*,
so it is not comparable with previously reported numbers.

## Related

- The step-cap end is *also* mis-handled downstream: `Trainer` reports it to
  `learn()` as a terminal state, so the bootstrap is zeroed on the last
  transition of every episode. That is a separate issue.
