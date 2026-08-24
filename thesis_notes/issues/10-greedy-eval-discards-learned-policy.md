## Summary

Every evaluation in this repository is greedy: `agent.eval()` switches DQN to
`argmax` and switches **PPO to `argmax` over its policy logits**. On Klondike
that turns a working policy into a broken one. Same checkpoint, same 200 deals,
only the action-selection rule differs:

| rule | cards to foundation | win rate | steps revisiting a seen position |
|---|---:|---:|---:|
| `argmax` over PPO's policy (what `eval()` does today) | **7.54** | **0.0 %** | 78.0 % |
| sampling that same policy | **22.45** | **28.5 %** | 44.8 % |

For reference on the identical pool: Random 11.59 / 0.0 %,
GreedyLookahead(1) 9.22 / 1.0 %, MCTS(20) 26.80 / 37.0 %, scripted heuristic
28.74 / 45.5 %.

So the PPO agent that the reports describe as "below random" is, when asked for
the policy it actually learned, roughly twice the random baseline and within
reach of MCTS.

## Evidence

```python
# packages/core/src/rl_card_lib/agents/ppo_agent.py:205-216
with torch.no_grad():
    state = torch.as_tensor(observation, device=self.device).unsqueeze(0)
    logits, value = self.network(state)
    mask_tensor = torch.as_tensor(mask, device=self.device).unsqueeze(0)
    logits = logits.masked_fill(~mask_tensor, MASK_VALUE)

    if not self.training:
        return int(logits.argmax(dim=1).item())     # <- the learned
                                                    #    distribution is dropped
    distribution = Categorical(logits=logits)
    action = distribution.sample()
```

PPO optimises a *stochastic* policy — the clipped surrogate, the entropy bonus
and the importance ratio are all defined over `π(a|s)`. Taking the argmax of it
produces a different policy, one that was never trained and whose value was
never estimated.

The same effect shows up on the value-based agents as a monotone response to
exploration at evaluation time (same weights throughout):

| agent | ε = 0 | ε = 0.05 | ε = 0.20 |
|---|---:|---:|---:|
| PPO | 7.54 | 13.90 | **20.61** |
| DQN | 5.84 | 8.13 | **13.55** |
| Double DQN | 5.14 | 6.44 | **10.46** |
| Q-learning | 11.24 | 12.07 | 11.91 |

and the revisit fraction falls as the score rises (PPO 78.0 → 63.8 → 46.8 %,
DQN 80.8 → 70.8 → 55.3 %). Q-learning is flat on both because it is already a
random policy (23 % revisits, same as `RandomAgent`).

Reproduce:

```
python thesis_notes/scripts/probe_greedy_vs_epsilon.py
```

Raw: `thesis_notes/raw/greedy_vs_epsilon.json`,
`thesis_notes/raw/ppo_argmax_vs_sampled.json`.
Figure: `thesis_notes/figures/action_rule_klondike.png`.

## Why it happens

A deterministic policy in a deterministic environment that returns to an
already-visited state repeats its whole future from there. Klondike has
reversible tableau moves and, with `max_passes=None`, an unbounded
draw/recycle loop — so the first revisit closes a cycle that lasts until the
300-step cap. ε > 0 during training is the only thing that has ever broken that
cycle.

Macao confirms the mechanism by not showing it: episodes are short and have no
reversible cycle (revisit fraction 0.0 %), and there PPO argmax 35.0 % vs
sampled 39.5 % — a small difference, not a 3× one.

## Suggested fix

Any one of these works; the first is the smallest.

1. **Let PPO evaluate the policy it learned.** Sample in eval mode, or expose
   the choice:

   ```python
   def select_action(self, observation, legal_actions=None, *, greedy=None):
       greedy = (not self.training) if greedy is None else greedy
       ...
       if greedy:
           return int(logits.argmax(dim=1).item())
       return int(Categorical(logits=logits).sample().item())
   ```

   with the evaluation harness defaulting to sampling for on-policy agents.

2. **Stop the greedy policy cycling**, by enabling `repeated_position_penalty`
   (tracked separately).

3. **Report both numbers.** For a game with reversible moves, the deterministic
   and the stochastic policy are two different policies and both are
   informative. The report already has a place for a secondary metric.

## Knock-on effect

The generated reports and the thesis text both conclude that no learner clears
the random baseline on Klondike. That conclusion is an artefact of the
evaluation rule, not a property of the trained agents, and should be revisited
once this is fixed.
