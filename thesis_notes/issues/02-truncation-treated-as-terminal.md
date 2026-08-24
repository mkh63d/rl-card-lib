## Summary

`Trainer` collapses `terminated` and `truncated` into one `done` flag before
handing the transition to `agent.learn()`. Every value target then multiplies
the bootstrap by `(1 - done)`, so a step that was cut by the **step cap** is
taught that the future is worth exactly zero.

On Klondike this is not an edge case: **100 % of episodes end by truncation**,
so it is the last transition of every single episode.

## Evidence

```python
# packages/core/src/rl_card_lib/trainer/trainer.py:209-217
next_observation, reward, terminated, truncated, info = self.env.step(action)
done = terminated or truncated
...
learn_result = self._learn(
    self.agent, observation, action, reward,
    next_observation, done, info,
)
```

`SelfPlayTrainer._run_episode` does the same at `trainer.py:449-463`.

Every learner consumes that flag as if it meant "the episode reached a terminal
state":

```python
# packages/core/src/rl_card_lib/agents/dqn_agent.py:384
target_q = rewards + (1 - dones) * has_actions * self.gamma * next_q
```

```python
# packages/core/src/rl_card_lib/agents/tabular.py:155-156
if done:
    target = reward
```

```python
# packages/core/src/rl_card_lib/agents/ppo_agent.py:306-308
non_terminal = 1.0 - dones[t]
delta = rewards[t] + self.gamma * next_value * non_terminal - values[t]
running = delta + self.gamma * self.gae_lambda * non_terminal * running
```

## Measured

200 TEST deals, random policy, Klondike (`max_steps=300`, default
`max_passes=None`):

| | `terminated` | `truncated` |
|---|---:|---:|
| random policy | **0.0 %** | **100.0 %** |
| scripted heuristic | 45.5 % | 54.5 % |

Instrumented training (5 Klondike episodes, 1500 transitions): 5 transitions
arrived with `done=True`, of which **0 were terminations and 5 were
truncations**; the reward on those transitions was `-0.01` each — the per-move
cost and nothing else.

The recorded runs agree: mean episode length was 299.4 (DQN), 299.2
(Double DQN), 300.0 (Q-learning) and 272.3 (PPO) against a 300-step cap.

Macao is affected too, though less severely: truncation is common with a weak
policy (97 % for two random players) but the game does pay a hand-size
differential at the cap (`macao.py:499-508`), so only the bootstrap is lost.

Reproduce: `python thesis_notes/scripts/probe_protocol.py` →
`truncation_and_terminal_reward` and `episode_shape` in
`thesis_notes/raw/protocol_probe.json`.

## Suggested fix

Keep ending the loop on either flag, but only report a real termination to the
learner — one line:

```python
done = terminated or truncated
learn_result = self._learn(self.agent, observation, action, reward,
                           next_observation, terminated, info)
```

(and the same in `SelfPlayTrainer._run_episode`). This is the standard
time-limit handling: the loop stops, the value function keeps bootstrapping.

A worked implementation that does this without touching `packages/` is
`TimeLimitBootstrapMixin` in `thesis_notes/scripts/harness.py`.

## Measured effect of the fix

Honest answer: **on the headline metric, none.** Re-ran both value learners on
both games, 3 initialisation seeds each, 5000 episodes, identical TRAIN deal
stream and the identical 200-deal TEST pool, with and without the fix:

| game | agent | as published | + fix | Δ |
|---|---|---:|---:|---:|
| Klondike | Double DQN | 5.88 ± 0.30 | 5.38 ± 0.44 | −0.50 cards |
| Klondike | DQN | 5.67 ± 0.41 | 5.77 ± **0.07** | +0.10 cards |
| Macao | Double DQN | 7.5 ± 1.0 % | 8.7 ± 0.6 % | +1.2 pp |
| Macao | DQN | 7.8 ± 0.6 % | 7.8 ± 1.0 % | 0.0 pp |

Two of four move up, one down, one flat, all within roughly one standard
deviation. The one consistent effect is on *variance*: Klondike DQN's spread
across seeds drops from 0.41 to 0.07.

That is not an argument for leaving the bug in place — the current target is
biased, provably, on 100 % of Klondike episodes, and the fix is one line. It
does mean the bootstrap bias is **not** what is holding these agents back. The
dominant effect turned out to be the evaluation rule (tracked separately): the
same PPO checkpoint scores 7.5 cards by argmax and 22.5 by sampling its own
policy.

Raw data: `thesis_notes/tables/ablation_fixes.csv`, `thesis_notes/raw/runs/`.

## Related

- The reason Klondike never terminates in the first place is a separate issue:
  `LOSS_REWARD` is unreachable with the default `max_passes=None`.
