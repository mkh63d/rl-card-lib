## Summary

`CardGameEnv.step()` returns early on an illegal action **before** incrementing
the step counter, so `max_steps` is never reached. An agent that only ever
proposes illegal actions produces an episode that never terminates and never
truncates — a livelock, not a long episode.

## Evidence

```python
# packages/core/src/rl_card_lib/env/card_game_env.py:120-136
def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
    legal = self.get_legal_actions()
    if legal and action not in legal:
        observation = self.game.get_observation()
        observation = np.asarray(observation, dtype=np.float32)
        info = {
            "invalid_action": True,
            "legal_actions": legal,
            "winner": self.game.winner,
        }
        return observation, float(self.invalid_action_reward), False, False, info
        #  ^ returns here; _step_count is never touched

    observation, reward, terminated, truncated, info = self.game.step(action)
    self._step_count += 1

    if self.max_steps is not None and self._step_count >= self.max_steps:
        truncated = True
```

## Measured

```
env = CardGameEnv(Macao(num_players=2), max_steps=200)
obs, info = env.reset(seed=0)
# repeat one action that is not in info["legal_actions"]

5000 illegal steps later:
  terminated       = False
  truncated        = False
  env._step_count  = 0
  reward per step  = -1.0
```

Reproduce: `python thesis_notes/scripts/probe_protocol.py` →
`invalid_action_livelock` in `thesis_notes/raw/protocol_probe.json`.

## Why it matters

The library's own agents never hit this, because they are always handed
`info["legal_actions"]` and never propose an illegal move. Any **external**
consumer does hit it. Concretely, a Stable-Baselines3 PPO driven through a thin
`gymnasium.Env` adapter produced a Macao episode of 300 steps with a total
reward of exactly `-300.0` (= 300 × `invalid_action_reward`) and was still
running when the harness loop cut it off — in Macao only 2–4 of 65 actions are
legal in a typical position, so an unmasked policy essentially never proposes a
legal one.

## Suggested fix

Count the step, or apply the cap on the illegal branch too:

```python
if legal and action not in legal:
    self._step_count += 1
    truncated = (self.max_steps is not None
                 and self._step_count >= self.max_steps)
    ...
    return observation, float(self.invalid_action_reward), False, truncated, info
```

Counting illegal actions against the step budget is also the more honest
accounting: they cost the agent a turn's worth of reward.
