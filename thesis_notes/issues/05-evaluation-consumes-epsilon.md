## Summary

Evaluation is not side-effect free. Every evaluation episode calls
`agent.reset()`, and `reset()` is exactly where epsilon decays — so measuring an
agent permanently advances its exploration schedule. The effect is visible in
the committed run records: the epsilon captured *before training starts* is
0.8647, not 1.0.

## Evidence

```python
# packages/core/src/rl_card_lib/agents/dqn_agent.py:403-414
def reset(self) -> None:
    if self.episodes > 0 and self.epsilon > self.epsilon_end:
        self.epsilon *= self.epsilon_decay
    self.episodes += 1
```

(the tabular agent does the same at `tabular.py:175-184`)

Both evaluation paths call it once per episode:

```python
# packages/core/src/rl_card_lib/trainer/trainer.py:196   (Trainer._run_episode,
#                                                         reached from evaluate())
self.agent.reset()

# packages/examples/src/rl_card_lib/harness/evaluation.py:50
agent.reset()
```

and `run_sweep.train_one` runs a full evaluation *before* it captures the
configuration:

```python
# packages/examples/scripts/run_sweep.py:72-75
tick = time.time()
before = spec.evaluate(agent, args.eval_episodes, args.seed)
...
# packages/examples/scripts/run_sweep.py:102-105
# Captured before training so the recorded epsilon is the start value ...
config = TrainingReport.from_trainer(...)
```

## Measured

The arithmetic matches to the last digit:

| Game | eval episodes before training | `0.995^(n-1)` | value recorded in `run.json` |
|---|---:|---:|---:|
| Klondike | 30 | 0.8647077305675337 | **0.8647077305675338** |
| Macao | 2 × 30 (two opponents) | 0.7439808620067382 | **0.7439808620067382** |

During training the sweep adds a further **200** decays per 5000-episode run
(`eval_interval = episodes // 10`, `eval_episodes = 20`).

Reproduce: `python thesis_notes/scripts/probe_protocol.py` →
`epsilon_schedule` in `thesis_notes/raw/protocol_probe.json`.

## Why it matters

1. The `epsilon` recorded in every `run.json` and rendered in the report is
   wrong — the comment above it says it is the start value, and it is not.
2. The exploration schedule now depends on how often you *measured*, i.e. on
   reporting settings rather than on training configuration.
3. The magnitude is small (the floor moves from episode ~599 to ~570 out of
   5000) but it breaks reproducibility: change `--eval-episodes` and you change
   the exploration schedule.

## Suggested fix

Either of:

- **A.** Move the decay out of `Agent.reset()` into an explicit
  `Agent.on_episode_end()` that only the training loop calls. This is the clean
  fix — `reset()` should mean "start a new episode", not "advance the
  curriculum".
- **B.** Make evaluation restore what it touched. A context manager that saves
  and restores `epsilon`, `episodes` and the `training` flag is implemented as
  `frozen_exploration()` in `thesis_notes/scripts/harness.py`.

A is preferable; B is a two-line stopgap.
