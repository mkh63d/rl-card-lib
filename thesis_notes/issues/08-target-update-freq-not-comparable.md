## Summary

`target_update_freq=500` is shared by both games, but it is counted in gradient
steps, and the two games differ by 6.5× in episode length. The same number
therefore means "every 1.7 episodes" on Klondike and "every 10.9 episodes" on
Macao — and a whole 5000-episode Macao run gets only **460** target-network
updates.

## Evidence

```python
# packages/examples/src/rl_card_lib/harness/learners.py:54  and  :62
buffer_size=50_000, batch_size=64, target_update_freq=500,
```

`DQNAgent.learn()` takes one gradient step per environment step once the buffer
holds `batch_size` transitions, so `train_steps` ≈ environment steps:

```python
# packages/core/src/rl_card_lib/agents/dqn_agent.py:395-399
self.train_steps += 1

if self.train_steps % self.target_update_freq == 0:
    self.target_network.load_state_dict(self.q_network.state_dict())
```

## Measured

| Game | mean episode length | 500 gradient steps = | target updates in a 5000-episode run |
|---|---:|---|---:|
| Klondike | 300.0 (always the cap) | **1.67 episodes** | **3 000** |
| Macao | 46.0 | **10.9 episodes** | **460** |

(Macao's mean episode length measured over 200 TEST deals and confirmed by the
recorded runs: 46.5 for DQN, 45.1 for Double DQN.)

Reproduce: `python thesis_notes/scripts/probe_protocol.py` →
`target_update_cadence` and `episode_shape` in
`thesis_notes/raw/protocol_probe.json`.

## Assessment

- **Klondike: fine.** ~0.6 refreshes per episode is a normal cadence.
- **Macao: probably too slow.** The terminal `+10` is rare early on and each
  target refresh propagates value one step further back; 460 refreshes is not
  many for a game whose reward is concentrated at the end.
- The asymmetry looks unintended rather than chosen — nothing in the code or the
  comments suggests the two games were meant to have different effective
  cadences.

## Suggested fix

Not a correctness bug, so no urgency, but two options:

- express the cadence in episodes (e.g. refresh every 20 episodes) so it means
  the same thing in every game; or
- make it a per-game value in `register_sweep_game(...)`, alongside
  `mcts_simulations` and `mcts_rollout_depth` which are already per-game.

If either is done, note it in the hyper-parameter table — the current table
reports "500 steps" without saying steps of what.
