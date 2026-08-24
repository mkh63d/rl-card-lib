## Summary

`CardGameEnv` implements a repeated-position penalty specifically to stop agents
cycling through reversible moves, and its docstring says so. It defaults to
`0.0`, and neither bundled game ever passes a non-zero value — so the mechanism
has never been active in any run. The trained greedy policies do exactly what
the mechanism exists to prevent: **80 % of their moves land in a position they
have already seen this episode.**

## Evidence

```python
# packages/core/src/rl_card_lib/env/card_game_env.py:20-27
def __init__(
    self,
    game: Any,
    max_steps: Optional[int] = None,
    render_mode: Optional[str] = None,
    invalid_action_reward: float = -1.0,
    repeated_position_penalty: float = 0.0,   # <- default
):
```

with the docstring:

> `repeated_position_penalty`: Added to the reward (use a negative value)
> whenever a step lands in a position already seen this episode. Games with
> reversible moves let an agent shuffle in circles forever; this makes each lap
> cost something.

```python
# packages/core/src/rl_card_lib/env/card_game_env.py:146-151
position = hash(observation.tobytes())
if position in self._seen_positions:
    info["repeated_position"] = True
    reward += self.repeated_position_penalty     # adds 0.0
else:
    self._seen_positions.add(position)
```

Neither registration passes it:

```python
# packages/examples/src/rl_card_lib/games/registration.py:58
env_factory=lambda: CardGameEnv(KlondikeSolitaire(), max_steps=KLONDIKE_MAX_STEPS),
# packages/examples/src/rl_card_lib/games/registration.py:82
env_factory=lambda: CardGameEnv(Macao(num_players=2), max_steps=MACAO_MAX_STEPS),
```

`grep -rn "repeated_position_penalty" packages/ tests/` finds it only in
`card_game_env.py` (the definition, the docstring, the two constructors and the
one use site) and in `tests/test_reward_design.py:474`. So the mechanism is
implemented *and unit-tested* — it is simply never switched on by anything that
actually trains an agent.

## Measured

Trained checkpoints from `checkpoints/klondike_*`, greedy, 30 TEST deals
(seeds 100000–100029), 300-step cap:

| Policy | steps landing in an already-seen position | distinct actions used per episode (of 68) | cards to foundation |
|---|---:|---:|---:|
| DQN (trained, greedy) | **80.3 %** | 12.5 | 6.43 |
| Double DQN (trained, greedy) | **83.2 %** | 14.0 | 6.73 |
| PPO (trained, greedy) | 78.9 % | 16.9 | 8.77 |
| **random** | **23.0 %** | 40.0 | **11.17** |

What the loop consists of:

| Policy | draw / recycle | tableau ↔ tableau | to foundation | waste → tableau |
|---|---:|---:|---:|---:|
| DQN | **70.2 %** | 24.7 % | **2.1 %** | 3.0 % |
| Double DQN | 43.7 % | 51.6 % | 2.2 % | 2.4 % |
| PPO | 52.9 % | 40.6 % | 2.9 % | 3.6 % |
| random | 29.4 % | 62.3 % | 3.7 % | 4.5 % |
| scripted heuristic | 33.0 % | 47.9 % | **12.7 %** | 6.4 % |

The trained DQN's greedy policy is "keep drawing" 70 % of the time. With the
default `max_passes=None` the stock can be recycled forever, so that is a closed
loop that only ends at the 300-step cap. This is why the greedy policies score
*below* random.

### Why it happens

Not because the Q-function is flat. Measured over the same 9 000 visited
Klondike positions, the mean spread of Q across the **legal** actions is 0.066
against a mean legal Q of 0.751 (8.8 %) for plain DQN, but 0.337 against 1.015
(33 %) for Double DQN — and Double DQN loops *more* (83.2 % vs 80.3 %). Flatness
is not the mechanism.

The mechanism is structural:

> In a deterministic environment, a deterministic policy that returns to an
> already-visited state repeats its entire future from there — i.e. it cycles.

Klondike satisfies both premises: tableau↔tableau moves are reversible, and with
`max_passes=None` the draw/recycle loop is deterministic for a fixed stock
order. A greedy policy is deterministic by definition. Nothing in the
environment breaks the cycle, because the one mechanism that would is set to
`0.0`.

This also explains the gap between the training curves and the greedy
evaluation that the project has been puzzling over: during training ε > 0 breaks
the cycle with a random move roughly every 20 steps. The same policy without
exploration cycles.

Reproduce: `thesis_notes/raw/greedy_loop_probe.json`,
`thesis_notes/raw/action_mix_probe.json`, `thesis_notes/raw/q_spread_probe.json`.

## Suggested fix

Turn it on where the games are registered:

```python
env_factory=lambda: CardGameEnv(
    KlondikeSolitaire(), max_steps=KLONDIKE_MAX_STEPS,
    repeated_position_penalty=-0.05,
),
```

and consider making the default non-zero for games that declare reversible
moves, since a silently-inert safeguard is worse than none.

## Measured effect of the fix

A retraining ablation (`repeated_position_penalty=-0.05`, DQN and Double DQN on
Klondike, 3 initialisation seeds each, 5000 episodes, same TRAIN stream and same
200-deal TEST pool) is running at the time of filing; the result will be added
here and in `thesis_notes/tables/ablation_fixes.csv` as the `noloop` arm.

What is already established without it:

- the loop is real and dominant (80–83 % of greedy steps revisit a position);
- it is what separates the greedy score from the exploratory one — the same
  weights score 3× higher when the cycle is broken by exploration at
  evaluation time;
- the mechanism that would prevent it exists in this repository, is documented
  as the remedy for exactly this, is unit-tested, and is off.

That last point stands on its own regardless of how the retraining lands.
