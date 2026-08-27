# Quickstart

## Quick demo

The fastest way to see the pieces fit together — create a game, wrap it, and run
a short training loop:

```bash
python packages/examples/scripts/quick_demo.py
```

## Train an agent in a few lines

`CardGameEnv` adapts any `Game` to the Gymnasium API that the agents and
`Trainer` expect:

```python
from rl_card_lib.games import bundled_klondike
from rl_card_lib.env import CardGameEnv
from rl_card_lib.trainer import Trainer
from rl_card_lib.agents import DoubleDQNAgent

game = bundled_klondike()
env = CardGameEnv(game, max_steps=200)

agent = DoubleDQNAgent(
    state_size=env.observation_space.shape[0],
    action_size=env.action_space.n,
)

trainer = Trainer(env, agent)
metrics = trainer.train(episodes=1000)
metrics.plot()   # reward / win-rate curves (needs matplotlib)
```

`bundled_klondike()` is the game this library's own experiments play — the
sweep, the evaluation protocols, the baselines and the Gymnasium ids all build
their game there, so anything you measure against it is comparable with the
numbers in the report. `KlondikeSolitaire(...)` is the constructor for a
deliberately customised game; it differs in one respect that matters, allowing
unlimited passes through the stock, under which a deal can never be *lost* and
an agent gets no terminal signal at all.

See [the agent zoo](../guides/agents.md) for the full list of agents and when to
reach for each one.

## Use it from Gymnasium

`CardGameEnv` and `MaskedCardGameEnv` are real `gymnasium.Env` subclasses, so
the standard tooling works on them unchanged — `check_env`, the wrappers, and
third-party trainers such as Stable-Baselines3. The bundled games register
themselves as ids when `rl_card_lib.games` is imported:

```python
import gymnasium
import rl_card_lib.games          # registers the ids below

env = gymnasium.make("rl_card_lib/Macao-v0")
observation, info = env.reset(seed=0)
observation, reward, terminated, truncated, info = env.step(info["legal_actions"][0])
```

| id | env | observation |
| --- | --- | --- |
| `rl_card_lib/Klondike-v0` | `CardGameEnv` | `Box` |
| `rl_card_lib/Macao-v0` | `CardGameEnv` | `Box` |
| `rl_card_lib/KlondikeMasked-v0` | `MaskedCardGameEnv` | `Dict(observation, action_mask)` |
| `rl_card_lib/MacaoMasked-v0` | `MaskedCardGameEnv` | `Dict(observation, action_mask)` |

Reach for the **masked** ids when something is going to learn on them. Neither
game makes more than a handful of its actions legal at once — in Macao only 2–4
of 65 in a typical position — so a policy that cannot see `info["legal_actions"]`
spends nearly every step collecting `invalid_action_reward` instead of playing.

### Masking, for an outside algorithm

There are two separate channels, and it is worth not confusing them:

- The **masked ids** put the mask *in the observation*, so a policy can read it
  as a feature and learn to use it.
- **Every** id, masked or not, also exposes `action_masks()` — the method name
  `sb3-contrib` looks for when it asks an environment what is legal. That is
  what actually constrains the sampled action.

So `MaskablePPO` works off the shelf, with no adapter:

```python
import gymnasium
import rl_card_lib.games                     # registers the ids
from sb3_contrib import MaskablePPO

env = gymnasium.make("rl_card_lib/MacaoMasked-v0")
MaskablePPO("MultiInputPolicy", env).learn(total_timesteps=300_000)
```

`sb3-contrib` is an optional extra — `pip install -e "./packages/examples[sb3]"`.
The worked version, which also measures the result on the held-out deals, is
[`scripts/train_maskable_ppo.py`](../guides/third-party-algorithms.md).

The step cap stays on the env (`CardGameEnv.max_steps`) rather than being set as
`max_episode_steps` on the registration, so a truncation has one cause and one
place to configure it:

```python
env = gymnasium.make("rl_card_lib/Macao-v0", max_steps=50)
```

## Play a game by hand

Every `Game` exposes its legal actions, so you can step it directly:

```python
from rl_card_lib.games import KlondikeSolitaire

game = KlondikeSolitaire()
game.reset()
done = False
while not done:
    actions = game.get_legal_actions()
    action = actions[0]                 # pick the first legal action
    obs, reward, terminated, truncated, info = game.step(action)
    done = terminated or truncated
    print(game.render())
```

## Train and benchmark from the command line

```bash
# Train every learner on both games and write results/index.html
python packages/examples/scripts/run_sweep.py --episodes 200

# Re-render the report from stored records, training nothing
python packages/examples/scripts/run_sweep.py --html-only

# Train a chosen agent on a chosen game
python packages/examples/scripts/train_agents.py

# Compare the agent zoo on the same deals
python packages/examples/scripts/benchmark_agents.py

# Single-game training scripts
python packages/examples/scripts/train_klondike.py
python packages/examples/scripts/train_macao.py
```

`run_sweep.py` produces a self-contained `results/index.html` — see
[Training and reports](../guides/training-and-reports.md).

## Run the tests

```bash
pytest tests/ -q                      # library test suite
pytest packages/examples/tests/ -q    # example-game tests (run separately in CI)
pytest --cov --cov-report=html        # with a coverage report
```
