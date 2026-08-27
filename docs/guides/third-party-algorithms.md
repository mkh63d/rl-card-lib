# Third-party algorithms

The bundled agents are not the only thing that can learn these games. This page
is about handing one of them to an algorithm the library did not write — in
practice, `sb3-contrib`'s `MaskablePPO`.

## Why an unmasked algorithm gets nowhere

The environments have been real `gymnasium.Env` subclasses for a while. They
pass `env_checker`, they accept wrappers, and Stable-Baselines3 will construct
`PPO("MlpPolicy", CardGameEnv(...))` without complaining.

That acceptance is not the same as being learnable. In a typical Macao position
2–4 of the 65 actions are legal, so a policy choosing uniformly has roughly a 5%
chance per step of proposing a move the game will accept. An unmasked SB3 PPO
run bears this out:

```
200 steps, total reward -198.9
```

That is almost exactly 200 × `invalid_action_reward` (−1.0). Across a whole
episode the policy essentially never picked a legal action. The environment was
accepted and then could not be learned from.

## The three mask channels

The fix is to give the algorithm the mask. The library offers it three separate
ways, and conflating them is the trap:

| channel | what it is | who reads it |
| --- | --- | --- |
| `info["legal_actions"]` | a list, alongside the observation | the bundled agents |
| `Dict(observation, action_mask)` | the mask **in the observation**, on the masked ids | a policy network, as a feature |
| `env.action_masks()` | a method, on **every** env | `sb3-contrib` |

The third row is the one that matters for `MaskablePPO`.
`sb3_contrib.common.maskable.utils.get_action_masks()` finds an environment's
mask by calling a method named exactly `action_masks()` — reaching it through
`env.get_wrapper_attr("action_masks")` on a bare env, and through
`VecEnv.env_method("action_masks")` on a vectorised one. It never looks at the
observation. Exposing the `Dict` shape alone left `MaskablePPO` sampling
illegally like any unmasked policy, which is what
[#40](https://github.com/mkh63d/rl-card-lib/issues/40) was really about.

`CardGameEnv.action_masks()` supplies it, delegating to the same
`get_legal_action_mask()` the rest of the library uses so the two spellings
cannot drift apart. It lives on the base class, not on `MaskedCardGameEnv`, so
the plain `Box` ids are maskable too — an algorithm that wants the mask but not
an extra 65 input features can take `rl_card_lib/Macao-v0`.

## Training MaskablePPO

`sb3-contrib` is an optional extra, not a dependency:

```bash
pip install -e "./packages/examples[sb3]"
```

No adapter and no `ActionMasker` are needed:

```python
import gymnasium
import rl_card_lib.games                     # registers the ids
from sb3_contrib import MaskablePPO

env = gymnasium.make("rl_card_lib/MacaoMasked-v0", deal_seeds=range(10_000))
model = MaskablePPO("MultiInputPolicy", env, seed=0)
model.learn(total_timesteps=300_000)
```

`MultiInputPolicy` because the masked observation is a `Dict`. `deal_seeds`
restricts training to the TRAIN pool, so the policy never trains on a deal it
will later be measured on.

### One policy, both seats

While training, the model plays *both* sides of the table. That is well posed
rather than a shortcut, because of two properties of `Macao`:

- the observation is written from the **current player's** side — its own hand,
  the other players' hand sizes;
- rewards are **actor-relative**: "every reward `step()` returns is paid to the
  player who took the action, including the terminal one."

So the policy always sees the position from the mover's point of view and is
paid for the mover's outcome. This is the same self-play arrangement
`SelfPlayTrainer` gives the bundled agents, which is what keeps the resulting
number comparable to theirs.

## Measuring it

The worked script trains and then evaluates through
`rl_card_lib.harness.evaluation` — the same protocol, the same opponents and the
same 200 held-out deals every bundled agent is scored on:

```bash
python packages/examples/scripts/train_maskable_ppo.py
```

### The result

300 000 steps, seed 0, evaluated on all 200 held-out deals (about 8 minutes of
CPU training):

| agent | vs random | vs heuristic |
| --- | --- | --- |
| **MaskablePPO (300k)** | **79.5%** | **37.0%** |
| Random | 1.5% | 2.5% |
| GreedyLookahead(1) | 64.0% | 23.5% |
| Heuristic | 95.0% | 54.0% |

Read this honestly. MaskablePPO **learned the game**: 79.5% against a random
opponent where a random policy itself manages 1.5%, and it beats
`GreedyLookahead(1)` on both columns. It does **not** beat the hand-written
`MacaoHeuristicAgent`, which knows Macao's rules directly and wins 54% of its own
mirror match. A longer run or tuned hyperparameters would likely narrow that;
the defaults are kept here because the claim being demonstrated is that an
off-the-shelf algorithm learns this game, not that it is the strongest player in
the repository.

The "vs heuristic" column is the harder one by construction — the heuristic beats
random 95% of the time, so 37.0% against it is a substantially stronger result
than the same number against random would be.

The script also measures the fixed-strength agents in the same run and writes
them to the same file. That is deliberate: `results/baselines/macao.json` was
recorded over 30 deals, so quoting it next to a 200-deal number would be
comparing two different measurements rather than two agents.

Like everything under `results/`, the output is a generated artifact and is not
committed — rerun the script to reproduce it.
