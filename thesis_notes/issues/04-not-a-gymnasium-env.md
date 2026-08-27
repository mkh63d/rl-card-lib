## Summary

`CardGameEnv`, `MaskedCardGameEnv` and `GymEnvWrapper` follow the Gymnasium
*conventions* (5-tuple, `terminated`/`truncated`, `spaces.Box` / `spaces.Discrete`)
but none of them subclasses `gymnasium.Env`. As a result
`gymnasium.utils.env_checker.check_env` rejects all of them on the first
assertion, Gymnasium wrappers refuse them, and Stable-Baselines3 refuses them.

The README advertises `gymnasium>=0.29.0` as "RL environment API", so this is
worth either fixing or documenting precisely.

## Evidence

Measured with gymnasium 1.3.0 and stable-baselines3 2.9.0
(`python thesis_notes/scripts/probe_gymnasium.py`, output in
`thesis_notes/raw/gymnasium_probe.json`):

```
check_env(CardGameEnv(KlondikeSolitaire(seed=0), max_steps=300))    -> TypeError
check_env(CardGameEnv(Macao(num_players=2, seed=0), max_steps=200)) -> TypeError
check_env(MaskedCardGameEnv(Macao(num_players=2, seed=0)))          -> TypeError
check_env(GymEnvWrapper(Macao(num_players=2, seed=0)))              -> TypeError

TypeError: The environment must inherit from the gymnasium.Env class,
actual class: <class 'rl_card_lib.env.card_game_env.CardGameEnv'>.
```

```
gym.wrappers.TimeLimit(CardGameEnv(Macao(...)), max_episode_steps=50)
  -> AssertionError: Expected env to be a `gymnasium.Env` but got
     <class 'rl_card_lib.env.card_game_env.CardGameEnv'>
gym.wrappers.RecordEpisodeStatistics(CardGameEnv(Macao(...)))
  -> AssertionError: (same)
```

```
PPO("MlpPolicy", CardGameEnv(Macao(num_players=2, seed=0), max_steps=200))
  -> ValueError: The environment is of type
     <class 'rl_card_lib.env.card_game_env.CardGameEnv'>, not a Gymnasium
     environment.
```

The checker stops at the `isinstance` test, so the rest of the contract has
**not** been verified — we only know it fails the entry test.

What is missing, measured across all four classes:

| Gymnasium contract element | Present? |
|---|---|
| subclasses `gymnasium.Env` | no (MRO is `['CardGameEnv', 'object']`) |
| `metadata` (e.g. `render_modes`) | no |
| `spec` | no |
| `self.np_random` from `Env.reset(seed=...)` | no |
| `unwrapped` | no |
| `gymnasium.register` / `gymnasium.make` anywhere in the repo | no |

## Suggested fix

Four mechanical changes to `CardGameEnv`:

1. `class CardGameEnv(gym.Env):` plus `super().__init__()`.
2. `metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}` as a class
   attribute.
3. `super().reset(seed=seed)` at the top of `reset()`, so `self.np_random` exists.
4. The `except Exception: gym = None` fallback then needs a stand-in base class,
   or the optional-dependency guarantee has to be dropped.

A 30-line adapter that does 1–3 externally is `GymnasiumAdapter` in
`thesis_notes/scripts/probe_gymnasium.py`; with it, `sb3.common.env_checker.check_env`
passes with zero warnings and `PPO.learn(2000)` runs on both games.

## Caveat worth documenting either way

Passing `check_env` would not make the environment *usable* by a generic
Gymnasium consumer, because action masking lives outside the observation
(`info["legal_actions"]`). Driven by an unmasked SB3 PPO, Macao produced an
episode whose total reward was exactly `-300.0` = 300 × `invalid_action_reward`:
not one legal action was chosen in 300 steps.

**Closed by #40.** The `Dict(observation, action_mask)` shape of
`MaskedCardGameEnv` was *not*, as this note previously claimed, the channel
`MaskablePPO` reads — sb3-contrib calls a method named `action_masks()` and
never inspects the observation, so exposing the Dict alone changed nothing.
`CardGameEnv.action_masks()` now supplies it, and
`packages/examples/scripts/train_maskable_ppo.py` trains MaskablePPO on
`rl_card_lib/MacaoMasked-v0` and scores it on the held-out deals. See §3c and
§5c of [`gymnasium.md`](../gymnasium.md) for the measured result.

Two related sub-points, if they are worth folding in here:

- `GymEnvWrapper` (`packages/core/src/rl_card_lib/core/gym_wrapper.py`)
  duplicates a subset of `CardGameEnv` and is not used by anything.
- `import gymnasium as gym` is unused in both files that import it; only
  `spaces` is ever referenced.
