"""Probe how far CardGameEnv actually satisfies the Gymnasium contract.

Writes thesis_notes/raw/gymnasium_probe.json and prints a human-readable log.
Nothing in packages/ is modified: the adapter used for the Stable-Baselines3
experiment is defined here, in thesis_notes, exactly so the library stays as the
thesis describes it.
"""

from __future__ import annotations

import inspect
import io
import json
import os
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout

import numpy as np

import gymnasium as gym
from gymnasium import spaces

from rl_card_lib.env import CardGameEnv, MaskedCardGameEnv
from rl_card_lib.core.gym_wrapper import GymEnvWrapper
from rl_card_lib.games import KlondikeSolitaire, Macao

OUT = os.path.join(os.path.dirname(__file__), "..", "raw")
LOGS = os.path.join(os.path.dirname(__file__), "..", "logs")


def api_surface(env) -> dict:
    """What of the Gymnasium API the object has, and what it lacks."""
    reset_sig = str(inspect.signature(env.reset))
    step_sig = str(inspect.signature(env.step))
    return {
        "class": type(env).__name__,
        "is_gym_Env_subclass": isinstance(env, gym.Env),
        "mro": [c.__name__ for c in type(env).__mro__],
        "reset_signature": reset_sig,
        "step_signature": step_sig,
        "has_metadata": hasattr(env, "metadata"),
        "has_spec": hasattr(env, "spec"),
        "has_np_random": hasattr(env, "np_random"),
        "has_unwrapped": hasattr(env, "unwrapped"),
        "has_render_mode": hasattr(env, "render_mode"),
        "observation_space": repr(getattr(env, "observation_space", None)),
        "action_space": repr(getattr(env, "action_space", None)),
        "observation_space_type": type(getattr(env, "observation_space", None)).__name__,
        "action_space_type": type(getattr(env, "action_space", None)).__name__,
        "extra_public_methods": sorted(
            name for name in dir(env)
            if not name.startswith("_")
            and callable(getattr(env, name, None))
            and name not in {
                "reset", "step", "render", "close",
            }
        ),
    }


def step_return_shape(env) -> dict:
    """Confirm the 5-tuple terminated/truncated convention at runtime."""
    obs, info = env.reset(seed=123)
    legal = info.get("legal_actions") or [0]
    out = env.step(legal[0])
    return {
        "reset_returns": ["obs", "info"] if isinstance(info, dict) else ["?"],
        "reset_obs_type": type(obs).__name__,
        "reset_info_keys": sorted(info.keys()) if isinstance(info, dict) else None,
        "step_tuple_len": len(out),
        "step_field_types": [type(x).__name__ for x in out],
        "obs_in_observation_space": bool(env.observation_space.contains(out[0]))
        if env.observation_space is not None else None,
        "terminated_is_bool": isinstance(out[2], bool),
        "truncated_is_bool": isinstance(out[3], bool),
    }


def run_check_env(env_factory, label: str) -> dict:
    """gymnasium.utils.env_checker.check_env on the raw library env."""
    from gymnasium.utils.env_checker import check_env

    buf = io.StringIO()
    result = {"env": label, "passed": False, "error_type": None, "error": None}
    try:
        with redirect_stdout(buf), redirect_stderr(buf):
            check_env(env_factory(), skip_render_check=True)
        result["passed"] = True
    except Exception as exc:  # noqa: BLE001 - the failure IS the finding
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc).strip().splitlines()[0] if str(exc) else ""
        result["traceback_tail"] = traceback.format_exc().strip().splitlines()[-4:]
    result["checker_output"] = buf.getvalue().strip()
    return result


# ---------------------------------------------------------------------------
# The 30-line adapter that makes the library env a real gymnasium.Env.
# ---------------------------------------------------------------------------

class GymnasiumAdapter(gym.Env):
    """Wrap a CardGameEnv as a genuine `gymnasium.Env`.

    Everything the library env already does is kept; this only adds what the
    Gymnasium contract requires and the library never declared: the base class,
    `metadata`, the seeded `self.np_random`, and -- crucially for a generic
    trainer such as Stable-Baselines3, which does not know about legal actions
    -- a translation of illegal actions into the library's invalid-action
    penalty (already CardGameEnv's behaviour) plus the mask exposed in `info`.
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(self, inner: CardGameEnv):
        super().__init__()
        self.inner = inner
        self.observation_space = inner.observation_space
        self.action_space = inner.action_space
        self.render_mode = inner.render_mode

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs, info = self.inner.reset(seed=seed, options=options)
        info = dict(info)
        info["action_mask"] = self.inner.get_legal_action_mask()
        return np.asarray(obs, dtype=np.float32), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.inner.step(int(action))
        info = dict(info)
        info["action_mask"] = self.inner.get_legal_action_mask()
        return (
            np.asarray(obs, dtype=np.float32),
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )

    def render(self):
        return self.inner.render()

    def close(self):
        self.inner.close()


def run_sb3(label: str, make_inner, timesteps: int = 2000) -> dict:
    """Try SB3 on the raw env, then on the adapter. Record both outcomes."""
    out = {"env": label, "sb3_version": None}
    try:
        import stable_baselines3 as sb3
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_checker import check_env as sb3_check
    except ImportError as exc:
        out["available"] = False
        out["error"] = f"stable-baselines3 not installed: {exc}"
        return out

    out["available"] = True
    out["sb3_version"] = sb3.__version__

    # 1. raw library env, straight into SB3
    raw = {"passed": False}
    buf = io.StringIO()
    try:
        with redirect_stdout(buf), redirect_stderr(buf):
            PPO("MlpPolicy", make_inner(), verbose=0)
        raw["passed"] = True
    except Exception as exc:  # noqa: BLE001
        raw["error_type"] = type(exc).__name__
        raw["error"] = str(exc).strip()
    raw["output"] = buf.getvalue().strip()
    out["raw_env"] = raw

    # 2. same game through the adapter
    adapted = {"passed": False}
    buf = io.StringIO()
    try:
        with redirect_stdout(buf), redirect_stderr(buf):
            env = GymnasiumAdapter(make_inner())
            sb3_check(env, warn=True, skip_render_check=True)
            model = PPO("MlpPolicy", env, verbose=0, n_steps=256, batch_size=64,
                        seed=0, device="cpu")
            model.learn(total_timesteps=timesteps, progress_bar=False)
            # roll one greedy episode to prove the trained model drives the game
            obs, info = env.reset(seed=100_000)
            steps, total = 0, 0.0
            for _ in range(300):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total += reward
                steps += 1
                if terminated or truncated:
                    break
        adapted["passed"] = True
        adapted["timesteps"] = timesteps
        adapted["demo_episode_steps"] = steps
        adapted["demo_episode_reward"] = round(total, 3)
        adapted["invalid_action_rate_note"] = (
            "SB3 has no action mask, so illegal actions are absorbed by "
            "CardGameEnv.invalid_action_reward"
        )
    except Exception as exc:  # noqa: BLE001
        adapted["error_type"] = type(exc).__name__
        adapted["error"] = str(exc).strip()
        adapted["traceback_tail"] = traceback.format_exc().strip().splitlines()[-6:]
    adapted["output"] = buf.getvalue().strip()
    out["adapter_env"] = adapted
    return out


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)

    report: dict = {"gymnasium_version": gym.__version__}

    envs = {
        "CardGameEnv(Klondike)": lambda: CardGameEnv(KlondikeSolitaire(seed=0), max_steps=300),
        "CardGameEnv(Macao)": lambda: CardGameEnv(Macao(num_players=2, seed=0), max_steps=200),
        "MaskedCardGameEnv(Macao)": lambda: MaskedCardGameEnv(Macao(num_players=2, seed=0), max_steps=200),
        "GymEnvWrapper(Macao)": lambda: GymEnvWrapper(Macao(num_players=2, seed=0)),
    }

    report["api_surface"] = {name: api_surface(f()) for name, f in envs.items()}
    report["step_contract"] = {}
    for name, f in envs.items():
        try:
            report["step_contract"][name] = step_return_shape(f())
        except Exception as exc:  # noqa: BLE001
            report["step_contract"][name] = {"error": f"{type(exc).__name__}: {exc}"}

    report["check_env"] = [run_check_env(f, name) for name, f in envs.items()]

    report["stable_baselines3"] = [
        run_sb3("Klondike", envs["CardGameEnv(Klondike)"], timesteps=2000),
        run_sb3("Macao", envs["CardGameEnv(Macao)"], timesteps=2000),
    ]

    # gymnasium wrappers: do they accept the library env at all?
    wrappers = {}
    for wname, wrapper in (
        ("TimeLimit", gym.wrappers.TimeLimit),
        ("RecordEpisodeStatistics", gym.wrappers.RecordEpisodeStatistics),
    ):
        try:
            kwargs = {"max_episode_steps": 50} if wname == "TimeLimit" else {}
            w = wrapper(envs["CardGameEnv(Macao)"](), **kwargs)
            w.reset(seed=1)
            wrappers[wname] = {"accepted": True}
        except Exception as exc:  # noqa: BLE001
            wrappers[wname] = {
                "accepted": False,
                "error_type": type(exc).__name__,
                "error": str(exc).strip(),
            }
    report["gymnasium_wrappers"] = wrappers

    path = os.path.join(OUT, "gymnasium_probe.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)
    print(json.dumps(report, indent=2, default=str))
    print(f"\nWrote {os.path.abspath(path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
