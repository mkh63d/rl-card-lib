"""Simple training loop and metrics collection for games/environments.

This Trainer is intentionally lightweight: it accepts either a Gym-like
env (with `reset`/`step`) or a `Game` instance and runs episodes while
collecting basic episode statistics.
"""
from typing import Any, Dict, List, Optional
import time
import numpy as np


class Trainer:
    """Run training episodes and collect simple metrics.

    Args:
        env: Either a Gym-like environment or an object exposing
             `reset()` and `step(action)`.
        agent: Optional agent object. If provided it must implement
               `select_action(observation, legal_actions=None)`.
    """

    def __init__(self, env: Any, agent: Optional[Any] = None, max_steps_per_episode: int = 1000):
        self.env = env
        self.agent = agent
        self.max_steps = max_steps_per_episode

    def _random_action(self, game: Any):
        # Try to use game-provided legal actions, else sample uniformly
        try:
            legal = game.get_legal_actions()
            if legal:
                return int(np.random.choice(legal))
        except Exception:
            pass
        try:
            # fall back to action_space if present
            if hasattr(self.env, "action_space") and getattr(self.env, "action_space") is not None:
                return int(self.env.action_space.sample())
        except Exception:
            pass
        return 0

    def train(self, num_episodes: int = 100) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []

        for ep in range(num_episodes):
            start = time.time()
            # reset API: support both gym-like (returns obs, info) and simple
            reset_res = self.env.reset()
            if isinstance(reset_res, tuple):
                obs = reset_res[0]
            else:
                obs = reset_res

            total_reward = 0.0
            steps = 0
            done = False

            while not done and steps < self.max_steps:
                if self.agent is not None:
                    try:
                        legal = None
                        if hasattr(self.env, "game"):
                            try:
                                legal = self.env.game.get_legal_actions()
                            except Exception:
                                legal = None
                        action = self.agent.select_action(obs, legal_actions=legal)
                    except Exception:
                        action = self._random_action(self.env)
                else:
                    action = self._random_action(self.env)

                step_res = self.env.step(action)
                # support gym's (obs, reward, terminated, truncated, info)
                if len(step_res) == 5:
                    obs, reward, terminated, truncated, info = step_res
                    done = bool(terminated or truncated)
                else:
                    obs, reward, done, info = step_res

                total_reward += float(reward)
                steps += 1

            results.append({
                "episode": ep,
                "reward": total_reward,
                "steps": steps,
                "time_s": time.time() - start,
            })

        return results
