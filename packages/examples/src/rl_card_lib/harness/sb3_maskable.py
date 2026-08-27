"""Training and evaluating `sb3-contrib`'s MaskablePPO on the bundled Macao.

This is the third-party half of the Gymnasium compatibility claim. The library's
own agents are handed `info["legal_actions"]` and never propose an illegal move,
so they say nothing about whether an outside algorithm can learn these games. An
unmasked SB3 PPO cannot: in a typical Macao position 2--4 of 65 actions are
legal, and a 2000-step run scored -198.9 over 200 steps, almost exactly
200 x `invalid_action_reward`. It never found a legal move.

MaskablePPO can, because it asks the environment for the mask. It finds it by
calling a method named exactly `action_masks()` -- *not* by reading the
`action_mask` field of `MaskedCardGameEnv`'s `Dict` observation, which is a
common misreading of what that field is for. `CardGameEnv.action_masks()`
supplies it, so `gymnasium.make(...)` is enough and no `ActionMasker` adapter is
needed.

Why one policy playing both seats is a well-posed problem: `Macao` encodes the
observation from the *current* player's side (its own hand, the others' hand
sizes) and pays every reward to the actor -- "Rewards are actor-relative: every
reward `step()` returns is paid to the player who took the action, including the
terminal one." So a single policy driving both seats always sees the position
from the mover's side and is paid for the mover's outcome. That is the same
self-play arrangement `SelfPlayTrainer` gives the bundled agents, which is what
keeps the resulting number comparable to theirs.

This module imports `sb3_contrib` at module scope, so importing it is itself the
dependency check: `sb3-contrib` is an optional extra
(`pip install -e "./packages/examples[sb3]"`), not a dependency of the library.
That is also why `rl_card_lib.harness.__init__` does not re-export it -- doing so
would make every harness import fail for anyone without the extra.
"""

from __future__ import annotations

from typing import Optional

import gymnasium
import numpy as np
from sb3_contrib import MaskablePPO

from rl_card_lib.agents import Agent
from rl_card_lib.harness.deals import TRAIN_SEEDS

#: The registered id this module trains on. Masked, so the `Dict` observation
#: carries the mask as a feature as well; `action_masks()` is what MaskablePPO
#: actually reads.
MACAO_MASKED_ID = "rl_card_lib/MacaoMasked-v0"

#: Matches `MACAO_MAX_STEPS` in `rl_card_lib.games.registration`, so a run here
#: is capped exactly like every other Macao measurement.
MACAO_MAX_STEPS = 200


def make_training_env(
    env_id: str = MACAO_MASKED_ID, *, max_steps: int = MACAO_MAX_STEPS,
) -> gymnasium.Env:
    """Build the training env, dealing only from the TRAIN pool.

    `deal_seeds` reaches the env constructor through the registered factory's
    `**kwargs`, so the agent never trains on a deal it will later be evaluated
    on -- the same guarantee the bundled training envs get in
    `rl_card_lib.games.registration`.
    """
    return gymnasium.make(env_id, max_steps=max_steps, deal_seeds=TRAIN_SEEDS)


def train_maskable_ppo(
    total_timesteps: int, *, seed: int = 0, env_id: str = MACAO_MASKED_ID,
    max_steps: int = MACAO_MAX_STEPS, n_steps: int = 2048,
    batch_size: int = 64, learning_rate: float = 3e-4,
    verbose: int = 1,
) -> MaskablePPO:
    """Train MaskablePPO on the masked id and return the fitted model.

    `MultiInputPolicy` because the masked env's observation is a `Dict`. The
    hyperparameters are sb3-contrib's own defaults apart from the seed: the
    point of this example is that a third-party algorithm learns the game
    off-the-shelf, which a tuned configuration would undercut.

    Args:
        total_timesteps: Environment steps to train for
        seed: Seeds the policy's initialisation and its action sampling
        env_id: Registered id to train on (must expose `action_masks()`)
        max_steps: Move cap per episode
        n_steps: Rollout length between updates
        batch_size: Minibatch size
        learning_rate: Adam step size
        verbose: Passed to MaskablePPO (1 prints its own progress table)

    Returns:
        The trained model
    """
    env = make_training_env(env_id, max_steps=max_steps)
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        seed=seed,
        n_steps=n_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        verbose=verbose,
    )
    model.learn(total_timesteps=total_timesteps)
    return model


class MaskablePPOAgent(Agent):
    """A trained MaskablePPO seated in the library's evaluation protocols.

    The protocols in `rl_card_lib.harness.evaluation` step a plain
    `CardGameEnv`, so they hand out the flat `Box` observation and the legal
    moves separately in `info["legal_actions"]`. The model was trained on the
    `Dict` observation, so this rebuilds that dict and passes the mask through
    `action_masks=` as well -- the observation field feeds the network, the
    keyword is what actually constrains the sampled action.

    Evaluating through those protocols rather than SB3's own
    `evaluate_policy` is the point: it puts this agent on the same 200 held-out
    deals, against the same opponents, as every bundled agent.
    """

    #: MaskablePPO.save() writes a zip archive, not a torch pickle.
    checkpoint_suffix = ".zip"

    def __init__(
        self, model: MaskablePPO, name: str = "MaskablePPO",
        deterministic: bool = True,
    ):
        super().__init__(name=name)
        self.model = model
        self.deterministic = deterministic
        self._action_size = int(model.action_space.n)

    def select_action(
        self, observation: np.ndarray, legal_actions: Optional[list[int]] = None,
    ) -> int:
        mask = np.zeros(self._action_size, dtype=bool)
        if legal_actions:
            mask[list(legal_actions)] = True
        else:
            # No mask offered: let the policy choose freely rather than
            # silently forbidding everything, which would make predict()
            # return an arbitrary action with no signal that it had.
            mask[:] = True

        observation = {
            "observation": np.asarray(observation, dtype=np.float32),
            "action_mask": mask.astype(np.int8),
        }
        # Greedy in eval mode, sampled while training, matching the convention
        # the library's own agents follow -- the protocols call eval() before
        # the first deal, so a measurement is reproducible.
        action, _ = self.model.predict(
            observation,
            action_masks=mask,
            deterministic=self.deterministic and not self.training,
        )
        return int(action)

    def save(self, path: str) -> None:
        self.model.save(path)

    def load(self, path: str) -> None:
        self.model = MaskablePPO.load(path)
        self._action_size = int(self.model.action_space.n)


__all__ = [
    "MACAO_MASKED_ID",
    "MACAO_MAX_STEPS",
    "MaskablePPOAgent",
    "make_training_env",
    "train_maskable_ppo",
]
