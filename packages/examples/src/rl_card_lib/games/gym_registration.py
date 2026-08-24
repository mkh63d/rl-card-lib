"""Register the bundled games as Gymnasium ids, so `gymnasium.make` works.

Distinct from `registration.py` next door: that one fills the library's own
sweep/report registry, which drives the experiment scripts. This one fills the
*Gymnasium* registry, which is what an outside consumer reaches for --

    import rl_card_lib.games          # registers on import
    env = gymnasium.make("rl_card_lib/Macao-v0")

Entry points are module-qualified strings rather than closures so that
`env.spec` stays re-creatable in another process, which is what vectorised
runners such as Stable-Baselines3's `SubprocVecEnv` rebuild an env from.

`max_episode_steps` is deliberately left unset on the registrations: the step
cap belongs to `CardGameEnv.max_steps`, which reports the cut as `truncated`
and keeps the library's `info` contract. Setting it here as well would stack a
second `TimeLimit` on top of the first, and the two would disagree about which
step ended the episode.
"""

from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym

from rl_card_lib.env import CardGameEnv, MaskedCardGameEnv
from rl_card_lib.games.klondike import KlondikeSolitaire
from rl_card_lib.games.macao import Macao
from rl_card_lib.games.registration import KLONDIKE_MAX_STEPS, MACAO_MAX_STEPS

#: Namespace prefix for every id this module registers.
GYM_NAMESPACE = "rl_card_lib"


def make_klondike(max_steps: Optional[int] = KLONDIKE_MAX_STEPS,
                  **kwargs: Any) -> CardGameEnv:
    """Klondike in a plain `CardGameEnv`."""
    return CardGameEnv(KlondikeSolitaire(), max_steps=max_steps, **kwargs)


def make_macao(max_steps: Optional[int] = MACAO_MAX_STEPS,
               num_players: int = 2, **kwargs: Any) -> CardGameEnv:
    """Macao in a plain `CardGameEnv`."""
    return CardGameEnv(Macao(num_players=num_players), max_steps=max_steps,
                       **kwargs)


def make_klondike_masked(max_steps: Optional[int] = KLONDIKE_MAX_STEPS,
                         **kwargs: Any) -> MaskedCardGameEnv:
    """Klondike with the action mask in the observation."""
    return MaskedCardGameEnv(KlondikeSolitaire(), max_steps=max_steps, **kwargs)


def make_macao_masked(max_steps: Optional[int] = MACAO_MAX_STEPS,
                      num_players: int = 2, **kwargs: Any) -> MaskedCardGameEnv:
    """Macao with the action mask in the observation.

    The shape `sb3-contrib`'s `MaskablePPO` expects. Worth preferring over the
    plain ids: neither game makes more than a handful of its actions legal at
    once, so an unmasked policy spends nearly every step on the invalid-action
    penalty.
    """
    return MaskedCardGameEnv(Macao(num_players=num_players), max_steps=max_steps,
                             **kwargs)


#: id -> factory, as "module:function" strings for cross-process re-creation.
_ENV_IDS = {
    f"{GYM_NAMESPACE}/Klondike-v0": f"{__name__}:make_klondike",
    f"{GYM_NAMESPACE}/Macao-v0": f"{__name__}:make_macao",
    f"{GYM_NAMESPACE}/KlondikeMasked-v0": f"{__name__}:make_klondike_masked",
    f"{GYM_NAMESPACE}/MacaoMasked-v0": f"{__name__}:make_macao_masked",
}


def register_gym_envs() -> None:
    """Register the bundled games with Gymnasium. Idempotent.

    Called on `rl_card_lib.games` import. Ids already present are left alone,
    so importing twice does not make Gymnasium warn about an override, and a
    caller who registered their own variant under one of these ids keeps it.
    """
    for env_id, entry_point in _ENV_IDS.items():
        if env_id in gym.registry:
            continue
        gym.register(id=env_id, entry_point=entry_point)


def registered_gym_ids() -> list[str]:
    """The ids `register_gym_envs()` installs, in registration order."""
    return list(_ENV_IDS)
