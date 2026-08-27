"""MaskablePPO actually learns Macao, rather than merely being accepted by it.

Skipped wholesale when `sb3-contrib` is absent. It is an optional extra, not a
dependency, and CI installs neither it nor stable-baselines3 -- the point of the
extra is that the library stays installable without a training framework
attached. Run these with:

    pip install -e "./packages/examples[sb3]"
"""

import numpy as np
import pytest

pytest.importorskip("sb3_contrib")

from rl_card_lib.env import CardGameEnv                        # noqa: E402
from rl_card_lib.games import Macao, MacaoHeuristicAgent       # noqa: E402
from rl_card_lib.harness.evaluation import evaluate_macao      # noqa: E402
from rl_card_lib.harness.sb3_maskable import (                 # noqa: E402
    MaskablePPOAgent,
    train_maskable_ppo,
)

#: Enough gradient steps for the run to be real, few enough to stay a test.
SMOKE_TIMESTEPS = 2048


@pytest.fixture(scope="module")
def trained_agent():
    """One short training run shared by every test here."""
    model = train_maskable_ppo(SMOKE_TIMESTEPS, seed=0, verbose=0)
    agent = MaskablePPOAgent(model)
    agent.eval()
    return agent


def test_masking_is_supported_on_the_registered_id():
    """sb3-contrib's own probe, not our reimplementation of it."""
    import gymnasium

    import rl_card_lib.games  # noqa: F401  (registration side effect)
    from sb3_contrib.common.maskable.utils import is_masking_supported

    assert is_masking_supported(gymnasium.make("rl_card_lib/MacaoMasked-v0"))


def test_trained_policy_plays_legally(trained_agent):
    """The regression guard for #40.

    Before `action_masks()` existed, an SB3 policy on Macao scored -198.9 over
    200 steps -- about 200 x `invalid_action_reward`, one illegal action per
    step, because 2--4 of 65 actions are legal in a typical position. Asserting
    on the illegal count rather than on strength keeps this a test of the
    plumbing: it fails if the mask stops reaching the algorithm, and does not
    fail merely because a 2048-step policy plays badly.
    """
    game = Macao(num_players=2)
    env = CardGameEnv(game, max_steps=200)
    observation, info = env.reset(seed=100_000)

    total, illegal = 0.0, 0
    for _ in range(200):
        action = trained_agent.select_action(observation, info.get("legal_actions"))
        observation, reward, terminated, truncated, info = env.step(action)
        total += reward
        illegal += bool(info.get("invalid_action"))
        if terminated or truncated:
            break

    assert illegal == 0
    assert total > 200 * env.invalid_action_reward


def test_select_action_only_ever_returns_a_legal_move(trained_agent):
    """Whatever the policy prefers, the mask is what it is allowed to do."""
    game = Macao(num_players=2)
    env = CardGameEnv(game, max_steps=200)
    observation, info = env.reset(seed=100_001)

    for _ in range(50):
        legal = info.get("legal_actions")
        action = trained_agent.select_action(observation, legal)
        assert action in legal
        observation, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break


def test_evaluates_through_the_shared_protocol(trained_agent):
    """It seats in `evaluate_macao` like any bundled agent, on TEST deals."""
    result = evaluate_macao(
        trained_agent, MacaoHeuristicAgent(seed=0), episodes=5,
    )
    assert set(result) == {"win_rate", "draw_rate"}
    assert 0.0 <= result["win_rate"] <= 1.0


def test_round_trips_through_save_and_load(trained_agent, tmp_path):
    """`checkpoint_suffix` is `.zip`, because MaskablePPO.save writes an archive."""
    assert trained_agent.checkpoint_suffix == ".zip"

    path = tmp_path / ("model" + trained_agent.checkpoint_suffix)
    trained_agent.save(str(path))
    assert path.exists()

    reloaded = MaskablePPOAgent(trained_agent.model)
    reloaded.load(str(path))
    reloaded.eval()

    game = Macao(num_players=2)
    env = CardGameEnv(game, max_steps=200)
    observation, info = env.reset(seed=100_002)
    legal = info.get("legal_actions")
    assert reloaded.select_action(observation, legal) in legal


def test_falls_back_to_every_action_when_no_mask_is_offered(trained_agent):
    """A missing mask must not forbid everything."""
    game = Macao(num_players=2)
    env = CardGameEnv(game, max_steps=200)
    observation, _ = env.reset(seed=100_003)

    action = trained_agent.select_action(observation, None)
    assert 0 <= action < int(np.asarray(env.action_space.n))
