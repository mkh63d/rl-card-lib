"""Tests for game implementations."""

import warnings

import pytest
import numpy as np
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

from rl_card_lib.games import KlondikeSolitaire, Macao, register_gym_envs, registered_gym_ids
from rl_card_lib.env import CardGameEnv, MaskedCardGameEnv
from rl_card_lib.core.gym_wrapper import GymEnvWrapper


class TestKlondikeSolitaire:
    """Tests for Klondike Solitaire game."""
    
    def test_game_init(self):
        """Test game initialization."""
        game = KlondikeSolitaire()
        assert game.num_players == 1
        assert not game.done
    
    def test_game_reset(self):
        """Test game reset."""
        game = KlondikeSolitaire()
        obs = game.reset()
        
        assert isinstance(obs, np.ndarray)
        assert obs.shape == game.get_observation_shape()
    
    def test_game_legal_actions(self):
        """Test legal action retrieval."""
        game = KlondikeSolitaire()
        game.reset()
        
        legal = game.get_legal_actions()
        assert isinstance(legal, list)
        assert len(legal) > 0  # Should always have at least draw action
        assert 0 in legal  # Draw from stock should be legal initially
    
    def test_game_step(self):
        """Test taking a step."""
        game = KlondikeSolitaire()
        game.reset()
        
        legal = game.get_legal_actions()
        action = legal[0]
        
        obs, reward, terminated, truncated, info = game.step(action)
        
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(info, dict)
    
    def test_game_render(self):
        """Test game rendering."""
        game = KlondikeSolitaire()
        game.reset()
        
        rendered = game.render()
        assert isinstance(rendered, str)
        assert "Klondike" in rendered


class TestMacao:
    """Tests for Macao game."""
    
    def test_game_init(self):
        """Test game initialization."""
        game = Macao(num_players=2)
        assert game.num_players == 2
    
    def test_game_reset(self):
        """Test game reset."""
        game = Macao()
        obs = game.reset()
        
        assert isinstance(obs, np.ndarray)
        assert len(game.players) == 2
        assert all(len(p.hand) == 5 for p in game.players)
    
    def test_game_legal_actions(self):
        """Test legal action retrieval."""
        game = Macao()
        game.reset()
        
        legal = game.get_legal_actions()
        assert isinstance(legal, list)
        assert len(legal) > 0
    
    def test_game_step(self):
        """Test taking a step."""
        game = Macao()
        game.reset()
        
        legal = game.get_legal_actions()
        action = legal[0]
        
        obs, reward, terminated, truncated, info = game.step(action)
        
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)


class TestCardGameEnv:
    """Tests for Gymnasium environment wrapper."""
    
    def test_env_creation(self):
        """Test environment creation."""
        game = KlondikeSolitaire()
        env = CardGameEnv(game)
        
        assert env.action_space.n == game.get_action_space_size()
    
    def test_env_reset(self):
        """Test environment reset."""
        game = KlondikeSolitaire()
        env = CardGameEnv(game)
        
        obs, info = env.reset(seed=42)
        
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)
        assert "legal_actions" in info
    
    def test_env_step(self):
        """Test environment step."""
        game = KlondikeSolitaire()
        env = CardGameEnv(game)
        
        obs, info = env.reset()
        legal = info["legal_actions"]
        action = legal[0]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert isinstance(obs, np.ndarray)
        assert obs.dtype == np.float32
    
    def test_env_invalid_action(self):
        """Test handling of invalid actions."""
        game = KlondikeSolitaire()
        env = CardGameEnv(game)
        
        obs, info = env.reset()
        legal = info["legal_actions"]
        
        # Find an illegal action
        invalid_action = None
        for i in range(game.get_action_space_size()):
            if i not in legal:
                invalid_action = i
                break
        
        if invalid_action is not None:
            obs, reward, _, _, info = env.step(invalid_action)
            assert reward < 0  # Should get negative reward
            assert info.get("invalid_action", False)


class TestKlondikeGameplay:
    """Extended tests for Klondike game mechanics."""

    def test_draw_from_stock(self):
        """Test drawing cards from stock."""
        game = KlondikeSolitaire()
        game.reset()
        
        initial_stock = len(game.stock)
        initial_waste = len(game.waste)
        
        # Action 0 is draw
        game.step(0)
        
        # Stock should decrease, waste should increase
        assert len(game.stock) < initial_stock or len(game.waste) > initial_waste

    def test_draw_from_empty_stock(self):
        """Test deck recycling when stock is empty."""
        game = KlondikeSolitaire()
        game.reset()
        
        # Empty stock by drawing all cards
        while game.stock:
            game.step(0)
        
        # Now draw should recycle waste
        if game.waste:
            game.step(0)

    def test_move_waste_to_tableau(self):
        """Test moving a card from waste to tableau."""
        game = KlondikeSolitaire()
        game.reset()
        
        # Draw until we have a waste card
        for _ in range(10):
            game.step(0)
        
        # Try all possible waste-to-tableau moves
        legal = game.get_legal_actions()
        for action in legal:
            if 1 <= action <= 7:
                game.step(action)
                break

    def test_move_to_foundation(self):
        """Test moving cards to foundation."""
        game = KlondikeSolitaire()
        game.reset()
        
        # Play random moves, including foundation moves if possible
        for _ in range(100):
            if game.is_game_over():
                break
            legal = game.get_legal_actions()
            # Prefer foundation moves
            for action in legal:
                if 8 <= action <= 11 or 12 <= action <= 18:
                    game.step(action)
                    break
            else:
                game.step(legal[0])

    def test_move_tableau_to_tableau(self):
        """Test moving cards between tableaux."""
        game = KlondikeSolitaire()
        game.reset()
        
        # Find a tableau-to-tableau move
        for _ in range(50):
            if game.is_game_over():
                break
            legal = game.get_legal_actions()
            for action in legal:
                if action >= 19:
                    game.step(action)
                    break
            else:
                game.step(legal[0])

    def test_play_multiple_moves(self):
        """Test playing many moves."""
        game = KlondikeSolitaire()
        game.reset()
        
        for _ in range(50):
            if game.is_game_over():
                break
            legal = game.get_legal_actions()
            action = legal[0]
            game.step(action)

    def test_action_to_string_all_types(self):
        """Test action string conversion for all action types."""
        game = KlondikeSolitaire()
        
        assert "Draw" in game.action_to_string(0)
        assert "waste" in game.action_to_string(1).lower()
        assert "foundation" in game.action_to_string(8).lower()
        assert "foundation" in game.action_to_string(12).lower()
        assert "tableau" in game.action_to_string(19).lower()

    def test_game_over_no_moves(self):
        """Test game over detection."""
        game = KlondikeSolitaire()
        game.reset()
        assert not game.is_game_over()


class TestMacaoGameplay:
    """Extended tests for Macao game mechanics."""

    def test_play_card(self):
        """Test playing a valid card."""
        game = Macao(num_players=2)
        game.reset()
        
        for _ in range(20):
            if game.is_game_over():
                break
            legal = game.get_legal_actions()
            action = legal[0]
            game.step(action)

    def test_draw_card(self):
        """Test drawing a card (action 52)."""
        game = Macao(num_players=2)
        game.reset()

        # Force a draw by trying action 52
        if 52 in game.get_legal_actions():
            game.step(52)
            # Hand size may change

    def test_full_game_simulation(self):
        """Simulate a complete game."""
        game = Macao(num_players=2, max_turns=100)
        game.reset()
        
        while not game.is_game_over():
            legal = game.get_legal_actions()
            if not legal:
                break
            game.step(legal[0])

    def test_special_cards(self):
        """Test special card effects by playing many moves."""
        import random
        game = Macao(num_players=2, max_turns=200)
        game.reset()
        
        random.seed(42)
        for _ in range(100):
            if game.is_game_over():
                break
            legal = game.get_legal_actions()
            action = random.choice(legal)
            game.step(action)

    def test_pass_action(self):
        """Test the pass action (53)."""
        game = Macao(num_players=2)
        game.reset()
        
        # Play until pass is needed
        for _ in range(50):
            if game.is_game_over():
                break
            legal = game.get_legal_actions()
            if 53 in legal:
                game.step(53)
                break
            game.step(legal[0])

    def test_action_to_string(self):
        """Test action string conversion."""
        game = Macao()
        
        assert "Draw" in game.action_to_string(52)
        assert "Pass" in game.action_to_string(53)
        assert "Play" in game.action_to_string(0)

    def test_render(self):
        """Test rendering."""
        game = Macao()
        game.reset()
        rendered = game.render()
        assert "Macao" in rendered
        assert "Player" in rendered

    def test_max_turns_truncation(self):
        """Test game truncation at max turns."""
        game = Macao(num_players=2, max_turns=5)
        game.reset()
        
        for _ in range(10):
            if game.is_game_over():
                break
            legal = game.get_legal_actions()
            game.step(legal[0])
        
        assert game._turn_count <= 10


class TestKlondikeEdgeCases:
    """Test edge cases in Klondike."""

    def test_max_passes_limit(self):
        """Test max passes limit."""
        game = KlondikeSolitaire(max_passes=3)
        game.reset()
        
        # Cycle through deck multiple times
        for _ in range(200):
            if game.is_game_over():
                break
            game.step(0)  # Keep drawing

    def test_draw_three_mode(self):
        """Test draw 3 cards mode."""
        game = KlondikeSolitaire(draw_count=3)
        game.reset()
        
        # Should draw 3 cards at a time
        initial_stock = len(game.stock)
        game.step(0)
        assert len(game.stock) <= initial_stock - 1


class TestKlondikeInternalMethods:
    """Direct tests for Klondike internal methods."""

    def test_can_place_on_empty_tableau(self):
        """Test placing King on empty tableau."""
        game = KlondikeSolitaire()
        game.reset()
        
        # Clear a tableau pile
        game.tableaux[0] = []
        
        # Only King can go on empty
        from rl_card_lib.cardgames import Card, Suit, Rank
        king = Card(Suit.HEARTS, Rank.KING)
        queen = Card(Suit.HEARTS, Rank.QUEEN)
        
        assert game._can_place_on_tableau(king, 0) is True
        assert game._can_place_on_tableau(queen, 0) is False

    def test_can_place_on_foundation(self):
        """Test placing on foundation."""
        game = KlondikeSolitaire()
        game.reset()
        
        from rl_card_lib.cardgames import Card, Suit, Rank
        
        # Empty foundation - only Ace can go
        ace = Card(Suit.HEARTS, Rank.ACE)
        two = Card(Suit.HEARTS, Rank.TWO)
        
        assert game._can_place_on_foundation(ace, int(Suit.HEARTS)) is True
        assert game._can_place_on_foundation(two, int(Suit.HEARTS)) is False

    def test_invalid_tableau_move(self):
        """Test invalid tableau to tableau move."""
        game = KlondikeSolitaire()
        game.reset()
        
        # Try to move from empty pile
        game.tableaux[0] = []
        reward = game._move_tableau_to_tableau(0, 1)
        assert reward < 0

    def test_invalid_waste_to_foundation(self):
        """Test invalid waste to foundation move."""
        game = KlondikeSolitaire()
        game.reset()
        game.waste = []
        
        reward = game._move_waste_to_foundation(0)
        assert reward < 0

    def test_invalid_waste_to_tableau(self):
        """Test invalid waste to tableau move."""
        game = KlondikeSolitaire()
        game.reset()
        game.waste = []
        
        reward = game._move_waste_to_tableau(0)
        assert reward < 0

    def test_invalid_tableau_to_foundation(self):
        """Test invalid tableau to foundation move."""
        game = KlondikeSolitaire()
        game.reset()
        
        # Empty tableau
        game.tableaux[0] = []
        reward = game._move_tableau_to_foundation(0)
        assert reward < 0


class TestCardGameEnvExtended:
    """Extended tests for CardGameEnv."""

    def test_max_steps_truncation(self):
        """Test truncation at max steps."""
        game = KlondikeSolitaire()
        env = CardGameEnv(game, max_steps=5)
        
        obs, info = env.reset()
        
        for _ in range(10):
            legal = info.get("legal_actions", [0])
            obs, reward, terminated, truncated, info = env.step(legal[0])
            if terminated or truncated:
                break
        
        assert truncated or terminated

    def test_illegal_actions_truncate_at_max_steps(self):
        """An episode of nothing but illegal actions still ends."""
        game = Macao(num_players=2)
        env = CardGameEnv(game, max_steps=20)

        obs, info = env.reset(seed=0)
        illegal = next(
            i
            for i in range(game.get_action_space_size())
            if i not in info["legal_actions"]
        )

        truncated = False
        for step in range(1, 101):
            obs, reward, terminated, truncated, info = env.step(illegal)
            assert info.get("invalid_action", False)
            assert not terminated
            if truncated:
                break

        assert truncated, "illegal actions livelocked instead of truncating"
        assert step == 20

    def test_illegal_and_legal_steps_share_the_budget(self):
        """An illegal action spends the same budget a legal one does."""
        game = Macao(num_players=2)
        env = CardGameEnv(game, max_steps=4)

        obs, info = env.reset(seed=0)
        illegal = next(
            i
            for i in range(game.get_action_space_size())
            if i not in info["legal_actions"]
        )

        for _ in range(3):
            obs, reward, terminated, truncated, info = env.step(illegal)
            assert not truncated

        # Fourth step is legal, and finds the budget already spent down to it.
        obs, reward, terminated, truncated, info = env.step(info["legal_actions"][0])
        assert truncated or terminated

    def test_render_ansi(self):
        """Test ANSI rendering."""
        game = KlondikeSolitaire()
        env = CardGameEnv(game, render_mode="ansi")
        env.reset()
        result = env.render()
        assert isinstance(result, str)

    def test_close(self):
        """Test env close."""
        game = KlondikeSolitaire()
        env = CardGameEnv(game)
        env.reset()
        env.close()  # Should not raise


class TestMaskedCardGameEnv:
    """Tests for masked environment."""

    def test_creation(self):
        from rl_card_lib.env.card_game_env import MaskedCardGameEnv
        game = KlondikeSolitaire()
        env = MaskedCardGameEnv(game)
        assert "observation" in env.observation_space.spaces
        assert "action_mask" in env.observation_space.spaces

    def test_reset(self):
        from rl_card_lib.env.card_game_env import MaskedCardGameEnv
        game = KlondikeSolitaire()
        env = MaskedCardGameEnv(game)
        
        obs, info = env.reset(seed=42)
        assert "observation" in obs
        assert "action_mask" in obs
        assert obs["action_mask"].dtype == np.int8

    def test_step(self):
        from rl_card_lib.env.card_game_env import MaskedCardGameEnv
        game = KlondikeSolitaire()
        env = MaskedCardGameEnv(game)
        
        obs, info = env.reset()
        legal = info["legal_actions"]
        obs, reward, terminated, truncated, info = env.step(legal[0])
        
        assert "observation" in obs
        assert "action_mask" in obs


class TestCardGameEnvMethods:
    """Test additional env methods."""

    def test_get_legal_actions(self):
        game = KlondikeSolitaire()
        env = CardGameEnv(game)
        env.reset()
        
        actions = env.get_legal_actions()
        assert isinstance(actions, list)

    def test_get_legal_action_mask(self):
        game = KlondikeSolitaire()
        env = CardGameEnv(game)
        env.reset()
        
        mask = env.get_legal_action_mask()
        assert mask.dtype == bool

    def test_action_to_string(self):
        game = KlondikeSolitaire()
        env = CardGameEnv(game)
        env.reset()
        
        s = env.action_to_string(0)
        assert isinstance(s, str)

    def test_render_human(self, capsys):
        game = KlondikeSolitaire()
        env = CardGameEnv(game, render_mode="human")
        env.reset()
        env.render()
        
        captured = capsys.readouterr()
        assert "Klondike" in captured.out

    def test_render_none_mode(self):
        game = KlondikeSolitaire()
        env = CardGameEnv(game, render_mode=None)
        env.reset()
        result = env.render()
        assert result is None

    def test_step_with_human_render_mode(self, capsys):
        """Test that step() calls render when render_mode='human'."""
        game = KlondikeSolitaire()
        env = CardGameEnv(game, render_mode="human")
        env.reset()
        
        # Take a step - should auto-render
        legal = env.get_legal_actions()
        env.step(legal[0])
        
        captured = capsys.readouterr()
        assert "Klondike" in captured.out


class TestGymnasiumContract:
    """The envs are real `gymnasium.Env`s, not just Gymnasium-shaped (#15).

    Every one of these went through the `isinstance(env, gymnasium.Env)` gate
    that `check_env`, the wrappers and Stable-Baselines3 all apply first, and
    every one of them used to fail it.
    """

    ENV_FACTORIES = [
        pytest.param(lambda: CardGameEnv(KlondikeSolitaire(seed=0), max_steps=300),
                     id="CardGameEnv-klondike"),
        pytest.param(lambda: CardGameEnv(Macao(num_players=2, seed=0), max_steps=200),
                     id="CardGameEnv-macao"),
        pytest.param(lambda: MaskedCardGameEnv(Macao(num_players=2, seed=0), max_steps=200),
                     id="MaskedCardGameEnv-macao"),
        pytest.param(lambda: GymEnvWrapper(Macao(num_players=2, seed=0)),
                     id="GymEnvWrapper-macao"),
    ]

    @pytest.mark.parametrize("factory", ENV_FACTORIES)
    def test_is_gymnasium_env_subclass(self, factory):
        """The entry test every Gymnasium consumer applies before any other."""
        assert isinstance(factory(), gym.Env)

    @pytest.mark.parametrize("factory", ENV_FACTORIES)
    def test_declares_render_modes(self, factory):
        """`metadata` has to advertise the modes `render()` actually honours."""
        env = factory()
        assert set(env.metadata["render_modes"]) == {"human", "ansi"}

    @pytest.mark.parametrize("factory", ENV_FACTORIES)
    def test_reset_seeds_np_random(self, factory):
        """`reset(seed=...)` must seed `self.np_random`, per the contract.

        Two envs reset with the same seed draw the same numbers; a different
        seed moves them. Without `super().reset(seed=seed)` the attribute did
        not exist at all.
        """
        a, b, c = factory(), factory(), factory()
        a.reset(seed=1234)
        b.reset(seed=1234)
        c.reset(seed=5678)

        assert a.np_random.random() == b.np_random.random()
        assert a.np_random.random() != c.np_random.random()

    @pytest.mark.parametrize("factory", ENV_FACTORIES)
    def test_passes_gymnasium_env_checker(self, factory):
        """The whole contract, not just the entry test.

        `check_env` used to stop at the `isinstance` assertion, so the rest was
        never actually verified. Render is checked separately -- the checker
        wants a `render_mode` set at construction.
        """
        check_env(factory(), skip_render_check=True)

    @pytest.mark.parametrize("factory", ENV_FACTORIES)
    def test_accepts_gymnasium_wrappers(self, factory):
        """Wrappers assert on the base class before they wrap anything."""
        limited = gym.wrappers.TimeLimit(factory(), max_episode_steps=50)
        assert limited.unwrapped is not limited

        recorded = gym.wrappers.RecordEpisodeStatistics(factory())
        obs, info = recorded.reset(seed=7)
        assert isinstance(info, dict)

    def test_unwrapped_reaches_the_library_env(self):
        """`unwrapped` has to hand back the env holding the game."""
        env = CardGameEnv(Macao(num_players=2, seed=0), max_steps=200)
        assert gym.wrappers.TimeLimit(env, 50).unwrapped is env
        assert env.unwrapped.game is env.game


class TestGymnasiumRegistration:
    """The bundled games are reachable through `gymnasium.make` (#15)."""

    def test_ids_are_registered(self):
        for env_id in registered_gym_ids():
            assert env_id in gym.registry

    @pytest.mark.parametrize("env_id", registered_gym_ids())
    def test_make_produces_a_working_env(self, env_id):
        """A full reset/step round trip through `gymnasium.make`."""
        env = gym.make(env_id)
        assert env.spec.id == env_id

        obs, info = env.reset(seed=11)
        assert obs in env.observation_space

        action = (info.get("legal_actions") or [0])[0]
        observation, reward, terminated, truncated, info = env.step(action)
        assert observation in env.observation_space
        assert isinstance(reward, float)
        assert isinstance(terminated, bool) and isinstance(truncated, bool)

    def test_masked_ids_expose_the_action_mask(self):
        """The masked ids carry the mask in the observation, as a feature.

        This is the `Dict` shape a policy can *see* the mask through. It is not
        how `sb3-contrib` finds the mask -- that is `action_masks()`, covered
        below -- and conflating the two is what left MaskablePPO sampling
        illegal actions despite this field being present.
        """
        env = gym.make("rl_card_lib/MacaoMasked-v0")
        obs, info = env.reset(seed=11)
        assert set(obs) == {"observation", "action_mask"}
        assert obs["action_mask"].sum() == len(info["legal_actions"])

    @pytest.mark.parametrize("env_id", registered_gym_ids())
    def test_action_masks_survives_the_wrapper_chain(self, env_id):
        """`get_wrapper_attr` reaches the mask through `gymnasium.make`.

        This is exactly how `sb3_contrib.common.maskable.utils.get_action_masks`
        asks a bare env for its mask, so it is the regression guard for #40:
        `gymnasium.make` returns an `OrderEnforcing` wrapper, and if the method
        stopped being reachable through it, MaskablePPO would silently fall back
        to sampling illegally rather than raising.

        Parametrized over *every* id, masked or not: the mask and the `Dict`
        observation are independent, and an algorithm that needs only the mask
        may take the plain `Box` env.
        """
        env = gym.make(env_id)
        _, info = env.reset(seed=11)

        mask = env.get_wrapper_attr("action_masks")()
        assert mask.dtype == np.bool_
        assert sorted(np.flatnonzero(mask)) == sorted(info["legal_actions"])

    @pytest.mark.parametrize("env_class", [CardGameEnv, MaskedCardGameEnv])
    def test_action_masks_agrees_with_get_legal_action_mask(self, env_class):
        """The ecosystem spelling and the library's own cannot drift apart."""
        env = env_class(Macao(num_players=2), max_steps=200)
        env.reset(seed=11)
        assert np.array_equal(env.action_masks(), env.get_legal_action_mask())

    def test_entry_points_are_importable_strings(self):
        """Strings, not closures, so another process can rebuild `env.spec`.

        A `SubprocVecEnv` worker reconstructs the env from its spec; a lambda
        entry point cannot survive that trip.
        """
        for env_id in registered_gym_ids():
            assert isinstance(gym.registry[env_id].entry_point, str)

    def test_registration_is_idempotent(self):
        """Importing twice must not warn about overriding an existing id."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            register_gym_envs()

    def test_step_cap_is_the_env_not_a_second_time_limit(self):
        """One truncation authority: CardGameEnv.max_steps.

        `max_episode_steps` is left unset on the registrations so Gymnasium
        does not stack a second TimeLimit that disagrees about which step
        ended the episode.
        """
        env = gym.make("rl_card_lib/Macao-v0")
        assert env.spec.max_episode_steps is None
        assert env.unwrapped.max_steps == 200
