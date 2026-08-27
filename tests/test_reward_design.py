"""Tests pinning the reward and action-space fixes from the TODO audit.

Each test here is a regression guard for a specific defect that distorted
results: the Klondike reward loop, the missing loss terminal, Macao's
player-0-centric terminal reward, the unlearnable Ace/Jack declarations, the
global RNG reseeding, and the search bugs those exposed in MCTSAgent.
"""

import inspect
import random

import numpy as np
import pytest

from rl_card_lib.agents import MCTSAgent, RandomAgent
from rl_card_lib.cardgames import Card, Rank, Suit
from rl_card_lib.env import CardGameEnv
from rl_card_lib.games import KlondikeSolitaire, Macao, bundled_klondike
from rl_card_lib.games.gym_registration import make_klondike, make_klondike_masked
from rl_card_lib.games.registration import KLONDIKE_REPEAT_PENALTY
from rl_card_lib.harness.baselines import run_klondike_baselines
from rl_card_lib.harness.evaluation import evaluate_klondike
from rl_card_lib.harness.registry import sweep_game


def fresh_macao_duel(hand0, hand1, top=None):
    """Two-player Macao with fixed hands and a neutral discard top."""
    game = Macao(num_players=2, seed=0)
    game.reset()
    game.discard_pile = [top or Card(Suit.CLUBS, Rank.NINE, True)]
    game.requested_suit = None
    game.requested_rank = None
    game.draw_penalty = 0
    game.skip_next = False
    game.pending_declaration = None
    game.current_player_idx = 0
    game.players[0].hand = list(hand0)
    game.players[1].hand = list(hand1)
    return game


class TestKlondikeRewardLoop:
    """The tableau-shuffle loop must never pay again."""

    @staticmethod
    def _two_pile_loop():
        """A red 7 movable back and forth between two black 8s."""
        game = KlondikeSolitaire(seed=0)
        game.foundations = [[] for _ in range(4)]
        game.tableaux = [[] for _ in range(7)]
        game.tableaux[0] = [Card(Suit.SPADES, Rank.EIGHT, True)]
        game.tableaux[1] = [
            Card(Suit.CLUBS, Rank.EIGHT, True),
            Card(Suit.HEARTS, Rank.SEVEN, True),
        ]
        game.stock = [Card(Suit.DIAMONDS, Rank.KING, False)]
        game.waste = []
        return game

    def test_non_revealing_tableau_move_nets_a_loss(self):
        game = self._two_pile_loop()
        # Move the 7 from pile 1 to pile 0: reveals nothing, reversible.
        _, reward, _, _, _ = game.step(19 + 1 * 7 + 0)
        assert reward == pytest.approx(-0.01)

    def test_shuffling_back_and_forth_only_loses(self):
        game = self._two_pile_loop()
        total = 0.0
        for _ in range(10):
            _, reward, _, _, _ = game.step(19 + 1 * 7 + 0)
            total += reward
            _, reward, _, _, _ = game.step(19 + 0 * 7 + 1)
            total += reward
        assert total == pytest.approx(-0.20)

    def test_reveal_still_pays(self):
        game = KlondikeSolitaire(seed=0)
        game.foundations = [[] for _ in range(4)]
        game.tableaux = [[] for _ in range(7)]
        game.tableaux[0] = [Card(Suit.SPADES, Rank.EIGHT, True)]
        game.tableaux[1] = [
            Card(Suit.DIAMONDS, Rank.TWO, False),
            Card(Suit.HEARTS, Rank.SEVEN, True),
        ]
        game.stock = [Card(Suit.DIAMONDS, Rank.KING, False)]
        game.waste = []

        _, reward, _, _, _ = game.step(19 + 1 * 7 + 0)
        assert reward == pytest.approx(0.2 - 0.01)
        assert game.tableaux[1][-1].face_up


class TestKlondikeTerminals:
    """A dead deal must end as a loss, not run out the clock."""

    @staticmethod
    def _one_draw_from_dead(**kwargs):
        """One legal draw left, after which nothing is playable."""
        game = KlondikeSolitaire(max_passes=1, **kwargs)
        game.foundations = [[] for _ in range(4)]
        game.tableaux = [[] for _ in range(7)]
        game.tableaux[0] = [Card(Suit.HEARTS, Rank.FIVE, True)]
        game.stock = [Card(Suit.DIAMONDS, Rank.NINE, False)]
        game.waste = []
        game.passes = 0
        return game

    def test_dead_deal_terminates_with_loss_reward(self):
        game = self._one_draw_from_dead()
        _, reward, terminated, _, _ = game.step(0)
        assert terminated
        assert game.winner is None
        assert reward == pytest.approx(-0.01 + game.LOSS_REWARD)
        assert game.is_game_over()

    def test_dead_deal_loss_in_sparse_mode(self):
        game = self._one_draw_from_dead(reward_mode="sparse")
        _, reward, terminated, _, _ = game.step(0)
        assert terminated
        assert reward == pytest.approx(game.LOSS_REWARD)

    def test_exhausted_passes_make_the_draw_illegal(self):
        game = KlondikeSolitaire(max_passes=2)
        game.foundations = [[] for _ in range(4)]
        game.tableaux = [[] for _ in range(7)]
        game.tableaux[0] = [Card(Suit.HEARTS, Rank.FIVE, True)]
        game.stock = [Card(Suit.DIAMONDS, Rank.NINE, False)]
        game.waste = []
        game.passes = 0

        game.step(0)  # draw the only stock card
        assert 0 in game.get_legal_actions()  # one recycle remains
        game.step(0)  # recycle
        assert game.passes == 1
        game.step(0)  # draw it again
        # Second recycle would be pass 3 of 2: withheld, so the deal is dead.
        assert 0 not in game.get_legal_actions()

    def test_unlimited_passes_keep_the_draw_legal(self):
        game = self._one_draw_from_dead()
        game.max_passes = None
        game.step(0)
        assert 0 in game.get_legal_actions()

    def test_win_pays_one_in_sparse_mode(self):
        game = KlondikeSolitaire(reward_mode="sparse")
        game.foundations = [
            [Card(Suit(s), Rank(r), True) for r in range(1, 14)] for s in range(4)
        ]
        game.foundations[0] = [Card(Suit.CLUBS, Rank(r), True) for r in range(1, 13)]
        game.tableaux = [[] for _ in range(7)]
        game.tableaux[0] = [Card(Suit.CLUBS, Rank.KING, True)]
        game.stock = []
        game.waste = []

        _, reward, terminated, _, _ = game.step(12)
        assert terminated and game.winner == 0
        assert reward == pytest.approx(1.0)

    def test_sparse_moves_pay_nothing(self):
        game = KlondikeSolitaire(reward_mode="sparse", seed=0)
        _, reward, _, _, _ = game.step(0)
        assert reward == 0.0

    def test_reward_mode_is_validated(self):
        with pytest.raises(ValueError, match="reward_mode"):
            KlondikeSolitaire(reward_mode="dense")


class TestBundledKlondikeCanActuallyLose:
    """The shipped Klondike must be able to reach LOSS_REWARD at all.

    `TestKlondikeTerminals` above covers the mechanism, and it passes -- but it
    builds its games with an explicit `max_passes=1`. The bundled game took the
    class default of None, under which draw/recycle is legal forever, so a deal
    can never run out of legal moves and the loss branch cannot fire. Over 200
    TEST deals a random policy terminated 0.0 % of the time: it never won and
    never lost, so `LOSS_REWARD = -1.0` was paid to nobody, ever, and the
    docstring promising it made a stuck deal distinguishable from running out
    of time described something that could not happen (issue #18).

    These tests pin the configuration rather than the mechanism.
    """

    def test_registered_klondike_limits_its_passes(self):
        env = sweep_game("klondike").env_factory()
        assert env.game.max_passes is not None

    def test_evaluation_plays_the_very_same_rules(self):
        """`max_passes` defines the game, so it cannot differ across the run.

        Unlike `repeated_position_penalty`, which is shaping and belongs to
        training alone, this changes what the environment *is*. A learner
        trained on three passes and scored on unlimited ones has been measured
        on a game it never played.
        """
        train = sweep_game("klondike").env_factory().game
        evaluate = sweep_game("klondike").eval_env_factory().game
        assert train.max_passes == evaluate.max_passes

    def test_the_library_default_is_left_unlimited(self):
        """Only the *bundled* game changes; a plain construction does not.

        Someone reaching for KlondikeSolitaire() wants ordinary Klondike, which
        has no pass limit. The finite value is a property of this project's
        experiment, so it lives on the entry points, not on the class.
        """
        assert KlondikeSolitaire().max_passes is None
        assert KlondikeSolitaire.BUNDLED_MAX_PASSES == 3

    def test_the_factory_gives_the_bundled_rules(self):
        """`bundled_klondike()` is the named way to get the finite game.

        The bare constructor stays permissive (the test above), so the project's
        own rules need a name of their own -- otherwise every entry point
        repeats `max_passes=BUNDLED_MAX_PASSES` and one of them eventually
        forgets to (issue #38).
        """
        assert bundled_klondike().max_passes == KlondikeSolitaire.BUNDLED_MAX_PASSES

    def test_the_factory_still_forwards_an_explicit_unlimited(self):
        """Asking for the unlimited game on purpose has to keep working.

        The factory only changes the *default*; a caller who writes
        `max_passes=None` has said what they want and gets it.
        """
        assert bundled_klondike(max_passes=None).max_passes is None

    def test_no_bundled_entry_point_plays_unlimited_passes(self):
        """The guard the five wrong evaluation paths needed (issue #38).

        `max_passes` defines what the environment *is*, so a bundled path that
        silently takes the class default has scored an agent on a game it never
        played. Measured on the same 200 held-out deals the gap is around three
        cards for the heuristic and six for MCTS(20) -- enough for an agent to
        look like it cleared random when it did not.

        Env-building entry points are checked on the game they construct; the
        two that build their game inside their own loop are checked on the
        signature default, which is the thing that regressed.
        """
        built = {
            "sweep env_factory": sweep_game("klondike").env_factory().game,
            "sweep eval_env_factory": sweep_game("klondike").eval_env_factory().game,
            "Klondike-v0": make_klondike().game,
            "KlondikeMasked-v0": make_klondike_masked().game,
        }
        for name, game in built.items():
            assert game.max_passes is not None, f"{name} plays unlimited passes"

        for fn in (evaluate_klondike, run_klondike_baselines):
            default = inspect.signature(fn).parameters["max_passes"].default
            assert default is not None, (
                f"{fn.__name__} defaults to the unlimited game"
            )

    def test_a_real_deal_really_does_die(self):
        """End to end: play the bundled rules until something actually loses.

        The unit tests above hand-build a dead position. This one deals real
        Klondike and checks that death is reachable by playing -- which is the
        claim the issue disproved for the shipped configuration.
        """
        game = KlondikeSolitaire(max_passes=KlondikeSolitaire.BUNDLED_MAX_PASSES)
        env = CardGameEnv(game, max_steps=300)
        agent = RandomAgent(action_size=game.get_action_space_size(), seed=0)

        deaths, final_rewards = 0, []
        for deal in range(100000, 100060):
            observation, info = env.reset(seed=deal)
            agent.reset()
            for _ in range(300):
                action = agent.select_action(observation, info["legal_actions"])
                observation, reward, terminated, truncated, info = env.step(action)
                if terminated:
                    deaths += 1
                    final_rewards.append(reward)
                    assert game.winner is None, "a random policy should not win"
                    break
                if truncated:
                    break

        # 4 of these 60 deals die; the assertion is >= 1 so that a change in
        # the shuffle does not make the test brittle, but 0 means the loss
        # branch went unreachable again.
        assert deaths >= 1
        for reward in final_rewards:
            assert reward <= KlondikeSolitaire.LOSS_REWARD

    def test_unlimited_passes_still_cannot_lose(self):
        """The defect itself, pinned so the contrast stays visible.

        Same deals, same policy, unlimited passes: nothing terminates. This is
        what the bundled game used to do.
        """
        game = KlondikeSolitaire(max_passes=None)
        env = CardGameEnv(game, max_steps=300)
        agent = RandomAgent(action_size=game.get_action_space_size(), seed=0)

        for deal in range(100000, 100020):
            observation, info = env.reset(seed=deal)
            agent.reset()
            for _ in range(300):
                action = agent.select_action(observation, info["legal_actions"])
                observation, _, terminated, truncated, info = env.step(action)
                assert not terminated, "unlimited passes cannot produce a loss"
                if truncated:
                    break


class TestKlondikeActionSpace:
    """The action space is 68 wide and every tableau move is unambiguous."""

    def test_action_space_is_tight(self):
        game = KlondikeSolitaire()
        assert game.get_action_space_size() == 68
        # Highest encodable action: from pile 6 to pile 6.
        assert 19 + 6 * 7 + 6 == 67

    def test_legal_actions_fit_the_declared_space(self):
        game = KlondikeSolitaire(seed=3)
        rng = random.Random(3)
        for _ in range(120):
            legal = game.get_legal_actions()
            if not legal:
                break
            assert all(0 <= action < 68 for action in legal)
            game.step(rng.choice(legal))

    def test_tableau_move_is_unambiguous(self):
        """
        For any (from, to) pair at most one face-up card can legally move.

        This is why the pile-pair encoding loses nothing: the face-up section
        of a pile is a single descending alternating-color run, so the
        destination determines the card. The TODO suspected hidden ambiguity
        here; this pins the analysis that found none.
        """
        game = KlondikeSolitaire(seed=7)
        rng = random.Random(7)
        for _ in range(200):
            for from_pile in range(7):
                for to_pile in range(7):
                    if from_pile == to_pile:
                        continue
                    candidates = [
                        idx
                        for idx, card in enumerate(game.tableaux[from_pile])
                        if card.face_up
                        and game._can_place_on_tableau(card, to_pile)
                    ]
                    assert len(candidates) <= 1, (
                        f"ambiguous move {from_pile}->{to_pile}: {candidates}"
                    )
            legal = game.get_legal_actions()
            if not legal:
                break
            game.step(rng.choice(legal))


class TestMacaoRewards:
    """Terminal rewards follow the actor; shaping is potential-based."""

    def test_winner_is_paid_whoever_wins(self):
        """Player 1 winning must be recorded as +10 for player 1's move."""
        game = fresh_macao_duel(
            [Card(Suit.SPADES, Rank.FIVE, True)] * 3,
            [Card(Suit.HEARTS, Rank.NINE, True)],
        )
        game.current_player_idx = 1
        _, reward, terminated, _, _ = game.step(
            Card(Suit.HEARTS, Rank.NINE).to_index()
        )
        assert terminated
        assert game.winner == 1
        assert reward == pytest.approx(game.WIN_REWARD)

    def test_losers_terminal_payoff_is_queryable(self):
        game = fresh_macao_duel(
            [Card(Suit.SPADES, Rank.FIVE, True)] * 3,
            [Card(Suit.HEARTS, Rank.NINE, True)],
        )
        game.current_player_idx = 1
        game.step(Card(Suit.HEARTS, Rank.NINE).to_index())
        assert game.get_reward(1) == pytest.approx(game.WIN_REWARD)
        assert game.get_reward(0) == pytest.approx(game.LOSS_REWARD)

    def test_positive_reward_requires_shedding_a_card(self):
        """
        Potential-based shaping: reward is positive only when the actor's hand
        shrank. Anything else would reopen a farmable side payment.
        """
        game = Macao(num_players=2, seed=11)
        rng = random.Random(11)
        for _ in range(200):
            if game.is_game_over():
                break
            actor = game.current_player_idx
            hand_before = len(game.players[actor].hand)
            _, reward, terminated, truncated, _ = game.step(
                rng.choice(game.get_legal_actions())
            )
            if terminated or truncated:
                break
            if reward > 0:
                assert len(game.players[actor].hand) < hand_before

    def test_draw_costs_the_card_potential(self):
        game = fresh_macao_duel(
            [Card(Suit.SPADES, Rank.FIVE, True)] * 2,
            [Card(Suit.SPADES, Rank.SIX, True)] * 2,
        )
        _, reward, _, _, _ = game.step(52)
        assert reward == pytest.approx(-game.CARD_POTENTIAL)

    def test_special_cards_pay_no_bonus(self):
        """A 2 must pay the same as a plain card: its value is strategic."""
        game = fresh_macao_duel(
            [Card(Suit.CLUBS, Rank.TWO, True), Card(Suit.SPADES, Rank.FIVE, True)],
            [Card(Suit.HEARTS, Rank.SIX, True)] * 3,
        )
        _, reward, _, _, _ = game.step(Card(Suit.CLUBS, Rank.TWO).to_index())
        assert reward == pytest.approx(game.CARD_POTENTIAL)

    def test_sparse_mode_pays_only_the_winner(self):
        game = fresh_macao_duel(
            [Card(Suit.CLUBS, Rank.FIVE, True)],
            [Card(Suit.HEARTS, Rank.SIX, True)] * 3,
        )
        game.reward_mode = "sparse"
        _, reward, terminated, _, _ = game.step(
            Card(Suit.CLUBS, Rank.FIVE).to_index()
        )
        assert terminated and game.winner == 0
        assert reward == pytest.approx(1.0)
        assert game.get_reward(0) == pytest.approx(1.0)
        assert game.get_reward(1) == pytest.approx(-1.0)

    def test_unplayed_card_raises(self):
        game = fresh_macao_duel(
            [Card(Suit.CLUBS, Rank.FIVE, True)],
            [Card(Suit.HEARTS, Rank.SIX, True)],
        )
        with pytest.raises(ValueError, match="does not hold"):
            game.step(Card(Suit.DIAMONDS, Rank.QUEEN).to_index())

    def test_reward_mode_is_validated(self):
        with pytest.raises(ValueError, match="reward_mode"):
            Macao(reward_mode="dense")


class TestMacaoDeclarations:
    """Ace/Jack requests are the agent's own actions now."""

    def test_ace_opens_a_suit_declaration(self):
        ace = Card(Suit.HEARTS, Rank.ACE, True)
        game = fresh_macao_duel(
            [ace, Card(Suit.SPADES, Rank.FIVE, True)],
            [Card(Suit.HEARTS, Rank.SIX, True)] * 3,
        )
        game.step(ace.to_index())

        assert game.pending_declaration == "suit"
        assert game.current_player_idx == 0, "turn must not pass yet"
        assert game.get_legal_actions() == [54, 55, 56, 57]

        game.step(57)
        assert game.requested_suit == Suit.SPADES
        assert game.pending_declaration is None
        assert game.current_player_idx == 1

    def test_jack_opens_a_rank_declaration(self):
        jack = Card(Suit.CLUBS, Rank.JACK, True)
        game = fresh_macao_duel(
            [jack, Card(Suit.SPADES, Rank.QUEEN, True)],
            [Card(Suit.HEARTS, Rank.SIX, True)] * 3,
        )
        game.step(jack.to_index())

        assert game.pending_declaration == "rank"
        assert game.get_legal_actions() == list(range(58, 65))

        game.step(58 + game.DECLARABLE_RANKS.index(Rank.QUEEN))
        assert game.requested_rank == Rank.QUEEN
        assert game.current_player_idx == 1

    def test_requested_suit_constrains_the_next_player(self):
        ace = Card(Suit.HEARTS, Rank.ACE, True)
        game = fresh_macao_duel(
            [ace, Card(Suit.SPADES, Rank.FIVE, True)],
            [Card(Suit.HEARTS, Rank.SIX, True), Card(Suit.CLUBS, Rank.SIX, True)],
        )
        game.step(ace.to_index())
        game.step(54 + int(Suit.HEARTS))

        legal = game.get_legal_actions()
        assert Card(Suit.HEARTS, Rank.SIX).to_index() in legal
        assert Card(Suit.CLUBS, Rank.SIX).to_index() not in legal

    def test_winning_ace_skips_the_declaration(self):
        ace = Card(Suit.HEARTS, Rank.ACE, True)
        game = fresh_macao_duel([ace], [Card(Suit.HEARTS, Rank.SIX, True)] * 3)
        _, reward, terminated, _, _ = game.step(ace.to_index())
        assert terminated and game.winner == 0
        assert game.pending_declaration is None
        assert reward == pytest.approx(game.WIN_REWARD)

    def test_declaration_out_of_phase_raises(self):
        game = fresh_macao_duel(
            [Card(Suit.CLUBS, Rank.FIVE, True)],
            [Card(Suit.HEARTS, Rank.SIX, True)],
        )
        with pytest.raises(ValueError, match="no declaration is pending"):
            game.step(55)

    def test_wrong_declaration_kind_raises(self):
        ace = Card(Suit.HEARTS, Rank.ACE, True)
        game = fresh_macao_duel(
            [ace, Card(Suit.SPADES, Rank.FIVE, True)],
            [Card(Suit.HEARTS, Rank.SIX, True)] * 3,
        )
        game.step(ace.to_index())
        with pytest.raises(ValueError, match="declaration"):
            game.step(58)  # a rank declaration during a suit phase

    def test_action_space_is_fully_used(self):
        game = Macao()
        assert game.get_action_space_size() == 65
        assert game.action_to_string(54).startswith("Declare suit")
        assert game.action_to_string(64).startswith("Declare rank")

    def test_observation_shape_includes_phase_flags(self):
        game = Macao(num_players=2)
        observation = game.reset()
        assert observation.shape == game.get_observation_shape()

        ace = Card(Suit.HEARTS, Rank.ACE, True)
        game.players[0].hand = [ace, Card(Suit.SPADES, Rank.FIVE, True)]
        game.current_player_idx = 0
        game.requested_suit = None
        game.requested_rank = None
        game.draw_penalty = 0
        game.skip_next = False
        observation, _, _, _, _ = game.step(ace.to_index())
        # The suit-phase flag sits right after the 52+52+4+13 encodings.
        assert observation[121] == 1.0
        assert observation[122] == 0.0

    def test_copy_preserves_the_declaration_phase(self):
        ace = Card(Suit.HEARTS, Rank.ACE, True)
        game = fresh_macao_duel(
            [ace, Card(Suit.SPADES, Rank.FIVE, True)],
            [Card(Suit.HEARTS, Rank.SIX, True)] * 3,
        )
        game.step(ace.to_index())
        clone = game.copy()
        assert clone.pending_declaration == "suit"
        assert clone.get_legal_actions() == game.get_legal_actions()


class TestRngIsolation:
    """Nothing may reseed the process-wide RNGs."""

    def test_seeded_deals_are_reproducible(self):
        game = KlondikeSolitaire()
        env = CardGameEnv(game)
        first, _ = env.reset(seed=123)
        second, _ = env.reset(seed=123)
        assert np.array_equal(first, second)

    def test_different_seeds_differ(self):
        env = CardGameEnv(KlondikeSolitaire())
        first, _ = env.reset(seed=1)
        second, _ = env.reset(seed=2)
        assert not np.array_equal(first, second)

    def test_global_rngs_are_untouched(self):
        python_state = random.getstate()
        numpy_state = np.random.get_state()

        env = CardGameEnv(Macao(num_players=2, seed=9))
        env.reset(seed=42)
        Card(Suit.CLUBS, Rank.ACE)  # arbitrary library use
        KlondikeSolitaire(seed=5).reset(seed=6)

        assert random.getstate() == python_state
        numpy_after = np.random.get_state()
        assert numpy_state[0] == numpy_after[0]
        assert np.array_equal(numpy_state[1], numpy_after[1])

    def test_deck_shuffle_seed_is_local(self):
        from rl_card_lib.cardgames import Deck

        state = random.getstate()
        Deck().shuffle(seed=42)
        assert random.getstate() == state

    def test_copy_does_not_share_rng_state(self):
        game = Macao(num_players=2, seed=4)
        clone = game.copy()
        # Advancing the clone's RNG must not advance the original's.
        clone._rng.random()
        assert game._rng.getstate() != clone._rng.getstate()


class TestDealPool:
    """An unseeded reset() draws its deal from a declared pool.

    Without one the game reshuffles from an RNG nothing seeded, which is what
    made every training run unreproducible and left its deals belonging to no
    declared set.
    """

    @staticmethod
    def _env(**kwargs):
        return CardGameEnv(KlondikeSolitaire(), max_steps=5, **kwargs)

    def test_no_pool_leaves_an_unseeded_reset_random(self):
        """The Gymnasium default survives: no pool, no seed, fresh deal."""
        env = self._env()
        first, _ = env.reset()
        second, _ = env.reset()
        assert not np.array_equal(first, second)
        assert env.dealt_seeds == []

    def test_pooled_reset_is_reproducible_from_the_deal_rng_seed(self):
        one = self._env(deal_seeds=range(100), deal_rng_seed=7)
        two = self._env(deal_seeds=range(100), deal_rng_seed=7)
        for _ in range(10):
            one.reset()
            two.reset()
        assert one.dealt_seeds == two.dealt_seeds
        assert len(one.dealt_seeds) == 10

    def test_a_different_deal_rng_seed_gives_a_different_stream(self):
        one = self._env(deal_seeds=range(1000), deal_rng_seed=0)
        two = self._env(deal_seeds=range(1000), deal_rng_seed=1)
        for _ in range(10):
            one.reset()
            two.reset()
        assert one.dealt_seeds != two.dealt_seeds

    def test_every_deal_comes_from_the_pool(self):
        env = self._env(deal_seeds=[3, 4, 5])
        for _ in range(20):
            env.reset()
        assert set(env.dealt_seeds) <= {3, 4, 5}

    def test_an_explicit_seed_still_wins(self):
        env = self._env(deal_seeds=range(100))
        first, _ = env.reset(seed=42)
        second, _ = env.reset(seed=42)
        assert np.array_equal(first, second)
        assert env.last_deal_seed == 42

    def test_cycle_walks_the_pool_in_order_and_wraps(self):
        env = self._env(deal_seeds=[10, 11, 12], deal_order="cycle")
        for _ in range(7):
            env.reset()
        assert env.dealt_seeds == [10, 11, 12, 10, 11, 12, 10]

    def test_reset_deal_cursor_replays_the_same_deals(self):
        env = self._env(deal_seeds=[10, 11, 12], deal_order="cycle")
        env.reset()
        env.reset()
        first = list(env.dealt_seeds)

        env.reset_deal_cursor()
        env.reset()
        env.reset()
        assert env.dealt_seeds == first + first

    def test_an_unknown_deal_order_is_rejected(self):
        with pytest.raises(ValueError, match="deal_order"):
            self._env(deal_seeds=[1], deal_order="shuffled")

    def test_drawing_deals_touches_no_global_rng(self):
        python_state = random.getstate()
        numpy_state = np.random.get_state()

        env = self._env(deal_seeds=range(100), deal_rng_seed=3)
        for _ in range(5):
            env.reset()

        assert random.getstate() == python_state
        numpy_after = np.random.get_state()
        assert numpy_state[0] == numpy_after[0]
        assert np.array_equal(numpy_state[1], numpy_after[1])


class TestRepeatedPositionHandling:
    """The env can flag and price position repeats generically."""

    @staticmethod
    def _loop_env(penalty=0.0):
        game = KlondikeSolitaire(seed=0)
        game.foundations = [[] for _ in range(4)]
        game.tableaux = [[] for _ in range(7)]
        game.tableaux[0] = [Card(Suit.SPADES, Rank.EIGHT, True)]
        game.tableaux[1] = [
            Card(Suit.CLUBS, Rank.EIGHT, True),
            Card(Suit.HEARTS, Rank.SEVEN, True),
        ]
        game.stock = [Card(Suit.DIAMONDS, Rank.KING, False)]
        game.waste = []
        env = CardGameEnv(game, repeated_position_penalty=penalty)
        env._seen_positions = {hash(np.asarray(
            game.get_observation(), dtype=np.float32).tobytes())}
        return env

    def test_repeat_is_flagged(self):
        env = self._loop_env()
        _, _, _, _, info = env.step(19 + 1 * 7 + 0)
        assert "repeated_position" not in info
        _, _, _, _, info = env.step(19 + 0 * 7 + 1)  # back where we started
        assert info.get("repeated_position") is True

    def test_repeat_penalty_is_applied(self):
        env = self._loop_env(penalty=-0.5)
        env.step(19 + 1 * 7 + 0)
        _, reward, _, _, _ = env.step(19 + 0 * 7 + 1)
        assert reward == pytest.approx(-0.01 - 0.5)

    def test_reset_forgets_the_history(self):
        env = CardGameEnv(KlondikeSolitaire(seed=1))
        env.reset(seed=1)
        assert len(env._seen_positions) == 1
        env.reset(seed=1)
        assert len(env._seen_positions) == 1


class TestRepeatedPositionPenaltyIsOffByDefault:
    """The bundled Klondike must train *unshaped*.

    `TestRepeatedPositionHandling` above covers the mechanism; these cover the
    switch. It was on for a while: the penalty had been implemented,
    documented as the remedy for reversible-move loops and unit-tested since
    the env was written, but no registration ever passed a value, so #17
    switched it on here.

    Measuring it refuted it. Over 3 seeds, 5000 episodes and 200 held-out
    deals, -0.05 buys 0.3-0.8 pp of the revisit fraction for the DQN family and
    moves it the wrong way for PPO, while costing PPO 17.09 -> 9.73 cards to
    foundation (the random baseline is 9.79) and 27.5 % -> 0.4 % of its solve
    rate on the proven-winnable pool. A default the repo's own measurement
    refutes is worse than none, so the fact that it is off is itself a
    regression test -- and so is the fact that the mechanism still works when
    a caller asks for it.
    """

    def test_klondike_trains_unshaped(self):
        env = sweep_game("klondike").env_factory()
        assert env.repeated_position_penalty == 0.0
        assert KLONDIKE_REPEAT_PENALTY == 0.0

    def test_klondike_evaluation_stays_unshaped(self):
        """Shaping is a training signal and must not reach the eval curve.

        Otherwise a rising evaluation reward cannot be told apart from the
        agent merely having stopped paying the penalty, and the number stops
        being comparable with the baselines in `harness.baselines`, which
        measure random and heuristic play on plain envs.
        """
        assert sweep_game("klondike").eval_env_factory(
            ).repeated_position_penalty == 0.0

    def test_macao_is_left_at_zero_because_it_cannot_loop(self):
        """Macao's 0.0 is a different zero from Klondike's.

        Klondike's is a shaping term that was measured and withdrawn, so a
        later reader could reasonably revisit it. Macao's never had anything to
        price: its positions are monotone -- every action moves a card out of a
        hand or the deck, and the observation carries the counts -- so no
        repeat can occur. Asserting the measurement keeps the two from being
        read as one decision.
        """
        assert sweep_game("macao").env_factory().repeated_position_penalty == 0.0

        game = Macao(num_players=2)
        env = CardGameEnv(game, max_steps=200)
        agent = RandomAgent(action_size=game.get_action_space_size(), seed=0)
        for deal in range(100000, 100010):
            observation, info = env.reset(seed=deal)
            agent.reset()
            for _ in range(200):
                action = agent.select_action(observation, info["legal_actions"])
                observation, _, terminated, truncated, info = env.step(action)
                assert not info.get("repeated_position")
                if terminated or truncated:
                    break

    def test_the_draw_loop_is_priced_when_a_caller_asks_for_it(self):
        """The mechanism still works, whatever the bundled default is.

        "Draw from stock" is action 0, and with the default `max_passes=None`
        the stock recycles forever, so pressing it is a cycle that only the
        step cap ends. A caller who passes a penalty has to see that cycle cost
        strictly more, or the opt-in the bundled zero leaves them is empty.
        Deliberately a literal rather than `KLONDIKE_REPEAT_PENALTY`: this
        pins the mechanism, not the configuration.
        """
        def always_draw(penalty):
            env = CardGameEnv(KlondikeSolitaire(), max_steps=300,
                              repeated_position_penalty=penalty)
            _, info = env.reset(seed=100000)
            total, repeats = 0.0, 0
            for _ in range(300):
                if 0 not in info["legal_actions"]:
                    break
                _, reward, terminated, truncated, info = env.step(0)
                total += reward
                repeats += bool(info.get("repeated_position"))
                if terminated or truncated:
                    break
            return total, repeats

        penalty = -0.05
        inert_return, inert_repeats = always_draw(0.0)
        priced_return, priced_repeats = always_draw(penalty)

        # Same trajectory either way -- the penalty prices the loop, it does
        # not change where a fixed action sequence goes.
        assert priced_repeats == inert_repeats
        assert inert_repeats > 200

        expected = inert_return + penalty * inert_repeats
        assert priced_return == pytest.approx(expected)
        assert priced_return < inert_return


class TestMCTSValuesTerminals:
    """The search must see instant wins and instant losses."""

    def test_takes_an_immediate_win(self):
        game = fresh_macao_duel(
            [Card(Suit.DIAMONDS, Rank.NINE, True)],
            [Card(Suit.HEARTS, Rank.NINE, True)],
        )
        agent = MCTSAgent(
            simulations=30, rollout_depth=8, use_determinization=False, seed=0
        ).bind(game)
        action = agent.select_action(game.get_observation(), game.get_legal_actions())
        assert action == Card(Suit.DIAMONDS, Rank.NINE).to_index()

    @pytest.mark.parametrize("seed", range(5))
    def test_blocks_an_imminent_opponent_win(self, seed):
        """
        Playing the 4 (skip) is the only move that stops the opponent's nine.

        This pins two fixes at once: edge rewards are part of a node's value
        (else the win/loss below is invisible at selection time), and the
        loser's terminal payoff reaches the search through get_reward() (else
        "opponent wins" costs nothing and blocking looks pointless).
        """
        game = fresh_macao_duel(
            [Card(Suit.CLUBS, Rank.FOUR, True), Card(Suit.DIAMONDS, Rank.NINE, True)],
            [Card(Suit.HEARTS, Rank.NINE, True)],
        )
        agent = MCTSAgent(
            simulations=100, rollout_depth=12, use_determinization=False, seed=seed
        ).bind(game)
        action = agent.select_action(game.get_observation(), game.get_legal_actions())
        assert action == Card(Suit.CLUBS, Rank.FOUR).to_index()
