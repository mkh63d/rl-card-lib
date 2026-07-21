"""Tests for the heuristic, search and advanced learning agents."""

import numpy as np
import pytest

from rl_card_lib.agents import (
    DoubleDQNAgent,
    DQNAgent,
    GameAwareAgent,
    GreedyLookaheadAgent,
    HeuristicAgent,
    MCTSAgent,
    PPOAgent,
    QLearningAgent,
    RandomAgent,
)
from rl_card_lib.agents.double_dqn_agent import DuelingQNetwork, MaskedReplayBuffer
from rl_card_lib.cardgames import Rank
from rl_card_lib.env import CardGameEnv
from rl_card_lib.games import (
    KlondikeHeuristicAgent,
    KlondikeSolitaire,
    Macao,
    MacaoHeuristicAgent,
)
from rl_card_lib.trainer import SelfPlayTrainer, Trainer


def play_episode(env, agent, max_steps=150):
    """Play one episode, asserting every action taken is legal."""
    observation, info = env.reset()
    agent.reset()
    total_reward = 0.0
    steps = 0

    for _ in range(max_steps):
        legal = info.get("legal_actions")
        action = agent.select_action(observation, legal)
        if legal:
            assert action in legal, f"{agent.name} chose illegal action {action}"
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        if terminated or truncated:
            break

    return total_reward, steps


@pytest.fixture
def klondike_env():
    game = KlondikeSolitaire()
    return CardGameEnv(game, max_steps=150)


@pytest.fixture
def macao_env():
    game = Macao(num_players=2)
    return CardGameEnv(game, max_steps=150)


class TestGameAwareAgent:
    """Tests for the game-binding plumbing."""

    def test_bind_accepts_env(self, klondike_env):
        agent = KlondikeHeuristicAgent()
        agent.bind(klondike_env)
        assert agent.game is klondike_env.game

    def test_bind_accepts_game(self, klondike_env):
        agent = KlondikeHeuristicAgent()
        agent.bind(klondike_env.game)
        assert agent.game is klondike_env.game

    def test_unbound_agent_explains_itself(self):
        agent = KlondikeHeuristicAgent()
        with pytest.raises(RuntimeError, match="bind"):
            agent.select_action(np.zeros(221, dtype=np.float32), [0])

    def test_trainer_binds_automatically(self, klondike_env):
        agent = KlondikeHeuristicAgent()
        Trainer(klondike_env, agent)
        assert agent.game is klondike_env.game

    def test_trainer_respects_existing_binding(self, klondike_env):
        other_game = KlondikeSolitaire()
        agent = KlondikeHeuristicAgent(game=other_game)
        Trainer(klondike_env, agent)
        assert agent.game is other_game

    def test_save_load_are_noops(self, tmp_path):
        agent = KlondikeHeuristicAgent()
        agent.save(str(tmp_path / "x"))
        agent.load(str(tmp_path / "x"))


class TestKlondikeHeuristicAgent:
    """Tests for the Klondike expert."""

    def test_plays_legal_actions(self, klondike_env):
        agent = KlondikeHeuristicAgent(seed=0).bind(klondike_env)
        play_episode(klondike_env, agent)

    def test_beats_random_on_reward(self, klondike_env):
        heuristic = KlondikeHeuristicAgent(seed=0).bind(klondike_env)
        random_agent = RandomAgent(action_size=klondike_env.action_space.n, seed=0)

        heuristic_rewards = [play_episode(klondike_env, heuristic)[0] for _ in range(15)]
        random_rewards = [play_episode(klondike_env, random_agent)[0] for _ in range(15)]

        assert np.mean(heuristic_rewards) > np.mean(random_rewards)

    def test_aces_go_up_immediately(self, klondike_env):
        """An ace on a tableau top outscores drawing."""
        game = klondike_env.game
        game.reset()
        agent = KlondikeHeuristicAgent(seed=0).bind(game)

        # Force an ace to the top of tableau 0 (action 12 moves it up).
        game.foundations = [[] for _ in range(4)]
        game.tableaux[0][-1] = type(game.tableaux[0][-1])(
            game.tableaux[0][-1].suit, Rank.ACE, True
        )

        assert agent.score_action(game, 12) > agent.score_action(game, 0)

    def test_unsafe_foundation_move_ranks_below_a_safe_one(self, klondike_env):
        """A card the tableau may still need should yield to better moves."""
        from rl_card_lib.cardgames import Card, Rank, Suit

        game = klondike_env.game
        game.reset()
        agent = KlondikeHeuristicAgent(seed=0).bind(game)

        # Clubs up to 9, everything else empty: the 10 of clubs is playable but
        # red 9s might still need a black 10 to sit on.
        game.foundations = [[] for _ in range(4)]
        game.foundations[int(Suit.CLUBS)] = [
            Card(Suit.CLUBS, Rank(r), True) for r in range(1, 10)
        ]
        game.tableaux[0] = [Card(Suit.CLUBS, Rank.TEN, True)]
        game.stock = [Card(Suit.SPADES, Rank.KING, False)]

        assert not agent._is_safe_to_foundation(game, Card(Suit.CLUBS, Rank.TEN))
        unsafe = agent.score_action(game, 12)

        # Same move, but with the red foundations caught up so nothing needs it.
        for suit in (Suit.DIAMONDS, Suit.HEARTS):
            game.foundations[int(suit)] = [
                Card(suit, Rank(r), True) for r in range(1, 10)
            ]
        game.foundations[int(Suit.SPADES)] = [
            Card(Suit.SPADES, Rank(r), True) for r in range(1, 9)
        ]
        assert agent._is_safe_to_foundation(game, Card(Suit.CLUBS, Rank.TEN))

        assert unsafe < agent.score_action(game, 12)
        # ...but still worth doing rather than stalling: demoting it below a
        # draw measured worse (38.7% vs 43.3% win rate over 150 deals).
        assert unsafe > agent.score_action(game, 0)

    def test_safe_foundation_move_scores_above_drawing(self):
        """With both red foundations caught up, the 10 of clubs is safe to send."""
        from rl_card_lib.cardgames import Card, Rank, Suit

        game = KlondikeSolitaire()
        agent = KlondikeHeuristicAgent(seed=0).bind(game)

        game.foundations = [[] for _ in range(4)]
        game.foundations[int(Suit.CLUBS)] = [
            Card(Suit.CLUBS, Rank(r), True) for r in range(1, 10)
        ]
        for suit in (Suit.DIAMONDS, Suit.HEARTS):
            game.foundations[int(suit)] = [
                Card(suit, Rank(r), True) for r in range(1, 10)
            ]
        game.foundations[int(Suit.SPADES)] = [
            Card(Suit.SPADES, Rank(r), True) for r in range(1, 9)
        ]
        game.tableaux[0] = [Card(Suit.CLUBS, Rank.TEN, True)]
        game.stock = [Card(Suit.SPADES, Rank.KING, False)]

        assert agent._is_safe_to_foundation(game, Card(Suit.CLUBS, Rank.TEN))
        assert agent.score_action(game, 12) > agent.score_action(game, 0)

    def test_recycling_waste_is_last_resort(self, klondike_env):
        game = klondike_env.game
        game.reset()
        agent = KlondikeHeuristicAgent(seed=0).bind(game)

        with_stock = agent.score_action(game, 0)
        game.waste = list(game.stock)
        game.stock = []
        without_stock = agent.score_action(game, 0)

        assert without_stock < with_stock

    def test_reset_clears_move_memory(self, klondike_env):
        agent = KlondikeHeuristicAgent(seed=0).bind(klondike_env)
        play_episode(klondike_env, agent, max_steps=20)
        assert len(agent._recent) > 0
        agent.reset()
        assert len(agent._recent) == 0


class TestMacaoHeuristicAgent:
    """Tests for the Macao expert."""

    def test_plays_legal_actions(self, macao_env):
        agent = MacaoHeuristicAgent(seed=0).bind(macao_env)
        play_episode(macao_env, agent)

    def test_takes_the_winning_move(self, macao_env):
        game = macao_env.game
        game.reset()
        agent = MacaoHeuristicAgent(seed=0).bind(game)

        card = game.players[0].hand[0]
        game.players[0].hand = [card]
        game.current_player_idx = 0

        # Every other score is bounded well below the winning bonus.
        assert agent.score_action(game, card.to_index()) == 1000.0

    def test_counters_penalty_rather_than_drawing(self, macao_env):
        from rl_card_lib.cardgames import Card, Rank, Suit

        game = macao_env.game
        game.reset()
        agent = MacaoHeuristicAgent(seed=0).bind(game)

        game.draw_penalty = 2
        game.discard_pile = [Card(Suit.CLUBS, Rank.TWO, True)]
        counter = Card(Suit.HEARTS, Rank.TWO, True)
        game.players[0].hand = [counter, Card(Suit.SPADES, Rank.NINE, True)]
        game.current_player_idx = 0

        assert agent.score_action(game, counter.to_index()) > agent.score_action(game, 52)

    def test_holds_wilds_over_plain_cards(self, macao_env):
        from rl_card_lib.cardgames import Card, Rank, Suit

        game = macao_env.game
        game.reset()
        agent = MacaoHeuristicAgent(seed=0).bind(game)

        game.draw_penalty = 0
        game.requested_suit = None
        game.requested_rank = None
        game.discard_pile = [Card(Suit.CLUBS, Rank.NINE, True)]
        jack = Card(Suit.HEARTS, Rank.JACK, True)
        plain = Card(Suit.CLUBS, Rank.SEVEN, True)
        game.players[0].hand = [jack, plain]
        game.players[1].hand = [Card(Suit.SPADES, Rank.FIVE, True)] * 5
        game.current_player_idx = 0

        assert agent.score_action(game, plain.to_index()) > agent.score_action(game, jack.to_index())

    def test_attacks_harder_when_opponent_is_nearly_out(self, macao_env):
        from rl_card_lib.cardgames import Card, Rank, Suit

        game = macao_env.game
        game.reset()
        agent = MacaoHeuristicAgent(seed=0).bind(game)

        game.draw_penalty = 0
        game.requested_suit = None
        game.requested_rank = None
        game.discard_pile = [Card(Suit.CLUBS, Rank.NINE, True)]
        two = Card(Suit.CLUBS, Rank.TWO, True)
        game.players[0].hand = [two, Card(Suit.HEARTS, Rank.FIVE, True)]
        game.current_player_idx = 0

        game.players[1].hand = [Card(Suit.SPADES, Rank.FIVE, True)] * 7
        relaxed = agent.score_action(game, two.to_index())
        game.players[1].hand = [Card(Suit.SPADES, Rank.FIVE, True)] * 2
        urgent = agent.score_action(game, two.to_index())

        assert urgent > relaxed


class TestGreedyLookaheadAgent:
    """Tests for the game-agnostic lookahead agent."""

    def test_plays_legal_actions(self, klondike_env):
        agent = GreedyLookaheadAgent(depth=1, seed=0).bind(klondike_env)
        play_episode(klondike_env, agent, max_steps=60)

    def test_depth_two_plays_legal_actions(self, klondike_env):
        agent = GreedyLookaheadAgent(depth=2, seed=0).bind(klondike_env)
        play_episode(klondike_env, agent, max_steps=30)

    def test_rejects_invalid_depth(self):
        with pytest.raises(ValueError, match="depth"):
            GreedyLookaheadAgent(depth=0)

    def test_rejects_deep_search_in_multiplayer(self, macao_env):
        agent = GreedyLookaheadAgent(depth=2, seed=0).bind(macao_env)
        observation, info = macao_env.reset()
        with pytest.raises(ValueError, match="multi-player|depth"):
            agent.select_action(observation, info["legal_actions"])

    def test_depth_one_works_in_multiplayer(self, macao_env):
        agent = GreedyLookaheadAgent(depth=1, seed=0).bind(macao_env)
        play_episode(macao_env, agent, max_steps=60)

    def test_prefers_the_higher_rewarding_move(self, klondike_env):
        """Foundation moves pay 1.0, tableau moves 0.1: greedy must see that."""
        game = klondike_env.game
        game.reset()
        agent = GreedyLookaheadAgent(depth=1, seed=0).bind(game)

        legal = game.get_legal_actions()
        values = {a: agent._value_of(game, a, 1) for a in legal}
        chosen = agent.select_action(game.get_observation(), legal)

        assert values[chosen] == pytest.approx(max(values.values()))

    def test_search_does_not_mutate_the_game(self, klondike_env):
        game = klondike_env.game
        game.reset()
        agent = GreedyLookaheadAgent(depth=2, seed=0).bind(game)
        before = game.get_observation().copy()

        agent.select_action(before, game.get_legal_actions())

        assert np.array_equal(game.get_observation(), before)


class TestMCTSAgent:
    """Tests for the search agent."""

    def test_plays_legal_actions_klondike(self, klondike_env):
        agent = MCTSAgent(simulations=12, rollout_depth=8, seed=0).bind(klondike_env)
        play_episode(klondike_env, agent, max_steps=12)

    def test_plays_legal_actions_macao(self, macao_env):
        agent = MCTSAgent(simulations=12, rollout_depth=8, seed=0).bind(macao_env)
        play_episode(macao_env, agent, max_steps=12)

    def test_search_does_not_mutate_the_game(self, klondike_env):
        game = klondike_env.game
        game.reset()
        agent = MCTSAgent(simulations=20, rollout_depth=8, seed=0).bind(game)
        before = game.get_observation().copy()

        agent.select_action(before, game.get_legal_actions())

        assert np.array_equal(game.get_observation(), before)

    def test_determinization_ensemble_runs(self, macao_env):
        agent = MCTSAgent(
            simulations=20, determinizations=4, rollout_depth=6, seed=0
        ).bind(macao_env)
        play_episode(macao_env, agent, max_steps=8)

    def test_perfect_information_mode_runs(self, klondike_env):
        agent = MCTSAgent(
            simulations=12, rollout_depth=6, use_determinization=False, seed=0
        ).bind(klondike_env)
        play_episode(klondike_env, agent, max_steps=8)

    def test_heuristic_rollout_policy(self, klondike_env):
        agent = MCTSAgent(
            simulations=12,
            rollout_depth=6,
            rollout_policy=KlondikeHeuristicAgent(seed=0),
            seed=0,
        ).bind(klondike_env)
        play_episode(klondike_env, agent, max_steps=8)

    def test_callable_rollout_policy(self, klondike_env):
        agent = MCTSAgent(
            simulations=12,
            rollout_depth=6,
            rollout_policy=lambda game, legal: legal[0],
            seed=0,
        ).bind(klondike_env)
        play_episode(klondike_env, agent, max_steps=8)

    def test_single_legal_action_skips_search(self, macao_env):
        agent = MCTSAgent(simulations=10_000_000, seed=0).bind(macao_env)
        observation, _ = macao_env.reset()
        # A search this large would never finish; the shortcut must fire.
        assert agent.select_action(observation, [7]) == 7

    def test_rejects_invalid_config(self):
        with pytest.raises(ValueError, match="simulations"):
            MCTSAgent(simulations=0)
        with pytest.raises(ValueError, match="determinizations"):
            MCTSAgent(determinizations=0)

    def test_more_simulations_beat_fewer(self):
        """
        Search quality should rise with the budget, not just the runtime.

        This is the property that distinguishes real search from an expensive
        random agent. It spent its early life as an xfail: Klondike's reward
        contained a farmable loop and Macao's terminal reward was credited to
        player 0 regardless of who won, so a bigger budget only bought better
        loop-farming or a more faithful model of an opponent "trying to lose".
        It passes now that rewards are actor-relative and loop-free, and it is
        the regression test that keeps them that way.
        """
        weak_wins, strong_wins = 0, 0

        for seed in range(10):
            for simulations, tally in ((2, "weak"), (48, "strong")):
                game = Macao(num_players=2, seed=seed)
                env = CardGameEnv(game, max_steps=160)
                agent = MCTSAgent(
                    simulations=simulations, rollout_depth=25, seed=seed
                ).bind(env)
                opponent = RandomAgent(action_size=env.action_space.n, seed=seed)

                observation, info = env.reset(seed=seed)
                for _ in range(160):
                    actor = game.current_player_idx
                    chooser = agent if actor == 0 else opponent
                    action = chooser.select_action(observation, info.get("legal_actions"))
                    observation, _, terminated, truncated, info = env.step(action)
                    if terminated or truncated:
                        break

                if game.winner == 0:
                    if tally == "weak":
                        weak_wins += 1
                    else:
                        strong_wins += 1

        assert strong_wins > weak_wins


class TestQLearningAgent:
    """Tests for tabular Q-learning."""

    def test_action_selection_respects_legality(self):
        agent = QLearningAgent(action_size=10, seed=0)
        observation = np.zeros(5, dtype=np.float32)
        agent.eval()
        for _ in range(50):
            assert agent.select_action(observation, [2, 4, 6]) in (2, 4, 6)

    def test_learns_a_rewarding_action(self):
        agent = QLearningAgent(action_size=4, epsilon_start=0.0, seed=0)
        observation = np.ones(3, dtype=np.float32)
        next_observation = np.zeros(3, dtype=np.float32)

        for _ in range(50):
            agent.learn(observation, 2, 1.0, next_observation, True)
            agent.learn(observation, 0, -1.0, next_observation, True)

        agent.eval()
        assert agent.select_action(observation, [0, 1, 2, 3]) == 2

    def test_bootstrap_only_uses_legal_next_actions(self):
        agent = QLearningAgent(action_size=3, learning_rate=1.0, gamma=1.0, seed=0)
        observation = np.ones(2, dtype=np.float32)
        next_observation = np.zeros(2, dtype=np.float32)

        # Make action 2 look great in the next state, then forbid it there.
        agent.q_table[agent._key(next_observation)] = np.array([1.0, 2.0, 99.0])

        agent.learn(observation, 0, 0.0, next_observation, False, next_legal_actions=[0, 1])
        assert agent.get_q_values(observation)[0] == pytest.approx(2.0)

    def test_empty_next_actions_do_not_bootstrap(self):
        agent = QLearningAgent(action_size=3, learning_rate=1.0, gamma=1.0, seed=0)
        observation = np.ones(2, dtype=np.float32)
        next_observation = np.zeros(2, dtype=np.float32)
        agent.q_table[agent._key(next_observation)] = np.array([5.0, 5.0, 5.0])

        agent.learn(observation, 0, 1.0, next_observation, False, next_legal_actions=[])
        assert agent.get_q_values(observation)[0] == pytest.approx(1.0)

    def test_done_transition_does_not_bootstrap(self):
        agent = QLearningAgent(action_size=3, learning_rate=1.0, gamma=1.0, seed=0)
        observation = np.ones(2, dtype=np.float32)
        next_observation = np.zeros(2, dtype=np.float32)
        agent.q_table[agent._key(next_observation)] = np.array([5.0, 5.0, 5.0])

        agent.learn(observation, 0, 1.0, next_observation, True)
        assert agent.get_q_values(observation)[0] == pytest.approx(1.0)

    def test_table_grows_with_new_states(self):
        agent = QLearningAgent(action_size=3, seed=0)
        agent.eval()  # exploring at random would never consult the table
        assert agent.table_size == 0
        for i in range(5):
            agent.select_action(np.full(2, float(i), dtype=np.float32), [0, 1])
        assert agent.table_size == 5

    def test_exploring_does_not_touch_the_table(self):
        """A random exploratory move needs no Q-value, so it should store none."""
        agent = QLearningAgent(action_size=3, epsilon_start=1.0, seed=0)
        for i in range(5):
            agent.select_action(np.full(2, float(i), dtype=np.float32), [0, 1])
        assert agent.table_size == 0

    def test_precision_merges_nearby_states(self):
        agent = QLearningAgent(action_size=3, precision=1, seed=0)
        agent.eval()
        agent.select_action(np.array([0.123], dtype=np.float32), [0])
        agent.select_action(np.array([0.124], dtype=np.float32), [0])
        assert agent.table_size == 1

    def test_epsilon_decays_per_episode(self):
        agent = QLearningAgent(action_size=3, epsilon_start=1.0, epsilon_decay=0.9, seed=0)
        observation = np.ones(2, dtype=np.float32)

        # Learning steps must not decay epsilon, and neither must the reset
        # that opens the first episode; the second reset applies one decay.
        agent.learn(observation, 0, 1.0, observation, False)
        assert agent.epsilon == 1.0
        agent.reset()
        assert agent.epsilon == 1.0
        agent.reset()
        assert agent.epsilon == pytest.approx(0.9)

    def test_save_load_roundtrip(self, tmp_path):
        agent = QLearningAgent(action_size=4, seed=0)
        observation = np.ones(3, dtype=np.float32)
        for _ in range(10):
            agent.learn(observation, 2, 1.0, np.zeros(3, dtype=np.float32), True)

        path = str(tmp_path / "q.pkl")
        agent.save(path)

        restored = QLearningAgent(action_size=4)
        restored.load(path)

        assert restored.table_size == agent.table_size
        assert restored.epsilon == agent.epsilon
        np.testing.assert_allclose(
            restored.get_q_values(observation), agent.get_q_values(observation)
        )

    def test_load_rejects_mismatched_action_size(self, tmp_path):
        agent = QLearningAgent(action_size=4, seed=0)
        path = str(tmp_path / "q.pkl")
        agent.save(path)

        with pytest.raises(ValueError, match="action_size"):
            QLearningAgent(action_size=9).load(path)

    def test_trains_on_a_real_game(self, klondike_env):
        agent = QLearningAgent(action_size=klondike_env.action_space.n, seed=0)
        metrics = Trainer(klondike_env, agent, eval_interval=10_000).train(
            episodes=2, max_steps_per_episode=30, verbose=False
        )
        assert len(metrics.rewards) == 2


class TestDQNAgent:
    """Tests for vanilla DQN, whose target now masks illegal next actions."""

    def test_declares_it_wants_next_legal_actions(self):
        assert DQNAgent(
            state_size=4, action_size=3, hidden_sizes=[8], device="cpu"
        ).accepts_next_legal_actions is True

    def test_uses_a_masked_replay_buffer(self):
        agent = DQNAgent(
            state_size=4, action_size=3, hidden_sizes=[8], device="cpu", seed=0
        )
        assert isinstance(agent.replay_buffer, MaskedReplayBuffer)

    def test_action_selection_respects_legality(self):
        agent = DQNAgent(
            state_size=8, action_size=6, hidden_sizes=[16], device="cpu", seed=0
        )
        agent.eval()
        observation = np.random.randn(8).astype(np.float32)
        for _ in range(30):
            assert agent.select_action(observation, [1, 3]) in (1, 3)

    def test_learning_returns_finite_loss_with_mask(self):
        agent = DQNAgent(
            state_size=8, action_size=6, hidden_sizes=[16],
            batch_size=4, device="cpu", seed=0
        )
        result = None
        for _ in range(30):
            result = agent.learn(
                np.random.randn(8).astype(np.float32), 1, 1.0,
                np.random.randn(8).astype(np.float32), False,
                next_legal_actions=[0, 1, 2],
            )
        assert result is not None and "loss" in result
        assert np.isfinite(result["loss"])

    def test_learning_without_next_legal_actions(self):
        """Omitting the mask must still work, treating everything as legal."""
        agent = DQNAgent(
            state_size=8, action_size=6, hidden_sizes=[16],
            batch_size=4, device="cpu", seed=0
        )
        result = None
        for _ in range(30):
            result = agent.learn(
                np.random.randn(8).astype(np.float32), 1, 1.0,
                np.random.randn(8).astype(np.float32), False,
            )
        assert result is not None and np.isfinite(result["loss"])

    def test_terminal_states_do_not_bootstrap(self):
        """A next state with no legal action must contribute no bootstrap value.

        This is exactly the case that made the unmasked target diverge: the max
        over all actions plus the empty-mask 0 * MASK_VALUE would poison the
        target. Masking keeps the Q-values finite.
        """
        agent = DQNAgent(
            state_size=4, action_size=3, hidden_sizes=[8],
            batch_size=2, device="cpu", seed=0
        )
        for _ in range(10):
            agent.learn(
                np.ones(4, dtype=np.float32), 0, 0.0,
                np.zeros(4, dtype=np.float32), False, next_legal_actions=[],
            )
        assert np.isfinite(agent.get_q_values(np.ones(4, dtype=np.float32))).all()

    def test_trains_on_a_real_game(self, klondike_env):
        agent = DQNAgent(
            state_size=klondike_env.observation_space.shape[0],
            action_size=klondike_env.action_space.n,
            hidden_sizes=[32], batch_size=8, device="cpu", seed=0,
        )
        metrics = Trainer(klondike_env, agent, eval_interval=10_000).train(
            episodes=2, max_steps_per_episode=30, verbose=False
        )
        assert len(metrics.rewards) == 2


class TestMaskingSharedAcrossDQN:
    """The masked buffer and mask constant are relocated but still re-exported."""

    def test_masked_buffer_is_the_same_class_from_both_modules(self):
        from rl_card_lib.agents import dqn_agent, double_dqn_agent

        assert double_dqn_agent.MaskedReplayBuffer is dqn_agent.MaskedReplayBuffer
        assert double_dqn_agent.MASK_VALUE == dqn_agent.MASK_VALUE

    def test_dueling_network_still_imports_from_double_dqn(self):
        from rl_card_lib.agents.double_dqn_agent import DuelingQNetwork as DQN2

        assert DQN2 is DuelingQNetwork


class TestDoubleDQNAgent:
    """Tests for Double DQN."""

    def test_action_selection_respects_legality(self):
        agent = DoubleDQNAgent(
            state_size=8, action_size=6, hidden_sizes=[16], device="cpu", seed=0
        )
        agent.eval()
        observation = np.random.randn(8).astype(np.float32)
        for _ in range(30):
            assert agent.select_action(observation, [1, 3]) in (1, 3)

    def test_learning_returns_loss(self):
        agent = DoubleDQNAgent(
            state_size=8, action_size=6, hidden_sizes=[16],
            batch_size=4, device="cpu", seed=0
        )
        result = None
        for _ in range(30):
            result = agent.learn(
                np.random.randn(8).astype(np.float32), 1, 1.0,
                np.random.randn(8).astype(np.float32), False,
                next_legal_actions=[0, 1, 2],
            )
        assert result is not None and "loss" in result
        assert np.isfinite(result["loss"])

    def test_learning_without_next_legal_actions(self):
        """Omitting the mask must still work, treating everything as legal."""
        agent = DoubleDQNAgent(
            state_size=8, action_size=6, hidden_sizes=[16],
            batch_size=4, device="cpu", seed=0
        )
        result = None
        for _ in range(30):
            result = agent.learn(
                np.random.randn(8).astype(np.float32), 1, 1.0,
                np.random.randn(8).astype(np.float32), False,
            )
        assert result is not None and np.isfinite(result["loss"])

    def test_terminal_states_do_not_bootstrap(self):
        """A state with no legal actions must contribute no bootstrap value."""
        agent = DoubleDQNAgent(
            state_size=4, action_size=3, hidden_sizes=[8],
            batch_size=2, device="cpu", seed=0
        )
        for _ in range(10):
            agent.learn(
                np.ones(4, dtype=np.float32), 0, 0.0,
                np.zeros(4, dtype=np.float32), False, next_legal_actions=[],
            )
        assert np.isfinite(agent.get_q_values(np.ones(4, dtype=np.float32))).all()

    def test_declares_it_wants_next_legal_actions(self):
        assert DoubleDQNAgent(
            state_size=4, action_size=3, hidden_sizes=[8], device="cpu"
        ).accepts_next_legal_actions is True

    def test_dueling_can_be_disabled(self):
        plain = DoubleDQNAgent(
            state_size=4, action_size=3, hidden_sizes=[8],
            dueling=False, device="cpu", seed=0
        )
        assert not isinstance(plain.q_network, DuelingQNetwork)

        dueling = DoubleDQNAgent(
            state_size=4, action_size=3, hidden_sizes=[8], device="cpu", seed=0
        )
        assert isinstance(dueling.q_network, DuelingQNetwork)

    def test_dueling_network_shape(self):
        import torch
        net = DuelingQNetwork(10, 5, hidden_sizes=[16, 16])
        assert net(torch.randn(3, 10)).shape == (3, 5)

    def test_dueling_network_needs_hidden_layers(self):
        with pytest.raises(ValueError, match="hidden layer"):
            DuelingQNetwork(10, 5, hidden_sizes=[])

    def test_save_load_roundtrip(self, tmp_path):
        agent = DoubleDQNAgent(
            state_size=8, action_size=6, hidden_sizes=[16],
            batch_size=4, device="cpu", seed=0
        )
        for _ in range(20):
            agent.learn(
                np.random.randn(8).astype(np.float32), 1, 1.0,
                np.random.randn(8).astype(np.float32), False, next_legal_actions=[1],
            )

        path = str(tmp_path / "ddqn.pt")
        agent.save(path)

        restored = DoubleDQNAgent(
            state_size=8, action_size=6, hidden_sizes=[16], device="cpu"
        )
        restored.load(path)

        observation = np.random.randn(8).astype(np.float32)
        np.testing.assert_allclose(
            restored.get_q_values(observation), agent.get_q_values(observation), rtol=1e-5
        )

    def test_trains_on_a_real_game(self, klondike_env):
        agent = DoubleDQNAgent(
            state_size=klondike_env.observation_space.shape[0],
            action_size=klondike_env.action_space.n,
            hidden_sizes=[32], batch_size=8, device="cpu", seed=0,
        )
        metrics = Trainer(klondike_env, agent, eval_interval=10_000).train(
            episodes=2, max_steps_per_episode=30, verbose=False
        )
        assert len(metrics.rewards) == 2


class TestMaskedReplayBuffer:
    """Tests for the mask-aware buffer."""

    def test_push_and_sample(self):
        buffer = MaskedReplayBuffer(capacity=10, action_size=4)
        for _ in range(5):
            buffer.push(np.zeros(3), 1, 1.0, np.ones(3), False, [0, 2])

        assert len(buffer) == 5
        _, _, _, _, _, masks = buffer.sample(3)
        assert masks.shape == (3, 4)
        assert masks[0].tolist() == [True, False, True, False]

    def test_missing_mask_means_all_legal(self):
        buffer = MaskedReplayBuffer(capacity=10, action_size=4)
        buffer.push(np.zeros(3), 1, 1.0, np.ones(3), False, None)
        _, _, _, _, _, masks = buffer.sample(1)
        assert masks[0].all()

    def test_respects_capacity(self):
        buffer = MaskedReplayBuffer(capacity=3, action_size=4)
        for _ in range(10):
            buffer.push(np.zeros(3), 0, 0.0, np.zeros(3), False, [0])
        assert len(buffer) == 3


class TestPPOAgent:
    """Tests for PPO."""

    def test_action_selection_respects_legality(self):
        agent = PPOAgent(
            state_size=8, action_size=6, hidden_sizes=[16], device="cpu", seed=0
        )
        observation = np.random.randn(8).astype(np.float32)
        for _ in range(30):
            assert agent.select_action(observation, [1, 3]) in (1, 3)

    def test_eval_mode_respects_legality(self):
        agent = PPOAgent(
            state_size=8, action_size=6, hidden_sizes=[16], device="cpu", seed=0
        )
        agent.eval()
        observation = np.random.randn(8).astype(np.float32)
        for _ in range(10):
            assert agent.select_action(observation, [4]) == 4

    def test_updates_once_the_rollout_fills(self):
        agent = PPOAgent(
            state_size=8, action_size=6, hidden_sizes=[16],
            rollout_steps=16, minibatch_size=8, device="cpu", seed=0
        )
        results = []
        for _ in range(16):
            observation = np.random.randn(8).astype(np.float32)
            action = agent.select_action(observation, [0, 1, 2])
            results.append(
                agent.learn(observation, action, 1.0, np.random.randn(8).astype(np.float32), False)
            )

        assert all(r is None for r in results[:-1])
        assert results[-1] is not None
        assert agent.updates == 1

    def test_update_losses_are_finite(self):
        """A single legal action makes the masked entropy term a NaN risk."""
        agent = PPOAgent(
            state_size=4, action_size=5, hidden_sizes=[8],
            rollout_steps=8, minibatch_size=4, device="cpu", seed=0
        )
        result = None
        for _ in range(8):
            observation = np.random.randn(4).astype(np.float32)
            action = agent.select_action(observation, [3])
            result = agent.learn(observation, action, 0.5, observation, False)

        assert result is not None
        for key in ("loss", "policy_loss", "value_loss", "entropy"):
            assert np.isfinite(result[key]), f"{key} is not finite: {result[key]}"

    def test_rollout_clears_after_update(self):
        agent = PPOAgent(
            state_size=4, action_size=3, hidden_sizes=[8],
            rollout_steps=8, minibatch_size=4, device="cpu", seed=0
        )
        for _ in range(8):
            observation = np.random.randn(4).astype(np.float32)
            action = agent.select_action(observation, [0, 1])
            agent.learn(observation, action, 1.0, observation, False)

        assert len(agent._states) == 0
        assert len(agent._rewards) == 0

    def test_learn_without_select_is_ignored(self):
        agent = PPOAgent(
            state_size=4, action_size=3, hidden_sizes=[8], device="cpu", seed=0
        )
        observation = np.zeros(4, dtype=np.float32)
        assert agent.learn(observation, 0, 1.0, observation, False) is None

    def test_orphaned_step_is_dropped(self):
        """A select_action with no matching learn must not shift later rewards."""
        agent = PPOAgent(
            state_size=4, action_size=3, hidden_sizes=[8],
            rollout_steps=100, device="cpu", seed=0
        )
        observation = np.zeros(4, dtype=np.float32)

        agent.select_action(observation, [0, 1])  # orphan: never gets a reward
        agent.select_action(observation, [0, 1])
        agent.learn(observation, 0, 1.0, observation, False)

        assert len(agent._states) == len(agent._rewards) == 1

    def test_action_probabilities_zero_out_illegal(self):
        agent = PPOAgent(
            state_size=4, action_size=5, hidden_sizes=[8], device="cpu", seed=0
        )
        probabilities = agent.get_action_probabilities(
            np.zeros(4, dtype=np.float32), [1, 3]
        )
        assert probabilities[0] == pytest.approx(0.0)
        assert probabilities[2] == pytest.approx(0.0)
        assert probabilities[4] == pytest.approx(0.0)
        assert probabilities.sum() == pytest.approx(1.0, abs=1e-5)

    def test_save_load_roundtrip(self, tmp_path):
        agent = PPOAgent(
            state_size=8, action_size=6, hidden_sizes=[16],
            rollout_steps=8, minibatch_size=4, device="cpu", seed=0
        )
        for _ in range(8):
            observation = np.random.randn(8).astype(np.float32)
            action = agent.select_action(observation, [0, 1, 2])
            agent.learn(observation, action, 1.0, observation, False)

        path = str(tmp_path / "ppo.pt")
        agent.save(path)

        restored = PPOAgent(
            state_size=8, action_size=6, hidden_sizes=[16], device="cpu"
        )
        restored.load(path)

        observation = np.random.randn(8).astype(np.float32)
        np.testing.assert_allclose(
            restored.get_action_probabilities(observation, [0, 1, 2]),
            agent.get_action_probabilities(observation, [0, 1, 2]),
            rtol=1e-5,
        )

    def test_trains_on_a_real_game(self, klondike_env):
        agent = PPOAgent(
            state_size=klondike_env.observation_space.shape[0],
            action_size=klondike_env.action_space.n,
            hidden_sizes=[32], rollout_steps=32, minibatch_size=8,
            device="cpu", seed=0,
        )
        metrics = Trainer(klondike_env, agent, eval_interval=10_000).train(
            episodes=2, max_steps_per_episode=30, verbose=False
        )
        assert len(metrics.rewards) == 2


class TestTrainerAgentIntegration:
    """Tests for the trainer plumbing the new agents need."""

    def test_next_legal_actions_reach_opted_in_agents(self, klondike_env):
        seen = []

        class SpyAgent(RandomAgent):
            accepts_next_legal_actions = True

            def learn(self, obs, action, reward, next_obs, done, next_legal_actions=None):
                seen.append(next_legal_actions)
                return None

        agent = SpyAgent(action_size=klondike_env.action_space.n, seed=0)
        Trainer(klondike_env, agent, eval_interval=10_000).train(
            episodes=1, max_steps_per_episode=10, verbose=False
        )

        assert seen and all(isinstance(x, list) for x in seen)

    def test_legacy_agents_keep_the_old_signature(self, klondike_env):
        """Agents that never opted in must not receive the extra keyword."""
        calls = []

        class LegacyAgent(RandomAgent):
            def learn(self, obs, action, reward, next_obs, done):
                calls.append(action)
                return None

        agent = LegacyAgent(action_size=klondike_env.action_space.n, seed=0)
        Trainer(klondike_env, agent, eval_interval=10_000).train(
            episodes=1, max_steps_per_episode=10, verbose=False
        )

        assert calls

    def test_selfplay_against_fixed_opponent(self, macao_env):
        agent = RandomAgent(action_size=macao_env.action_space.n, seed=0)
        opponent = MacaoHeuristicAgent(seed=0)

        trainer = SelfPlayTrainer(macao_env, agent, opponent=opponent)
        assert trainer.self_play is False
        assert trainer.opponent is opponent
        assert opponent.game is macao_env.game

        metrics = trainer.train(episodes=3, max_steps_per_episode=60, verbose=False)
        assert len(metrics.rewards) == 3

    def test_selfplay_defaults_to_frozen_snapshot(self, macao_env):
        agent = RandomAgent(action_size=macao_env.action_space.n, seed=0)
        trainer = SelfPlayTrainer(macao_env, agent)
        assert trainer.self_play is True
        # The opponent is a frozen copy of the agent, not the agent itself.
        assert trainer.opponent is not agent
        assert type(trainer.opponent) is type(agent)

    def test_selfplay_zero_lag_mirror_is_optional(self, macao_env):
        agent = RandomAgent(action_size=macao_env.action_space.n, seed=0)
        trainer = SelfPlayTrainer(macao_env, agent, opponent_update_interval=None)
        assert trainer.self_play is True
        assert trainer.opponent is agent

    def test_selfplay_snapshot_lags_the_learner(self, macao_env):
        """The frozen opponent must only pick up new weights at the interval."""
        agent = QLearningAgent(action_size=macao_env.action_space.n, seed=0)
        trainer = SelfPlayTrainer(macao_env, agent, opponent_update_interval=2)

        first_opponent = trainer.opponent
        # Teach the learner something after the snapshot was taken.
        observation = np.ones(4, dtype=np.float32)
        agent.learn(observation, 0, 5.0, observation, True)
        taught_key = agent._key(observation)
        assert taught_key in agent.q_table

        trainer.train(episodes=1, max_steps_per_episode=5, verbose=False)
        assert trainer.opponent is first_opponent, "refreshed before the interval"
        assert taught_key not in trainer.opponent.q_table, \
            "snapshot saw later learning"

        trainer.train(episodes=2, max_steps_per_episode=5, verbose=False)
        assert trainer.opponent is not first_opponent, "never refreshed"

    def test_selfplay_tracks_turns_from_the_game(self, macao_env):
        """The agent must only be credited for the seat it actually plays."""
        agent = RandomAgent(action_size=macao_env.action_space.n, seed=0)
        opponent = MacaoHeuristicAgent(seed=0)
        trainer = SelfPlayTrainer(macao_env, agent, opponent=opponent)

        trainer._run_episode(training=False, max_steps=40)
        assert trainer._current_player(0) in (0, 1)
