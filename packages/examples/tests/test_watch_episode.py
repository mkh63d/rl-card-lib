"""Replaying a real bundled game, not a fake env.

`tests/test_episode_replay.py` pins the engine's contract against a stub. This
pins the other half: that the contract it assumes is the one the bundled games
and the registry actually offer -- a seeded deal that reproduces, boards that
render, and action labels a reader can follow.
"""

from __future__ import annotations

import rl_card_lib.games  # noqa: F401  (registers the bundled games)
from rl_card_lib.agents import RandomAgent
from rl_card_lib.harness import sweep_game
from rl_card_lib.visualizer import episode_to_html, play_episode


def klondike_episode(seed=0, max_steps=20, **kwargs):
    env = sweep_game("klondike").env_factory()
    agent = RandomAgent(action_size=env.action_space.n, seed=0)
    return play_episode(env, agent, seed=seed, max_steps=max_steps, **kwargs), env


class TestKlondikeReplay:
    def test_episode_records_moves_with_readable_labels(self):
        record, _ = klondike_episode()
        assert record.step_count == 20
        assert record.game == "KlondikeSolitaire"
        assert record.agent == "RandomAgent"
        # Klondike names every action; nothing should fall back to "Action N".
        assert all(not s.action_label.startswith("Action ") for s in record.steps)

    def test_seed_round_trips_through_the_env(self):
        record, env = klondike_episode(seed=12345)
        assert record.seed == 12345 == env.last_deal_seed

    def test_same_seed_replays_identically(self):
        first, _ = klondike_episode(seed=7)
        second, _ = klondike_episode(seed=7)
        assert first.opening_board == second.opening_board
        assert [s.action for s in first.steps] == [s.action for s in second.steps]

    def test_boards_are_captured_and_change(self):
        record, _ = klondike_episode()
        assert "Klondike" in record.opening_board
        assert "Tableaux" in record.steps[0].board
        # The deal is not the position after twenty moves.
        assert record.steps[-1].board != record.opening_board

    def test_env_is_left_as_it_was_found(self):
        _, env = klondike_episode()
        assert env.render_mode is None

    def test_record_renders_to_a_page(self):
        record, _ = klondike_episode()
        page = episode_to_html(record)
        assert page.startswith("<!DOCTYPE html>")
        # One frame per move plus the opening deal.
        assert page.count('"action":') == record.step_count + 1
