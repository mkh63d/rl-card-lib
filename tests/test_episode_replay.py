"""Tests for the episode replay engine and its renderers."""

import io

import numpy as np

from rl_card_lib.visualizer.episode import (
    EpisodeRecord,
    EpisodeStep,
    iter_episode,
    play_episode,
)
from rl_card_lib.visualizer.html_episode import episode_to_html, write_episode_html
from rl_card_lib.visualizer.replay import (
    episode_summary,
    format_episode,
    format_step,
    print_episode,
)


class FakeGame:
    winner = None


class FakeEnv:
    """The slice of the CardGameEnv contract the replay actually leans on.

    Written here rather than driving a real game because the engine is
    duck-typed on purpose: a fake is the only way to assert it makes no
    assumption a custom game would fail.
    """

    def __init__(self, *, rewards=(1.0,), terminate_at=None, max_steps=None,
                 flags=None, observation=None, board="board ♠"):
        self.game = FakeGame()
        self.render_mode = None
        self.max_steps = max_steps
        self.last_deal_seed = None
        self.reset_seed = None
        self.reset_calls = 0
        self.render_modes_seen = []
        self.steps_taken = 0
        self._rewards = list(rewards)
        self._terminate_at = terminate_at
        self._flags = flags or {}
        self._board = board
        self._observation = (
            observation if observation is not None
            else np.zeros(3, dtype=np.float32)
        )

    def reset(self, *, seed=None, options=None):
        self.reset_seed = seed
        self.last_deal_seed = 7 if seed is None else seed
        self.reset_calls += 1
        self.steps_taken = 0
        return self._observation, {"legal_actions": [0, 1, 2]}

    def step(self, action):
        reward = self._rewards[self.steps_taken % len(self._rewards)]
        terminated = (
            self._terminate_at is not None and self.steps_taken == self._terminate_at
        )
        info = {"legal_actions": [0, 1, 2], "winner": 0 if terminated else None}
        info.update(self._flags.get(self.steps_taken, {}))
        self.steps_taken += 1
        return self._observation, reward, terminated, False, info

    def render(self):
        self.render_modes_seen.append(self.render_mode)
        return f"{self._board} {self.steps_taken}"

    def action_to_string(self, action):
        return f"move {action}"


class BareEnv:
    """The floor: an env that neither renders nor names its actions.

    A custom game is not obliged to implement either, and the replay has to
    survive both being missing rather than refusing to run.
    """

    def __init__(self, max_steps=2):
        self.game = FakeGame()
        self.render_mode = None
        self.max_steps = max_steps
        self.steps_taken = 0

    def reset(self, *, seed=None, options=None):
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action):
        self.steps_taken += 1
        return np.zeros(2, dtype=np.float32), 0.0, False, False, {}


class FakeAgent:
    name = "Fake"

    def __init__(self):
        self.training = True
        self.observations = []
        self.legal_seen = []
        self.reset_calls = 0

    def select_action(self, observation, legal_actions=None):
        self.observations.append(observation)
        self.legal_seen.append(legal_actions)
        return 1

    def reset(self):
        self.reset_calls += 1

    def eval(self):
        self.training = False

    def train(self):
        self.training = True


class BindingAgent(FakeAgent):
    """An MCTS-shaped agent: it wants the env handed to it before it plays."""

    def __init__(self):
        super().__init__()
        self.bound = None

    def bind(self, env):
        self.bound = env


def record_of(**kwargs):
    env = FakeEnv(**kwargs)
    return play_episode(env, FakeAgent(), seed=3), env


class TestPlayEpisode:
    def test_seed_is_forwarded_to_reset(self):
        record, env = record_of(max_steps=4)
        assert env.reset_seed == 3
        assert record.seed == 3

    def test_unseeded_replay_records_the_seed_the_env_dealt(self):
        env = FakeEnv(max_steps=2)
        record = play_episode(env, FakeAgent())
        assert record.seed == 7

    def test_rewards_accumulate(self):
        record, _ = record_of(rewards=(1.0, -0.5), max_steps=4)
        assert [s.reward for s in record.steps] == [1.0, -0.5, 1.0, -0.5]
        assert [s.total_reward for s in record.steps] == [1.0, 0.5, 1.5, 1.0]
        assert record.total_reward == 1.0

    def test_action_label_comes_from_the_env(self):
        record, _ = record_of(max_steps=1)
        assert record.steps[0].action == 1
        assert record.steps[0].action_label == "move 1"

    def test_env_without_action_to_string_falls_back(self):
        record = play_episode(BareEnv(max_steps=1), FakeAgent(), seed=0)
        assert record.steps[0].action_label == "Action 1"

    def test_info_without_legal_actions_is_tolerated(self):
        env, agent = BareEnv(max_steps=2), FakeAgent()
        play_episode(env, agent, seed=0)
        assert agent.legal_seen == [None, None]

    def test_stops_at_the_envs_step_cap(self):
        record, _ = record_of(max_steps=5)
        assert record.step_count == 5

    def test_explicit_max_steps_overrides_the_env(self):
        env = FakeEnv(max_steps=50)
        record = play_episode(env, FakeAgent(), seed=0, max_steps=3)
        assert record.step_count == 3
        # Cut short by the viewer, not by the env: nothing terminated and the
        # env never raised truncated, so the episode is simply unfinished.
        assert record.outcome == "unfinished"

    def test_termination_ends_the_episode(self):
        record, _ = record_of(max_steps=50, terminate_at=2)
        assert record.step_count == 3
        assert record.terminated
        assert record.solved
        assert record.outcome == "solved"

    def test_flags_are_read_from_info(self):
        record, _ = record_of(
            max_steps=3,
            flags={0: {"invalid_action": True}, 1: {"repeated_position": True}},
        )
        assert record.steps[0].invalid and not record.steps[0].repeated
        assert record.steps[1].repeated and not record.steps[1].invalid
        assert not record.steps[2].invalid and not record.steps[2].repeated

    def test_legal_actions_are_passed_to_the_agent(self):
        env, agent = FakeEnv(max_steps=2), FakeAgent()
        play_episode(env, agent, seed=0)
        assert agent.legal_seen == [[0, 1, 2], [0, 1, 2]]

    def test_dict_observation_reaches_the_agent_unchanged(self):
        # MaskedCardGameEnv hands back {observation, action_mask}; an agent
        # trained on that shape needs it whole, so the engine must not coerce.
        observation = {"observation": np.zeros(2), "action_mask": np.ones(3)}
        env, agent = FakeEnv(max_steps=2, observation=observation), FakeAgent()
        play_episode(env, agent, seed=0)
        assert agent.observations[0] is observation

    def test_agent_is_reset_and_evaluated_then_restored(self):
        env, agent = FakeEnv(max_steps=2), FakeAgent()
        assert agent.training
        play_episode(env, agent, seed=0)
        assert agent.reset_calls == 1
        # Watching an agent must not leave it changed.
        assert agent.training

    def test_agent_already_in_eval_stays_there(self):
        env, agent = FakeEnv(max_steps=2), FakeAgent()
        agent.eval()
        play_episode(env, agent, seed=0)
        assert not agent.training

    def test_binding_agent_is_bound(self):
        env, agent = FakeEnv(max_steps=2), BindingAgent()
        play_episode(env, agent, seed=0)
        assert agent.bound is env

    def test_names_default_to_the_game_and_the_agent(self):
        record, _ = record_of(max_steps=1)
        assert record.game == "FakeGame"
        assert record.agent == "Fake"

    def test_names_can_be_overridden(self):
        env = FakeEnv(max_steps=1)
        record = play_episode(env, FakeAgent(), seed=0, game="klondike", agent_name="ppo")
        assert (record.game, record.agent) == ("klondike", "ppo")


class TestRenderMode:
    def test_boards_are_captured_through_ansi_mode(self):
        record, env = record_of(max_steps=2)
        # "human" would make CardGameEnv print from inside step(); None would
        # make render() return None and capture nothing.
        assert set(env.render_modes_seen) == {"ansi"}
        assert record.steps[0].board.startswith("board ♠")

    def test_original_render_mode_is_restored(self):
        env = FakeEnv(max_steps=2)
        env.render_mode = "human"
        play_episode(env, FakeAgent(), seed=0)
        assert env.render_mode == "human"

    def test_nothing_is_printed_during_play(self, capsys):
        env = FakeEnv(max_steps=3)
        env.render_mode = "human"
        play_episode(env, FakeAgent(), seed=0)
        assert capsys.readouterr().out == ""

    def test_capture_board_off_skips_rendering(self):
        env = FakeEnv(max_steps=3)
        record = play_episode(env, FakeAgent(), seed=0, capture_board=False)
        assert env.render_modes_seen == []
        assert all(step.board == "" for step in record.steps)

    def test_env_that_cannot_render_still_replays(self):
        record = play_episode(BareEnv(max_steps=2), FakeAgent(), seed=0)
        assert record.step_count == 2
        assert record.opening_board == ""
        assert all(step.board == "" for step in record.steps)


class TestIterEpisode:
    def test_steps_arrive_one_at_a_time(self):
        env = FakeEnv(max_steps=5)
        steps = iter_episode(env, FakeAgent(), seed=0)
        first = next(steps)
        # The generator body has run exactly one step, not the whole episode --
        # which is what lets a slow agent be watched while it plays.
        assert first.index == 0
        assert env.steps_taken == 1

    def test_on_reset_receives_the_opening_board(self):
        env, seen = FakeEnv(max_steps=1), []
        list(iter_episode(env, FakeAgent(), seed=0,
                          on_reset=lambda board, info: seen.append((board, info))))
        assert seen[0][0].startswith("board ♠")
        assert seen[0][1]["legal_actions"] == [0, 1, 2]

    def test_abandoning_the_generator_restores_the_agent(self):
        env, agent = FakeEnv(max_steps=50), FakeAgent()
        steps = iter_episode(env, agent, seed=0)
        next(steps)
        assert not agent.training
        steps.close()
        assert agent.training


class TestFormatting:
    def step(self, **kwargs):
        base = dict(
            index=4, action=11, action_label="Move waste to foundation 4",
            reward=0.99, total_reward=1.5, terminated=False, truncated=False,
            invalid=False, repeated=False, winner=None, board="BOARD",
        )
        base.update(kwargs)
        return EpisodeStep(**base)

    def test_step_line_carries_index_label_and_numbers(self):
        text = format_step(self.step(), board=False)
        assert "[   4]" in text
        assert "Move waste to foundation 4" in text
        assert "+0.990" in text and "+1.500" in text
        assert "BOARD" not in text

    def test_board_is_appended_when_asked(self):
        assert format_step(self.step()).endswith("\nBOARD")

    def test_flags_are_named(self):
        text = format_step(self.step(invalid=True, repeated=True), board=False)
        assert "illegal" in text and "repeat" in text

    def test_summary_reports_the_outcome(self):
        record = EpisodeRecord(
            game="klondike", seed=12, agent="ppo", opening_board="",
            steps=[self.step(terminated=True, winner=0)],
            winner=0, total_reward=1.5, terminated=True,
        )
        text = episode_summary(record)
        assert "klondike" in text and "12" in text and "ppo" in text
        assert "solved" in text

    def test_format_episode_covers_deal_moves_and_summary(self):
        record = EpisodeRecord(
            game="g", seed=1, agent="a", opening_board="DEAL", steps=[self.step()],
        )
        text = format_episode(record)
        assert text.startswith("DEAL")
        assert "Move waste to foundation 4" in text
        assert "outcome" in text


class Cp1250Stream(io.StringIO):
    """A stream reporting the code page a Polish Windows console runs.

    Subclassed rather than assigned to: `io.StringIO` takes no attributes of
    its own, and its inherited `encoding` reads as None -- which `console_safe`
    correctly treats as "nothing to downgrade for".
    """

    encoding = "cp1250"


class Utf8Stream(io.StringIO):
    encoding = "utf-8"


class TestPrinting:
    def test_suit_glyphs_are_downgraded_for_a_legacy_console(self):
        env = FakeEnv(max_steps=2)
        record = play_episode(env, FakeAgent(), seed=0)
        stream = Cp1250Stream()
        print_episode(record, stream=stream)
        out = stream.getvalue()
        # cp1250 cannot encode U+2660..U+2666, and print() loses the whole
        # frame to one glyph it cannot take (#47).
        assert "♠" not in out
        assert "board" in out

    def test_utf8_stream_keeps_the_glyphs(self):
        env = FakeEnv(max_steps=1)
        record = play_episode(env, FakeAgent(), seed=0)
        stream = Utf8Stream()
        print_episode(record, stream=stream)
        assert "♠" in stream.getvalue()

    def test_a_live_iterator_prints_move_by_move(self):
        env = FakeEnv(max_steps=3)
        stream = io.StringIO()
        print_episode(iter_episode(env, FakeAgent(), seed=0), stream=stream, board=False)
        lines = [line for line in stream.getvalue().splitlines() if line]
        assert len(lines) == 3
        # No summary: an iterator is not a finished record.
        assert "outcome" not in stream.getvalue()

    def test_pause_runs_once_per_move(self):
        env = FakeEnv(max_steps=4)
        record = play_episode(env, FakeAgent(), seed=0)
        calls = []
        print_episode(record, stream=io.StringIO(), pause=lambda: calls.append(1))
        assert len(calls) == 4


class TestHtml:
    def make(self, **kwargs):
        env = FakeEnv(max_steps=kwargs.pop("max_steps", 3), **kwargs)
        return play_episode(env, FakeAgent(), seed=42)

    def test_document_is_self_contained_and_describes_the_episode(self):
        page = episode_to_html(self.make())
        assert page.startswith("<!DOCTYPE html>")
        assert "42" in page and "FakeGame" in page
        # One frame per move plus the opening deal.
        assert page.count('"action":') == 4
        assert "http://" not in page and "https://" not in page

    def test_glyphs_survive_into_the_page(self):
        # The page declares UTF-8, so unlike the terminal it shows the real
        # suits; downgrading here would lose what the record still holds.
        assert "♠" in episode_to_html(self.make())

    def test_markup_in_a_label_cannot_break_out(self):
        env = FakeEnv(max_steps=1)
        env.action_to_string = lambda action: "</script><img src=x onerror=alert(1)>"
        page = episode_to_html(play_episode(env, FakeAgent(), seed=0))
        assert "<img src=x" not in page
        assert page.count("</script>") == 1

    def test_title_can_be_overridden(self):
        page = episode_to_html(self.make(), title="Deal 42")
        assert "<title>Deal 42</title>" in page

    def test_write_creates_parent_directories(self, tmp_path):
        target = tmp_path / "nested" / "episode.html"
        written = write_episode_html(self.make(), target)
        assert written.is_file()
        assert written.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")
