# Adding your own game

The library trains and reports on any game that implements the `Game` contract.
Klondike and Macao are not special-cased — they register themselves through the
same public API described here (see
[`games/registration.py`](https://github.com/mkh63d/rl-card-lib/blob/main/packages/examples/src/rl_card_lib/games/registration.py)),
so those two calls are your reference.

There are two things you might want:

- **A report only** — you drive training yourself and want the HTML report over
  your results. You need the `Game` contract and one `register_game(...)` call.
- **The full sweep** — the library trains every learner on your game, measures
  baselines, and renders the report. You additionally register an *execution*
  spec with `register_sweep_game(...)`.

---

## 1. Implement the `Game` contract

Subclass `rl_card_lib.core.Game` (or `rl_card_lib.cardgames.CardGame` if you
want a 52-card `Deck` and `Player` objects). Seven methods are abstract:

```python
from rl_card_lib.core import Game
import numpy as np

class Hearts(Game):
    def __init__(self):
        super().__init__(num_players=4)
        self.reset()

    def reset(self) -> np.ndarray:            # start a new game, return obs
        ...
        self.done = False
        self.winner = None
        return self.get_observation()

    def step(self, action: int):              # -> (obs, reward, terminated, truncated, info)
        ...
        return self.get_observation(), reward, self.done, False, {}

    def get_legal_actions(self) -> list[int]:
        ...

    def get_observation(self) -> np.ndarray:
        ...

    def get_action_space_size(self) -> int:
        ...

    def get_observation_shape(self) -> tuple[int, ...]:
        ...

    def is_game_over(self) -> bool:
        return self.done
```

Maintain the plain attributes yourself: set `self.done` and `self.winner` when
the game ends, and advance `self.current_player_idx` for a multiplayer game (the
`next_player()` helper does this). `winner` is the index of the winning player,
or `None`.

### What you get for free

| Method | Default | When to override |
|---|---|---|
| `copy()` | deep copy of the game | large state or an RNG whose stream you must reproduce exactly (the bundled games override for speed) |
| `determinize()` | `self.copy()` | your game has **hidden** information a search agent should re-sample rather than read |
| `get_reward(player_idx)` | `0.0` | **multiplayer** — MCTS needs each non-actor's terminal payoff, or losses are invisible to the search |
| `render()`, `action_to_string()`, `get_legal_action_mask()` | generic | cosmetic |

Because `copy()` deep-copies by default, the search agents work on your game
with no extra code.

### Which agents you get

| Agent | Available for your game? |
|---|---|
| `RandomAgent`, `QLearningAgent`, `DQNAgent`, `DoubleDQNAgent`, `PPOAgent` | Yes — they only read `get_observation_shape()` and `get_action_space_size()` |
| `GreedyLookaheadAgent`, `MCTSAgent` | Yes — they use `copy()`, `step()`, `get_legal_actions()`; MCTS also needs `get_reward()` for multiplayer |
| `HeuristicAgent` | Only if you write one — `score_action(game, action)` is game knowledge and cannot be derived |

---

## 2. Report only

If you train the models yourself, build a `RunRecord` per model and hand them to
the report. First declare how your game is presented:

```python
from rl_card_lib.report import register_game, RunRecord, RunStore
from rl_card_lib.report.html_report import HtmlReport

register_game(
    "hearts",
    label="Hearts",
    headline_key="penalty_points",       # the metric your game is judged on
    headline_label="Penalty points",
    headline_max=26,
    higher_is_better=False,              # fewer penalty points is better
    episode_curves=["penalty_points"],  # a per-episode series to chart
)

store = RunStore("./results")
store.reset_run_dir("hearts", "dqn")     # before training, not after

record = RunRecord.from_training(
    game="hearts", agent="dqn", agent_class="DQNAgent",
    metrics=metrics,                     # your TrainingMetrics
    episode_extras={"penalty_points": [...]},   # one value per episode
    baseline_before={"penalty_points": 20.0},
    baseline_after={"penalty_points": 8.0},
)
store.save_run(record)

HtmlReport.build(store).write("./results/index.html")
```

The custom `penalty_points` series is stored and charted, its declared `0-26`
bound appears in the range column, and a custom agent name gets its own colour.

`register_game` auto-registers the headline metric with its bound. For any other
custom metric, declare it explicitly with `register_metric("cards_saved",
label="Cards saved", kind="count", min=0, max=13, unit="cards")`.

---

## 3. The full sweep

To have the library train and evaluate for you, register an execution spec.
This is one call and it forwards the presentation fields to `register_game`, so
you do not call both.

```python
from rl_card_lib.env import CardGameEnv
from rl_card_lib.harness import register_sweep_game

def evaluate_hearts(agent, episodes, seed):
    """Play `episodes` games and return a dict of metrics -- whatever your
    game is judged on. The keys become columns in the report."""
    ...
    return {"penalty_points": mean_penalty, "win_rate": win_rate}

register_sweep_game(
    "hearts",
    env_factory=lambda: CardGameEnv(Hearts(), max_steps=200),
    max_steps=200,
    evaluate=evaluate_hearts,
    heuristic_factory=lambda seed: HeartsHeuristicAgent(seed=seed),  # optional
    episode_extras=lambda game, agent: {"penalty_points": game.penalty()},
    # presentation (forwarded to register_game)
    label="Hearts",
    headline_key="penalty_points",
    headline_label="Penalty points",
    headline_max=26,
    higher_is_better=False,
    episode_curves=["penalty_points"],
)
```

Then run the sweep. Make sure the module that calls `register_sweep_game` is
imported first (the bundled games do this from `rl_card_lib.games/__init__.py`):

```bash
python packages/examples/scripts/run_sweep.py --games hearts --episodes 5000
```

Everything else follows automatically: the learners train, `Random` /
`GreedyLookahead` / `MCTS` baselines are measured (MCTS only if your game is
copyable), your `penalty_points` curve is recorded per episode, and
`results/index.html` shows it all. Use `--games all` to include every registered
game, or `python -m rl_card_lib.report.cli --exclude-builtin-games` to render a
report of your games alone.

### Self-play games

For a game trained against an opponent (like Macao), declare it:

```python
register_sweep_game(
    "hearts",
    ...,
    self_play=True,
    opponent_factory=lambda seed: HeartsHeuristicAgent(seed=seed),
)
```

The sweep then uses `SelfPlayTrainer`, and `--self-play` switches to the
zero-lag mirror match.

---

## Checklist

- [ ] Subclass `Game`, implement the seven abstract methods, maintain `done` /
      `winner` / `current_player_idx`.
- [ ] Multiplayer? Override `get_reward(player_idx)` with actor-relative
      terminal payoffs.
- [ ] Hidden information? Override `determinize()`.
- [ ] `register_sweep_game(...)` (or `register_game(...)` for report-only),
      declaring `headline_key`, `headline_max` and `higher_is_better`.
- [ ] Optionally supply a `HeuristicAgent` subclass and a `heuristic_factory`.
- [ ] Import your registration module before running the sweep.
