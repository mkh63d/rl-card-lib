# visualizer

Text rendering of cards and tableaux, and step-by-step replay of a single
episode. Training curves are rendered by `TrainingMetrics.plot()` in the
[core package](env-trainer.md#metrics).

## Cards and tableaux

::: rl_card_lib.visualizer.visualization.render_cards

::: rl_card_lib.visualizer.visualization.render_tableau

::: rl_card_lib.visualizer.visualization.create_simple_board_view

## Episode replay

Watching one agent play one seeded deal, move by move. The engine is duck-typed
against the env and agent contracts, so any game the library can run replays
without a line here knowing its name.

```python
from rl_card_lib.harness import sweep_game
from rl_card_lib.visualizer import play_episode, print_episode

env = sweep_game("klondike").env_factory()
record = play_episode(env, agent, seed=0)
print_episode(record)
```

`packages/examples/scripts/watch_episode.py` is the entry point built on this:
it resolves the game from the registry, the agent from a baseline or a
checkpoint, and the deal from the pool of seeds a solver proved winnable.

::: rl_card_lib.visualizer.episode.EpisodeStep

::: rl_card_lib.visualizer.episode.EpisodeRecord

::: rl_card_lib.visualizer.episode.iter_episode

::: rl_card_lib.visualizer.episode.play_episode

## Printing a replay

::: rl_card_lib.visualizer.replay.format_step

::: rl_card_lib.visualizer.replay.format_episode

::: rl_card_lib.visualizer.replay.episode_summary

::: rl_card_lib.visualizer.replay.step_printer

::: rl_card_lib.visualizer.replay.print_episode

## Exporting a replay

::: rl_card_lib.visualizer.html_episode.episode_to_html

::: rl_card_lib.visualizer.html_episode.write_episode_html
