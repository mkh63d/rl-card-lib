# Visualizer Package - TODO

Status 2026-09-08: the package ships ASCII rendering helpers
(`render_cards`, `render_tableau`, `create_simple_board_view`), the episode
replay engine (`play_episode` / `iter_episode`) with its terminal and HTML
backends, and the training-visualizer CLI; training-curve plotting lives with
`TrainingMetrics.plot()` in core. Everything below is roadmap, kept small on
purpose — the thesis needs plots more than it needs rendering backends.

## Code Structure
- [x] **Correct pathing** — imports verified (depends on cardgames for `Card`,
  on core for `console_safe`; never on `games`/`harness`/`report`, which would
  close a cycle through `examples`)
- [ ] **Backend abstraction** — two backends (terminal, HTML) still share too
  little to be worth a common interface; revisit at a third
- [ ] **Plugin system** — same

## Rendering
- [x] **ASCII rendering** — terminal-friendly card/tableau display
- [x] **Episode replay** — `play_episode` records a seeded deal move by move;
  `print_episode` prints it live or after the fact
- [x] **HTML rendering** — `write_episode_html` writes a self-contained,
  steppable page of one episode
- [ ] **Matplotlib board rendering** — needs per-game layout knowledge, so it
  generalises badly; the ASCII board is what every game already provides
- [ ] **Animation support** — `--delay` auto-play covers the need for now

## Metrics Visualization
- [x] **Training curves** — `TrainingMetrics.plot()` (core) saves
  reward/win-rate/loss plots; the training scripts use it
- [ ] **Comparison plots** for multiple runs
- [ ] **Live dashboard** (optional)
- [ ] **Export capabilities** beyond PNG

## Testing
- [x] **Unit tests** — `tests/test_episode_replay.py` covers the engine and
  both backends against a stub env; `packages/examples/tests/test_watch_episode.py`
  pins the same contract against real Klondike. The card/tableau helpers are
  covered in `tests/test_utils.py`.
- [ ] **Visual regression** — premature before the API settles

## Documentation
- [x] **Usage examples / API reference** — README quick start and
  `docs/reference/visualizer.md`
- [ ] **Gallery**
