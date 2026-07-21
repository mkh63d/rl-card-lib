# Visualizer Package - TODO

Status 2026-07-17: the package ships ASCII rendering helpers
(`render_cards`, `render_tableau`, `create_simple_board_view`) and the
training-visualizer CLI; training-curve plotting lives with
`TrainingMetrics.plot()` in core. Everything below is roadmap, kept small on
purpose — the thesis needs plots more than it needs rendering backends.

## Code Structure
- [x] **Correct pathing** — imports verified (depends on cardgames for `Card`)
- [ ] **Backend abstraction** — only worthwhile once a second backend exists
- [ ] **Plugin system** — same

## Rendering
- [x] **ASCII rendering** — terminal-friendly card/tableau display
- [ ] **Matplotlib board rendering**
- [ ] **HTML rendering**
- [ ] **Animation support**

## Metrics Visualization
- [x] **Training curves** — `TrainingMetrics.plot()` (core) saves
  reward/win-rate/loss plots; the training scripts use it
- [ ] **Comparison plots** for multiple runs
- [ ] **Live dashboard** (optional)
- [ ] **Export capabilities** beyond PNG

## Testing
- [ ] **Unit tests** for the rendering helpers (pure string functions, cheap
  to add when the API settles)
- [ ] **Visual regression** — premature before the API settles

## Documentation
- [ ] **Usage examples / API reference / gallery**
