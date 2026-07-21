# Examples Package - TODO

## Code Structure
- [x] **Correct pathing** — verified by the package's own test suite and CI
- [ ] **Script organization** — organize scripts by complexity level

## Scripts
- [x] **benchmark_agents.py** — benchmarks all agents on both games over
  shared deals; reward and cards-up now agree in ranking (the reward loop that
  made them diverge is fixed)
- [x] **train_agents.py / train_klondike.py / train_macao.py** — epsilon
  decays retuned for the per-episode semantics; stale reward-loop caveats
  removed
- [ ] **Quick demo** — keep under ~5 seconds, add error handling
- [ ] **Hyperparameter tuning example** (grid search)
- [ ] **Macao tournament example** (multi-agent round-robin)

## Advanced Examples
- [ ] **Custom game** — example of creating a new game
- [ ] **Custom agent** — example of implementing a custom agent
- [ ] **Distributed training** — multi-process training example
- [ ] **Web integration** — Flask/FastAPI integration example

## Testing
- [x] **Game logic tests** — `packages/examples/tests/` plus the root suites
- [ ] **Smoke tests for the scripts themselves** — the library under them is
  tested; the argparse entry points are not

## Documentation
- [x] **Expected output** — benchmark/train scripts print labeled tables;
  module docstrings say what to expect
- [ ] **README per example** with troubleshooting notes
