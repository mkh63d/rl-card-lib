# harness

The importable pieces of the training and benchmarking scripts: the one
definition of the sweep-registration API, the evaluation protocols, and the
baseline agent sets. See [Add your own game](../custom_game.md) for how to
register a game for the full sweep.

## Sweep registration

::: rl_card_lib.harness.registry.register_sweep_game

::: rl_card_lib.harness.registry.SweepGame

## Learners

::: rl_card_lib.harness.learners.build_learner

::: rl_card_lib.harness.learners.load_trained_learner

## Solve-time benchmark

::: rl_card_lib.harness.solve_benchmark.curate_solvable_pool

::: rl_card_lib.harness.solve_benchmark.measure_agent_on_pool

::: rl_card_lib.harness.solve_benchmark.run_solve_benchmark

## Evaluation protocols

::: rl_card_lib.harness.evaluation.evaluate_klondike

::: rl_card_lib.harness.evaluation.evaluate_macao

## Third-party algorithms

`rl_card_lib.harness.sb3_maskable` seats `sb3-contrib`'s `MaskablePPO` in the
protocols above:

- `train_maskable_ppo(total_timesteps, seed=0, ...)` — trains on
  `rl_card_lib/MacaoMasked-v0`, dealing only from the TRAIN pool.
- `MaskablePPOAgent(model)` — wraps a trained model in the library's `Agent`
  interface, so `evaluate_macao` scores it on the same held-out deals as every
  bundled agent.

It is documented in prose rather than through mkdocstrings on purpose: the
module imports `sb3_contrib` at module scope, so a `:::` directive would make
building these docs require the optional extra. See
[Third-party algorithms](../guides/third-party-algorithms.md).

## Baselines

::: rl_card_lib.harness.baselines.measure_baselines
