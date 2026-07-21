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

## Baselines

::: rl_card_lib.harness.baselines.measure_baselines
