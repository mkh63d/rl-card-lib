# env & trainer

Gymnasium-compatible environment wrappers, the training loop, and metrics.

## Environment wrappers

::: rl_card_lib.env.card_game_env.CardGameEnv

::: rl_card_lib.env.card_game_env.MaskedCardGameEnv

## Gymnasium registration

The bundled games as `gymnasium.make` ids. Registered on `rl_card_lib.games`
import; see [the quickstart](../getting-started/quickstart.md) for the id table.

::: rl_card_lib.games.gym_registration.register_gym_envs

::: rl_card_lib.games.gym_registration.registered_gym_ids

## Trainer

::: rl_card_lib.trainer.trainer.Trainer

::: rl_card_lib.trainer.trainer.SelfPlayTrainer

## Metrics

::: rl_card_lib.trainer.metrics.TrainingMetrics
