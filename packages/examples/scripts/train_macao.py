"""
Example: Training a DQN agent to play Macao.

This script demonstrates training an agent in the Macao card game
using self-play or against a random opponent.
"""

from rl_card_lib.games import Macao
from rl_card_lib.env import CardGameEnv
from rl_card_lib.agents import DQNAgent, RandomAgent
from rl_card_lib.trainer import Trainer


def main():
    # Create the game and environment
    game = Macao(num_players=2, max_turns=200)
    env = CardGameEnv(game, max_steps=200)

    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.n}")

    # Initialize the DQN agent
    agent = DQNAgent(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n,
        hidden_sizes=[128, 128],
        learning_rate=5e-4,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.999,
        buffer_size=30000,
        batch_size=32,
        target_update_freq=500,
    )

    print(f"Agent: {agent}")
    print(f"Using device: {agent.device}")

    # Create trainer
    trainer = Trainer(
        env=env,
        agent=agent,
        checkpoint_dir="./checkpoints/macao",
        log_interval=100,
        eval_interval=500,
        eval_episodes=100,
        checkpoint_interval=2000,
    )

    # Train
    print("\nStarting training...")
    metrics = trainer.train(
        episodes=10000,
        max_steps_per_episode=200,
        verbose=True
    )

    # Final evaluation
    print("\nFinal evaluation...")
    eval_results = trainer.evaluate(episodes=200)
    print(f"Mean reward: {eval_results['mean_reward']:.2f}")
    print(f"Win rate: {eval_results['win_rate']:.2%}")
    print(f"Mean steps: {eval_results['mean_steps']:.1f}")

    # Compare against random agent
    print("\nComparing against random agent...")
    random_agent = RandomAgent(action_size=env.action_space.n)

    wins_vs_random = 0
    for _ in range(100):
        obs, info = env.reset()
        done = False
        current_player = 0

        while not done:
            legal = info.get("legal_actions", None)

            if current_player == 0:
                action = agent.select_action(obs, legal)
            else:
                action = random_agent.select_action(obs, legal)

            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            current_player = 1 - current_player

        if info.get("winner") == 0:
            wins_vs_random += 1

    print(f"Win rate vs random agent: {wins_vs_random}%")

    # Save final model
    agent.save("./checkpoints/macao/final_model.pt")
    metrics.save("./checkpoints/macao/final_metrics.json")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
