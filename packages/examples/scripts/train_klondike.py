"""
Example: Training a DQN agent to play Klondike Solitaire.

This script demonstrates the basic workflow:
1. Create a game environment
2. Initialize a DQN agent
3. Train the agent
4. Evaluate performance
"""

from rl_card_lib.games import KlondikeSolitaire
from rl_card_lib.env import CardGameEnv
from rl_card_lib.agents import DQNAgent
from rl_card_lib.trainer import Trainer


def main():
    # Create the game and environment
    game = KlondikeSolitaire(draw_count=1)
    env = CardGameEnv(game, max_steps=500)

    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.n}")

    # Initialize the DQN agent
    agent = DQNAgent(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n,
        hidden_sizes=[256, 256, 128],
        learning_rate=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.999,  # per episode: reaches the 0.05 floor around episode 3000
        buffer_size=50000,
        batch_size=64,
        target_update_freq=1000,
    )

    print(f"Agent: {agent}")
    print(f"Using device: {agent.device}")

    # Create trainer
    trainer = Trainer(
        env=env,
        agent=agent,
        checkpoint_dir="./checkpoints/klondike",
        log_interval=100,
        eval_interval=500,
        eval_episodes=50,
        checkpoint_interval=1000,
    )

    # Train
    print("\nStarting training...")
    metrics = trainer.train(
        episodes=5000,
        max_steps_per_episode=500,
        verbose=True
    )

    # Final evaluation
    print("\nFinal evaluation...")
    eval_results = trainer.evaluate(episodes=100)
    print(f"Mean reward: {eval_results['mean_reward']:.2f}")
    print(f"Win rate: {eval_results['win_rate']:.2%}")
    print(f"Mean steps: {eval_results['mean_steps']:.1f}")

    # Save final model
    agent.save("./checkpoints/klondike/final_model.pt")
    metrics.save("./checkpoints/klondike/final_metrics.json")

    # Plot results
    try:
        metrics.plot(metrics=["reward", "win"], save_path="./checkpoints/klondike/training_plot.png")
        print("\nTraining plot saved to ./checkpoints/klondike/training_plot.png")
    except Exception as e:
        print(f"Could not create plot: {e}")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
