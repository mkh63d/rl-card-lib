"""
Example: Quick demo of the library functionality.

This script provides a minimal example of using the RL Card Library.
"""

import numpy as np

from rl_card_lib.cardgames import (
    Card,
    Deck,
    Suit,
    Rank,
    is_alternating_color,
    is_next_lower,
    count_by_color,
)
from rl_card_lib.games import KlondikeSolitaire, Macao
from rl_card_lib.env import CardGameEnv
from rl_card_lib.agents import RandomAgent, DQNAgent


def demo_cards():
    """Demonstrate basic card operations."""
    print("=== Card Demo ===\n")

    # Create individual cards
    ace_of_spades = Card(Suit.SPADES, Rank.ACE)
    king_of_hearts = Card(Suit.HEARTS, Rank.KING, face_up=False)

    print(f"Ace of Spades: {ace_of_spades}")
    print(f"King of Hearts (face down): {king_of_hearts}")

    king_of_hearts.flip()
    print(f"King of Hearts (flipped): {king_of_hearts}")

    # Create and shuffle a deck
    deck = Deck()
    print(f"\nFresh deck: {deck}")

    deck.shuffle(seed=42)
    drawn = deck.draw(5, face_up=True)
    print(f"Drew 5 cards: {' '.join(str(c) for c in drawn)}")
    print(f"Remaining: {deck}")

    # Rule helpers
    if len(drawn) >= 2:
        print("\nRule helpers:")
        print(f"Alternating colors? {is_alternating_color(drawn[0], drawn[1])}")
        print(f"Is next lower? {is_next_lower(drawn[0], drawn[1])}")
    print(f"Color counts: {count_by_color(drawn)}")


def demo_klondike():
    """Demonstrate Klondike Solitaire game."""
    print("\n=== Klondike Solitaire Demo ===\n")

    game = KlondikeSolitaire()
    obs = game.reset()

    print(game.render())
    print(f"\nObservation shape: {obs.shape}")

    # Play a few random moves
    print("\nPlaying 5 random moves...")
    for i in range(5):
        legal = game.get_legal_actions()
        action = np.random.choice(legal)
        obs, reward, done, _, info = game.step(action)
        print(f"Move {i+1}: {game.action_to_string(action)} (reward: {reward:.2f})")

        if done:
            print("Game over!")
            break


def demo_macao():
    """Demonstrate Macao game."""
    print("\n=== Macao Demo ===\n")

    game = Macao(num_players=2)
    obs = game.reset()

    print(game.render())
    print(f"\nObservation shape: {obs.shape}")

    # Play a few moves
    print("\nPlaying 10 moves...")
    for i in range(10):
        legal = game.get_legal_actions()
        action = np.random.choice(legal)
        obs, reward, done, _, info = game.step(action)
        print(f"Player {info['current_player']}: {game.action_to_string(action)}")

        if done:
            print(f"Game over! Winner: Player {info['winner']}")
            break


def demo_training():
    """Demonstrate quick training setup."""
    print("\n=== Training Demo ===\n")

    # Create environment
    game = KlondikeSolitaire()
    env = CardGameEnv(game, max_steps=100)

    # Create agents
    random_agent = RandomAgent(action_size=env.action_space.n)
    dqn_agent = DQNAgent(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n,
        hidden_sizes=[64, 64],
        device="cpu"
    )

    # Run a few episodes with random agent
    print("Running 10 episodes with random agent...")
    total_reward = 0
    for _ in range(10):
        obs, info = env.reset()
        done = False
        while not done:
            action = random_agent.select_action(obs, info.get("legal_actions"))
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

    print(f"Random agent average reward: {total_reward / 10:.2f}")

    # Run a few episodes with DQN agent (untrained)
    print("\nRunning 10 episodes with untrained DQN agent...")
    total_reward = 0
    for _ in range(10):
        obs, info = env.reset()
        done = False
        while not done:
            action = dqn_agent.select_action(obs, info.get("legal_actions"))
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

    print(f"Untrained DQN agent average reward: {total_reward / 10:.2f}")

    print("\nNote: Train the agent for many more episodes to see improvement!")


def main():
    """Run all demos."""
    demo_cards()
    demo_klondike()
    demo_macao()
    demo_training()

    print("\n" + "=" * 50)
    print("Demo complete! Check scripts/train_*.py for full training scripts.")


if __name__ == "__main__":
    main()
