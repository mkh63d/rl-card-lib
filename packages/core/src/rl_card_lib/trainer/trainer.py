"""Training loop and utilities."""

from typing import Optional, Callable
import time
import numpy as np
from tqdm import tqdm

from rl_card_lib.env.card_game_env import CardGameEnv
from rl_card_lib.agents.base import Agent
from rl_card_lib.trainer.metrics import TrainingMetrics


class Trainer:
    """
    Training manager for RL agents in card game environments.
    
    Features:
    - Configurable training loops
    - Automatic metric tracking
    - Checkpointing
    - Early stopping
    - Evaluation runs
    """
    
    def __init__(
        self,
        env: CardGameEnv,
        agent: Agent,
        eval_env: Optional[CardGameEnv] = None,
        checkpoint_dir: Optional[str] = None,
        log_interval: int = 100,
        eval_interval: int = 1000,
        eval_episodes: int = 100,
        checkpoint_interval: int = 5000,
    ):
        """
        Initialize the trainer.
        
        Args:
            env: Training environment
            agent: Agent to train
            eval_env: Optional separate environment for evaluation
            checkpoint_dir: Directory to save checkpoints
            log_interval: Episodes between logging
            eval_interval: Episodes between evaluations
            eval_episodes: Number of episodes per evaluation
            checkpoint_interval: Episodes between checkpoints
        """
        self.env = env
        self.agent = agent
        self.eval_env = eval_env or env
        self.checkpoint_dir = checkpoint_dir

        # Rule-based and search agents read the game object rather than the
        # observation vector. Bind them here so callers get one less step to
        # forget; an agent bound elsewhere on purpose is left alone.
        if hasattr(agent, "bind") and getattr(agent, "game", None) is None:
            agent.bind(env)

        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.eval_episodes = eval_episodes
        self.checkpoint_interval = checkpoint_interval
        
        self.metrics = TrainingMetrics()
        self._episode_count = 0
        self._total_steps = 0

    def _learn(
        self,
        agent: Agent,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
        info: dict,
    ) -> Optional[dict]:
        """
        Hand a transition to an agent, including next-state legality if it wants it.

        Agents that mask their bootstrap need to know which actions the next
        state allows, which only the post-step info dict knows. Passing it
        unconditionally would break every agent with a five-argument learn(), so
        it goes only to those advertising accepts_next_legal_actions.

        Args:
            agent: Agent to update
            observation: State before action
            action: Action taken
            reward: Reward received
            next_observation: State after action
            done: Whether episode ended
            info: Info dict returned by the step, holding the next legal actions

        Returns:
            Whatever the agent's learn() returned
        """
        if getattr(agent, "accepts_next_legal_actions", False):
            return agent.learn(
                observation, action, reward, next_observation, done,
                next_legal_actions=info.get("legal_actions"),
            )
        return agent.learn(observation, action, reward, next_observation, done)

    def train(
        self,
        episodes: int,
        max_steps_per_episode: Optional[int] = None,
        verbose: bool = True,
        callback: Optional[Callable[[dict], bool]] = None
    ) -> TrainingMetrics:
        """
        Train the agent for a number of episodes.
        
        Args:
            episodes: Number of episodes to train
            max_steps_per_episode: Max steps per episode (None for no limit)
            verbose: Whether to show progress bar
            callback: Optional callback function called after each episode
                      Returns False to stop training
        
        Returns:
            Training metrics
        """
        self.agent.train()
        start_time = time.time()
        
        iterator = range(episodes)
        if verbose:
            iterator = tqdm(iterator, desc="Training", unit="ep")
        
        for episode in iterator:
            episode_metrics = self._run_episode(
                training=True,
                max_steps=max_steps_per_episode
            )
            
            self._episode_count += 1
            self.metrics.add_episode(episode_metrics)
            
            # Update progress bar
            if verbose and isinstance(iterator, tqdm):
                avg_reward = self.metrics.get_recent_average("reward", 100)
                win_rate = self.metrics.get_recent_average("win", 100)
                iterator.set_postfix({
                    "reward": f"{avg_reward:.2f}",
                    "win_rate": f"{win_rate:.2%}"
                })
            
            # Logging
            if self._episode_count % self.log_interval == 0:
                self._log_progress()
            
            # Evaluation
            if self._episode_count % self.eval_interval == 0:
                eval_metrics = self.evaluate(self.eval_episodes, verbose=False)
                self.metrics.add_evaluation(self._episode_count, eval_metrics)
            
            # Checkpointing
            if (self.checkpoint_dir
                    and self._episode_count % self.checkpoint_interval == 0):
                self._save_checkpoint()
            
            # Callback
            if callback:
                if not callback(episode_metrics):
                    break
        
        elapsed = time.time() - start_time
        self.metrics.training_time = elapsed
        
        if verbose:
            print(f"\nTraining completed in {elapsed:.1f}s")
            print(f"Final metrics: {self.metrics.summary()}")
        
        return self.metrics
    
    def _run_episode(
        self,
        training: bool = True,
        max_steps: Optional[int] = None
    ) -> dict:
        """
        Run a single episode.
        
        Args:
            training: Whether to train during episode
            max_steps: Maximum steps
            
        Returns:
            Episode metrics dictionary
        """
        observation, info = self.env.reset()
        self.agent.reset()
        
        episode_reward = 0.0
        episode_steps = 0
        done = False
        losses = []
        
        while not done:
            # Select action
            legal_actions = info.get("legal_actions", None)
            action = self.agent.select_action(observation, legal_actions)
            
            # Execute action
            next_observation, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Learn
            if training:
                learn_result = self._learn(
                    self.agent, observation, action, reward,
                    next_observation, done, info,
                )
                if learn_result and "loss" in learn_result:
                    losses.append(learn_result["loss"])

            observation = next_observation
            episode_reward += reward
            episode_steps += 1
            self._total_steps += 1
            
            if max_steps and episode_steps >= max_steps:
                break
        
        return {
            "reward": episode_reward,
            "steps": episode_steps,
            "win": 1 if info.get("winner") == 0 else 0,
            "loss": np.mean(losses) if losses else 0.0,
        }
    
    def evaluate(
        self,
        episodes: int = 100,
        verbose: bool = True
    ) -> dict:
        """
        Evaluate the agent without training.
        
        Args:
            episodes: Number of evaluation episodes
            verbose: Whether to show progress
            
        Returns:
            Evaluation metrics
        """
        self.agent.eval()
        
        rewards = []
        wins = 0
        steps = []
        
        iterator = range(episodes)
        if verbose:
            iterator = tqdm(iterator, desc="Evaluating", unit="ep")
        
        for _ in iterator:
            metrics = self._run_episode(training=False)
            rewards.append(metrics["reward"])
            wins += metrics["win"]
            steps.append(metrics["steps"])
        
        self.agent.train()
        
        return {
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "min_reward": np.min(rewards),
            "max_reward": np.max(rewards),
            "win_rate": wins / episodes,
            "mean_steps": np.mean(steps),
        }
    
    def _log_progress(self) -> None:
        """Log training progress."""
        avg_reward = self.metrics.get_recent_average("reward", self.log_interval)
        win_rate = self.metrics.get_recent_average("win", self.log_interval)
        avg_loss = self.metrics.get_recent_average("loss", self.log_interval)
        
        print(f"Episode {self._episode_count}: "
              f"Reward={avg_reward:.2f}, "
              f"Win Rate={win_rate:.2%}, "
              f"Loss={avg_loss:.4f}")
    
    def _save_checkpoint(self) -> None:
        """Save training checkpoint."""
        import os
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        path = os.path.join(
            self.checkpoint_dir,
            f"checkpoint_ep{self._episode_count}.pt"
        )
        self.agent.save(path)
        
        # Save metrics
        metrics_path = os.path.join(self.checkpoint_dir, "metrics.json")
        self.metrics.save(metrics_path)


class SelfPlayTrainer(Trainer):
    """
    Trainer for multi-player games where the agent needs someone to play against.

    By default the agent plays itself, which is the classic self-play setup: the
    opponent improves exactly as fast as the agent, so the difficulty tracks it
    and never becomes trivial or hopeless. Pass an `opponent` to train against a
    fixed policy instead, which is what you want when self-play is drifting into
    strategies that only work against a mirror of itself, or when you want the
    win rate to mean something absolute rather than "half, by construction".

    Only the agent's own transitions are learned from, and only its own rewards
    count towards the episode reward.

    Note that self-play here has zero lag: `self.opponent` is the agent itself,
    so the opponent always plays the learner's current weights. There is no
    frozen snapshot. See `opponent_update_interval` below and TODO.md.

    Args:
        env: Game environment
        agent: Agent to train, always seated as player 0
        opponent: Fixed policy for the other seats (None to play against self)
        opponent_update_interval: Currently unused, reserved for the frozen-
            opponent snapshot described in TODO.md. Accepted and stored so the
            eventual implementation does not break call sites, but it has no
            effect on training today: do not read episode counts into it
        **kwargs: Additional arguments passed to Trainer
    """

    def __init__(
        self,
        env: CardGameEnv,
        agent: Agent,
        opponent: Optional[Agent] = None,
        opponent_update_interval: int = 1000,
        **kwargs
    ):
        super().__init__(env, agent, **kwargs)
        self.opponent_update_interval = opponent_update_interval
        self.self_play = opponent is None
        self.opponent = agent if opponent is None else opponent

        if (hasattr(self.opponent, "bind")
                and getattr(self.opponent, "game", None) is None):
            self.opponent.bind(env)

    def _current_player(self, fallback: int) -> int:
        """
        Ask the game whose turn it is.

        Alternating on every step would be wrong: a Macao four skips a turn, an
        invalid action does not advance the game at all, and neither shows up in
        a blind toggle.

        Args:
            fallback: Value to use for games that do not track a current player

        Returns:
            Index of the player to move
        """
        return int(getattr(self.env.game, "current_player_idx", fallback))

    def _run_episode(
        self,
        training: bool = True,
        max_steps: Optional[int] = None
    ) -> dict:
        """Run one episode of the agent against the opponent."""
        observation, info = self.env.reset()
        self.agent.reset()
        if not self.self_play:
            self.opponent.reset()

        episode_reward = 0.0
        episode_steps = 0
        done = False
        losses = []

        current_player = self._current_player(0)

        while not done:
            # Select action based on current player
            legal_actions = info.get("legal_actions", None)

            if current_player == 0:
                # Training agent's turn
                action = self.agent.select_action(observation, legal_actions)
            else:
                # Opponent's turn (use eval mode)
                self.opponent.eval()
                action = self.opponent.select_action(observation, legal_actions)
                self.opponent.train()

            # Execute action
            next_observation, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # Only learn from agent's own actions
            if training and current_player == 0:
                learn_result = self._learn(
                    self.agent, observation, action, reward,
                    next_observation, done, info,
                )
                if learn_result and "loss" in learn_result:
                    losses.append(learn_result["loss"])
                episode_reward += reward

            observation = next_observation
            episode_steps += 1
            self._total_steps += 1

            current_player = self._current_player(1 - current_player)

            if max_steps and episode_steps >= max_steps:
                break

        return {
            "reward": episode_reward,
            "steps": episode_steps,
            "win": 1 if info.get("winner") == 0 else 0,
            "loss": np.mean(losses) if losses else 0.0,
        }
