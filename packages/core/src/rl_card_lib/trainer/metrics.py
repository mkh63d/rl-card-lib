"""Training metrics tracking and analysis."""

from typing import Optional, Any
from collections import deque
import json
import numpy as np


class TrainingMetrics:
    """
    Tracks and analyzes training metrics over time.
    
    Collects episode metrics and provides aggregation functions
    for analysis and visualization.
    """
    
    def __init__(self, window_size: int = 100):
        """
        Initialize metrics tracker.
        
        Args:
            window_size: Window size for moving averages
        """
        self.window_size = window_size
        
        # Episode metrics
        self.rewards: list[float] = []
        self.steps: list[int] = []
        self.wins: list[int] = []
        self.losses: list[float] = []
        
        # Evaluation metrics
        self.evaluations: list[dict] = []
        
        # Timing
        self.training_time: float = 0.0
        
        # Moving averages
        self._reward_window = deque(maxlen=window_size)
        self._win_window = deque(maxlen=window_size)
    
    def add_episode(self, metrics: dict) -> None:
        """
        Add metrics from a single episode.
        
        Args:
            metrics: Dictionary containing episode metrics
        """
        reward = metrics.get("reward", 0.0)
        steps = metrics.get("steps", 0)
        win = metrics.get("win", 0)
        loss = metrics.get("loss", 0.0)
        
        self.rewards.append(reward)
        self.steps.append(steps)
        self.wins.append(win)
        self.losses.append(loss)
        
        self._reward_window.append(reward)
        self._win_window.append(win)
    
    def add_evaluation(self, episode: int, metrics: dict) -> None:
        """
        Add evaluation results.
        
        Args:
            episode: Episode number when evaluation occurred
            metrics: Evaluation metrics dictionary
        """
        self.evaluations.append({
            "episode": episode,
            **metrics
        })
    
    def get_recent_average(self, metric: str, n: int = 100) -> float:
        """
        Get average of recent values for a metric.
        
        Args:
            metric: Metric name ("reward", "win", "loss", "steps")
            n: Number of recent values to average
            
        Returns:
            Average value
        """
        data = getattr(self, f"{metric}s" if metric != "loss" else "losses", [])
        if not data:
            return 0.0
        recent = data[-n:]
        return np.mean(recent)
    
    def get_moving_average(self, metric: str) -> list[float]:
        """
        Get moving average over all episodes.
        
        Args:
            metric: Metric name
            
        Returns:
            List of moving average values
        """
        data = getattr(self, f"{metric}s" if metric != "loss" else "losses", [])
        if not data:
            return []
        
        result = []
        for i in range(len(data)):
            start = max(0, i - self.window_size + 1)
            result.append(np.mean(data[start:i + 1]))
        return result
    
    def summary(self) -> dict:
        """
        Get summary statistics.
        
        Returns:
            Dictionary with summary metrics
        """
        return {
            "total_episodes": len(self.rewards),
            "total_wins": sum(self.wins),
            "win_rate": sum(self.wins) / len(self.wins) if self.wins else 0.0,
            "mean_reward": np.mean(self.rewards) if self.rewards else 0.0,
            "std_reward": np.std(self.rewards) if self.rewards else 0.0,
            "max_reward": max(self.rewards) if self.rewards else 0.0,
            "min_reward": min(self.rewards) if self.rewards else 0.0,
            "mean_steps": np.mean(self.steps) if self.steps else 0.0,
            "training_time": self.training_time,
        }
    
    def save(self, path: str) -> None:
        """
        Save metrics to JSON file.
        
        Args:
            path: File path to save to
        """
        data = {
            "summary": self.summary(),
            "rewards": self.rewards,
            "steps": self.steps,
            "wins": self.wins,
            "losses": self.losses,
            "evaluations": self.evaluations,
        }
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    
    def load(self, path: str) -> None:
        """
        Load metrics from JSON file.
        
        Args:
            path: File path to load from
        """
        with open(path, "r") as f:
            data = json.load(f)
        
        self.rewards = data.get("rewards", [])
        self.steps = data.get("steps", [])
        self.wins = data.get("wins", [])
        self.losses = data.get("losses", [])
        self.evaluations = data.get("evaluations", [])
    
    def plot(
        self,
        metrics: list[str] = ["reward", "win"],
        figsize: tuple[int, int] = (12, 4),
        save_path: Optional[str] = None
    ) -> Any:
        """
        Plot training metrics.
        
        Args:
            metrics: List of metrics to plot
            figsize: Figure size
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed, cannot plot")
            return None
        
        n_plots = len(metrics)
        fig, axes = plt.subplots(1, n_plots, figsize=figsize)
        
        if n_plots == 1:
            axes = [axes]
        
        for ax, metric in zip(axes, metrics):
            data = getattr(self, f"{metric}s" if metric != "loss" else "losses", [])
            if not data:
                continue
            
            # Raw data (light)
            ax.plot(data, alpha=0.3, label="Raw")
            
            # Moving average (bold)
            ma = self.get_moving_average(metric)
            ax.plot(ma, linewidth=2, label=f"MA({self.window_size})")
            
            ax.set_xlabel("Episode")
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f"{metric.capitalize()} over Training")
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        
        return fig
