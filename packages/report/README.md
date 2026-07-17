# RL Card Lib - Report

Utilities for summarizing training process parameters for RL agents.

## Installation

```bash
pip install -e ./packages/report
```

## Usage

```python
from rl_card_lib.agents import DQNAgent
from rl_card_lib.env import CardGameEnv
from rl_card_lib.trainer import Trainer
from rl_card_lib.report import TrainingReport

# Create your game, env, agent, and trainer first
report = TrainingReport.from_trainer(
    trainer,
    episodes=5000,
    max_steps_per_episode=500,
)

print(report.to_markdown())
```
