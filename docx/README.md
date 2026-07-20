# RL Card Library - PlantUML Diagrams

This folder contains PlantUML diagrams documenting the architecture of the RL Card Library.

## Diagrams

### 1. Package Diagram (`package_diagram.puml`)
Shows the complete modular structure of the library:
- **cardgames**: Card, Deck, Player, CardGame base classes and the reusable `rules` helpers
- **core**: Game base class, Gym wrapper, and the minimal standalone trainer
- **games**: Concrete game implementations (Klondike, Macao), the Klondike solvability
  search (`solve_klondike`) and the per-game heuristic agents
- **agents**: three families —
  baselines (`RandomAgent`, `HeuristicAgent`, `GreedyLookaheadAgent`),
  search (`MCTSAgent`), and
  learners (`QLearningAgent`, `DQNAgent`, `DoubleDQNAgent`, `PPOAgent`)
  with their network and replay-buffer components
- **env**: Gymnasium-like environment wrappers (CardGameEnv, MaskedCardGameEnv)
- **trainer**: Training loop, self-play trainer with a frozen opponent snapshot, and metrics tracking
- **utils**: Encoding utilities
- **visualizer**: Rendering helpers for cards and boards
- **report**: `TrainingReport`, which collects the run's parameters and renders them as Markdown or JSON

### 2. Use Case Diagram (`use_case_diagram.puml`)
Shows the main use cases for different actors:
- **Developer/Researcher**: Creating custom games and environment wrappers, reusing the rule
  helpers, analysing Klondike solvability
- **ML Engineer**: Creating agents from all three families, training, evaluating, benchmarking,
  self-play, tracking metrics, and generating training reports
- **Game Designer**: Implementing and testing game rules and visualizations

### 3. Sequence Diagram (`sequence_diagram.puml`)
Shows the complete DQN agent training workflow:
- Initialization phase
- Per-episode `agent.reset()`, where epsilon decays once per episode
- Episode loop with action selection (exploration vs exploitation)
- The optional self-play path, where a frozen opponent snapshot takes the other seats and is
  refreshed every `opponent_update_interval` episodes
- Environment stepping and reward calculation
- Experience replay and Q-network updates
- Metrics tracking, evaluation, and checkpointing

## Rendering

To render these diagrams:

### Online
- Use [PlantUML Web Server](https://www.plantuml.com/plantuml/uml/)
- Paste the `.puml` file content

### VS Code
- Install the "PlantUML" extension
- Open a `.puml` file and use `Alt+D` to preview

### Command Line
```bash
# Install PlantUML (requires Java)
java -jar plantuml.jar diagram_name.puml
```

### Programmatic (Python)
```python
from plantuml import PlantUML

server = PlantUML(url='http://www.plantuml.com/plantuml/png/')
server.processes_file('package_diagram.puml')
```
