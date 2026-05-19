# RL Card Library - PlantUML Diagrams

This folder contains PlantUML diagrams documenting the architecture of the RL Card Library.

## Diagrams

### 1. Package Diagram (`package_diagram.puml`)
Shows the complete modular structure of the library:
- **cardgames**: Card, Deck, Player, CardGame base classes
- **core**: Game base class, Gym wrapper, and simple trainer
- **games**: Concrete game implementations (Klondike, Macao)
- **agents**: RL agents (Random, DQN) with neural network components
- **env**: Gymnasium-like environment wrappers (CardGameEnv, MaskedCardGameEnv)
- **trainer**: Training loop, self-play trainer, and metrics tracking
- **utils**: Encoding utilities
- **visualizer**: Rendering helpers for cards and boards

### 2. Use Case Diagram (`use_case_diagram.puml`)
Shows the main use cases for different actors:
- **Developer/Researcher**: Creating custom games and environment wrappers
- **ML Engineer**: Training, evaluating, self-play, and tracking metrics
- **Game Designer**: Implementing and testing game rules and visualizations

### 3. Sequence Diagram (`sequence_diagram.puml`)
Shows the complete DQN agent training workflow:
- Initialization phase
- Episode loop with action selection (exploration vs exploitation)
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
