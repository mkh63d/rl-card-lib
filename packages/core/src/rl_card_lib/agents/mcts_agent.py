"""Monte Carlo Tree Search agent (UCT) with determinization for hidden cards."""

from typing import Any, Callable, Optional, Union
from collections import defaultdict
import math
import random
import numpy as np

from rl_card_lib.agents.base import Agent
from rl_card_lib.agents.heuristic import GameAwareAgent


class _MinMaxStats:
    """
    Tracks the value range seen so far, so UCT can normalize Q into [0, 1].

    Textbook UCT assumes rewards in [0, 1] and picks the exploration constant
    against that scale. The games here return whatever their reward shaping
    happens to produce, which would make a single constant either drown out Q or
    be drowned by it. Rescaling by the observed range keeps one constant usable
    across games.
    """

    def __init__(self) -> None:
        self.minimum = math.inf
        self.maximum = -math.inf

    def update(self, value: float) -> None:
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


class _Node:
    """One position in the search tree."""

    __slots__ = (
        "state", "parent", "action", "actor", "reward",
        "children", "untried", "visits", "value_sums", "terminal",
    )

    def __init__(
        self,
        state: Any,
        parent: Optional["_Node"],
        action: Optional[int],
        actor: Optional[int],
        reward: float,
        num_players: int,
        terminal: bool = False,
    ):
        self.state = state
        self.parent = parent
        #: Action that led into this node, and the player who took it. Rewards
        #: are credited to that player, not to whoever moves next.
        self.action = action
        self.actor = actor
        self.reward = reward
        self.children: list["_Node"] = []
        self.untried: list[int] = []
        self.visits = 0
        self.value_sums = np.zeros(num_players, dtype=np.float64)
        self.terminal = terminal


class MCTSAgent(GameAwareAgent):
    """
    Plans each move by growing a search tree of simulated games (UCT).

    Four steps per iteration: descend the tree by UCB1 to a promising unexpanded
    node, add one child, play a fast rollout to the end (or a depth cap), and
    credit the result back up the path. Actions that keep producing good
    outcomes get visited more, so the tree deepens where it matters.

    Returns are tracked per player, so this works unchanged for single-player
    Klondike and two-player Macao: each node is scored from the point of view of
    whoever is to move there, which is plain UCT in the first case and max^n in
    the second.

    On hidden information: the game object holds the true face-down cards, so a
    search that reads it straight would cheat. Instead each search runs on a
    determinization, a random re-deal of the cards the agent cannot see that
    stays consistent with what it can. Setting determinizations > 1 searches
    several such worlds and pools the visit counts, which is the standard
    Perfect Information Monte Carlo treatment: it never confuses two worlds
    inside one tree, but it also cannot value gathering information, since every
    world it searches is already fully known to it.

    This agent does not learn. It is a compute-for-strength baseline: quality
    scales with `simulations`, and so does the time per move.

    Args:
        game: Game or environment to read state from (can be bound later)
        simulations: Total rollouts per move, split across determinizations
        exploration_weight: UCB1 constant; higher explores more
        rollout_depth: Steps a rollout plays before giving up on reaching an end
        gamma: Discount applied to rewards further into a simulation
        determinizations: Independent hidden-card samples to search per move
        use_determinization: Set False to search the true state, which cheats at
            Klondike and Macao but is a useful upper-bound baseline
        rollout_policy: Agent or callable(game, legal_actions) -> action used
            inside rollouts; None means uniform random. A decent policy here
            sharpens the value estimates but costs time on every rollout step
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        game: Optional[Any] = None,
        simulations: int = 100,
        exploration_weight: float = 1.4,
        rollout_depth: int = 40,
        gamma: float = 0.98,
        determinizations: int = 1,
        use_determinization: bool = True,
        rollout_policy: Optional[Union[Agent, Callable[[Any, list[int]], int]]] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(game=game, name="MCTSAgent", seed=seed)

        if simulations < 1:
            raise ValueError(f"simulations must be at least 1, got {simulations}")
        if determinizations < 1:
            raise ValueError(
                f"determinizations must be at least 1, got {determinizations}"
            )

        self.simulations = simulations
        self.exploration_weight = exploration_weight
        self.rollout_depth = rollout_depth
        self.gamma = gamma
        self.determinizations = determinizations
        self.use_determinization = use_determinization
        self.rollout_policy = rollout_policy

        self._py_rng = random.Random(seed)
        self._stats = _MinMaxStats()
        self._num_players = 1

    def select_action(
        self,
        observation: np.ndarray,
        legal_actions: Optional[list[int]] = None
    ) -> int:
        """
        Search from the current position and play the most-visited action.

        Visit count is the choice rather than mean value because it is the more
        robust of the two: an action can post a high mean off one lucky rollout,
        but only sustained agreement from UCB1 gets it visited often.

        Args:
            observation: Unused, the search runs on copies of the bound game
            legal_actions: Valid action indices (read from the game if None)

        Returns:
            Selected action index
        """
        game = self._require_game()

        legal = list(legal_actions) if legal_actions else list(game.get_legal_actions())
        if not legal:
            return 0
        if len(legal) == 1:
            return legal[0]

        self._num_players = int(getattr(game, "num_players", 1))
        observer = int(getattr(game, "current_player_idx", 0))
        legal_set = set(legal)

        per_tree = max(1, self.simulations // self.determinizations)
        visits: dict[int, int] = defaultdict(int)
        values: dict[int, float] = defaultdict(float)

        for _ in range(self.determinizations):
            if self.use_determinization:
                root_state = game.determinize(observer, self._py_rng)
            else:
                root_state = game.copy()

            # Values are not comparable across trees built on different worlds,
            # so each search gets its own normalization range.
            self._stats = _MinMaxStats()
            root = self._search(root_state, per_tree)

            for child in root.children:
                # A determinization should not change what the observer may play,
                # but never return something the real game would reject.
                if child.action in legal_set:
                    visits[child.action] += child.visits
                    values[child.action] += child.value_sums[observer]

        if not visits:
            return int(legal[self.rng.randint(len(legal))])

        return int(max(visits, key=lambda a: (visits[a], values[a] / visits[a])))

    def _new_node(
        self,
        state: Any,
        parent: Optional[_Node],
        action: Optional[int],
        actor: Optional[int],
        reward: float,
        terminal: bool = False,
    ) -> _Node:
        """Create a node and cache the actions still to try from it."""
        node = _Node(
            state=state,
            parent=parent,
            action=action,
            actor=actor,
            reward=reward,
            num_players=self._num_players,
            terminal=terminal,
        )
        if not terminal:
            node.untried = list(state.get_legal_actions())
            self._py_rng.shuffle(node.untried)
            if not node.untried:
                # Nothing legal left: the game is stuck, which ends it.
                node.terminal = True
        return node

    def _search(self, root_state: Any, iterations: int) -> _Node:
        """
        Grow a tree from `root_state` for a fixed number of iterations.

        Args:
            root_state: Position to search from (owned by the search)
            iterations: Number of select-expand-rollout-backup cycles

        Returns:
            The root, whose children carry the visit counts
        """
        root = self._new_node(root_state, None, None, None, 0.0)

        for _ in range(iterations):
            node = root
            path = [root]

            # Selection: follow UCB1 while the node is fully expanded.
            while not node.terminal and not node.untried and node.children:
                node = self._uct_select(node)
                path.append(node)

            # Expansion: add one child for an untried action.
            if not node.terminal and node.untried:
                node = self._expand(node)
                path.append(node)

            returns = self._rollout(node)
            self._backpropagate(path, returns)

        return root

    def _uct_select(self, node: _Node) -> _Node:
        """
        Pick the child with the best UCB1 score, judged by whoever moves at `node`.

        Args:
            node: Fully expanded, non-terminal node

        Returns:
            The chosen child
        """
        perspective = int(node.state.current_player_idx)
        log_visits = math.log(node.visits + 1)

        best_score = -math.inf
        best_child = node.children[0]

        for child in node.children:
            exploit = self._stats.normalize(child.value_sums[perspective] / child.visits)
            explore = self.exploration_weight * math.sqrt(log_visits / child.visits)
            score = exploit + explore
            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def _expand(self, node: _Node) -> _Node:
        """
        Play one untried action and attach the resulting position as a child.

        Args:
            node: Node with at least one untried action

        Returns:
            The new child node
        """
        action = node.untried.pop()
        actor = int(node.state.current_player_idx)

        child_state = node.state.copy()
        _, reward, terminated, truncated, _ = child_state.step(action)

        child = self._new_node(
            state=child_state,
            parent=node,
            action=action,
            actor=actor,
            reward=float(reward),
            terminal=bool(terminated or truncated),
        )
        node.children.append(child)
        return child

    def _rollout_action(self, state: Any, legal: list[int]) -> int:
        """Choose a rollout move with the configured policy, or at random."""
        if self.rollout_policy is None:
            return legal[self.rng.randint(len(legal))]

        if isinstance(self.rollout_policy, GameAwareAgent):
            self.rollout_policy.bind(state)
            return self.rollout_policy.select_action(state.get_observation(), legal)

        if isinstance(self.rollout_policy, Agent):
            return self.rollout_policy.select_action(state.get_observation(), legal)

        return self.rollout_policy(state, legal)

    def _rollout(self, node: _Node) -> np.ndarray:
        """
        Play out from a node and return each player's discounted return.

        Args:
            node: Leaf to play out from (its state is not mutated)

        Returns:
            Array of returns indexed by player
        """
        returns = np.zeros(self._num_players, dtype=np.float64)
        if node.terminal:
            return returns

        state = node.state.copy()
        discount = 1.0

        for _ in range(self.rollout_depth):
            legal = state.get_legal_actions()
            if not legal:
                break

            actor = int(state.current_player_idx)
            action = self._rollout_action(state, legal)
            _, reward, terminated, truncated, _ = state.step(action)

            returns[actor] += discount * float(reward)
            discount *= self.gamma

            if terminated or truncated:
                break

        return returns

    def _backpropagate(self, path: list[_Node], returns: np.ndarray) -> None:
        """
        Credit a simulation's result to every node on the path.

        Walking up, the return seen from a parent is its edge reward plus the
        discounted return of the child, credited to the player who actually took
        that edge.

        Args:
            path: Root-to-leaf node path
            returns: Per-player returns from the rollout
        """
        value = returns.copy()

        for node in reversed(path):
            node.visits += 1
            node.value_sums += value

            if node.parent is not None:
                self._stats.update(node.value_sums[node.actor] / node.visits)
                value = self.gamma * value
                value[node.actor] += node.reward
