"""Abstract base class for strategy spaces (Module 0)."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple


class StrategySpace(ABC):
    """
    A strategy space maintains the agent's knowledge structure and supports
    path selection via an exploration method.

    Tree-based spaces call exploration_method.select() at each internal node
    to do MCTS-style traversal. Flat-list spaces call it once over all candidates.
    """

    @property
    def space_type(self) -> str:
        """Return the space type identifier (e.g. 'tree', 'dag', 'flat_list')."""
        return 'unknown'

    @abstractmethod
    def has_strategies(self) -> bool:
        """Return True if the space has any actionable strategies (beyond root/empty)."""

    @abstractmethod
    def get_candidates(self, node_id: str = None) -> List[Dict[str, Any]]:
        """Return candidate children at a given node (or top-level candidates).

        Each candidate dict must contain at least:
            - 'id': unique identifier
            - 'visits': int
            - 'total_reward': float
            - 'reward_sq_sum': float  (for Thompson sampling variance)
        """

    @abstractmethod
    def select_path(self, exploration_method) -> List[str]:
        """Select a path through the space using the given exploration method.

        For tree spaces: MCTS-style traversal from root to leaf.
        For flat spaces: select one strategy from the list.

        Returns a list of node/strategy IDs representing the selected path.
        """

    @abstractmethod
    def path_to_strategy(self, path: List[str]) -> Optional[Dict[str, Any]]:
        """Convert a path (list of IDs) into a strategy dict for prompt injection.

        The returned dict should contain:
            - 'id': str
            - 'description': str
            - 'steps': List[str]  (formatted step strings)
            - 'high_level_steps': List[dict]  (with 'step' and 'key_action')
        """

    @abstractmethod
    def backpropagate(self, path: List[str], score: float,
                      milestone_rewards: Dict[int, float], **kwargs):
        """Update statistics along the selected path after an episode."""

    @abstractmethod
    def apply_operations(self, operations: List[Dict[str, Any]]):
        """Apply tree/space modification operations from reflection."""

    @abstractmethod
    def save(self, path: str):
        """Persist the space to disk."""

    @abstractmethod
    def load(self, path: str):
        """Load the space from disk."""

    @abstractmethod
    def format_for_reflection(self) -> str:
        """Format the space as text for inclusion in reflection prompts."""

    @abstractmethod
    def format_tree_display(self) -> str:
        """Format the space for human-readable display/logging."""

    @abstractmethod
    def active_count(self) -> int:
        """Return number of active strategies/nodes (excluding root)."""

    def evolve(self, episode_summaries, llm_model: str, args) -> List[Dict[str, Any]]:
        """Evolve the strategy space using the space's built-in evolution method.

        Each strategy space bundles its own evolution logic:
          - PlanFlatList uses Free Reflection
          - MilestoneTree/DAG uses Decision Point Mining

        Returns list of operations applied, or empty list.
        """
        return []

    def get_milestone_list(self) -> List[str]:
        """Return a list of milestone/strategy descriptions for tree-aware summary generation.

        Default returns empty list. Subclasses should override to provide
        their milestone vocabulary.
        """
        return []
