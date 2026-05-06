"""Abstract base class for exploration methods (Module 2)."""
from abc import ABC, abstractmethod
from typing import Dict, List, Any


class ExplorationMethod(ABC):
    """
    Stateless exploration method that scores candidates and selects one.

    Used by StrategySpace.select_path() — tree spaces call select() at each
    level, flat spaces call it once over all candidates.
    """

    @abstractmethod
    def score(self, candidate: Dict[str, Any], parent_visits: int) -> float:
        """Compute the exploration score for a single candidate.

        Args:
            candidate: dict with at least 'visits', 'total_reward', 'reward_sq_sum'
            parent_visits: total visits of the parent node (or sum for flat lists)

        Returns:
            float score (higher = more likely to be selected)
        """

    @abstractmethod
    def select(self, candidates: List[Dict[str, Any]], parent_visits: int) -> int:
        """Select one candidate index from the list.

        Args:
            candidates: list of candidate dicts
            parent_visits: parent visit count

        Returns:
            index into candidates list
        """
