"""Collect-all exploration: treat siblings as independent tasks, not mutually exclusive."""
from typing import Any, Dict, List

from .base import ExplorationMethod


class CollectAllExploration(ExplorationMethod):
    """Non-exclusive exploration: select ALL nodes, not one per level.

    Used with milestone_tree to treat siblings as independent tasks
    rather than mutually exclusive alternatives. DFS traversal collects
    every active node; siblings are ordered by avg_reward descending
    (best subtree first).
    """
    collect_all = True  # Flag for select_path to detect

    def score(self, candidate: Dict[str, Any], parent_visits: int) -> float:
        """Score for ordering siblings (best subtree first)."""
        visits = candidate.get('visits', 0)
        if visits == 0:
            return float('inf')
        return candidate['total_reward'] / visits

    def select(self, candidates: List[Dict[str, Any]], parent_visits: int) -> int:
        """Required by interface but not used in collect_all mode."""
        scores = [self.score(c, parent_visits) for c in candidates]
        return max(range(len(scores)), key=lambda i: scores[i])
