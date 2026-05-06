"""UCB1 + softmax exploration method."""
import math
import random
from typing import Any, Dict, List

from .base import ExplorationMethod


class UCBExploration(ExplorationMethod):
    """UCB1 with softmax probabilistic selection.

    Untried candidates (visits=0) get infinite score and are selected randomly.
    Among tried candidates, UCB1 scores are computed and softmax is applied.
    """

    def __init__(self, c: float = 1.414):
        """
        Args:
            c: exploration constant for UCB1
        """
        self.c = c

    def score(self, candidate: Dict[str, Any], parent_visits: int) -> float:
        visits = candidate.get('visits', 0)
        if visits == 0:
            return float('inf')
        avg_reward = candidate['total_reward'] / visits
        exploration = self.c * math.sqrt(math.log(max(parent_visits, 1)) / visits)
        return avg_reward + exploration

    def select(self, candidates: List[Dict[str, Any]], parent_visits: int) -> int:
        if not candidates:
            raise ValueError("No candidates to select from")

        scores = [self.score(c, parent_visits) for c in candidates]

        # Untried candidates get priority (random among them)
        untried = [i for i, s in enumerate(scores) if s == float('inf')]
        if untried:
            return random.choice(untried)

        # Softmax probabilistic selection
        max_score = max(scores)
        exp_values = [math.exp(s - max_score) for s in scores]
        total_exp = sum(exp_values)
        probs = [e / total_exp for e in exp_values]
        return random.choices(range(len(candidates)), weights=probs, k=1)[0]
