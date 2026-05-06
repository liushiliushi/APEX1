"""Epsilon-greedy exploration method."""
import random
from typing import Any, Dict, List

from .base import ExplorationMethod


class EpsilonGreedy(ExplorationMethod):
    """Epsilon-greedy: with probability epsilon choose random, else argmax(avg_reward).

    Unvisited candidates are always prioritized (random among them).
    """

    def __init__(self, epsilon: float = 0.1):
        self.epsilon = epsilon

    def score(self, candidate: Dict[str, Any], parent_visits: int) -> float:
        visits = candidate.get('visits', 0)
        if visits == 0:
            return float('inf')
        return candidate['total_reward'] / visits

    def select(self, candidates: List[Dict[str, Any]], parent_visits: int) -> int:
        if not candidates:
            raise ValueError("No candidates to select from")

        # Unvisited candidates get priority
        unvisited = [i for i, c in enumerate(candidates) if c.get('visits', 0) == 0]
        if unvisited:
            return random.choice(unvisited)

        # Epsilon-greedy
        if random.random() < self.epsilon:
            return random.randrange(len(candidates))

        # Greedy: argmax of average reward
        scores = [self.score(c, parent_visits) for c in candidates]
        return max(range(len(candidates)), key=lambda i: scores[i])
