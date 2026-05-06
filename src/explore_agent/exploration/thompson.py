"""Thompson Sampling exploration method."""
import math
import random
from typing import Any, Dict, List

from .base import ExplorationMethod


class ThompsonSampling(ExplorationMethod):
    """Thompson Sampling: sample from posterior distribution.

    For each candidate, sample reward ~ Normal(mean, stderr).
    Unvisited candidates (visits=0) get inf to match the protocol
    used by select_path() which separates untried vs tried nodes.
    Candidates with 1 visit use a calibrated prior.

    The prior_std should be on the order of a single milestone's max
    reward (~10-25 for Zork1) so that samples are meaningful.
    min_stderr prevents over-exploitation when variance collapses
    after many visits.
    """

    def __init__(self, prior_std: float = 15.0, min_stderr: float = 1.0):
        """
        Args:
            prior_std: std dev for prior (visits=1). Should match the scale
                       of per-milestone rewards (default 15 for Zork-like games).
            min_stderr: minimum stderr to maintain exploration even at high
                        visit counts (prevents collapsing to greedy).
        """
        self.prior_std = prior_std
        self.min_stderr = min_stderr

    def score(self, candidate: Dict[str, Any], parent_visits: int) -> float:
        visits = candidate.get('visits', 0)
        total_reward = candidate.get('total_reward', 0.0)
        reward_sq_sum = candidate.get('reward_sq_sum', 0.0)

        # Unvisited: return inf to match select_path() untried detection
        if visits == 0:
            return float('inf')

        mean = total_reward / visits

        if visits == 1:
            # Single observation: use calibrated prior
            return random.gauss(mean, self.prior_std)

        # Compute sample variance: Var = (sum(x^2) - n*mean^2) / (n-1)
        variance = max(0, (reward_sq_sum - visits * mean ** 2) / (visits - 1))
        stderr = math.sqrt(variance / visits) if variance > 0 else self.min_stderr
        # Enforce minimum stderr to maintain exploration
        stderr = max(stderr, self.min_stderr)
        return random.gauss(mean, stderr)

    def select(self, candidates: List[Dict[str, Any]], parent_visits: int) -> int:
        if not candidates:
            raise ValueError("No candidates to select from")

        # Separate untried (inf) from tried
        untried = [i for i in range(len(candidates))
                   if candidates[i].get('visits', 0) == 0]
        if untried:
            return random.choice(untried)

        samples = [self.score(c, parent_visits) for c in candidates]
        return max(range(len(candidates)), key=lambda i: samples[i])
