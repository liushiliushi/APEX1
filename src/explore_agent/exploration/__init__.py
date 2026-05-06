from .base import ExplorationMethod
from .ucb import UCBExploration
from .thompson import ThompsonSampling
from .epsilon_greedy import EpsilonGreedy
from .collect_all import CollectAllExploration

EXPLORATION_METHODS = {
    'ucb': UCBExploration,
    'thompson': ThompsonSampling,
    'epsilon_greedy': EpsilonGreedy,
    'collect_all': CollectAllExploration,
}

__all__ = ['ExplorationMethod', 'UCBExploration', 'ThompsonSampling', 'EpsilonGreedy', 'CollectAllExploration', 'EXPLORATION_METHODS']
