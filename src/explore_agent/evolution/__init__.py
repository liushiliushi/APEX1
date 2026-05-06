from .base import SpaceEvolution
from .free_reflection import FreeReflection
from .decision_point_mining import DecisionPointMining
from .info_gain_mining import InfoGainMining

EVOLUTION_METHODS = {
    'free_reflection': FreeReflection,
    'decision_point_mining': DecisionPointMining,
    'info_gain_mining': InfoGainMining,
}

__all__ = ['SpaceEvolution', 'FreeReflection', 'DecisionPointMining', 'InfoGainMining', 'EVOLUTION_METHODS']
