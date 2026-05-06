"""Abstract base class for space evolution methods (Module 3)."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List


class SpaceEvolution(ABC):
    """
    Evolves the strategy space between episodes based on accumulated experience.

    Takes episode summaries and the current strategy space, produces a list
    of modification operations to apply.
    """

    @abstractmethod
    def reflect(self, episode_summaries: List[Dict[str, Any]],
                strategy_space,  # StrategySpace instance
                llm_model: str,
                args) -> List[Dict[str, Any]]:
        """Run reflection on accumulated episodes and return operations.

        Args:
            episode_summaries: list of episode summary dicts
            strategy_space: the current StrategySpace instance
            llm_model: LLM model identifier
            args: full args namespace

        Returns:
            list of operation dicts to pass to strategy_space.apply_operations()
        """
