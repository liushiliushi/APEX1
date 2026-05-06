"""Abstract base class for guidance modes (Module 1)."""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class GuidanceMode(ABC):
    """
    Controls how strategy information is presented to the agent in prompts.

    Different modes vary in how much of the plan is shown at once and how
    milestone transitions are communicated.
    """

    @abstractmethod
    def build_strategy_prompt(self, strategy: Dict[str, Any],
                              milestone_idx: int,
                              step_limit: int,
                              current_step: int = 0,
                              navigation_graph: Dict = None,
                              current_state: str = '',
                              **kwargs) -> str:
        """Build the strategy section of the system prompt.

        Args:
            strategy: strategy dict with 'high_level_steps', 'steps', 'description', etc.
            milestone_idx: current milestone index (0-based)
            step_limit: max steps per episode
            current_step: current step number in the episode

        Returns:
            str to append to system prompt
        """

    @abstractmethod
    def build_response_schema(self) -> Dict[str, Any]:
        """Build the JSON response format schema for the LLM.

        Returns:
            response_format dict for the OpenAI API
        """

    @abstractmethod
    def on_milestone_advance(self, old_idx: int, new_idx: int,
                             strategy: Dict[str, Any]) -> Optional[str]:
        """Called when the agent reports milestone advancement.

        Returns optional text to inject into the next prompt, or None.
        """

    def reset(self):
        """Reset per-episode state. Override in subclasses if needed."""
        pass
