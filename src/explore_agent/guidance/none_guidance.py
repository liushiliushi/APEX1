"""No guidance mode — strategy info is shown but agent is NOT forced to follow any sequence.

The agent sees the strategy description and milestones as context/reference only,
and makes its own decisions based on the current game state. No key_action sequences
are shown, no "follow this step by step" instructions.

This is useful for games where the optimal action depends heavily on the current
game state (stats, suspicion, energy, etc.) rather than a fixed plan.
"""
from typing import Any, Dict, Optional

from .base import GuidanceMode


class NoneGuidance(GuidanceMode):
    """Show strategy as context only, no execution guidance."""

    def build_strategy_prompt(self, strategy: Dict[str, Any],
                              milestone_idx: int,
                              step_limit: int,
                              current_step: int = 0,
                              navigation_graph: Dict = None,
                              current_state: str = '',
                              **kwargs) -> str:
        description = strategy.get('description', '')
        hl_steps = strategy.get('high_level_steps', [])

        if not description and not hl_steps:
            return ""

        text = f"\n\n**STRATEGY CONTEXT** (for reference — adapt based on current game state):\n"
        text += f"{description}\n"

        if hl_steps:
            text += "\nGoals to pursue (in rough order, but adapt as needed):\n"
            for i, hl in enumerate(hl_steps):
                step_name = hl.get('step', '')
                text += f"  {i+1}. {step_name}\n"
            text += (
                "\nThese are guidelines, not strict instructions. "
                "Read the current game state and make the best decision for THIS moment. "
                "Skip, reorder, or adapt goals based on what you observe."
            )

        return text

    def build_response_schema(self) -> Dict[str, Any]:
        properties = {
            "progress_analysis": {"type": "string"},
            "next_objective": {"type": "string"},
            "reasoning": {"type": "string"},
            "action": {"type": "string"},
        }
        required = ["progress_analysis", "next_objective", "reasoning", "action"]

        return {
            "type": "json_schema",
            "json_schema": {
                "name": "game_action",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False
                }
            }
        }

    def on_milestone_advance(self, old_idx: int, new_idx: int,
                             strategy: Dict[str, Any]) -> Optional[str]:
        return None
