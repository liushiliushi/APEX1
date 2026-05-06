"""Step-by-Step guidance mode (Mode B).

Only shows the current milestone. When completed, reveals the next one.
The agent doesn't see future milestones, reducing cognitive overload.

DAG mode: shows all available milestones (deps satisfied + not completed),
letting the agent pick the closest one to its current position.
"""
from typing import Any, Dict, List, Optional, Set

from .base import GuidanceMode


class StepByStepGuidance(GuidanceMode):
    """Show only the current milestone; reveal next upon completion.

    In DAG mode, shows available/completed/locked milestone groups.
    """

    def __init__(self):
        self.completed_milestones: Set[int] = set()
        self._is_dag: bool = False
        self._last_target_idx: Optional[int] = None

    def reset(self):
        """Reset per-episode state."""
        self.completed_milestones = set()
        self._last_target_idx = None

    def build_strategy_prompt(self, strategy: Dict[str, Any],
                              milestone_idx: int,
                              step_limit: int,
                              current_step: int = 0,
                              navigation_graph: Dict = None,
                              current_state: str = '',
                              **kwargs) -> str:
        hl_steps = strategy.get('high_level_steps', [])
        self._is_dag = strategy.get('is_dag', False)

        if not hl_steps:
            return (
                f"\n\n**CURRENT STRATEGY**: {strategy.get('description', '')}\n"
                f"Choose actions that work toward this goal."
            )

        if self._is_dag:
            return self._build_dag_prompt(hl_steps, step_limit, current_step)
        else:
            return self._build_linear_prompt(hl_steps, milestone_idx)

    def _build_linear_prompt(self, hl_steps: List[Dict], milestone_idx: int) -> str:
        """Original linear step-by-step logic."""
        cur = milestone_idx

        if cur >= len(hl_steps):
            text = f"\n\nYou have completed all {len(hl_steps)} assigned milestones.\n"
            text += (
                "Use remaining steps to SCORE MORE POINTS:\n"
                "- Explore new areas you haven't visited\n"
                "- Collect treasures and interact with new objects/NPCs\n"
                "- Do NOT revisit already-explored areas\n"
            )
            return text

        hl = hl_steps[cur]
        step_name = hl.get('step', '')
        key_action = hl.get('key_action', '')

        text = f"\n\nCURRENT MILESTONE ({cur + 1} of {len(hl_steps)}):\n"
        text += f"  {step_name}\n"
        if key_action:
            text += f"  Key action sequence: {key_action}\n"
            text += f"  Follow this sequence step by step.\n"

        text += (
            f"\nComplete this milestone, then report the updated milestone number "
            f"in your response. You will receive the next milestone after completion."
        )

        if cur > 0:
            text += f"\n\nCompleted so far:"
            for i in range(cur):
                text += f"\n  \u2713 {hl_steps[i].get('step', '')}"

        return text

    def _build_dag_prompt(self, hl_steps: List[Dict],
                          step_limit: int = 80, current_step: int = 0) -> str:
        """DAG mode: show available/completed/locked milestones."""
        completed = self.completed_milestones
        num = len(hl_steps)
        remaining_steps = max(step_limit - current_step, 0)

        # Classify each milestone
        lines = []
        available_count = 0
        first_available_idx: Optional[int] = None
        has_explore = False
        for i in range(num):
            hl = hl_steps[i]
            step_name = hl.get('step', '')
            key_action = hl.get('key_action', '')
            deps_info = hl.get('deps_indices', [])

            # deps_indices is List[int] (or legacy List[dict])
            dep_indices = []
            for d in deps_info:
                if isinstance(d, dict):
                    dep_indices.append(d['idx'])
                else:
                    dep_indices.append(d)

            ka_text = f" (key: {key_action})" if key_action else ""
            if 'explore' in key_action.lower():
                has_explore = True

            if i in completed:
                lines.append(f"  \u2713 {i + 1}. {step_name}")
            elif all(d in completed for d in dep_indices):
                lines.append(f"  \u2192 {i + 1}. {step_name}{ka_text}")
                available_count += 1
                if first_available_idx is None:
                    first_available_idx = i
            else:
                needed = [str(d + 1) for d in dep_indices if d not in completed]
                lines.append(f"  \U0001f512 {i + 1}. {step_name} (needs: #{', #'.join(needed)})")

        # Track the current target for completion/auto-skip in agent.py
        self._last_target_idx = first_available_idx

        if available_count == 0 and len(completed) >= num:
            text = f"\n\nYou have completed all {num} assigned milestones.\n"
            text += (
                "Use remaining steps to SCORE MORE POINTS:\n"
                "- Explore new areas you haven't visited\n"
                "- Collect treasures and interact with new objects/NPCs\n"
                "- Do NOT revisit already-explored areas\n"
            )
            return text

        text = f"\n\nYOUR MILESTONES (step {current_step}/{step_limit}, {remaining_steps} steps remaining):\n"
        text += "\n".join(lines)
        text += (
            "\n\nPick the AVAILABLE milestone (\u2192) closest to your current location and pursue it."
            "\nWhen you complete the current milestone, set current_milestone_completed to true."
            "\n\nIf you have been trying the same milestone for many steps without progress, "
            "SKIP it and move to a different AVAILABLE milestone. Do not waste steps repeating "
            "the same failed actions \u2014 move on and try something else."
        )
        if has_explore:
            text += (
                "\n\nEXPLORE milestones (key action ends with '\u2192 explore'): after reaching "
                "the destination, spend a few steps trying the valid actions available there "
                "(take items, examine objects, try exits) before marking it complete and moving on."
            )
        return text

    def build_response_schema(self) -> Dict[str, Any]:
        if self._is_dag:
            properties = {
                "progress_analysis": {"type": "string"},
                "current_milestone_completed": {"type": "boolean"},
                "next_objective": {"type": "string"},
                "reasoning": {"type": "string"},
                "action": {"type": "string"},
            }
            required = ["progress_analysis", "current_milestone_completed", "next_objective", "reasoning", "action"]
        else:
            properties = {
                "progress_analysis": {"type": "string"},
                "current_milestone": {"type": "integer"},
                "next_objective": {"type": "string"},
                "reasoning": {"type": "string"},
                "action": {"type": "string"},
            }
            required = ["progress_analysis", "current_milestone", "next_objective", "reasoning", "action"]

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
