"""Full Plan guidance mode (Mode A).

Shows all milestones with progress markers (completed/current/pending).

DAG mode: path already linearized by select_path(). Shows full milestone
list, tracks completion via completed_milestones set, LLM reports
current_milestone_completed (bool).

Linear mode: uses milestone_idx integer, LLM reports current_milestone (int).
"""
from typing import Any, Dict, List, Optional, Set

from .base import GuidanceMode


class FullPlanGuidance(GuidanceMode):
    """Show all milestones with completion status markers."""

    def __init__(self):
        self.completed_milestones: Set[int] = set()
        self._is_dag: bool = False
        self._last_target_idx: Optional[int] = None

    def reset(self):
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
        steps = strategy.get('steps', [])
        self._is_dag = strategy.get('is_dag', False)

        if hl_steps:
            if self._is_dag:
                return self._build_dag_prompt(hl_steps, step_limit, current_step)
            else:
                return self._build_linear_prompt(hl_steps, milestone_idx, step_limit)

        elif steps:
            shown_steps = steps[:step_limit]
            steps_text = "\n".join(f"  {s}" for s in shown_steps)
            remaining = len(steps) - len(shown_steps)
            text = (
                f"\n\n**CURRENT STRATEGY**: {strategy['description']}\n"
                f"PLAN (first {len(shown_steps)} steps for this episode):\n{steps_text}\n"
            )
            if remaining > 0:
                text += f"  ... ({remaining} more steps for future episodes)\n"
            text += (
                f"Follow these steps in order. Choose the action that corresponds "
                f"to the next incomplete step in the plan."
            )
            return text

        else:
            return (
                f"\n\n**CURRENT STRATEGY**: {strategy['description']}\n"
                f"Your current goal is to pursue the above strategy. "
                f"Choose actions that directly work toward this goal."
            )

    def _build_linear_prompt(self, hl_steps: List[Dict], milestone_idx: int,
                              step_limit: int) -> str:
        """Original linear mode: use milestone_idx integer."""
        cur = milestone_idx
        if cur >= len(hl_steps):
            text = "\n\nALL MILESTONES COMPLETED:\n"
            for i, hl in enumerate(hl_steps):
                text += f"  \u2713 Milestone {i+1}: {hl.get('step', '')}\n"
            text += (
                "\nYou have completed your assigned strategy. Use remaining steps to SCORE MORE POINTS:\n"
                "- Explore new areas you haven't visited yet\n"
                "- Collect treasures and bring them to a safe storage location\n"
                "- Solve puzzles, interact with objects and NPCs you haven't tried\n"
                "- Do NOT revisit areas you already explored — push into unknown territory\n"
            )
            return text

        text = "\n\nYOUR MILESTONES FOR THIS EPISODE:\n"
        for i, hl in enumerate(hl_steps):
            step_name = hl.get('step', '')
            key_action = hl.get('key_action', '')
            if i < cur:
                text += f"  \u2713 Milestone {i+1}: {step_name}\n"
            elif i == cur:
                text += f"  \u2192 Milestone {i+1}: {step_name}  (CURRENT)\n"
                if key_action:
                    text += f"    \U0001f4a1 Key action sequence: {key_action}\n"
                    text += f"    \u26a0\ufe0f Follow this sequence step by step. Do NOT deviate to manage inventory or explore — the sequence already includes all needed preparation.\n"
            else:
                text += f"  \u25cb Milestone {i+1}: {step_name}\n"
        text += (
            f"\nFocus on completing the CURRENT milestone. "
            f"Figure out the specific actions needed to achieve each milestone goal. "
            f"Once done, report the updated milestone number in your response."
        )
        return text

    def _build_dag_prompt(self, hl_steps: List[Dict], step_limit: int,
                           current_step: int) -> str:
        """DAG mode: show full plan, track completion via completed_milestones set."""
        completed = self.completed_milestones
        num = len(hl_steps)

        # Find first uncompleted milestone as CURRENT
        current_idx = None
        for i in range(num):
            if i not in completed:
                current_idx = i
                break

        self._last_target_idx = current_idx

        if current_idx is None:
            text = "\n\nALL MILESTONES COMPLETED:\n"
            for i, hl in enumerate(hl_steps):
                text += f"  \u2713 Milestone {i+1}: {hl.get('step', '')}\n"
            text += "\nAll milestones done! Explore new areas for bonus points.\n"
            return text

        text = f"\n\nYOUR MILESTONES ({len(completed)}/{num} completed):\n"
        for i, hl in enumerate(hl_steps):
            step_name = hl.get('step', '')
            key_action = hl.get('key_action', '')
            if i in completed:
                text += f"  \u2713 Milestone {i+1}: {step_name}\n"
            elif i == current_idx:
                text += f"  \u2192 Milestone {i+1}: {step_name}  (CURRENT)\n"
                if key_action:
                    text += f"    \U0001f4a1 Key action sequence: {key_action}\n"
                    text += f"    \u26a0\ufe0f Follow this sequence step by step.\n"
            else:
                text += f"  \u25cb Milestone {i+1}: {step_name}\n"
                if key_action:
                    text += f"    Key actions: {key_action}\n"

        text += (
            f"\nFocus on the CURRENT milestone. Once complete, set "
            f"current_milestone_completed to true."
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
            required = ["progress_analysis", "current_milestone_completed",
                        "next_objective", "reasoning", "action"]
        else:
            properties = {
                "progress_analysis": {"type": "string"},
                "current_milestone": {"type": "integer"},
                "next_objective": {"type": "string"},
                "reasoning": {"type": "string"},
                "action": {"type": "string"},
            }
            required = ["progress_analysis", "current_milestone",
                        "next_objective", "reasoning", "action"]

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
