"""Information-Gain Mining space evolution (Module 3, Config C).

Instead of looking for local decision points (like DPM), this module takes
a global view: it maintains a World Knowledge table of what the agent knows,
identifies knowledge gaps, and asks the LLM to propose the most informative
exploration targets — actions that would teach us something fundamentally new.
"""
from typing import Any, Dict, List

from ..openai_helpers_proxy import chat_completion, parse_json_response


class InfoGainMining:
    """Propose exploration targets based on information gain over world knowledge."""

    def reflect(self, episode_summaries: List[Dict[str, Any]],
                strategy_space,
                llm_model: str,
                args) -> List[Dict[str, Any]]:
        n = len(episode_summaries)
        print(f"[InfoGainMining] Analyzing {n} episodes for information gaps...")

        # Python games use dedicated prompt
        if getattr(args, 'game_name', '').startswith('catnip'):
            return self._reflect_python_game(episode_summaries, strategy_space, llm_model, args)

        episodes_str = self._format_episodes(episode_summaries)
        existing_str = strategy_space.format_for_reflection()
        is_dag = getattr(strategy_space, 'space_type', 'tree') == 'dag'

        # Update action coverage from recent episodes
        if hasattr(strategy_space, 'update_action_coverage'):
            for ep in episode_summaries:
                gh = ep.get('game_history', [])
                if gh:
                    strategy_space.update_action_coverage(gh)

        # Build context
        abandoned_str = ""
        if hasattr(strategy_space, 'get_abandoned_summary'):
            abandoned_str = strategy_space.get_abandoned_summary()

        world_knowledge_str = ""
        if hasattr(strategy_space, 'get_world_knowledge'):
            world_knowledge_str = strategy_space.get_world_knowledge()

        room_list = []
        if is_dag and hasattr(strategy_space, 'get_room_list'):
            room_list = strategy_space.get_room_list()

        sys_prompt, user_prompt = self._build_prompts(
            existing_str, episodes_str,
            abandoned_str=abandoned_str,
            world_knowledge_str=world_knowledge_str,
            room_list=room_list,
            is_dag=is_dag)

        max_retries = 5
        for attempt in range(max_retries):
            try:
                raw = chat_completion(
                    model=llm_model,
                    sys_prompt=sys_prompt,
                    prompt=user_prompt,
                    max_tokens=6000,
                    temperature=0.4,
                )

                if not raw:
                    print(f"[InfoGainMining] LLM call failed (attempt {attempt + 1}/{max_retries})")
                    continue

                result = parse_json_response(raw)
                if not result:
                    print(f"[InfoGainMining] JSON parse failed (attempt {attempt + 1}/{max_retries})")
                    continue

                gaps = result.get('knowledge_gaps', [])
                operations = result.get('proposed_operations', [])

                self._print_gaps(gaps)
                print(f"[InfoGainMining] Found {len(gaps)} knowledge gaps, "
                      f"proposed {len(operations)} operations")

                return {
                    'operations': operations,
                    'recurring_mistakes': [],
                    'knowledge_gaps': gaps,
                }

            except Exception as e:
                print(f"[InfoGainMining] Error (attempt {attempt + 1}/{max_retries}): {e}")

        print(f"[InfoGainMining] Failed after {max_retries} attempts")
        return {'operations': [], 'recurring_mistakes': []}

    # ------------------------------------------------------------------ #
    #  Prompt builder
    # ------------------------------------------------------------------ #

    def _build_prompts(self, existing_str, episodes_str,
                       abandoned_str="", world_knowledge_str="",
                       room_list=None, is_dag=True):

        rooms_section = ""
        if room_list:
            rooms_lines = [f"  {rid}: {name}" for rid, name in room_list]
            rooms_section = "\nKNOWN ROOMS (use room_id for the \"destination\" field):\n" + "\n".join(rooms_lines)

        abandoned_section = ""
        if abandoned_str:
            abandoned_section = f"""
ABANDONED NODES (tried ≥3 times, never earned reward — do NOT re-propose):
{abandoned_str}"""

        knowledge_section = ""
        if world_knowledge_str:
            knowledge_section = f"""
WORLD KNOWLEDGE (what we know so far — look for gaps):
{world_knowledge_str}"""

        if is_dag:
            dep_section = """
DAG SEMANTICS:
- Each node has a "deps" list = prerequisites whose OUTCOME (item/key/state change) is required
- deps=[] means the milestone is independent (no prerequisites)
- Only add deps for causal requirements, NOT for geographic convenience or sequencing"""
            op_format = """{
            "op": "add_child",
            "deps": ["node_id", ...],
            "milestone": "milestone text",
            "key_action": "",
            "destination": <room_id or -1>,
            "estimated_milestone_reward": 0,
            "reason": "why this would fill a knowledge gap"
        }"""
        else:
            dep_section = ""
            op_format = """{
            "op": "add_child",
            "parent_id": "node_id to attach to (or 'root')",
            "milestone": "milestone text",
            "key_action": "",
            "estimated_milestone_reward": 0,
            "reason": "why this would fill a knowledge gap"
        }"""

        sys_prompt = f"""You are an expert analyzer for text adventure games.

Your goal: examine what the agent knows about the game world, identify the biggest knowledge gaps, and propose exploration targets that would yield the most new information.

Think about:
- Rooms with unexplored exits or untried actions
- Items that were seen but never picked up or used
- Game mechanics that are only partially understood
- Entire areas of the game world that have never been reached
- Objects or features mentioned in game text but never investigated
{dep_section}

CONSTRAINTS:
- At most 3 proposed operations per reflection
- Every proposed milestone must be a concrete, executable game command — not an abstract goal
- Do not propose actions that match ABANDONED nodes
- Do not duplicate nodes already in the existing space
- For navigation milestones, prefix directions with [from: Room Name] (e.g. "[from: Kitchen] up → explore")
- Leave key_action empty ("") for speculative nodes
- Leave prune_suggestions as an empty list []

Respond with valid JSON only:
{{
    "knowledge_gaps": [
        {{
            "description": "what we don't know",
            "evidence": "what hints at this gap (game text, untried actions, frontier rooms)",
            "potential_value": "what we might discover"
        }}
    ],
    "proposed_operations": [
        {op_format}
    ],
    "prune_suggestions": []
}}"""

        user_prompt = f"""CURRENT MILESTONE {"DAG" if is_dag else "TREE"}:
{existing_str}
{rooms_section}
{abandoned_section}
{knowledge_section}

RECENT EPISODES:
{episodes_str}

Based on the world knowledge and recent episodes, identify the biggest gaps in our understanding of the game world. Then propose the exploration targets most likely to fill those gaps.

Respond with valid JSON only."""
        return sys_prompt, user_prompt

    def _format_episodes(self, episode_summaries: List[Dict[str, Any]]) -> str:
        from .tree_update import TreeUpdate
        keep = {'milestones achieved', 'milestones not', 'penalties', 'new discoveries',
                'failed attempts', 'untried valid', 'unexplored observations'}
        lines = []
        for ep in episode_summaries:
            ep_num = ep.get('episode', '?')
            score = ep.get('score', 0)
            summary = ep.get('summary', '')
            strategy_desc = ep.get('strategy_description')

            header = f"\nEpisode {ep_num} (Score: {score})"
            if strategy_desc:
                header += f' — Strategy: "{strategy_desc}"'
            lines.append(header + ":")
            trimmed = TreeUpdate._trim_summary(summary, keep)
            lines.append(f"  {trimmed}")

        return "\n".join(lines)

    def _print_gaps(self, gaps: List[Dict]):
        if not gaps:
            return
        print("\n" + "=" * 60)
        print("  Knowledge Gaps Found")
        print("=" * 60)
        for gap in gaps:
            desc = gap.get('description', '?')
            evidence = gap.get('evidence', '?')
            print(f"\n  Gap: {desc}")
            print(f"    Evidence: {evidence}")
        print()

    # ------------------------------------------------------------------ #
    #  Python game reflection — game-agnostic knowledge gap discovery
    # ------------------------------------------------------------------ #

    def _reflect_python_game(self, episode_summaries, strategy_space, llm_model, args):
        """IGM for python games: find knowledge gaps from raw game data.

        Game-agnostic — discovers what's untried by comparing valid actions vs
        chosen actions, and looking for unexplored combinations across episodes.
        """
        # Format episodes with raw game history (same as free_reflection)
        lines = []
        for ep in episode_summaries:
            ep_num = ep.get('episode', '?')
            score = ep.get('score', 0)
            summary = ep.get('summary', '')
            strategy_desc = ep.get('strategy_description')
            header = f"\nEpisode {ep_num} (Score: {score})"
            if strategy_desc:
                header += f' — Strategy: "{strategy_desc}"'
            else:
                header += " — Free exploration"
            lines.append(header + ":")
            lines.append(f"  {summary}")
        episodes_str = "\n".join(lines)

        existing_str = strategy_space.format_for_reflection()

        json_schema = """{
    "knowledge_gaps": [
        {
            "category": "untried_action|untried_choice|unknown_mechanic|untested_combination",
            "description": "what we don't know",
            "evidence": "what in the game data suggests this gap exists",
            "potential_value": "what we might learn by exploring this"
        }
    ],
    "proposed_operations": [
        {
            "op": "add_branch",
            "description": "a high-level strategy designed to EXPLORE this knowledge gap (1-3 sentences)",
            "reason": "which knowledge gap this addresses"
        }
    ]
}"""

        sys_prompt = f"""You are an expert at analyzing game data to find knowledge gaps — things the agent has NOT yet tried or understood.

Your goal is to find what's UNKNOWN and propose exploration experiments.

Look for these types of knowledge gaps:

1. **Untried actions**: Valid actions that appeared in the action list but were NEVER chosen across all episodes. The game offers these actions for a reason — they might be powerful.

2. **Untried choices**: Reward/selection options that were always skipped (e.g., always chose option A, never tried B or C). We don't know what B or C give.

3. **Unknown mechanics**: Game behaviors that are only partially understood. For example, if the agent sometimes gets a warning but we don't know exactly what triggers it or what happens if we ignore it.

4. **Untested combinations**: Combinations of choices that were never tried together. For example, if we always use item X on target Y, but never tried item X on target Z.

IMPORTANT:
- Compare VALID ACTIONS vs CHOSEN ACTIONS across all episodes to find what was never tried
- Look for patterns in rewards — is there a choice the agent always makes that might not be optimal?
- Proposed strategies should be designed to LEARN something new, not just maximize score
- At most 3 proposed operations per reflection

You MUST respond with valid JSON only. Use this exact schema:
{json_schema}"""

        user_prompt = f"""Analyze these episodes to find knowledge gaps — what has the agent NOT tried yet?

CURRENT STRATEGY LIST:
{existing_str}

RECENT EPISODES:
{episodes_str}

Step 1 — Inventory what we know:
List what actions, choices, and combinations the agent HAS tried across all episodes.

Step 2 — Find what's missing:
Compare against the valid actions lists. What options appeared but were NEVER chosen? What choices were always the same?

Step 3 — Propose exploration strategies:
For each significant gap, propose a HIGH-LEVEL strategy (not a fixed action sequence) designed to explore that gap. The strategy should be different from existing strategies.

Respond with valid JSON only."""

        max_retries = 5
        for attempt in range(max_retries):
            try:
                raw = chat_completion(
                    model=llm_model,
                    sys_prompt=sys_prompt,
                    prompt=user_prompt,
                    max_tokens=6000,
                    temperature=0.4,
                )
                if not raw:
                    print(f"[InfoGainMining] LLM call failed (attempt {attempt + 1}/{max_retries})")
                    continue

                result = parse_json_response(raw)
                if not result:
                    print(f"[InfoGainMining] JSON parse failed (attempt {attempt + 1}/{max_retries})")
                    continue

                gaps = result.get('knowledge_gaps', [])
                operations = result.get('proposed_operations', [])

                self._print_gaps(gaps)
                print(f"[InfoGainMining] Found {len(gaps)} knowledge gaps, "
                      f"proposed {len(operations)} operations")

                return {
                    'operations': operations,
                    'recurring_mistakes': [],
                    'knowledge_gaps': gaps,
                }

            except Exception as e:
                print(f"[InfoGainMining] Error (attempt {attempt + 1}/{max_retries}): {e}")

        print(f"[InfoGainMining] Failed after {max_retries} attempts")
        return {'operations': [], 'recurring_mistakes': []}
