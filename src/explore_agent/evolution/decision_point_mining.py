"""Decision Point Mining space evolution (Module 3, Config B).

Analyzes episode trajectories to find decision points — moments where
the agent chose one action but other promising actions were available.
These untaken forks become new branches in the strategy space.
"""
from typing import Any, Dict, List

from ..openai_helpers_proxy import chat_completion, parse_json_response


class DecisionPointMining:
    """Extract decision points from episodes and propose new strategy branches."""

    def reflect(self, episode_summaries: List[Dict[str, Any]],
                strategy_space,
                llm_model: str,
                args) -> List[Dict[str, Any]]:
        n = len(episode_summaries)
        print(f"[DecisionPointMining] Analyzing {n} episodes for decision points...")

        episodes_str = self._format_episodes(episode_summaries)
        existing_str = strategy_space.format_for_reflection()
        step_limit = getattr(args, 'env_step_limit', 50)
        is_dag = getattr(strategy_space, 'space_type', 'tree') == 'dag'
        # Games without valid_actions (e.g. balances) need DPM to infer
        # interactions from room descriptions instead of relying on action lists.
        game_name = getattr(args, 'game_name', '')
        has_valid_actions = game_name != 'balances'

        if is_dag:
            sys_prompt, user_prompt = self._build_dag_prompts(
                existing_str, episodes_str, step_limit,
                has_valid_actions=has_valid_actions)
        else:
            sys_prompt, user_prompt = self._build_tree_prompts(
                existing_str, episodes_str, step_limit,
                has_valid_actions=has_valid_actions)

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
                    print(f"[DecisionPointMining] LLM call failed (attempt {attempt + 1}/{max_retries})")
                    continue

                result = parse_json_response(raw)
                if not result:
                    print(f"[DecisionPointMining] JSON parse failed (attempt {attempt + 1}/{max_retries})")
                    continue

                decision_points = result.get('decision_points', [])
                operations = result.get('proposed_operations', [])
                prune_ops = result.get('prune_suggestions', [])

                # Print decision points
                self._print_decision_points(decision_points)

                # Combine operations
                all_ops = operations + prune_ops

                print(f"[DecisionPointMining] Found {len(decision_points)} decision points, "
                      f"proposed {len(all_ops)} operations")

                return {
                    'operations': all_ops,
                    'recurring_mistakes': [],
                    'decision_points': decision_points,
                }

            except Exception as e:
                print(f"[DecisionPointMining] Error (attempt {attempt + 1}/{max_retries}): {e}")

        print(f"[DecisionPointMining] Failed after {max_retries} attempts")
        return {'operations': [], 'recurring_mistakes': []}

    # ------------------------------------------------------------------ #
    #  Prompt builders
    # ------------------------------------------------------------------ #

    def _build_tree_prompts(self, existing_str, episodes_str, step_limit,
                            has_valid_actions=True):
        if has_valid_actions:
            focus_section = """Focus on TWO types of missed opportunities:

A. DECISION POINTS — specific steps where the agent chose one action but other options existed:
- The agent was at a location with multiple exits/interactions
- An item was visible but not taken
- An NPC was present but not interacted with
- A puzzle element was noticed but not explored
- The valid actions list contained promising options that were ignored

B. KNOWLEDGE GAPS — things the agent has NEVER tried across ALL episodes:
- Valid actions that appeared in action lists but were NEVER chosen in ANY episode
- Room descriptions mentioning areas, objects, or features that were never visited or interacted with
- Directions/exits available at visited rooms that were never taken"""
            evidence_rule = "- The action in a proposed milestone MUST have appeared in at least one episode's valid actions list or room description. Do NOT invent actions that were never seen in the game."
        else:
            focus_section = """Focus on TWO types of missed opportunities:

A. DECISION POINTS — specific steps where the agent chose one action but other options existed:
- The agent was at a location with multiple exits/interactions
- An item was visible but not taken
- An NPC was present but not interacted with
- A puzzle element was noticed but not explored

B. KNOWLEDGE GAPS — things the agent has NEVER tried across ALL episodes:
- Room descriptions mentioning areas, objects, or features that were never visited or interacted with
- Standard text adventure interactions never attempted (examine, take, open, read, push, press, pull, look under, etc.) with objects mentioned in room text
- Directions/exits available at visited rooms that were never taken"""
            evidence_rule = "- The action should be grounded in evidence from the game text: it appeared in a summary, OR can be reasonably inferred from room descriptions (e.g. if a room mentions \"furniture\", then \"examine furniture\" is a valid proposal; if it mentions \"control panel\", then \"push button\" or \"press button\" is valid). Use standard text adventure verbs (examine, take, open, read, push, press, pull, look under, etc.) with objects mentioned in the game."

        sys_prompt = f"""You are an expert analyzer for text adventure games.
Your task is to identify DECISION POINTS and KNOWLEDGE GAPS in gameplay trajectories — actions
the agent never tried that could unlock new game content.

A decision point is:
1. A specific step where the agent had meaningful choices
2. The agent chose action A, but action B (or C) was also available and potentially rewarding
3. The unchosen action leads to unexplored game content

{focus_section}

CRITICAL CONSTRAINTS on proposed operations:
- Every proposed milestone MUST be a CONCRETE, EXECUTABLE action (e.g. "look under stairs", "give herring to painting"), NOT an abstract goal (e.g. "Discover eastern corridor", "Understand printer software")
{evidence_rule}
- Do NOT propose branches for actions that were already tried in some episode without earning reward — they are likely dead ends.
- Limit yourself to at most 6 proposed operations per reflection. Quality over quantity.
- For EXPLORATION milestones (navigating to a new, unexplored area), set key_action to the navigation command followed by "→ explore" (e.g. "[from: Kitchen] up → explore", "[from: Cellar] south → explore"). Direction words MUST have [from: Room Name] prefix so the agent knows where to execute the direction.

You MUST respond with valid JSON only:
{{
    "decision_points": [
        {{
            "episode": <int>,
            "step": <int>,
            "location": "where this happened",
            "chosen_action": "what the agent did",
            "unchosen_alternatives": [
                {{
                    "action": "the action not taken",
                    "potential": "why this might be worth exploring",
                    "leads_to": "best guess of what this might unlock"
                }}
            ]
        }}
    ],
    "proposed_operations": [
        {{
            "op": "add_child|add_branch",
            "parent_id": "node_id to attach to (or 'root')",
            "milestone": "milestone text (for add_child)",
            "key_action": "",
            "estimated_milestone_reward": 0,
            "milestones": [],
            "reason": "derived from decision point at episode X, step Y"
        }}
    ],
    "prune_suggestions": [
        {{
            "op": "prune",
            "node_id": "node to prune",
            "reason": "why this path should be abandoned"
        }}
    ]
}}"""

        user_prompt = f"""Analyze these episode trajectories to find decision points.

CURRENT STRATEGY SPACE:
{existing_str}

EPISODE TRAJECTORIES:
{episodes_str}

For each episode, scan the raw action sequence and identify moments where:
1. The agent was at a choice point with multiple valid options
2. The agent chose one action but other actions could have led to unexplored content
3. The unchosen action has NOT been tried in any other episode

Then propose tree modifications to encode these unchosen paths as new branches.

IMPORTANT:
- Only propose branches for genuinely unexplored alternatives, not minor variations
- Each proposed branch should represent a meaningfully different strategy direction
- Leave key_action empty ("") for speculative branches — only use observed action sequences
- Check the existing strategy space to avoid duplicating already-known paths
- Do NOT use the prune operation. Leave prune_suggestions as an empty list [].
- CONCRETE ACTIONS ONLY: Every proposed milestone must be a specific game command (e.g. "give herring to painting", "look under stairs"), NOT an abstract description (e.g. "Explore the corridor", "Discover what's behind the picture")
{evidence_rule}

Each episode has {step_limit} steps. Focus on early decision points (first 1/3 of the episode)
since those have the most impact on overall strategy direction.

Respond with valid JSON only."""
        return sys_prompt, user_prompt

    def _build_dag_prompts(self, existing_str, episodes_str, step_limit,
                           has_valid_actions=True):
        if has_valid_actions:
            focus_section = """Focus on TWO types of missed opportunities:

A. DECISION POINTS — specific steps where the agent chose one action but other options existed:
- The agent was at a location with multiple exits/interactions
- An item was visible but not taken
- An NPC was present but not interacted with
- A puzzle element was noticed but not explored
- The valid actions list contained promising options that were ignored

B. KNOWLEDGE GAPS — things the agent has NEVER tried across ALL episodes:
- Valid actions that appeared in action lists but were NEVER chosen in ANY episode (compare valid actions vs chosen actions across all episodes)
- Room descriptions mentioning areas, objects, or features that were never visited or interacted with (e.g. "mining operations lie to the south" but agent never went south; "control panel" but agent never pushed/pressed it)
- Directions/exits available at visited rooms that were never taken"""
            evidence_rule = "- The action MUST have appeared in at least one episode's valid actions list or room description. Do NOT invent actions."
        else:
            focus_section = """Focus on TWO types of missed opportunities:

A. DECISION POINTS — specific steps where the agent chose one action but other options existed:
- The agent was at a location with multiple exits/interactions
- An item was visible but not taken
- An NPC was present but not interacted with
- A puzzle element was noticed but not explored
- The room description mentions objects, furniture, or features the agent never examined or interacted with

B. KNOWLEDGE GAPS — things the agent has NEVER tried across ALL episodes:
- Room descriptions mentioning areas, objects, or features that were never visited or interacted with (e.g. "mining operations lie to the south" but agent never went south; "control panel" but agent never pushed/pressed it)
- Standard text adventure interactions never attempted (examine, take, open, read, push, pull, press, look under, etc.) with objects mentioned in room text
- Directions/exits available at visited rooms that were never taken"""
            evidence_rule = '- The action should be grounded in evidence from the game text: it appeared in a summary, OR can be reasonably inferred from room descriptions (e.g. if a room mentions "furniture", then "examine furniture" is a valid proposal; if it mentions "control panel", then "push button" or "press button" is valid). Use standard text adventure verbs (examine, take, open, read, push, press, pull, look under, etc.) with objects mentioned in the game.'

        sys_prompt = f"""You are an expert analyzer for text adventure games.
Your task is to identify DECISION POINTS and KNOWLEDGE GAPS in gameplay trajectories — actions
the agent never tried that could unlock new game content.

The agent's knowledge is organized as a MILESTONE DAG (directed acyclic graph).

IMPORTANT — DAG uses "deps", NOT "parent_id":
- In a TREE, each node has a single "parent_id" meaning "this milestone comes after the parent in a linear sequence". The parent relationship is about ORDER.
- In a DAG, each node has a "deps" list meaning "this milestone REQUIRES these specific prerequisites to be completed first".
- deps=[] means the milestone is INDEPENDENT — it can be done from the game start with no prerequisites.
- deps=["node_005"] means this milestone requires the OUTCOME of node_005 (item, key, tool, state change).
- Two independent milestones MUST both have deps=[] — do NOT make one depend on the other just because they happened in sequence.

A decision point is:
1. A specific step where the agent had meaningful choices
2. The agent chose action A, but action B (or C) was also available and potentially rewarding
3. The unchosen action leads to unexplored game content

{focus_section}

EXECUTION LOG: Some nodes include "exec[epN]" lines showing what happened when the agent tried the key_action in-game. Use this to identify broken key_actions:
- If a command is consistently rejected by the game (e.g. "that can't contain things", "won't help"), it should be REMOVED from the key_action.
- If a command never earns reward across many attempts, consider whether it's actually necessary.
- Propose prune for nodes whose key_action is entirely rejected by the game engine.

CRITICAL CONSTRAINTS on proposed operations:
- Every proposed milestone MUST be a CONCRETE, EXECUTABLE action (e.g. "look under stairs", "give herring to painting"), NOT an abstract goal
{evidence_rule}
- Do NOT propose branches for actions already tried without earning reward.
- Limit yourself to at most 6 proposed operations per reflection. Quality over quantity.
- For EXPLORATION milestones (navigating to a new, unexplored area), set key_action to the navigation command followed by "→ explore" (e.g. "[from: Kitchen] up → explore", "[from: Cellar] south → explore"). Direction words MUST have [from: Room Name] prefix.
- key_action FORMAT: Direction words (north/south/east/west/up/down/ne/nw/se/sw) MUST be prefixed with [from: Room Name] to indicate WHERE to execute that direction. Non-direction actions don't need annotation. Example: "[from: Troll Room] east, [from: North of House] north, echo, take bar". This is CRITICAL — without [from:] on directions, the agent gets lost.

You MUST respond with valid JSON only:
{{
    "decision_points": [
        {{
            "episode": <int>,
            "step": <int>,
            "location": "where this happened",
            "chosen_action": "what the agent did",
            "unchosen_alternatives": [
                {{
                    "action": "the action not taken",
                    "potential": "why this might be worth exploring",
                    "leads_to": "best guess of what this might unlock"
                }}
            ]
        }}
    ],
    "proposed_operations": [
        {{
            "op": "add_child",
            "deps": ["node_id", ...],
            "milestone": "milestone text",
            "key_action": "",
            "destination": <room_id where agent must BE to start key_action, or -1 if unknown>,
            "estimated_milestone_reward": 0,
            "reason": "derived from decision point at episode X, step Y"
        }}
    ],
    "prune_suggestions": [
        {{
            "op": "prune",
            "node_id": "node to prune",
            "reason": "why this path should be abandoned"
        }}
    ]
}}

For add_child: "deps" is a list of node_id strings (e.g. ["node_001"]). Use [] for independent milestones.
  - Only list nodes whose OUTCOME is required (item, key, tool, state change)."""

        user_prompt = f"""Analyze these episode trajectories to find decision points.

CURRENT MILESTONE DAG:
{existing_str}

EPISODE TRAJECTORIES:
{episodes_str}

Perform TWO analyses:

ANALYSIS 1 — Decision Points: For each episode, find moments where:
1. The agent was at a choice point with multiple valid options
2. The agent chose one action but other actions could have led to unexplored content
3. The unchosen action has NOT been tried in any other episode

ANALYSIS 2 — Knowledge Gaps: Across ALL episodes, find:
1. Valid actions that appeared in action lists but were NEVER chosen in ANY episode
2. Room descriptions mentioning areas/objects/features the agent never visited or interacted with
3. Directions or exits at visited rooms that were never taken

Then propose DAG modifications to encode these missed opportunities as new nodes.

CRITICAL — DEPENDENCY RULES:
- deps=[] for milestones with no prerequisites
- deps=["node_001"] means this milestone REQUIRES node_001's outcome (item/key/tool/state)
- Only add deps when the milestone is IMPOSSIBLE without the prerequisite
- Do NOT add deps for geographic convenience or routing — deps are for CAUSAL requirements only
- Do NOT chain independent milestones. Two independent actions MUST both have deps=[]
- LEARN FROM FAILURES: If a milestone failed due to missing prerequisite (e.g. "no light source"), add that prerequisite as a dep.

IMPORTANT:
- Only propose nodes for genuinely unexplored alternatives, not minor variations
- Each proposed node should represent a meaningfully different strategy direction
- Leave key_action empty ("") for speculative nodes — only use observed action sequences
- DEDUP CHECK: Before proposing a new node, check if the existing DAG already has a node for the same location or action. If so, do NOT create a duplicate.
- Do NOT use the prune operation. Leave prune_suggestions as an empty list [].
- CONCRETE ACTIONS ONLY: Every milestone must be a specific game command, NOT an abstract description
{evidence_rule}

Each episode has {step_limit} steps. Focus on early decision points (first 1/3 of the episode).

For destination: use the room_id from KNOWN ROOMS where the agent needs to BE to start the key_action. For exploration milestones (e.g. "Kitchen + Go Up to Attic"), use the departure room (Kitchen), not the unknown target. Only use -1 if the starting room is truly unknown.

Respond with valid JSON only."""
        return sys_prompt, user_prompt

    def _format_episodes(self, episode_summaries: List[Dict[str, Any]]) -> str:
        # DPM needs everything except raw action sequence (already removed by not appending it)
        from .tree_update import TreeUpdate
        keep = {'milestones achieved', 'milestones not', 'penalties', 'new discoveries', 'failed attempts', 'untried valid', 'unexplored observations'}
        lines = []
        for ep in episode_summaries:
            ep_num = ep.get('episode', '?')
            score = ep.get('score', 0)
            initial_score = ep.get('initial_score', 0)
            summary = ep.get('summary', '')
            strategy_desc = ep.get('strategy_description')

            if initial_score > 0:
                earned = score - initial_score
                header = f"\nEpisode {ep_num} (Score: {score}, initial={initial_score} earned={earned})"
            else:
                header = f"\nEpisode {ep_num} (Score: {score})"
            if strategy_desc:
                header += f' — Strategy: "{strategy_desc}"'
            lines.append(header + ":")
            trimmed = TreeUpdate._trim_summary(summary, keep)
            lines.append(f"  {trimmed}")

        return "\n".join(lines)

    def _print_decision_points(self, points: List[Dict]):
        if not points:
            return
        print("\n" + "=" * 60)
        print("  Decision Points Found")
        print("=" * 60)
        for dp in points:
            ep = dp.get('episode', '?')
            step = dp.get('step', '?')
            location = dp.get('location', '?')
            chosen = dp.get('chosen_action', '?')
            print(f"\n  Episode {ep}, Step {step} at [{location}]:")
            print(f"    Chose: {chosen}")
            for alt in dp.get('unchosen_alternatives', []):
                print(f"    Alternative: {alt.get('action', '?')}")
                print(f"      Potential: {alt.get('potential', '?')}")
        print()
