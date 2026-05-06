"""Tree Update — encode observed facts into the milestone tree.

This step runs BEFORE space evolution (free_reflection / decision_point_mining).
It looks at episode summaries and the current tree, then proposes factual
tree operations: adding milestones that were achieved, correcting key_actions,
pruning consistently failing paths.

This is conservative and evidence-based — it only encodes what actually happened,
not speculative future branches.
"""
from typing import Any, Dict, List

from ..openai_helpers_proxy import chat_completion, parse_json_response


class TreeUpdate:
    """Update the milestone tree based on observed episode data."""

    def update(self, episode_summaries: List[Dict[str, Any]],
               strategy_space,
               llm_model: str,
               args) -> Dict[str, Any]:
        """Analyze episodes and return factual tree operations."""
        n = len(episode_summaries)
        print(f"[TreeUpdate] Encoding {n} episodes into tree...")

        episodes_str = self._format_episodes(episode_summaries)
        existing_str = strategy_space.format_for_reflection()
        step_limit = getattr(args, 'env_step_limit', 50)
        is_dag = getattr(strategy_space, 'space_type', 'tree') == 'dag'
        is_python_game = getattr(args, 'game_name', '').startswith('catnip')

        if is_python_game:
            sys_prompt, user_prompt = self._build_python_game_prompts(
                existing_str, episodes_str, step_limit)
        elif is_dag:
            sys_prompt, user_prompt = self._build_dag_prompts(
                existing_str, episodes_str, step_limit)
        else:
            sys_prompt, user_prompt = self._build_tree_prompts(
                existing_str, episodes_str, step_limit)

        max_retries = 5
        for attempt in range(max_retries):
            try:
                raw = chat_completion(
                    model=llm_model,
                    sys_prompt=sys_prompt,
                    prompt=user_prompt,
                    max_tokens=6000,
                    temperature=0.3,
                )

                if not raw:
                    print(f"[TreeUpdate] LLM call failed (attempt {attempt + 1}/{max_retries})")
                    continue

                result = parse_json_response(raw)
                if not result:
                    print(f"[TreeUpdate] JSON parse failed (attempt {attempt + 1}/{max_retries})")
                    continue

                analysis = result.get('analysis', [])
                operations = result.get('operations', [])

                self._print_analysis(analysis)
                self._print_operations(operations)

                print(f"[TreeUpdate] Proposed {len(operations)} operations")

                return {
                    'operations': operations,
                    'analysis': analysis,
                }

            except Exception as e:
                print(f"[TreeUpdate] Error (attempt {attempt + 1}/{max_retries}): {e}")

        print(f"[TreeUpdate] Failed after {max_retries} attempts")
        return {'operations': []}

    # ------------------------------------------------------------------ #
    #  Prompt builders
    # ------------------------------------------------------------------ #

    def _build_tree_prompts(self, existing_str, episodes_str, step_limit):
        sys_prompt = f"""You are an expert at analyzing text adventure game results and maintaining a milestone tree.

The agent's knowledge is organized as a MILESTONE TREE:
- The root represents the game start
- Each node is a milestone (a goal like "Collect Sword in Living Room")
- Children of a node represent alternative next milestones after completing the parent
- A path from root to a leaf is one complete strategy

Your job is to UPDATE the tree based on OBSERVED FACTS from the episodes below.
You must ONLY encode things that actually happened — do NOT speculate or propose exploratory branches.

RULES:
1. Every score-changing action in the episodes should correspond to a tree node — NO EXCEPTIONS, including NEGATIVE rewards (penalties like -10 for dying). If a score change occurred and no matching node exists, you MUST add one. Penalty nodes help the agent learn to AVOID dangerous actions.
2. If an existing node's key_action is wrong or incomplete compared to what episodes actually did, fix it.
   IMPORTANT: Do NOT change a node's destination room via update_node. If the action happens at a DIFFERENT room, create a NEW node instead.
3. Do NOT prune any nodes. Nodes may become useful later even if they currently have low reward.
4. SEQUENTIAL ORDER = PARENT-CHILD RELATIONSHIP. If milestone B requires milestone A to be completed first, B must be a child of A, not a sibling.
5. Each milestone must cover exactly ONE thing. Milestone name MUST start with the exact room name where the action happens, followed by "+" and the action. Examples:
   - GOOD: "Loud Room + Solve Echo Puzzle", "Troll Room + Defeat Troll", "Living Room + Take Lantern", "Cellar + Go Down"
   - BAD: "Solve Echo Puzzle for Platinum Bar" (missing room name), "Defeat Troll" (missing room name), "Acquire Lantern and Sword" (missing room name)
6. key_action FORMAT: Write ALL actions starting FROM the destination room until the reward is earned, in order. This MUST include every direction/navigation step after arriving at the destination — do NOT skip intermediate rooms. The agent cannot infer missing steps.
   Do NOT include navigation TO the destination — that is computed automatically by BFS.
   DIRECTION ANNOTATION RULE: If key_action contains ANY direction word (north/south/east/west/up/down/ne/nw/se/sw), you MUST prefix it with [from: Room Name] so the agent knows WHERE to execute that direction. Every direction must have this prefix, no exceptions.
   - GOOD: "echo, take bar" (no direction words, no annotation needed)
   - GOOD: "hit troll with sword" (no direction words)
   - GOOD: "[from: Strange Passage] east, [from: Cyclops Room] up" (each direction annotated with departure room)
   - GOOD: "take lantern, [from: Living Room] down" (direction annotated, non-direction action is fine without)
   - BAD: "east, up" (direction words WITHOUT [from:] annotation — agent won't know where to go)
   - BAD: "[from: Living Room] north, down, echo, take bar" (includes navigation TO destination — only annotate directions that are PART of the task)
   Remove loops and wasted steps. Do NOT include reward annotations like [+10].
7. Do NOT add speculative branches for things that were never tried. That is a separate step.
8. DEDUP CHECK: Before adding any node, check if a similar milestone already exists in the tree. If so, use update_node to improve it instead of adding a duplicate.
9. SCORING NODE PROTECTION: If a node has max_reward > 0 (confirmed scoring), do NOT change its milestone text via update_node. You may only update its key_action and deps. A node that has earned reward is a CONFIRMED scoring point — renaming it to a different milestone would destroy that confirmed knowledge.
10. OPTIMIZE ORDERING: Do NOT blindly follow the chronological order of the episode. Instead, reason about the OPTIMAL order to minimize backtracking and wasted steps:
   - If a milestone can be completed near the starting location, place it EARLY in the path (even if the episode happened to do it late).
   - If a milestone requires an item or key obtained from another milestone, it MUST come after that dependency.
   - Think about geography: group milestones in the same area together to avoid going back and forth between floors/locations.
   - Example: if the agent starts at the Lobby and can get a key there, "Get key at Lobby" should be an early milestone, NOT placed after "Go upstairs and explore" — otherwise the agent wastes steps going upstairs, coming back down for the key, then going upstairs again.

You MUST respond with valid JSON only:
{{
    "analysis": [
        {{
            "episode": <int>,
            "score": <int>,
            "milestones_achieved": [
                {{
                    "description": "what was achieved",
                    "reward": <int>,
                    "action": "the reward-triggering action",
                    "key_steps": "minimum action sequence from previous milestone",
                    "existing_node": "node_id if this matches an existing tree node, or 'none'"
                }}
            ]
        }}
    ],
    "operations": [
        {{
            "op": "add_child|add_branch|update_node",
            "parent_id": "node_id (for add_child/add_branch)",
            "node_id": "node_id (for update_node)",
            "milestone": "milestone text",
            "key_action": "actions starting FROM destination. Direction words MUST have [from: Room] prefix. No navigation TO destination.",
            "destination": "exact room name where this milestone is completed (e.g. 'Kitchen', 'Loud Room')",
            "estimated_milestone_reward": <float>,
            "milestones": [{{"milestone": "...", "key_action": "actions from destination, with [Room] annotations for multi-room tasks", "destination": "room name", "estimated_milestone_reward": 0}}],
            "reason": "which episode/step this is based on"
        }}
    ]
}}"""

        user_prompt = f"""Update the milestone tree based on these episode results.

CURRENT MILESTONE TREE:
{existing_str}

EPISODE SUMMARIES:
{episodes_str}

Step 1 — Analyze each episode:
For each episode, list every reward-earning milestone achieved. For each, check if it already exists in the tree.

Step 2 — Propose tree operations:
- For milestones NOT in the tree: add_child or add_branch to encode them
- For milestones IN the tree but with wrong/incomplete key_action: update_node
- Do NOT use the prune operation
- When adding a chain of milestones from a high-scoring episode, use add_branch with the full sequence

IMPORTANT:
- Build the tree to reflect the BEST known path. If Episode X scored highest, its milestone sequence should be a complete path in the tree.
- Respect parent-child ordering: if milestone B only happens after A, B is a child of A.
- key_action: actions starting FROM the destination room. Direction words (north/south/east/west/up/down) MUST have [from: Room Name] prefix. Do NOT include navigation TO the destination. No loops, no wasted steps, no reward annotations.
- Each episode has {step_limit} steps. Keep milestones focused and achievable.
- OPTIMIZE MILESTONE ORDERING: Do NOT just replay the episode's chronological order. Think about the optimal order:
  * Milestones available at/near the starting location should come FIRST.
  * Group milestones in the same area together to avoid backtracking.
  * If a milestone provides an item/key needed later (e.g. getting a key before going to a locked room), place it BEFORE the milestones that depend on it.
  * When building add_branch, reorder milestones to minimize total navigation steps, even if episodes did them in a different order.

Respond with valid JSON only."""
        return sys_prompt, user_prompt

    def _build_dag_prompts(self, existing_str, episodes_str, step_limit):
        sys_prompt = f"""You are an expert at analyzing text adventure game results and maintaining a milestone DAG (directed acyclic graph).

The agent's knowledge is organized as a MILESTONE DAG (directed acyclic graph).

IMPORTANT — DAG uses "deps", NOT "parent_id":
- In a TREE, each node has a single "parent_id" meaning "this milestone comes after the parent in a linear sequence". The parent relationship is about ORDER.
- In a DAG, each node has a "deps" list meaning "this milestone REQUIRES these specific prerequisites to be completed first". The deps relationship is about CAUSAL DEPENDENCY.
- deps=[] means the milestone is INDEPENDENT — it can be done from the game start with no prerequisites.
- deps=["node_005"] means this milestone requires node_005 to be completed first.
- INDEPENDENT milestones (no causal dependency) MUST have separate deps — do NOT chain them together.

CRITICAL: deps means "what must be completed BEFORE this milestone can be attempted".
- "Exchange ID card for key" has deps=[] (can be done immediately at game start)
- "Unlock rare books door" has deps=["node that gets the key"] (need key first)
- "Search shelves for novel" has deps=[] (can be done anytime, no prerequisite)
- "Give encyclopedia to librarian" has deps=["node that obtains encyclopedia"] (need item first)

Your job is to UPDATE the DAG based on OBSERVED FACTS from the episodes below.
You must ONLY encode things that actually happened — do NOT speculate.

EXECUTION LOG: Some nodes include "exec[epN]" lines showing what happened when the agent tried the key_action in-game. Use this evidence:
- If a command is consistently rejected by the game (e.g. "that can't contain things", "won't help"), REMOVE it from key_action via update_node.
- If a command never earns reward across many attempts, consider whether it's actually necessary.

INITIAL SCORE: Some games award points automatically when the game starts, BEFORE the player takes any action. If an episode header shows "initial=N", those N points are NOT earned by any action and must NOT be attributed to any milestone. Only count points earned by actual player actions (earned=score-initial).

RULES:
1. Every SIGNIFICANT event should correspond to a node — this includes score changes (positive OR negative), important state changes (opening gates, draining water), and key item acquisitions. Do NOT limit yourself to positive rewards.
2. If an existing node's key_action is SUBSTANTIALLY wrong (wrong command, missing critical step, incorrect order), fix it with update_node. Do NOT update nodes just to rephrase or make minor wording changes — only update when the current key_action would cause the agent to fail.
   IMPORTANT: Do NOT change a node's destination room via update_node. If the action happens at a DIFFERENT room than the existing node's destination, create a NEW node instead. Changing the destination transforms it into a completely different milestone.
3. PRUNE ONLY FOR DEDUP: Do NOT prune nodes for low performance. Only use prune to remove duplicate nodes (see Rule 8).
4. DEPENDENCY RULES — deps means "this milestone REQUIRES these prerequisites":
   - Only add a dep when the milestone is IMPOSSIBLE without completing the prerequisite first (needs item, key, tool, or irreversible state change).
   - Example: "Kill troll" needs sword → deps=["sword_node"]. "Unlock door" needs key → deps=["key_node"].
   - KEY_ACTION PREREQUISITE CHECK: Examine each step in the key_action — if any step requires a specific item, tool, or game state to succeed, the node that PROVIDES that item/tool/state MUST be listed in deps. Examples: entering a dark area needs a light source, combat needs a weapon, using machinery needs a tool, opening a lock needs a key.
   - Do NOT add deps for geographic convenience or routing optimization. If a milestone CAN be done independently, use deps=[].
   - DO NOT use deps at all for unrelated milestones. deps=[] means fully independent.
5. Each milestone name MUST start with the exact room name: "Room Name + Action" (e.g. "Loud Room + Solve Echo Puzzle", NOT "Solve Echo Puzzle").
6. key_action FORMAT: Write ALL actions starting FROM the destination room until the reward is earned. This MUST include every direction/navigation step after the destination — do NOT skip intermediate rooms. Do NOT include navigation TO the destination — routing is computed automatically.
   DIRECTION ANNOTATION RULE: Every direction word (north/south/east/west/up/down/ne/nw/se/sw) MUST have [from: Room Name] prefix. No exceptions.
   - Example: "echo, take bar" (no direction words, no annotation needed)
   - Example: "hit troll with sword" (no direction words)
   - Example: "[from: Strange Passage] east, [from: Cyclops Room] up" (each direction annotated)
   - BAD: "east, up" (direction words missing [from:] annotation)
   - BAD: "touch mirror, take trident" (skipped intermediate rooms between mirror and trident)
7. SPLITTING A COARSE NODE: If a node's key_action needs to be broken into sequential sub-steps (e.g. it covers two distinct milestones), do NOT use split_node. Instead:
   a) Use update_node to narrow the original node to the FIRST sub-step only.
   b) Use add_child for each subsequent sub-step, with deps chaining each to the previous (e.g. step2 deps=[step1], step3 deps=[step2]).
   Example: node_002 "Prepare and Enter Underground" key_action="take lantern, push rug, open trap, down". Split into:
   - update_node node_002: milestone="Living Room + Take Lantern", key_action="take lantern"
   - add_child: milestone="Living Room + Enter Underground", key_action="push rug, open trap, down", deps=["node_002"]
8. DEDUP CHECK — Before adding a node, check ALL existing nodes for overlap:
   a) SAME LOCATION + SAME/SIMILAR ACTION: An existing node covers the same location and similar action → use update_node on the existing node instead.
   b) SAME DESTINATION ROOM: If the new node has the SAME destination room as an existing node, this is a STRONG duplicate signal. Two milestones at the same room almost always represent the same reward event (e.g. "Behind House + Open Window" and "Behind House + Enter House" are the same +10 reward). Do NOT create a new node — use update_node on the existing one instead.
   c) If you discover two existing nodes that are duplicates (same location + overlapping key_action), merge them:
      - Use prune on the lower-quality node (fewer visits or lower avg_reward), with reason="dedup: merged into node_XXX".
      - Use update_node on the kept node to absorb any missing deps from the pruned node's dependents.
   d) SUPERSET KEY_ACTION: An existing node's key_action already covers everything the new node does → skip it.
9. deps must reference EXISTING node IDs (e.g. "node_001", "node_005"). Check the CURRENT MILESTONE DAG section for valid IDs. Do NOT invent symbolic names like "node_get_key" — use the actual node_id shown in brackets like [node_001].
10. DEPENDENCY AUDIT — Review ALL existing nodes for missing or incorrect deps:
   - If a milestone needs an item/key/tool/state produced by another node but deps is missing it → use update_node to add the dep.
   - "Passing through a location" is NEVER a valid reason for a dep. Only CAUSAL requirements (item, key, tool, state change) count.
   - When in doubt, do NOT add a dep. False deps are worse than missing deps.
11. SCORING NODE PROTECTION: If a node has max_reward > 0 (confirmed scoring), do NOT change its milestone text via update_node. You may only update its key_action and deps. A node that has earned reward is a CONFIRMED scoring point — renaming it to a different milestone would destroy that confirmed knowledge.

You MUST respond with valid JSON only:
{{
    "analysis": [
        {{
            "episode": <int>,
            "score": <int>,
            "milestones_achieved": [
                {{
                    "description": "what was achieved",
                    "reward": <int>,
                    "action": "the reward-triggering action",
                    "key_steps": "minimum action sequence",
                    "existing_node": "node_id if matches existing node, or 'none'"
                }}
            ]
        }}
    ],
    "operations": [
        {{
            "op": "add_child|update_node|prune",
            "deps": ["node_id", ...],
            "node_id": "node_id (for update_node/prune)",
            "milestone": "milestone text",
            "key_action": "actions starting FROM destination. Direction words MUST have [from: Room] prefix. No navigation TO destination.",
            "destination": <room_id integer from KNOWN ROOMS list, or -1 if room not yet discovered>,
            "estimated_milestone_reward": <float>,
            "reason": "which episode/step this is based on"
        }}
    ]
}}

For add_child: "deps" is a list of node_id strings (e.g. ["node_001", "node_003"]). Use [] for independent milestones.
  - Only list nodes whose OUTCOME is required (item, key, tool, state change).
For prune: only use for dedup merging. Include reason="dedup: merged into node_XXX".
For destination: Set the room_id (integer) from KNOWN ROOMS where the agent needs to BE to start executing the key_action. For exploration milestones (e.g. "Kitchen + Go Up to Attic"), use the departure room (Kitchen), not the unknown target room. Only use -1 if the starting room is truly unknown."""

        user_prompt = f"""Update the milestone DAG based on these episode results.

CURRENT MILESTONE DAG:
{existing_str}

EPISODE SUMMARIES:
{episodes_str}

Step 1 — Analyze each episode:
For each episode, list every significant event: score changes (positive AND negative), important state changes, and key item acquisitions. Check if each already exists in the DAG. Analyze EVERY episode thoroughly — do NOT skip or abbreviate any episode even if it overlaps with others.

Step 2 — Analyze milestone failures:
Check the "Milestones NOT Achieved" sections in the summaries. For each failed milestone:
- If it failed because of a missing prerequisite (e.g. "no light source", "needs item X"), find which existing DAG node provides that prerequisite and ensure the failed node's deps include it.
- If a SIMILAR node already exists in the DAG for the same location/action, do NOT create a duplicate. Use update_node on the existing one instead.

Step 3 — Dedup check and dependency audit:
- Scan ALL existing nodes: are there two nodes at the same location with overlapping key_actions? If so, merge them (prune the weaker, update the keeper).
- For each node, verify deps are correct: does this milestone need an ITEM/KEY/TOOL/STATE from another node? If the dep is missing, add it via update_node.

Step 4 — Propose DAG operations:
- For milestones NOT in the DAG (and passing dedup check): add_child with correct deps
- For milestones with wrong key_action or incomplete deps: update_node
- For duplicate nodes: prune the weaker one with reason="dedup: merged into node_XXX"
- To split a coarse node: update_node to narrow it + add_child for the continuation


CRITICAL — DEPENDENCY RULES:
- deps=[] for milestones with no prerequisites
- deps=["node_001"] means this milestone REQUIRES node_001's outcome (item/key/tool/state)
- Only add deps when the milestone is IMPOSSIBLE without the prerequisite
- KEY_ACTION PREREQUISITE CHECK: For each step in key_action, ask "does this step need a specific item/tool/state?" If yes, the node providing it MUST be in deps
- Do NOT add deps for geographic convenience or routing — deps are for CAUSAL requirements only
- deps must use EXISTING node IDs from the DAG (e.g. "node_001"), NOT invented names

Examples:
- "Get key at Lobby" → deps=[]
- "Find herring at Ground Floor" → deps=[] (no prerequisite)
- "Explore Attic" → deps=["node_lamp"] (needs lamp for dark area)
- "Unlock door with key" → deps=["node_key"] (NEED the key)
- "Enter Dark Cellar" → deps=["node_lamp"] (entering a dark area requires light source)
- "Open mailbox" → deps=[] (can be done independently, no item/tool needed)

Each episode has {step_limit} steps. Keep milestones focused and achievable.

CRITICAL REMINDER — key_action format:
- Write actions starting FROM the destination room, in order (e.g. "echo, take bar", "hit troll with sword")
- Do NOT include navigation TO the destination — routing to the destination is computed automatically
- If key_action contains direction words (north/south/east/west/up/down/ne/nw/se/sw), MUST prefix each with [from: Room Name] (e.g. "[from: Strange Passage] east, [from: Cyclops Room] up")

CRITICAL REMINDER — destination format:
- Use the integer room_id from KNOWN ROOMS where the agent must BE to start the key_action
- For exploration milestones (e.g. "Kitchen + Go Up to Attic"), use the departure room (Kitchen's room_id), NOT the unknown target
- Only use -1 if the starting room is truly not in KNOWN ROOMS

Respond with valid JSON only."""
        return sys_prompt, user_prompt

    def _build_python_game_prompts(self, existing_str, episodes_str, step_limit):
        """Build prompts for python games (catnip etc.).

        Strategies are high-level approaches, NOT fixed action sequences.
        No milestones, no key_actions — just descriptions of what approach to take.
        """
        sys_prompt = f"""You are an expert at analyzing strategy game results and maintaining a strategy list.

The agent's knowledge is organized as a STRATEGY LIST:
- Each strategy is a HIGH-LEVEL APPROACH description (1-3 sentences)
- Strategies describe WHAT to prioritize, not specific action sequences
- The agent decides specific actions at runtime based on game state

Your job is to UPDATE the strategy list based on OBSERVED FACTS from the episodes below.

RULES:
1. Strategies should encode DISCOVERED KNOWLEDGE — e.g. "calming actions work best on comfort targets", "use items to reduce suspicion on Inspector"
2. Do NOT write fixed action sequences. The agent adapts based on current game state.
3. Focus on the ROOT CAUSE of why some episodes scored higher than others.

You MUST respond with valid JSON only:
{{
    "analysis": [
        {{
            "episode": <int>,
            "score": <int>,
            "milestones_achieved": [
                {{
                    "description": "what was achieved",
                    "reward": <int>,
                    "action": "the reward-triggering action",
                    "existing_node": "strategy_id if matches existing strategy, or 'none'"
                }}
            ]
        }}
    ],
    "operations": [
        {{
            "op": "add_branch|update_strategy|prune",
            "strategy_id": "strategy_id (for update/prune)",
            "description": "high-level strategy description (1-3 sentences, NOT action sequences)",
            "reason": "which episode/step this is based on"
        }}
    ]
}}"""

        user_prompt = f"""Update the strategy list based on these episode results.

CURRENT STRATEGY LIST:
{existing_str}

EPISODE SUMMARIES:
{episodes_str}

Step 1 — Analyze each episode:
For each episode, identify what approach was used and how effective it was.
Pay special attention to CROSS-TARGET effects: how did reward choices on early targets affect performance on later targets? (e.g., "chose items from Target 3 → had Salmon Treat for Target 5 → higher score" vs "chose progress from Target 3 → no items for Target 5 → lower score")

Step 2 — Propose operations:
- add_branch: For new approaches discovered from high-scoring episodes. Description should be a high-level approach (1-3 sentences), NOT a fixed action sequence. MUST include reward selection strategy across targets.
- update_strategy: To refine an existing strategy's description based on new cross-target insights.
- prune: To remove consistently failing strategies.

Each episode has {step_limit} steps.

Respond with valid JSON only."""
        return sys_prompt, user_prompt

    @staticmethod
    def _trim_summary(summary: str, keep_sections: set) -> str:
        """Keep only specified sections from a structured summary.

        Sections are identified by **Section Name:** headers on their own line.
        keep_sections: set of lowercase section name prefixes to keep,
                       e.g. {'milestones achieved', 'milestones not', 'penalties', 'new discoveries'}
        """
        import re
        # Split on lines that start with **...**
        sections = re.split(r'(?m)(?=^\*\*.+?\*\*)', summary)
        kept = []
        for section in sections:
            header_match = re.match(r'\*\*(.+?)\*\*', section)
            if not header_match:
                if section.strip():
                    kept.append(section)
                continue
            header = header_match.group(1).strip().lower()
            if any(header.startswith(prefix) for prefix in keep_sections):
                kept.append(section)
        return ''.join(kept).strip()

    def _format_episodes(self, episode_summaries: List[Dict[str, Any]]) -> str:
        # TreeUpdate needs everything except Untried Valid Actions (that's for DPM)
        keep = {'milestones achieved', 'milestones not', 'penalties', 'new discoveries', 'failed attempts'}
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
            else:
                header += " — Free exploration (no strategy assigned)"
            lines.append(header + ":")
            trimmed = self._trim_summary(summary, keep)
            lines.append(f"  {trimmed}")

        return "\n".join(lines)

    def _print_analysis(self, analysis: List[Dict]):
        if not analysis:
            return
        print("\n" + "=" * 60)
        print("  Tree Update — Episode Analysis")
        print("=" * 60)
        for ep in analysis:
            ep_num = ep.get('episode', '?')
            score = ep.get('score', '?')
            print(f"\n  Episode {ep_num} (Score: {score}):")
            for m in ep.get('milestones_achieved', []):
                desc = m.get('description', '?')
                reward = m.get('reward', '?')
                existing = m.get('existing_node', 'none')
                tag = f"→ {existing}" if existing != 'none' else "→ NEW"
                print(f"    [+{reward}] {desc} {tag}")
        print()

    def _print_operations(self, operations: List[Dict]):
        if not operations:
            print("  No tree updates needed.")
            return
        print("=" * 60)
        print("  Tree Update — Operations")
        print("=" * 60)
        for i, op in enumerate(operations, 1):
            op_type = op.get('op', '?')
            reason = op.get('reason', '')
            print(f"\n  Op {i}: {op_type}")
            if op_type == 'add_child':
                # Show deps (DAG) or parent_id (tree) depending on what's present
                if 'deps' in op:
                    print(f"    Deps: {op.get('deps', [])}")
                else:
                    print(f"    Parent: {op.get('parent_id', '?')}")
                print(f"    Milestone: {op.get('milestone', '?')}")
                key_action = op.get('key_action', '')
                if key_action:
                    print(f"    Key Action: {key_action}")
            elif op_type == 'add_branch':
                print(f"    Parent: {op.get('parent_id', '?')}")
                for j, m in enumerate(op.get('milestones', []), 1):
                    ka = m.get('key_action', '')
                    ka_str = f" — key: {ka}" if ka else ""
                    print(f"    {j}. {m.get('milestone', '?')}{ka_str}")
            elif op_type == 'prune':
                print(f"    Node: {op.get('node_id', '?')}")
            elif op_type == 'update_node':
                print(f"    Node: {op.get('node_id', '?')}")
                print(f"    New milestone: {op.get('milestone', '?')}")
                key_action = op.get('key_action', '')
                if key_action:
                    print(f"    New key_action: {key_action}")
            if reason:
                print(f"    Reason: {reason}")
        print()
