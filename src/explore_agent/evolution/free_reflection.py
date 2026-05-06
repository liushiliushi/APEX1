"""Free Reflection space evolution (from existing ReflectiveExplorationAgent).

Three-phase pipeline:
  Phase 1: Per-episode strategy extraction
  Phase 2: Comparative reasoning
  Phase 3: Tree modification operations
"""
from typing import Any, Dict, List

from ..openai_helpers_proxy import chat_completion, parse_json_response


class FreeReflection:
    """Free-form LLM reflection that proposes tree modifications."""

    def reflect(self, episode_summaries: List[Dict[str, Any]],
                strategy_space,
                llm_model: str,
                args) -> List[Dict[str, Any]]:
        """Run 3-phase reflection and return tree operations + metadata."""
        n = len(episode_summaries)
        print(f"[FreeReflection] Running cross-episode reflection on {n} episodes...")

        # Use dedicated prompt for python games (catnip etc.)
        if getattr(args, 'game_name', '').startswith('catnip'):
            return self._reflect_python_game(episode_summaries, strategy_space, llm_model, args)

        # Use flat-list prompt for plan_flat_list strategy space
        is_flat_list = getattr(strategy_space, 'space_type', None) == 'plan_flat_list' or \
                       hasattr(strategy_space, 'strategies')
        if is_flat_list:
            return self._reflect_flat_list(episode_summaries, strategy_space, llm_model, args)

        episodes_str = self._format_episodes(episode_summaries)
        existing_str = strategy_space.format_for_reflection()

        step_limit = getattr(args, 'env_step_limit', 50)

        json_schema = """{
    "phase_1_episode_strategies": [
        {
            "episode": <int>,
            "score": <int>,
            "strategies": [
                {"description": "...", "outcome": "succeeded|failed|partial", "key_actions": ["..."]}
            ],
            "key_findings": "..."
        }
    ],
    "phase_2_reasoning": {
        "score_comparisons": [
            {
                "high_scorer": {"episode": <int>, "score": <int>},
                "low_scorer": {"episode": <int>, "score": <int>},
                "score_difference": <int>,
                "key_actions_that_made_difference": ["action [+N]", ...],
                "why_high_scored_more": "..."
            }
        ],
        "best_known_sequence": {
            "source_episode": <int>,
            "score": <int>,
            "action_sequence_summary": "...",
            "total_points_from_sequence": <int>
        },
        "unexplored_leads": [
            {
                "where": "location or context where the clue appeared",
                "what": "the item, path, NPC, or interaction mentioned",
                "suggested_action": "best guess for what to try",
                "which_episodes": [1, 3]
            }
        ],
        "tree_vs_episodes": {
            "orphan_rewards": [
                {
                    "episode": <int>,
                    "action": "the reward-triggering action",
                    "reward": <int>,
                    "nearest_tree_node": "node_id of closest related node, or 'none'",
                    "suggested_milestone": "what milestone should capture this reward"
                }
            ],
            "key_action_corrections": [
                {
                    "node_id": "tree node whose key_action is wrong or outdated",
                    "current_key_action": "what the tree currently says",
                    "observed_action": "what the episode actually did",
                    "source_episode": <int>,
                    "issue": "wrong|outdated|incomplete|suboptimal"
                }
            ],
            "untried_branches": [
                {
                    "node_id": "tree node that was never selected in these episodes",
                    "milestone": "the milestone text",
                    "visits": <int>,
                    "reason": "why it might be worth trying or pruning"
                }
            ]
        }
    },
    "phase_3_tree_operations": [
        {
            "op": "add_child|add_branch|prune|update_node",
            "parent_id": "node_id of parent (for add_child, add_branch)",
            "node_id": "node_id (for prune, update_node)",
            "milestone": "milestone text (for add_child, update_node)",
            "key_action": "critical action hint (for add_child, update_node)",
            "estimated_milestone_reward": "<float or 0, for add_child>",
            "milestones": [{"milestone": "...", "key_action": "...", "estimated_milestone_reward": "<float or 0>"}],
            "reason": "why this operation is needed"
        }
    ]
}"""

        sys_prompt = f"""You are an expert analyst for text adventure games.
Your task is to analyze gameplay summaries in three phases and propose tree modifications.

The agent's knowledge is organized as a MILESTONE TREE:
- The root represents the game start
- Each node is a milestone (a goal like "Collect Sword in Living Room")
- Children of a node represent alternative next milestones after completing the parent
- A path from root to a leaf is one complete strategy

STRICT RULES for milestones:
1. Each milestone must cover exactly ONE thing. Do not combine multiple interactions or goals into one milestone.
2. Each milestone name must describe WHERE + WHAT — mention the key location, object, NPC, or item involved (e.g. "Give ID card to attendant at Lobby", "Examine bookshelf in Study Room").
3. Do NOT write specific game commands in the milestone field. Milestones are goals, not commands.
   Instead, put the COMPLETE action sequence needed to achieve this milestone in the "key_action" field.
   Include ALL prerequisite steps: navigation from the previous milestone's location, inventory preparation
   (e.g. dropping items to free capacity, equipping needed tools), and the goal actions themselves.
   Format as a sequence: "drop sword → take wrench → go north → squeeze tube on pipe".
   For simple milestones that only need one obvious action, use a short hint or leave as "".
   CRITICAL: Each episode includes a "Raw action sequence" — the exact actions taken step by step.
   When writing key_actions, you MUST extract the subsequence directly from these raw actions.
   Do NOT reconstruct or guess action sequences from the narrative summary — use the raw data.
4. NEVER include repetitive or loop patterns. If episodes show back-and-forth movement, these are WASTED actions.
5. SEQUENTIAL ORDER = PARENT-CHILD RELATIONSHIP. If milestone B can only happen after milestone A, then B must be a CHILD of A — never a sibling. Only milestones that are truly independent alternatives to each other should share the same parent. Before placing any node, ask: "Does this require a previous milestone to be completed first?" If yes, it is a child of that milestone, not a sibling.

6. When writing key_action, do NOT include reward annotations like [+10] or [-5].
   Write only game commands: "echo → take bar", NOT "echo → take bar [+10]".

IMPORTANT CONSTRAINT: Each episode has only {step_limit} actions. Keep milestones small and focused. Early milestones should independently earn points so the agent makes progress even if it cannot complete the full strategy in one episode.

GOOD milestone examples:
- "Retrieve platinum bar from Loud Room" (key_action: "go up from Round Room → echo → take bar") — from observed trajectory
- "Store bar via slide in Mirror Room" (key_action: "go up → go northwest → go south → go south → put bar in slide") — from observed trajectory
- "Give the ID card to the front desk attendant" (key_action: "") — simple/obvious action
- "Navigate to Dome Room via Engravings Cave" (key_action: "") — speculative node, never walked this path, leave empty

BAD milestone examples:
- "go north, then east, then open door" (contains specific commands)
- "explore" (too vague, no WHERE or WHAT)

You MUST respond with valid JSON only, no other text. Use this exact schema:
{json_schema}"""

        user_prompt = f"""Analyze the agent's gameplay across the following episode summaries in THREE phases.

CURRENT MILESTONE TREE:
{existing_str}

EPISODE SUMMARIES:
{episodes_str}

Phase 1 — Per-Episode Strategy Extraction:
For each episode, identify the high-level strategies attempted and tag each as succeeded/failed/partial.
List the key actions taken for each strategy.
CRITICAL: Actions in the raw action sequence that triggered rewards are annotated like "action [+25]". These reward-triggering actions are the MOST important to capture — every single one must appear in your Phase 1 key_actions list. Missing a reward action means missing a critical milestone.

Phase 2 — Comparative Reasoning:
Compare episodes to extract actionable insights.

Score Comparisons:
Find pairs of episodes that visited similar areas but got very different scores. For each pair:
- Identify the HIGH scorer and LOW scorer (with episode numbers and scores)
- Pinpoint the SPECIFIC actions that caused the score difference (use reward-annotated actions like "action [+N]")
- Explain WHY the high scorer earned more (e.g. "took the platinum bar before leaving the room", "solved the puzzle instead of wandering")
Focus on the most informative comparisons — pairs where one episode clearly did something better.

Best Known Sequence:
Identify the single best scoring sequence across ALL episodes. Record:
- Which episode it came from and its score
- A summary of its action sequence (the key milestones it hit, in order)
- The total points accounted for by its reward-triggering actions
This is the "gold standard" the tree should encode as a path.

Unexplored Leads:
Scan the episode summaries for clues that appeared in game text but were NEVER followed up on by any episode. These include:
- Locations mentioned or seen but never visited (e.g. "a path leads south" but no episode went south)
- Items seen but never taken or used (e.g. "there is a rope here" but no episode took it)
- Interactions never attempted (e.g. an NPC that was never talked to, a locked door never opened)
- Game hints or descriptions suggesting hidden content (e.g. "you notice something unusual about the wall")
List each lead with: WHERE it was seen, WHAT was mentioned, and your BEST GUESS for what action to try.
These leads are critical — they represent the frontier of unexplored game content.

Tree vs Episodes:
Compare what the episodes actually did against what the milestone tree currently encodes. Identify three types of gaps:

Orphan Rewards — rewards earned by episodes that NO existing tree milestone accounts for.
For each, note the episode, the reward-triggering action, the reward amount, the nearest related tree node (or "none"), and suggest what milestone should be added to capture it.

Key Action Corrections — tree nodes whose key_action field is wrong, outdated, incomplete, or suboptimal compared to what episodes actually did.
For each, note the node_id, its current key_action, what the episode actually did (from raw action sequence), which episode, and the type of issue (wrong/outdated/incomplete/suboptimal).

Untried Branches — tree nodes that were NOT selected by any episode in this batch. These are exploration blind spots.
For each, note the node_id, its milestone text, its visit count, and your assessment of whether it's worth trying next or should be pruned.

Phase 3 — Tree Modifications (based on Phase 1 & 2):
Now propose CONCRETE modifications to the milestone tree. Look at the current tree structure and decide what to add, prune, or update.

Available operations:
- **add_child**: Add a single new milestone as a child of an existing node. Use when a node needs a new alternative next step.
  Required: parent_id, milestone, key_action, reason
- **add_branch**: Add a complete path (sequence of milestones) under an existing node. Use when proposing a whole new strategy branch.
  Required: parent_id, milestones (array of {{milestone, key_action}}), reason
- **prune**: Remove a node and its entire subtree. Use when a path has repeatedly failed and should be abandoned.
  Required: node_id, reason
- **update_node**: Modify an existing node's milestone or key_action. Use when a milestone needs refinement based on new experience.
  Required: node_id, milestone (new text), key_action (new text), reason

Guidelines:
- IMPORTANT: At least one tree operation MUST come from Phase 2 unexplored leads. The tree should not only record past behavior — it must also propose speculative branches for things the agent has NOT yet tried but has clues about. For these speculative nodes, leave key_action as "" (empty) — do NOT guess navigation sequences for paths the agent has never actually walked. The agent will figure out the route during gameplay. Only write key_actions that are directly extracted from the "Raw action sequence" of episodes that actually completed that path.
- When Phase 2 score_comparisons reveal high-scoring action sequences, ensure the tree has a path that encodes those actions as milestones. If the best_known_sequence is not represented in the tree, add it.
- Every orphan_reward from tree_vs_episodes should result in an add_child or add_branch to capture that reward in the tree.
- Every key_action_correction should result in an update_node to fix the key_action.
- Add branches for genuinely new approaches discovered in Phase 2 unexplored leads
- **PRUNE aggressively**: If a node has been selected multiple times but consistently gets low rewards (avg_reward near 0), the agent cannot execute that strategy — PRUNE IT. A node that consistently fails wastes exploration budget.
- Update nodes when we learn better key_actions from experience
- Prefer extending existing successful paths rather than always creating new root-level branches
- Do NOT add duplicate milestones under the same parent
- DEDUP CHECK: Before adding any new node, scan the ENTIRE tree for existing nodes with the same or very similar milestone goal (e.g. same item, same location, same objective). If a near-duplicate exists anywhere in the tree, use update_node to improve the existing node instead of add_child to create a duplicate. Having multiple nodes for the same goal wastes tree capacity.
- For add_child and add_branch, provide estimated_milestone_reward: the reward this milestone earns, from reward annotations in raw actions (e.g., "take bar [+10]" → 10). For speculative milestones with no observed reward, set to 0.

Respond with valid JSON only."""

        max_retries = 5
        for attempt in range(max_retries):
            try:
                raw = chat_completion(
                    model=llm_model,
                    sys_prompt=sys_prompt,
                    prompt=user_prompt,
                    max_tokens=8000,
                    temperature=0.4,
                )

                if not raw:
                    print(f"[FreeReflection] LLM call failed (attempt {attempt + 1}/{max_retries})")
                    continue

                reflection = parse_json_response(raw)
                if not reflection:
                    print(f"[FreeReflection] Failed to parse JSON (attempt {attempt + 1}/{max_retries})")
                    print(f"[FreeReflection] Raw (first 500 chars): {raw[:500]}")
                    continue

                phase_1 = reflection.get('phase_1_episode_strategies', [])
                phase_2 = reflection.get('phase_2_reasoning', {})
                phase_3 = reflection.get('phase_3_tree_operations', [])
                if not phase_1:
                    print(f"[FreeReflection] Missing Phase 1 (attempt {attempt + 1}/{max_retries})")
                    continue

                # Print phases
                self._print_phase_1(phase_1)
                self._print_phase_2(phase_2)
                self._print_phase_3(phase_3)

                # Return operations + metadata for the orchestrator to apply
                return {
                    'operations': phase_3,
                    'phase_1': phase_1,
                    'phase_2': phase_2,
                }

            except Exception as e:
                print(f"[FreeReflection] Error (attempt {attempt + 1}/{max_retries}): {e}")

        print(f"[FreeReflection] Reflection failed after {max_retries} attempts")
        return {'operations': []}

    # ------------------------------------------------------------------ #
    #  Episode formatting
    # ------------------------------------------------------------------ #

    def _format_episodes(self, episode_summaries: List[Dict[str, Any]]) -> str:
        lines = []
        for ep in episode_summaries:
            ep_num = ep.get('episode', '?')
            score = ep.get('score', 0)
            summary = ep.get('summary', 'No summary available.')
            actions = ep.get('actions', [])
            strategy_desc = ep.get('strategy_description')

            header = f"\nEpisode {ep_num} (Score: {score})"
            if strategy_desc:
                header += f' — Strategy: "{strategy_desc}"'
            else:
                header += " — Free exploration (no strategy assigned)"
            lines.append(header + ":")
            lines.append(f"  {summary}")
            if actions:
                actions_str = " → ".join(actions)
                lines.append(f"  Raw action sequence: [{actions_str}]")

        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  Pretty-printing
    # ------------------------------------------------------------------ #

    def _print_phase_1(self, data):
        print("\n" + "=" * 60)
        print("  Phase 1: Per-Episode Strategy Extraction")
        print("=" * 60)
        for ep in data:
            ep_num = ep.get('episode', '?')
            score = ep.get('score', '?')
            print(f"\n  Episode {ep_num} (Score: {score}):")
            for s in ep.get('strategies', []):
                outcome = s.get('outcome', 'unknown')
                tag = {'succeeded': '+', 'failed': '-', 'partial': '~'}.get(outcome, '?')
                print(f"    [{tag}] {s.get('description', '?')}")
                for action in s.get('key_actions', []):
                    print(f"        - {action}")
            findings = ep.get('key_findings', '')
            if findings:
                print(f"    Findings: {findings}")
        print()

    def _print_phase_2(self, data):
        print("=" * 60)
        print("  Phase 2: Comparative Reasoning")
        print("=" * 60)

        comparisons = data.get('score_comparisons', [])
        if comparisons:
            print("\n  Score Comparisons:")
            for comp in comparisons:
                high = comp.get('high_scorer', {})
                low = comp.get('low_scorer', {})
                diff = comp.get('score_difference', '?')
                print(f"    Ep {high.get('episode', '?')} ({high.get('score', '?')}) vs "
                      f"Ep {low.get('episode', '?')} ({low.get('score', '?')})  [+{diff}]")
                for action in comp.get('key_actions_that_made_difference', []):
                    print(f"      - {action}")
                why = comp.get('why_high_scored_more', '')
                if why:
                    print(f"      Why: {why}")

        best = data.get('best_known_sequence', {})
        if best:
            print(f"\n  Best Known Sequence:")
            print(f"    Source: Episode {best.get('source_episode', '?')} "
                  f"(Score: {best.get('score', '?')}, "
                  f"Points accounted: {best.get('total_points_from_sequence', '?')})")
            summary = best.get('action_sequence_summary', '')
            if summary:
                print(f"    Sequence: {summary}")

        leads = data.get('unexplored_leads', [])
        if leads:
            print(f"\n  Unexplored Leads:")
            for lead in leads:
                where = lead.get('where', '?')
                what = lead.get('what', '?')
                action = lead.get('suggested_action', '')
                print(f"    - [{where}] {what}")
                if action:
                    print(f"      Try: {action}")

        tree_vs = data.get('tree_vs_episodes', {})
        if tree_vs:
            orphans = tree_vs.get('orphan_rewards', [])
            if orphans:
                print(f"\n  Orphan Rewards (not in tree):")
                for o in orphans:
                    print(f"    - Ep {o.get('episode', '?')}: {o.get('action', '?')} [+{o.get('reward', '?')}]")
                    nearest = o.get('nearest_tree_node', 'none')
                    print(f"      Nearest node: {nearest}")
                    suggested = o.get('suggested_milestone', '')
                    if suggested:
                        print(f"      Suggested milestone: {suggested}")

            corrections = tree_vs.get('key_action_corrections', [])
            if corrections:
                print(f"\n  Key Action Corrections:")
                for c in corrections:
                    print(f"    - [{c.get('node_id', '?')}] {c.get('issue', '?')}")
                    print(f"      Tree says: {c.get('current_key_action', '?')}")
                    print(f"      Actually: {c.get('observed_action', '?')} (Ep {c.get('source_episode', '?')})")

            untried = tree_vs.get('untried_branches', [])
            if untried:
                print(f"\n  Untried Branches:")
                for u in untried:
                    print(f"    - [{u.get('node_id', '?')}] {u.get('milestone', '?')} "
                          f"(visits={u.get('visits', 0)})")
                    reason = u.get('reason', '')
                    if reason:
                        print(f"      {reason}")

        print()

    def _print_phase_3(self, data):
        print("=" * 60)
        print("  Phase 3: Tree Modifications")
        print("=" * 60)
        for i, op in enumerate(data, 1):
            op_type = op.get('op', '?')
            reason = op.get('reason', '')
            print(f"\n  Op {i}: {op_type}")
            if op_type == 'add_child':
                print(f"    Parent: {op.get('parent_id', '?')}")
                print(f"    Milestone: {op.get('milestone', '?')}")
                key_action = op.get('key_action', '')
                if key_action:
                    print(f"    Key Action: {key_action}")
            elif op_type == 'add_branch':
                print(f"    Parent: {op.get('parent_id', '?')}")
                for j, m in enumerate(op.get('milestones', []), 1):
                    print(f"    {j}. {m.get('milestone', '?')}")
            elif op_type == 'prune':
                print(f"    Node: {op.get('node_id', '?')}")
            elif op_type == 'update_node':
                print(f"    Node: {op.get('node_id', '?')}")
                print(f"    New milestone: {op.get('milestone', '?')}")
            if reason:
                print(f"    Reason: {reason}")
        print()

    # ------------------------------------------------------------------ #
    #  Flat list reflection — strategies are independent complete plans
    # ------------------------------------------------------------------ #

    def _reflect_flat_list(self, episode_summaries, strategy_space, llm_model, args):
        """Reflection for plan_flat_list: each strategy is a complete start-to-finish plan."""
        episodes_str = self._format_episodes(episode_summaries)
        existing_str = strategy_space.format_for_reflection()
        step_limit = getattr(args, 'env_step_limit', 50)

        json_schema = """{
    "phase_1_episode_strategies": [
        {
            "episode": <int>,
            "score": <int>,
            "strategies": [
                {"description": "...", "outcome": "succeeded|failed|partial", "key_actions": ["..."]}
            ],
            "key_findings": "..."
        }
    ],
    "phase_2_reasoning": {
        "score_comparisons": [
            {
                "high_scorer": {"episode": <int>, "score": <int>},
                "low_scorer": {"episode": <int>, "score": <int>},
                "score_difference": <int>,
                "key_actions_that_made_difference": ["action [+N]", ...],
                "why_high_scored_more": "..."
            }
        ],
        "best_known_sequence": {
            "source_episode": <int>,
            "score": <int>,
            "action_sequence_summary": "...",
            "total_points_from_sequence": <int>
        },
        "unexplored_leads": [
            {
                "where": "location or context where the clue appeared",
                "what": "the item, path, NPC, or interaction mentioned",
                "suggested_action": "best guess for what to try",
                "which_episodes": [1, 3]
            }
        ]
    },
    "phase_3_operations": [
        {
            "op": "add_branch|prune|update_strategy",
            "strategy_id": "strategy_id (for prune, update_strategy)",
            "description": "strategy description (for add_branch)",
            "milestones": [{"milestone": "...", "key_action": "...", "estimated_milestone_reward": "<float or 0>"}],
            "reason": "why this operation is needed"
        }
    ]
}"""

        sys_prompt = f"""You are an expert analyst for text adventure games.
Your task is to analyze gameplay summaries and propose strategy modifications.

The agent's knowledge is organized as a STRATEGY LIST:
- Each strategy is a COMPLETE plan from game start to finish
- Each strategy has ordered milestones the agent follows step by step
- Strategies are INDEPENDENT — they do NOT share or inherit steps from each other
- The agent starts every episode from the same starting location

RULES for strategies:
1. Every strategy must begin from the game's starting location. Do NOT assume the agent starts mid-game.
2. Each milestone describes WHERE + WHAT. Put the complete action sequence in "key_action".
3. key_action MUST NOT be empty. Extract from the "Raw action sequence" of episodes only.
   If you cannot provide concrete key_actions from observed data, do NOT add the strategy.
4. Do NOT include reward annotations like [+10] in key_action. Write only game commands.
5. Each episode has only {step_limit} actions. Order milestones so early ones earn points independently.

You MUST respond with valid JSON only. Use this exact schema:
{json_schema}"""

        user_prompt = f"""Analyze the agent's gameplay across the following episode summaries.

CURRENT STRATEGY LIST:
{existing_str}

EPISODE SUMMARIES:
{episodes_str}

Phase 1 — Per-Episode Strategy Extraction:
For each episode, identify the strategies attempted. Tag each as succeeded/failed/partial.
Actions annotated with rewards like "action [+25]" are the most important — capture every one.

Phase 2 — Comparative Reasoning:
- Score Comparisons: Find pairs with different scores. What actions caused the difference?
- Best Known Sequence: The single highest-scoring sequence across all episodes.
- Unexplored Leads: Locations, items, or interactions mentioned but never followed up.

Phase 3 — Strategy Modifications:
Available operations:
- **add_branch**: Add a new complete strategy with milestones. Every milestone must have non-empty key_action.
  The best approach: take the highest-scoring strategy as a base, then extend or vary it.
- **prune**: Remove a consistently underperforming strategy. Required: strategy_id, reason.
- **update_strategy**: Replace milestones of an existing strategy. Required: strategy_id, milestones, reason.

Guidelines:
- New strategies should build on the best known sequence — include the proven high-scoring steps, then diverge.
- Prune strategies that caused penalties or loops, or that have low avg_reward after 3+ visits.
- Each strategy should differ from others in meaningful ways, not just minor variations.

Respond with valid JSON only."""

        max_retries = 5
        for attempt in range(max_retries):
            try:
                raw = chat_completion(
                    model=llm_model,
                    sys_prompt=sys_prompt,
                    prompt=user_prompt,
                    max_tokens=8000,
                    temperature=0.4,
                )
                if not raw:
                    print(f"[FreeReflection] LLM call failed (attempt {attempt + 1}/{max_retries})")
                    continue
                reflection = parse_json_response(raw)
                if not reflection:
                    print(f"[FreeReflection] Failed to parse JSON (attempt {attempt + 1}/{max_retries})")
                    continue

                phase_1 = reflection.get('phase_1_episode_strategies', [])
                phase_2 = reflection.get('phase_2_reasoning', {})
                phase_3 = reflection.get('phase_3_operations',
                                         reflection.get('phase_3_tree_operations', []))
                if not phase_1:
                    print(f"[FreeReflection] Missing Phase 1 (attempt {attempt + 1}/{max_retries})")
                    continue

                self._print_phase_1(phase_1)
                self._print_phase_2(phase_2)
                self._print_phase_3(phase_3)

                return {
                    'operations': phase_3,
                    'phase_1': phase_1,
                    'phase_2': phase_2,
                }
            except Exception as e:
                print(f"[FreeReflection] Error (attempt {attempt + 1}/{max_retries}): {e}")

        print(f"[FreeReflection] Reflection failed after {max_retries} attempts")
        return {'operations': []}

    # ------------------------------------------------------------------ #
    #  Python game (catnip) reflection — separate prompt, no room/nav logic
    # ------------------------------------------------------------------ #

    def _reflect_python_game(self, episode_summaries, strategy_space, llm_model, args):
        """Reflection for python games (catnip etc.).

        Strategies are HIGH-LEVEL DIRECTIONS only — no milestones, no key_actions.
        The agent decides specific actions at runtime based on game state.
        """
        episodes_str = self._format_episodes(episode_summaries)
        existing_str = strategy_space.format_for_reflection()
        step_limit = getattr(args, 'env_step_limit', 50)

        json_schema = """{
    "phase_1_episode_analysis": [
        {
            "episode": <int>,
            "score": <int>,
            "what_worked": "what decisions led to high rewards",
            "what_failed": "what decisions led to low rewards or wasted turns",
            "key_findings": "important patterns observed in game state"
        }
    ],
    "phase_2_reasoning": {
        "score_comparisons": [
            {
                "high_scorer": {"episode": <int>, "score": <int>},
                "low_scorer": {"episode": <int>, "score": <int>},
                "score_difference": <int>,
                "why_high_scored_more": "root cause analysis — compare per-target rewards, not just totals"
            }
        ],
        "cross_target_dependencies": [
            {
                "early_choice": "what reward/item was chosen on an early target",
                "later_effect": "how it helped or hurt on a later target",
                "net_impact": "was the trade-off positive or negative overall",
                "evidence": "which episodes demonstrate this"
            }
        ],
        "speed_analysis": {
            "fastest_episode": <int>,
            "turns_used": <int>,
            "score": <int>,
            "observation": "did faster episodes tend to score higher?"
        },
        "best_approach": {
            "source_episode": <int>,
            "score": <int>,
            "description": "what made this episode's approach the best"
        },
        "discovered_rules": [
            {
                "rule": "a game mechanic or pattern discovered from the data",
                "evidence": "which episodes/steps demonstrate this"
            }
        ]
    },
    "phase_3_operations": [
        {
            "op": "add_branch|prune|update_strategy",
            "strategy_id": "strategy_id (for prune, update_strategy)",
            "description": "strategy description — a high-level approach, NOT a fixed action sequence",
            "reason": "why this operation is needed"
        }
    ]
}"""

        sys_prompt = f"""You are an expert analyst for strategy games.
Your task is to analyze gameplay data, discover game mechanics, and propose high-level strategies.

The agent's knowledge is organized as a STRATEGY LIST:
- Each strategy is a HIGH-LEVEL APPROACH — a general direction, not a fixed action sequence
- The agent reads the current game state each turn and decides what to do in the moment
- Strategies should describe WHAT APPROACH to take, not WHICH SPECIFIC ACTIONS to execute

GOOD strategy examples:
- "Black Cat stealth build: prioritize calming actions on comfort targets, use Shadow Cloak for high-suspicion targets, save items for Inspector"
- "Speedrun: convert each target in 2-3 turns at LOW/MED tier to maximize early completion bonus"
- "Item farming: choose stat/item rewards from early targets to prepare for high-value Inspector conversion"

BAD strategy examples:
- "purr → knead → convert → a → purr → knead → convert → a" (this is a fixed action sequence, not a strategy)
- "Select Black Cat" (too vague, doesn't describe the approach)

Each episode has {step_limit} actions. The game has hidden mechanics the agent must discover through experience.

You MUST respond with valid JSON only. Use this exact schema:
{json_schema}"""

        user_prompt = f"""Analyze the agent's gameplay across the following episode summaries.

CURRENT STRATEGY LIST:
{existing_str}

EPISODE SUMMARIES:
{episodes_str}

Phase 1 — Per-Episode Analysis:
For each episode, analyze what worked and what didn't. Focus on:
- Which breed was chosen and how it affected performance
- How the agent handled each target (action choices, convert timing, reward selection)
- What game state conditions (suspicion, stress, energy, influence) led to success or failure
- CRITICAL — Cross-target causality: How did reward choices on EARLIER targets affect LATER targets?
  For example: "Chose item reward from Target 3 → had Salmon Treat for Target 5 → used it to reduce suspicion → got HIGH tier conversion." vs "Chose progress reward from Target 3 → had no items for Target 5 → couldn't reduce suspicion → got kicked out."

Phase 2 — Pattern Discovery:
- Score Comparisons: Why did some episodes score much higher than others? Find the ROOT CAUSE.
  IMPORTANT: Don't just compare total scores. Compare what happened at each TARGET:
  "EP 3 got +40 from Researcher but +10 from Inspector (no items). EP 7 got +20 from Researcher (chose items) but +50 from Inspector (used Salmon Treat). EP 7's trade was +30 net better."
- Best Approach: What overall approach produced the highest score?
- Cross-Target Dependencies: What reward choices on early targets enabled or prevented success on later targets? Which items are most valuable to save for which targets?
- Speed Analysis: How many turns did each episode use? Did episodes that finished faster score higher (due to potential early completion bonuses)?
- Discovered Rules: What GAME MECHANICS have you discovered from the data? Examples:
  * "Targets have different preferences — calming actions work better on some, forceful on others"
  * "Converting too slowly wastes turns and reduces final score"
  * "Suspicion above X level causes the target to kick you out"
  * "Using item X on target Y reduces suspicion by a large amount"
  * "Choosing item rewards early and saving them for high-suspicion targets later yields higher TOTAL score"

Phase 3 — Strategy Modifications:
- **add_branch**: Add a new HIGH-LEVEL strategy. Description should be 1-3 sentences describing the approach, NOT a list of actions.
- **prune**: Remove a consistently underperforming strategy. Required: strategy_id, reason.
- **update_strategy**: Refine a strategy's description based on new discoveries. Required: strategy_id, description, reason.

Guidelines:
- Strategies should encode DISCOVERED KNOWLEDGE about game mechanics, not replay specific episodes
- A good strategy tells the agent WHAT TO PRIORITIZE, not WHAT BUTTONS TO PRESS
- CRITICAL: Strategies MUST include cross-target planning — which rewards to choose on early targets to prepare for later targets. A strategy that says "always pick highest cp" is WORSE than one that says "pick items from Target 3 to use on Target 5"
- When updating a strategy, incorporate new cross-target dependency discoveries into the description
- Prune strategies that consistently score below average after 3+ attempts
- Each strategy should represent a meaningfully different approach

Respond with valid JSON only."""

        max_retries = 5
        for attempt in range(max_retries):
            try:
                raw = chat_completion(
                    model=llm_model,
                    sys_prompt=sys_prompt,
                    prompt=user_prompt,
                    max_tokens=8000,
                    temperature=0.4,
                )
                if not raw:
                    print(f"[FreeReflection] LLM call failed (attempt {attempt + 1}/{max_retries})")
                    continue
                reflection = parse_json_response(raw)
                if not reflection:
                    print(f"[FreeReflection] Failed to parse JSON (attempt {attempt + 1}/{max_retries})")
                    continue

                phase_1 = reflection.get('phase_1_episode_analysis',
                                         reflection.get('phase_1_episode_strategies', []))
                phase_2 = reflection.get('phase_2_reasoning', {})
                phase_3 = reflection.get('phase_3_operations',
                                         reflection.get('phase_3_tree_operations', []))
                if not phase_1:
                    print(f"[FreeReflection] Missing Phase 1 (attempt {attempt + 1}/{max_retries})")
                    continue

                self._print_phase_1(phase_1)
                self._print_phase_2(phase_2)
                self._print_phase_3(phase_3)

                return {
                    'operations': phase_3,
                    'phase_1': phase_1,
                    'phase_2': phase_2,
                }
            except Exception as e:
                print(f"[FreeReflection] Error (attempt {attempt + 1}/{max_retries}): {e}")

        print(f"[FreeReflection] Reflection failed after {max_retries} attempts")
        return {'operations': []}
