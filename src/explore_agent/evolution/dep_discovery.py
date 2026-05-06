"""Retroactive Dependency Discovery for Milestone DAG.

After each episode, compares failed milestones (that have succeeded before)
against their successful game_history to identify missing dependencies
and wrong/incomplete key_actions.
Outputs update_node operations to patch deps and key_actions.
"""
from typing import Any, Dict, List, Optional

from ..openai_helpers_proxy import chat_completion, parse_json_response


class DepDiscovery:
    """Discover missing deps by comparing success vs failure game histories."""

    def __init__(self):
        # Cooldown: track (node_id -> last_analyzed_episode) to avoid spam
        self._last_analyzed: Dict[str, int] = {}
        self._cooldown = 3  # skip if analyzed within last N episodes

    def run(self, current_path: List[str], milestone_rewards: Dict[int, float],
            strategy_space, game_history: List[Dict], episode_num: int,
            llm_model: str, history_loader=None) -> List[Dict[str, Any]]:
        """Run dep discovery for failed milestones in current episode.

        Args:
            current_path: path selected this episode (e.g. ['root', 'node_001', ...])
            milestone_rewards: {path_index: reward} from summary parsing
            strategy_space: MilestoneDAGSpace instance
            game_history: full game_history of current episode
            episode_num: current episode number
            llm_model: LLM model identifier
            history_loader: callable(episode_num) -> List[Dict] to load saved episode histories

        Returns:
            list of update_node operations to apply
        """
        if not hasattr(strategy_space, 'nodes') or len(current_path) <= 1:
            return []

        # Find failed milestones that have success_game_history
        failed_nodes = []
        for idx, node_id in enumerate(current_path[1:]):
            node = strategy_space.nodes.get(node_id)
            if not node:
                continue

            # Check this milestone has succeeded before
            if node.get('max_reward', 0.0) <= 0:
                continue

            # Check this milestone failed in current episode
            reward = milestone_rewards.get(idx, 0.0)
            if reward > 0:
                continue  # succeeded, no need to analyze

            # Check we have a success history to compare against
            if not node.get('success_episode') and not node.get('success_game_history'):
                continue

            # Cooldown check
            last = self._last_analyzed.get(node_id, -999)
            if episode_num - last < self._cooldown:
                continue

            failed_nodes.append((idx, node_id, node))

        if not failed_nodes:
            return []

        print(f"[DepDiscovery] EP{episode_num}: {len(failed_nodes)} failed milestone(s) with success history — analyzing")

        all_ops = []
        all_lessons = []
        # Condense current episode history (skip full_response)
        current_history = [
            {k: v for k, v in entry.items() if k != 'full_response'}
            for entry in game_history
        ]

        for idx, node_id, node in failed_nodes:
            self._last_analyzed[node_id] = episode_num
            ops, lessons = self._analyze_single_node(
                node_id, node, current_history, strategy_space, llm_model,
                history_loader=history_loader)
            all_ops.extend(ops)
            all_lessons.extend(lessons)

        # Apply lessons to global_lessons
        if all_lessons:
            self._apply_lessons(all_lessons, strategy_space, episode_num)

        return all_ops

    def _analyze_single_node(self, node_id: str, node: Dict,
                             current_history: List[Dict],
                             strategy_space, llm_model: str,
                             history_loader=None):
        """Compare success vs failure for a single milestone. Returns (ops, lessons)."""

        # Load success history: prefer success_episode (file-based), fall back to inline
        success_history = None
        if node.get('success_episode') and history_loader:
            success_history = history_loader(node['success_episode'])
        if not success_history:
            success_history = node.get('success_game_history')
        if not success_history:
            print(f"[DepDiscovery] [{node_id}]: no success history available, skipping")
            return [], []
        dag_str = strategy_space.format_for_reflection()

        # Format histories
        current_str = self._format_history(current_history)
        success_str = self._format_history(success_history)

        milestone = node.get('milestone', '?')
        current_deps = node.get('deps', [])
        deps_str = ', '.join(current_deps) if current_deps else 'none'

        key_action = node.get('key_action', '')
        key_action_str = ', '.join(key_action) if isinstance(key_action, list) else str(key_action)

        # Build dependency info string for the analyzed node
        dep_nodes_info = []
        for dep_id in current_deps:
            dep_node = strategy_space.nodes.get(dep_id, {})
            if dep_node:
                dep_ka = dep_node.get('key_action', '')
                dep_milestone = dep_node.get('milestone', '?')
                dep_visits = dep_node.get('visits', 0)
                dep_avg = dep_node.get('total_reward', 0) / max(dep_visits, 1)
                dep_nodes_info.append(
                    f"  [{dep_id}] {dep_milestone} — key_action: [{dep_ka}], "
                    f"visits={dep_visits}, avg_reward={dep_avg:.1f}")
        dep_details_str = '\n'.join(dep_nodes_info) if dep_nodes_info else '  (none)'

        sys_prompt = f"""You are an expert analyzer for text adventure games.
You are examining why a milestone FAILED in the current episode but SUCCEEDED in a previous episode.

ANALYSIS METHOD — follow these steps in order:
1. REWARD MOMENT: Find the exact step where the reward was earned in the successful episode. Note the game state at that moment (what room, what conditions were active).
2. MINIMUM PATH: Working backwards from the reward, identify only the actions that were NECESSARY to reach that state. An action that was later reversed (e.g. turn off X → turn on X) had NO net effect and should be EXCLUDED.
3. DIVERGENCE: In the failed episode, what was different at the point where the reward action should have been executed? Did the agent not execute the correct action, or was a required state/item missing?
4. ROOT CAUSE: Is the failure due to:
   a) A missing dependency on this node?
   b) A wrong key_action on this node?
   c) A wrong key_action on one of its DEPENDENCY nodes (listed below)? For example, if a dependency node's key_action has never worked (avg_reward=0 after many visits), the dependency's key_action is likely wrong.
   d) Just bad execution?

DEPENDENCY NODES of [{node_id}]:
{dep_details_str}

IMPORTANT RULES:
- Only propose deps from EXISTING nodes in the DAG. Do NOT invent new nodes.
- A dep means this milestone is IMPOSSIBLE without that prerequisite. This includes both causal requirements (need an item, need to unlock something) AND geographic requirements (must pass through a room that is blocked or requires a prior action to access).
- KEY_ACTION MUST ONLY contain actions performed AT THE DESTINATION — the core interaction (take, open, push, pull, dig, read, etc.). NEVER include navigation/movement commands (go north, south, east, west, up, down, etc.) in key_action. Navigation to the destination is handled separately by the guidance system. If the agent failed because it didn't reach the destination, that is a navigation failure, NOT a key_action problem — output no changes.
- For key_action: propose the MINIMUM actions needed. Exclude any action that was reversed before the reward.
- If the failure was due to bad luck, wrong navigation, or time pressure, output no changes.
- CRITICAL: If a dependency node has avg_reward=0.0 after multiple visits, its key_action is probably WRONG. Check the successful episode to find what action ACTUALLY worked for that dependency, and propose a fix via dep_key_action_fix.

Respond with valid JSON:
{{
    "reward_moment": "The exact step where reward was earned in the successful episode. Describe the game state at that moment.",
    "minimum_actions": "Working backwards from the reward, what is the minimum action sequence needed? Exclude reversed/undone actions.",
    "divergence": "In the failed episode, what was different compared to the successful reward moment?",
    "analysis": "Based on the above, explain the root cause of failure.",
    "missing_dep": "node_id of the missing dependency, or null if none",
    "reason": "Why this dep is causally required",
    "updated_key_action": ["action1", "action2", "..."] or null. MINIMUM sequence only.",
    "key_action_reason": "Why the key_action needs updating, or null",
    "dep_key_action_fix": {{"node_id": "the dependency node to fix", "updated_key_action": ["action1", "..."], "reason": "why"}} or null. Use this when the root cause is a DEPENDENCY node having a wrong key_action.",
    "lesson": "General lesson if any. Format: [CATEGORY] lesson text. Null if none."
}}"""

        user_prompt = f"""MILESTONE UNDER ANALYSIS:
  Node: [{node_id}] {milestone}
  Current key_action: [{key_action_str}]
  Current deps: [{deps_str}]

CURRENT DAG:
{dag_str}

SUCCESSFUL EPISODE (this milestone was achieved):
{success_str}

FAILED EPISODE (this milestone was NOT achieved):
{current_str}

Compare these two episodes:
1. Was the failure caused by a missing dependency — a prerequisite from the DAG that should be in [{node_id}]'s deps but isn't?
2. Is the current key_action wrong or incomplete compared to what the successful episode actually did?

Respond with valid JSON only."""

        try:
            raw = chat_completion(
                model=llm_model,
                sys_prompt=sys_prompt,
                prompt=user_prompt,
                max_tokens=2000,
                temperature=0.3,
            )
            if not raw:
                print(f"[DepDiscovery] LLM call failed for [{node_id}]")
                return [], []

            result = parse_json_response(raw)
            if not result:
                print(f"[DepDiscovery] JSON parse failed for [{node_id}]")
                return [], []

            analysis = result.get('analysis', '')
            missing_dep = result.get('missing_dep')
            reason = result.get('reason', '')
            updated_key_action = result.get('updated_key_action')
            key_action_reason = result.get('key_action_reason', '')

            print(f"[DepDiscovery] [{node_id}] '{node.get('milestone','')}': {analysis}")

            ops = []
            lessons = []

            # --- Handle missing dep ---
            has_dep_update = False
            new_deps = None
            if missing_dep and missing_dep != 'null':
                if missing_dep not in strategy_space.nodes:
                    print(f"[DepDiscovery] [{node_id}]: proposed dep '{missing_dep}' not in DAG, skipping")
                elif strategy_space.nodes[missing_dep].get('max_reward', 0.0) < 0:
                    print(f"[DepDiscovery] [{node_id}]: proposed dep '{missing_dep}' is a penalty node, skipping")
                elif missing_dep in node.get('deps', []):
                    print(f"[DepDiscovery] [{node_id}]: '{missing_dep}' already in deps, skipping")
                else:
                    new_deps = node.get('deps', []) + [missing_dep]
                    has_dep_update = True
                    dep_node = strategy_space.nodes.get(missing_dep, {})
                    print(f"[DepDiscovery] [{node_id}]: adding dep [{missing_dep}] '{dep_node.get('milestone','')}'")

            # --- Handle key_action update ---
            has_key_action_update = False
            if updated_key_action and updated_key_action != 'null' and isinstance(updated_key_action, list):
                # Only update if meaningfully different from current
                current_ka = node.get('key_action', [])
                if isinstance(current_ka, str):
                    current_ka = [a.strip() for a in current_ka.split(',')]
                if updated_key_action != current_ka:
                    has_key_action_update = True
                    print(f"[DepDiscovery] [{node_id}]: updating key_action: {current_ka} -> {updated_key_action}")
                    if key_action_reason:
                        print(f"[DepDiscovery] [{node_id}]: reason: {key_action_reason}")

            # --- Handle dep_key_action_fix (fix a dependency node's key_action) ---
            dep_fix = result.get('dep_key_action_fix')
            if dep_fix and isinstance(dep_fix, dict) and dep_fix.get('node_id'):
                dep_fix_id = dep_fix['node_id']
                dep_fix_ka = dep_fix.get('updated_key_action')
                dep_fix_reason = dep_fix.get('reason', '')
                if dep_fix_id in strategy_space.nodes and dep_fix_ka and isinstance(dep_fix_ka, list):
                    dep_node = strategy_space.nodes[dep_fix_id]
                    current_dep_ka = dep_node.get('key_action', [])
                    if isinstance(current_dep_ka, str):
                        current_dep_ka = [a.strip() for a in current_dep_ka.split(',')]
                    if dep_fix_ka != current_dep_ka:
                        print(f"[DepDiscovery] [{node_id}]: fixing dependency [{dep_fix_id}] key_action: {current_dep_ka} -> {dep_fix_ka}")
                        if dep_fix_reason:
                            print(f"[DepDiscovery] [{dep_fix_id}]: reason: {dep_fix_reason}")
                        ops.append({
                            'op': 'update_node',
                            'node_id': dep_fix_id,
                            'key_action': ', '.join(dep_fix_ka),
                            'reason': f'dep_discovery: {dep_fix_reason}',
                        })

            # --- Build combined op if any update ---
            if has_dep_update or has_key_action_update:
                op = {
                    'op': 'update_node',
                    'node_id': node_id,
                    'reason': f'dep_discovery: {reason or key_action_reason}',
                }
                if has_dep_update:
                    op['deps'] = new_deps
                if has_key_action_update:
                    op['key_action'] = ', '.join(updated_key_action)
                ops.append(op)
            else:
                print(f"[DepDiscovery] [{node_id}]: no missing dep or key_action fix found")
                # Check for lesson
                lesson_text = result.get('lesson')
                if lesson_text and lesson_text != 'null':
                    import re
                    m = re.match(r'\[(\w+)\]\s*(.*)', lesson_text)
                    if m:
                        category, text = m.group(1), m.group(2)
                    else:
                        category, text = 'MECHANIC', lesson_text
                    lessons.append({'category': category, 'lesson': text,
                                    'evidence': analysis, 'confidence': 'medium'})

            return ops, lessons

        except Exception as e:
            print(f"[DepDiscovery] Error analyzing [{node_id}]: {e}")
            return [], []

    def _apply_lessons(self, lessons: List[Dict], strategy_space, episode_num: int):
        """Add lessons to strategy_space.global_lessons."""
        if not hasattr(strategy_space, 'global_lessons'):
            strategy_space.global_lessons = []

        max_id = max((l['id'] for l in strategy_space.global_lessons), default=0)
        for lesson in lessons:
            max_id += 1
            strategy_space.global_lessons.append({
                'id': max_id,
                'category': lesson['category'],
                'lesson': lesson['lesson'],
                'evidence': lesson.get('evidence', ''),
                'confidence': lesson.get('confidence', 'medium'),
                'added_at_episode': episode_num,
            })
            print(f"[DepDiscovery] Lesson: [{lesson['category']}] {lesson['lesson']}")

    def _format_history(self, history: List[Dict]) -> str:
        """Format game_history into readable text for LLM comparison."""
        lines = []
        for i, entry in enumerate(history):
            action = entry.get('action', '')
            state = entry.get('state', '')
            reward = entry.get('reward', 0)
            score = entry.get('score', 0)

            parts = [f"Step {i+1}:"]
            if action:
                parts.append(f"Action: {action}")
            if state:
                # Truncate very long states
                state_str = state[:500] + '...' if len(state) > 500 else state
                parts.append(f"Observation: {state_str}")
            if reward:
                parts.append(f"Reward: {reward:+d}")
            parts.append(f"Score: {score}")
            lines.append(' | '.join(parts))
        return '\n'.join(lines)
