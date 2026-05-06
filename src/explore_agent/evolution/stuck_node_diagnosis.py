"""Stuck Node Diagnosis — analyze underperforming nodes and inject actionable guidance.

After reflection, scans the DAG for nodes with many visits but non-positive reward.
Uses LLM to diagnose why they keep failing and generates guidance that gets shown
to the agent next time it encounters these milestones.
"""
from typing import Any, Dict, List

from ..openai_helpers_proxy import chat_completion, parse_json_response


class StuckNodeDiagnosis:
    """Diagnose nodes with many visits but non-positive reward."""

    def diagnose(self, strategy_space, llm_model: str, current_episode: int,
                 reflect_interval: int = 2) -> int:
        """Scan for stuck nodes and inject LLM-generated diagnostics.

        Returns count of nodes diagnosed.
        """
        if not hasattr(strategy_space, 'nodes'):
            return 0

        stuck_nodes = self._find_stuck_nodes(
            strategy_space, current_episode, reflect_interval)

        if not stuck_nodes:
            print("[StuckNodeDiagnosis] No stuck nodes found")
            return 0

        print(f"[StuckNodeDiagnosis] Found {len(stuck_nodes)} stuck nodes, diagnosing one by one...")

        count = 0
        for sn in stuck_nodes:
            sys_prompt, user_prompt = self._build_prompt(sn)
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    raw = chat_completion(
                        model=llm_model,
                        sys_prompt=sys_prompt,
                        prompt=user_prompt,
                        max_tokens=1000,
                        temperature=0.3,
                    )
                    if not raw:
                        continue
                    result = parse_json_response(raw)
                    if not result:
                        continue
                    count += self._apply_diagnostics(
                        [result], strategy_space, current_episode)
                    break
                except Exception as e:
                    print(f"[StuckNodeDiagnosis] Error on {sn['node_id']} (attempt {attempt + 1}): {e}")

        print(f"[StuckNodeDiagnosis] Diagnosed {count} stuck nodes")
        return count

    def _find_stuck_nodes(self, strategy_space, current_episode: int,
                          reflect_interval: int) -> List[Dict]:
        """Find nodes with visits >= 3 and avg_reward <= 0, excluding recently diagnosed."""
        stuck = []
        for nid, node in strategy_space.nodes.items():
            if nid == 'root' or node.get('status') != 'active':
                continue
            visits = node.get('visits', 0)
            if visits < 3:
                continue
            avg = node.get('total_reward', 0) / visits
            if avg > 0:
                continue

            # Skip if recently diagnosed
            diag = node.get('diagnostic')
            if diag:
                diagnosed_at = diag.get('diagnosed_at_episode', 0)
                if current_episode - diagnosed_at < 2 * reflect_interval:
                    continue

            # Gather context
            downstream = []
            if hasattr(strategy_space, 'get_downstream_nodes'):
                downstream = strategy_space.get_downstream_nodes(nid)

            dep_ids = strategy_space._dep_node_ids(node.get('deps', []))
            upstream = []
            for did in dep_ids:
                dep_node = strategy_space.nodes.get(did)
                if dep_node:
                    dep_visits = dep_node.get('visits', 0)
                    dep_avg = dep_node.get('total_reward', 0) / dep_visits if dep_visits > 0 else 0
                    upstream.append({
                        'node_id': did,
                        'milestone': dep_node.get('milestone', ''),
                        'avg_reward': round(dep_avg, 2),
                    })

            stuck.append({
                'node_id': nid,
                'milestone': node.get('milestone', ''),
                'key_action': node.get('key_action', ''),
                'visits': visits,
                'avg_reward': round(avg, 2),
                'attempt_notes': (node.get('attempt_notes') or [])[-5:],
                'downstream': downstream,
                'upstream': upstream,
            })
        return stuck

    def _build_prompt(self, sn: Dict) -> tuple:
        node_text = f"\n--- {sn['node_id']}: {sn['milestone']} ---\n"
        node_text += f"  key_action: {sn['key_action']}\n"
        node_text += f"  visits: {sn['visits']}, avg_reward: {sn['avg_reward']}\n"
        if sn['attempt_notes']:
            node_text += "  Recent attempt notes:\n"
            for note in sn['attempt_notes']:
                node_text += f"    - {note}\n"
        if sn['upstream']:
            node_text += "  Depends on:\n"
            for u in sn['upstream']:
                node_text += f"    - {u['node_id']}: {u['milestone']} (avg: {u['avg_reward']})\n"
        if sn['downstream']:
            node_text += "  Downstream (depends on this):\n"
            for d in sn['downstream']:
                node_text += f"    - {d['node_id']}: {d['milestone']} (avg: {d['avg_reward']})\n"

        sys_prompt = """You are an expert at diagnosing why an agent keeps failing at certain milestones.

Analyze the stuck node's attempt notes and context to determine:
1. WHY the agent keeps failing (the root cause)
2. What SPECIFIC action the agent should take next time (concrete guidance)
3. Whether the failure is because a PREREQUISITE is missing — identify which upstream dependency is unmet.

Respond with valid JSON only:
{
    "node_id": "node_XXX",
    "diagnosis": "1-2 sentences explaining the root cause of failure",
    "guidance": "Specific actionable advice for next attempt",
    "is_setup_node": false,
    "add_dep": "node_YYY or null"
}

Rules:
- diagnosis should explain the PATTERN of failure, not just restate the notes
- guidance should be CONCRETE and actionable
- add_dep: ONLY set when the action REQUIRES another node's outcome. Set null otherwise."""

        user_prompt = f"""Analyze this stuck node and diagnose why it keeps failing:

{node_text}

Provide a diagnosis and actionable guidance."""

        return sys_prompt, user_prompt

    def _apply_diagnostics(self, diagnostics: List[Dict], strategy_space,
                           current_episode: int) -> int:
        """Write diagnostic results into DAG nodes. Returns count applied."""
        count = 0
        for diag in diagnostics:
            node_id = diag.get('node_id', '')
            node = strategy_space.nodes.get(node_id)
            if not node:
                continue
            node['diagnostic'] = {
                'diagnosis': diag.get('diagnosis', ''),
                'guidance': diag.get('guidance', ''),
                'is_setup_node': diag.get('is_setup_node', False),
                'diagnosed_at_episode': current_episode,
            }
            print(f"  [{node_id}] {diag.get('diagnosis', '')[:80]}")

            # Add missing prerequisite dependency if diagnosed
            add_dep = diag.get('add_dep')
            if add_dep and add_dep in strategy_space.nodes and add_dep != node_id:
                current_deps = strategy_space._dep_node_ids(node.get('deps', []))
                if add_dep not in current_deps:
                    node['deps'] = node.get('deps', []) + [add_dep]
                    print(f"  [{node_id}] Added prerequisite dep: {add_dep} "
                          f"({strategy_space.nodes[add_dep].get('milestone', '')})")

            count += 1
        return count
