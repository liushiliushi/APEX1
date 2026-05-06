"""Milestone × Tree strategy space (Config B).

Extracted from ReflectiveExplorationAgent. Each node is a milestone (goal),
children represent alternative next milestones. MCTS-style traversal via
exploration_method.select() at each level.
"""
import json
import os
import re
from typing import Any, Dict, List, Optional

from .base import StrategySpace


class MilestoneTreeSpace(StrategySpace):

    @property
    def space_type(self) -> str:
        return 'tree'

    def __init__(self, backprop_gamma: float = 0.6,
                 backprop_method: str = 'per_node',
                 backprop_unreached_discount: float = 0.3):
        self.tree: Dict[str, Dict] = {}
        self.next_node_id = 1
        self.total_episodes = 0
        self.backprop_gamma = backprop_gamma
        self.backprop_method = backprop_method
        self.backprop_unreached_discount = backprop_unreached_discount
        self._ensure_root()

    # ------------------------------------------------------------------ #
    #  Tree node helpers
    # ------------------------------------------------------------------ #

    def _ensure_root(self):
        if 'root' not in self.tree:
            self.tree['root'] = {
                'id': 'root',
                'milestone': None,
                'key_action': None,
                'parent': None,
                'children': [],
                'visits': 0,
                'total_reward': 0.0,
                'reward_sq_sum': 0.0,
                'status': 'active',
                'depth': 0,
            }

    def _create_node(self, parent_id: str, milestone: str,
                     key_action: str = '') -> dict:
        node_id = f"node_{self.next_node_id:03d}"
        self.next_node_id += 1

        parent = self.tree.get(parent_id)
        depth = (parent['depth'] + 1) if parent else 1

        node = {
            'id': node_id,
            'milestone': milestone,
            'key_action': key_action or '',
            'parent': parent_id,
            'children': [],
            'visits': 0,
            'total_reward': 0.0,
            'reward_sq_sum': 0.0,
            'status': 'active',
            'depth': depth,
        }
        self.tree[node_id] = node

        if parent and node_id not in parent['children']:
            parent['children'].append(node_id)

        return node

    def _prune_subtree(self, node_id: str):
        node = self.tree.get(node_id)
        if not node:
            return
        node['status'] = 'pruned'
        for child_id in node.get('children', []):
            self._prune_subtree(child_id)
        parent = self.tree.get(node.get('parent'))
        if parent and node_id in parent['children']:
            parent['children'].remove(node_id)

    def _has_similar_sibling(self, parent_id: str, milestone: str) -> bool:
        parent = self.tree.get(parent_id)
        if not parent or not milestone:
            return False
        milestone_lower = milestone.lower().strip()
        for child_id in parent.get('children', []):
            child = self.tree.get(child_id)
            if child and child['status'] == 'active':
                existing = (child.get('milestone') or '').lower().strip()
                if existing == milestone_lower:
                    return True
                if len(existing) > 10 and len(milestone_lower) > 10:
                    if existing in milestone_lower or milestone_lower in existing:
                        return True
        return False

    def _resolve_parent_id(self, raw_id: str) -> Optional[str]:
        if not raw_id:
            return 'root'
        if raw_id in self.tree:
            return raw_id
        raw_lower = raw_id.lower().strip()
        for nid, node in self.tree.items():
            if (node.get('milestone') or '').lower().strip() == raw_lower:
                print(f"[MilestoneTree] Resolved '{raw_id}' -> {nid} (exact match)")
                return nid
        for nid, node in self.tree.items():
            ms = (node.get('milestone') or '').lower().strip()
            if ms and (raw_lower in ms or ms in raw_lower):
                print(f"[MilestoneTree] Resolved '{raw_id}' -> {nid} (partial match)")
                return nid
        return None

    # ------------------------------------------------------------------ #
    #  StrategySpace interface
    # ------------------------------------------------------------------ #

    def has_strategies(self) -> bool:
        return any(
            n.get('status') == 'active' and n['id'] != 'root'
            for n in self.tree.values()
        )

    def get_candidates(self, node_id: str = None) -> List[Dict[str, Any]]:
        node_id = node_id or 'root'
        node = self.tree.get(node_id)
        if not node:
            return []
        return [
            self.tree[cid] for cid in node.get('children', [])
            if self.tree.get(cid, {}).get('status') == 'active'
        ]

    def select_path(self, exploration_method) -> List[str]:
        """MCTS selection: use exploration_method at each level from root to leaf."""
        self._ensure_root()

        if getattr(exploration_method, 'collect_all', False):
            return self._select_all_nodes(exploration_method)

        path = ['root']
        current_id = 'root'

        while True:
            candidates = self.get_candidates(current_id)
            if not candidates:
                break

            parent_visits = max(self.tree[current_id]['visits'], 1)
            idx = exploration_method.select(candidates, parent_visits)
            selected_id = candidates[idx]['id']
            path.append(selected_id)
            current_id = selected_id

        return path

    def _select_all_nodes(self, exploration_method) -> List[str]:
        """BFS traversal collecting all active nodes.

        Siblings ordered by avg_reward descending (best subtree first).
        All nodes at depth d appear before any node at depth d+1,
        so agent tackles all independent top-level tasks first.
        """
        from collections import deque

        path = ['root']
        queue = deque(['root'])

        while queue:
            node_id = queue.popleft()
            candidates = self.get_candidates(node_id)
            if not candidates:
                continue
            parent_visits = max(self.tree[node_id]['visits'], 1)
            scored = []
            for c in candidates:
                s = exploration_method.score(c, parent_visits)
                scored.append((s, c['id']))
            scored.sort(reverse=True)

            for _, child_id in scored:
                path.append(child_id)
                queue.append(child_id)

        return path

    def path_to_strategy(self, path: List[str]) -> Optional[Dict[str, Any]]:
        if not path or path == ['root']:
            return None

        high_level_steps = []
        description_parts = []

        for node_id in path[1:]:
            node = self.tree.get(node_id, {})
            milestone = node.get('milestone', '')
            if milestone:
                high_level_steps.append({
                    'step': milestone,
                    'key_action': node.get('key_action', ''),
                })
                description_parts.append(milestone)

        if not high_level_steps:
            return None

        steps = self._flatten_high_level_steps(high_level_steps)

        if len(description_parts) == 1:
            description = description_parts[0]
        else:
            description = f"{description_parts[0]} -> ... -> {description_parts[-1]}"

        return {
            'id': ' -> '.join(path[1:]),
            'description': description,
            'steps': steps,
            'high_level_steps': high_level_steps,
            'total_reward': 0.0,
            'times_selected': 0,
            'status': 'active',
        }

    def backpropagate(self, path: List[str], score: float,
                      milestone_rewards: Dict[int, float],
                      collect_all: bool = False, **kwargs):
        if collect_all:
            self._backpropagate_collect_all(path, score, milestone_rewards)
        elif self.backprop_method == 'per_node':
            self._backpropagate_per_node(path, score, milestone_rewards)
        else:
            self._backpropagate_frontier(path, score)

    def apply_operations(self, operations: List[Dict[str, Any]],
                         source: str = 'tree_update'):
        self._ensure_root()
        for op_entry in operations:
            if not isinstance(op_entry, dict):
                continue
            op = op_entry.get('op', '')
            reason = op_entry.get('reason', '')

            if op == 'add_child':
                raw_parent = op_entry.get('parent_id') or 'root'
                parent_id = self._resolve_parent_id(raw_parent)
                milestone = op_entry.get('milestone', '')
                key_action = re.sub(r'\s*\[[\+\-]?\d+\]', '', op_entry.get('key_action', ''))

                if not milestone:
                    print(f"[MilestoneTree] Skipping add_child: empty milestone")
                    continue
                if parent_id is None:
                    print(f"[MilestoneTree] Skipping add_child: parent '{raw_parent}' not found")
                    continue
                if self._has_similar_sibling(parent_id, milestone):
                    print(f"[MilestoneTree] Skipping add_child: similar sibling exists under {parent_id}")
                    continue

                node = self._create_node(parent_id, milestone, key_action)
                if source == 'tree_update':
                    try:
                        reward_est = float(op_entry.get('estimated_milestone_reward', 0) or 0)
                    except (ValueError, TypeError):
                        reward_est = 0.0
                    if reward_est > 0:
                        node['visits'] = 1
                        node['total_reward'] = reward_est
                        node['reward_sq_sum'] = reward_est ** 2
                print(f"[MilestoneTree] Added [{node['id']}] '{milestone}' under {parent_id} — {reason}")

            elif op == 'add_branch':
                raw_parent = op_entry.get('parent_id') or 'root'
                parent_id = self._resolve_parent_id(raw_parent)
                milestones = op_entry.get('milestones', [])

                if not milestones:
                    # Fallback: LLM may put single milestone in 'milestone' field instead of 'milestones' array
                    single = op_entry.get('milestone', '')
                    if single:
                        milestones = [{'milestone': single, 'key_action': op_entry.get('key_action', ''), 'estimated_milestone_reward': op_entry.get('estimated_milestone_reward', 0)}]
                    else:
                        print(f"[MilestoneTree] Skipping add_branch: no milestones")
                        continue
                if parent_id is None:
                    print(f"[MilestoneTree] Skipping add_branch: parent '{raw_parent}' not found")
                    continue

                # Normalize milestones: handle both dict and string entries
                normalized = []
                for m in milestones:
                    if isinstance(m, str):
                        normalized.append({'milestone': m, 'key_action': ''})
                    elif isinstance(m, dict):
                        normalized.append(m)
                milestones = normalized

                first_milestone = milestones[0].get('milestone', '') if milestones else ''
                if first_milestone and self._has_similar_sibling(parent_id, first_milestone):
                    print(f"[MilestoneTree] Skipping add_branch: first milestone similar to existing child under {parent_id}")
                    continue

                current_parent = parent_id
                created_nodes = []
                for m in milestones:
                    ms = m.get('milestone', '')
                    if not ms:
                        continue
                    ka = re.sub(r'\s*\[[\+\-]?\d+\]', '', m.get('key_action', ''))
                    node = self._create_node(current_parent, ms, ka)
                    if source == 'tree_update':
                        try:
                            reward_est = float(m.get('estimated_milestone_reward', 0) or 0)
                        except (ValueError, TypeError):
                            reward_est = 0.0
                        if reward_est > 0:
                            node['visits'] = 1
                            node['total_reward'] = reward_est
                            node['reward_sq_sum'] = reward_est ** 2
                    created_nodes.append(node)
                    current_parent = node['id']

                # Gamma propagation from leaf to root
                gamma = self.backprop_gamma
                for i in range(len(created_nodes) - 2, -1, -1):
                    child = created_nodes[i + 1]
                    parent_node = created_nodes[i]
                    if child.get('visits', 0) > 0:
                        if parent_node.get('visits', 0) == 0:
                            parent_node['visits'] = 1
                            parent_node['total_reward'] = 0.0
                        parent_node['total_reward'] += gamma * child['total_reward']

                created_ids = [n['id'] for n in created_nodes]
                if created_ids:
                    print(f"[MilestoneTree] Added branch [{' -> '.join(created_ids)}] under {parent_id} — {reason}")

            elif op == 'prune':
                node_id = op_entry.get('node_id', '')
                if not node_id or node_id == 'root':
                    print(f"[MilestoneTree] Skipping prune: cannot prune root or empty node_id")
                    continue
                if node_id not in self.tree:
                    print(f"[MilestoneTree] Skipping prune: {node_id} not found")
                    continue

                node = self.tree[node_id]
                milestone = node.get('milestone', '?')
                self._prune_subtree(node_id)
                print(f"[MilestoneTree] Pruned [{node_id}] '{milestone}' and subtree — {reason}")

            elif op == 'update_node':
                node_id = op_entry.get('node_id', '')
                if not node_id or node_id not in self.tree:
                    print(f"[MilestoneTree] Skipping update_node: {node_id} not found")
                    continue

                node = self.tree[node_id]
                old_milestone = node.get('milestone', '?')
                new_milestone = op_entry.get('milestone', '')
                if new_milestone:
                    node['milestone'] = new_milestone
                new_key_action = op_entry.get('key_action')
                if new_key_action is not None:
                    node['key_action'] = re.sub(r'\s*\[[\+\-]?\d+\]', '', new_key_action)
                print(f"[MilestoneTree] Updated [{node_id}] '{old_milestone}' -> '{node.get('milestone', '?')}' — {reason}")

    def save(self, path: str):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            data = {
                'version': 2,
                'space_type': 'milestone_tree',
                'nodes': self.tree,
                'next_node_id': self.next_node_id,
                'total_episodes': self.total_episodes,
            }
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"[MilestoneTree] Saved tree with {self.active_count()} active nodes")
        except Exception as e:
            print(f"[MilestoneTree] Error saving: {e}")

    def load(self, path: str):
        if not os.path.exists(path):
            print(f"[MilestoneTree] No existing space found (first run)")
            self._ensure_root()
            return

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            version = data.get('version', 1)
            if version >= 2:
                self.tree = data.get('nodes', {})
                self.next_node_id = data.get('next_node_id', 1)
                self.total_episodes = data.get('total_episodes', 0)
                self._ensure_root()
                # Ensure reward_sq_sum exists on all nodes
                for node in self.tree.values():
                    if 'reward_sq_sum' not in node:
                        node['reward_sq_sum'] = 0.0
                print(f"[MilestoneTree] Loaded tree with {self.active_count()} active nodes")
            else:
                # v1 flat strategy list -> migrate
                print(f"[MilestoneTree] Detected v1 format, migrating to tree...")
                strategies = data.get('strategies', [])
                self.total_episodes = data.get('total_episodes', 0)
                self._migrate_v1_to_v2(strategies)
                print(f"[MilestoneTree] Migration complete")

        except Exception as e:
            print(f"[MilestoneTree] Error loading: {e}")
            self.tree = {}
            self._ensure_root()

    def format_for_reflection(self) -> str:
        if not self.has_strategies():
            return "No milestones discovered yet. The tree is empty (root only)."
        lines = self._format_tree_recursive('root', indent=0, lines=[])
        return "\n".join(lines)

    def format_tree_display(self) -> str:
        header = f"Milestone Tree ({self.active_count()} active nodes)"
        sep = "=" * 60
        lines = [f"\n{sep}", f"  {header}", sep]
        self._format_tree_recursive('root', indent=1, lines=lines)
        lines.append("")
        return "\n".join(lines)

    def active_count(self) -> int:
        return sum(1 for n in self.tree.values()
                   if n.get('status') == 'active' and n['id'] != 'root')

    def evolve(self, episode_summaries, llm_model: str, args):
        """Milestone Tree uses Decision Point Mining for evolution."""
        from ..evolution.decision_point_mining import DecisionPointMining
        evolver = DecisionPointMining()
        result = evolver.reflect(episode_summaries, self, llm_model, args)
        if result:
            operations = result.get('operations', [])
            if operations:
                self.apply_operations(operations, source='dpm')
            return operations
        return []

    def get_milestone_list(self) -> List[str]:
        """Return list of '[node_id] milestone' strings for tree-aware summaries."""
        milestones = []
        for node in self.tree.values():
            if node.get('status') == 'active' and node['id'] != 'root':
                milestone = node.get('milestone', '')
                if milestone:
                    milestones.append(f"[{node['id']}] {milestone}")
        return milestones

    def sync_root_visits(self):
        """Ensure root visits >= sum of children visits."""
        root = self.tree.get('root', {})
        children_total = sum(
            self.tree.get(cid, {}).get('visits', 0)
            for cid in root.get('children', [])
        )
        if children_total > root.get('visits', 0):
            root['visits'] = children_total

    # ------------------------------------------------------------------ #
    #  Backpropagation
    # ------------------------------------------------------------------ #

    def _backpropagate_per_node(self, path: List[str], score: float,
                                milestone_rewards: Dict[int, float]):
        # Free exploration
        if path == ['root']:
            node = self.tree.get('root')
            if node:
                node['visits'] += 1
                node['total_reward'] += score
                node['reward_sq_sum'] = node.get('reward_sq_sum', 0.0) + score ** 2
            return

        gamma = self.backprop_gamma
        num_nodes = len(path) - 1
        node_rewards = [milestone_rewards.get(i, 0.0) for i in range(num_nodes)]

        # leaf to root: value(i) = reward(i) + gamma * value(i+1)
        node_values = [0.0] * num_nodes
        node_values[-1] = node_rewards[-1]
        for i in range(num_nodes - 2, -1, -1):
            node_values[i] = node_rewards[i] + gamma * node_values[i + 1]

        root_value = gamma * node_values[0] if node_values else 0.0
        root_node = self.tree.get('root')
        if root_node:
            root_node['visits'] += 1
            root_node['total_reward'] += root_value
            root_node['reward_sq_sum'] = root_node.get('reward_sq_sum', 0.0) + root_value ** 2

        for i in range(num_nodes):
            node = self.tree.get(path[i + 1])
            if node:
                node['visits'] += 1
                node['total_reward'] += node_values[i]
                node['reward_sq_sum'] = node.get('reward_sq_sum', 0.0) + node_values[i] ** 2

    def _backpropagate_frontier(self, path: List[str], score: float):
        for node_id in path:
            node = self.tree.get(node_id)
            if node:
                node['visits'] += 1
                node['total_reward'] += score
                node['reward_sq_sum'] = node.get('reward_sq_sum', 0.0) + score ** 2

    def _backpropagate_collect_all(self, path: List[str], score: float,
                                    milestone_rewards: Dict[int, float]):
        """Backprop for collect_all: respect tree structure, not linear chain.

        Each node's value = its direct reward + gamma * sum(children values).
        Bottom-up computation through actual tree parent-child relationships.
        """

        gamma = self.backprop_gamma
        node_set = set(path[1:])  # exclude root

        # Map node_id → direct reward from milestone_rewards
        # milestone_rewards is keyed by index in path[1:]
        node_direct = {}
        for i, node_id in enumerate(path[1:]):
            node_direct[node_id] = milestone_rewards.get(i, 0.0)

        # Compute node values bottom-up through tree structure
        node_values = {}

        def compute_value(node_id):
            if node_id in node_values:
                return node_values[node_id]
            direct = node_direct.get(node_id, 0.0)
            children_in_path = [c for c in self.tree.get(node_id, {}).get('children', [])
                                if c in node_set]
            child_sum = sum(compute_value(c) for c in children_in_path)
            value = direct + gamma * child_sum
            node_values[node_id] = value
            return value

        for node_id in path[1:]:
            compute_value(node_id)

        # Update root
        root_children = [c for c in self.tree['root'].get('children', []) if c in node_set]
        root_value = gamma * sum(node_values.get(c, 0.0) for c in root_children)
        root = self.tree['root']
        root['visits'] += 1
        root['total_reward'] += root_value
        root['reward_sq_sum'] = root.get('reward_sq_sum', 0.0) + root_value ** 2

        # Update each node
        for node_id in path[1:]:
            node = self.tree[node_id]
            node['visits'] += 1
            v = node_values[node_id]
            node['total_reward'] += v
            node['reward_sq_sum'] = node.get('reward_sq_sum', 0.0) + v ** 2

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    def _flatten_high_level_steps(self, high_level_steps):
        steps = []
        for i, hl in enumerate(high_level_steps, 1):
            step_name = hl.get('step', '')
            key_action = hl.get('key_action', '')
            if step_name:
                if key_action:
                    steps.append(f"{i}. {step_name} — key action: {key_action}")
                else:
                    steps.append(f"{i}. {step_name}")
        return steps

    def _format_tree_recursive(self, node_id: str, indent: int = 0,
                                lines: list = None) -> list:
        if lines is None:
            lines = []
        node = self.tree.get(node_id)
        if not node:
            return lines

        prefix = "  " * indent
        visits = node.get('visits', 0)
        avg_r = node['total_reward'] / visits if visits > 0 else 0.0

        if node_id == 'root':
            line = f"{prefix}[root] visits={visits} avg_reward={avg_r:.1f}"
        else:
            status_tag = f" [PRUNED]" if node.get('status') == 'pruned' else ""
            milestone = node.get('milestone', '?')
            key_action = node.get('key_action', '')
            ka_text = f" (key: {key_action})" if key_action else ""
            line = f"{prefix}[{node_id}]{status_tag} {milestone}{ka_text}  visits={visits} avg={avg_r:.1f}"

        lines.append(line)

        for child_id in node.get('children', []):
            child = self.tree.get(child_id)
            if child and child.get('status') == 'active':
                self._format_tree_recursive(child_id, indent + 1, lines)

        return lines

    def _migrate_v1_to_v2(self, strategies: list):
        self.tree = {}
        self._ensure_root()
        self.next_node_id = 1

        for strategy in strategies:
            if strategy.get('status') != 'active':
                continue
            hl_steps = strategy.get('high_level_steps', [])
            times_selected = strategy.get('times_selected', 0)
            total_reward = strategy.get('total_reward', 0.0)

            if not hl_steps:
                continue

            parent_id = 'root'
            for i, hl in enumerate(hl_steps):
                milestone = hl.get('step', '')
                key_action = hl.get('key_action', '')

                if not milestone:
                    continue

                if self._has_similar_sibling(parent_id, milestone):
                    parent = self.tree[parent_id]
                    for child_id in parent['children']:
                        child = self.tree.get(child_id)
                        if child and child['status'] == 'active':
                            existing = (child.get('milestone') or '').lower().strip()
                            if existing == milestone.lower().strip() or \
                               existing in milestone.lower().strip() or \
                               milestone.lower().strip() in existing:
                                parent_id = child_id
                                break
                    continue

                node = self._create_node(parent_id, milestone, key_action)
                if i == 0 and times_selected > 0:
                    node['visits'] = times_selected
                    node['total_reward'] = total_reward
                parent_id = node['id']

        root = self.tree['root']
        root['visits'] = sum(s.get('times_selected', 0) for s in strategies if s.get('status') == 'active')
        root['total_reward'] = sum(s.get('total_reward', 0.0) for s in strategies if s.get('status') == 'active')
