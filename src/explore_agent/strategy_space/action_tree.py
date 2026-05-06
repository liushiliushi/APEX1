"""Action × Tree strategy space (Config C).

Single-step action nodes. Classic MCTS-style: the tree is expanded during
gameplay, not during reflection. Each node represents a single game action,
and the tree grows as the agent explores new actions in new states.
"""
import json
import os
from typing import Any, Dict, List, Optional

from .base import StrategySpace


class ActionTreeSpace(StrategySpace):
    """Single-action node tree with MCTS expansion during gameplay."""

    def __init__(self):
        self.tree: Dict[str, Dict] = {}
        self.next_node_id = 1
        self.total_episodes = 0
        self._ensure_root()

    def _ensure_root(self):
        if 'root' not in self.tree:
            self.tree['root'] = {
                'id': 'root',
                'action': None,
                'state_summary': 'game_start',
                'parent': None,
                'children': [],
                'visits': 0,
                'total_reward': 0.0,
                'reward_sq_sum': 0.0,
                'status': 'active',
                'depth': 0,
            }

    def _create_node(self, parent_id: str, action: str,
                     state_summary: str = '') -> dict:
        node_id = f"act_{self.next_node_id:04d}"
        self.next_node_id += 1

        parent = self.tree.get(parent_id)
        depth = (parent['depth'] + 1) if parent else 1

        node = {
            'id': node_id,
            'action': action,
            'state_summary': state_summary,
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
        """Select action path from root to a leaf using exploration method."""
        self._ensure_root()
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

    def path_to_strategy(self, path: List[str]) -> Optional[Dict[str, Any]]:
        if not path or path == ['root']:
            return None

        actions = []
        for node_id in path[1:]:
            node = self.tree.get(node_id, {})
            action = node.get('action', '')
            if action:
                actions.append(action)

        if not actions:
            return None

        high_level_steps = [{'step': a, 'key_action': a} for a in actions]
        steps = [f"{i+1}. {a}" for i, a in enumerate(actions)]

        return {
            'id': ' -> '.join(path[1:]),
            'description': f"Action sequence: {' -> '.join(actions[:3])}{'...' if len(actions) > 3 else ''}",
            'steps': steps,
            'high_level_steps': high_level_steps,
            'total_reward': 0.0,
            'times_selected': 0,
            'status': 'active',
        }

    def expand(self, parent_id: str, action: str, state_summary: str = '') -> str:
        """Expand the tree with a new action node. Returns the new node ID.
        Called during gameplay when the agent takes an action."""
        # Check if action already exists as a child
        parent = self.tree.get(parent_id)
        if parent:
            for child_id in parent.get('children', []):
                child = self.tree.get(child_id)
                if child and child.get('action') == action and child.get('status') == 'active':
                    return child_id

        node = self._create_node(parent_id, action, state_summary)
        return node['id']

    def backpropagate(self, path: List[str], score: float,
                      milestone_rewards: Dict[int, float], **kwargs):
        """Backpropagate episode score to all nodes in the path."""
        for node_id in path:
            node = self.tree.get(node_id)
            if node:
                node['visits'] += 1
                node['total_reward'] += score
                node['reward_sq_sum'] = node.get('reward_sq_sum', 0.0) + score ** 2

    def apply_operations(self, operations: List[Dict[str, Any]]):
        """Apply operations from reflection. Supports: prune, add_branch, add_child, update_node."""
        for op_entry in operations:
            if not isinstance(op_entry, dict):
                continue
            op = op_entry.get('op', '')
            reason = op_entry.get('reason', '')

            if op == 'prune':
                node_id = op_entry.get('node_id', '') or op_entry.get('strategy_id', '')
                if node_id and node_id != 'root' and node_id in self.tree:
                    node = self.tree[node_id]
                    node['status'] = 'pruned'
                    parent = self.tree.get(node.get('parent'))
                    if parent and node_id in parent['children']:
                        parent['children'].remove(node_id)
                    print(f"[ActionTree] Pruned [{node_id}] — {reason}")

            elif op == 'add_branch' or op == 'add_strategy':
                # Convert milestones to a chain of action nodes
                milestones = op_entry.get('milestones', [])
                parent_id = op_entry.get('parent_id', 'root')
                if parent_id not in self.tree:
                    parent_id = 'root'

                # Normalize milestones
                normalized = []
                for m in milestones:
                    if isinstance(m, str):
                        normalized.append({'milestone': m, 'key_action': ''})
                    elif isinstance(m, dict):
                        normalized.append(m)
                milestones = normalized

                if not milestones:
                    continue

                # Build a chain: each milestone becomes a node
                current_parent = parent_id
                first_node_id = None
                for m in milestones:
                    action = m.get('key_action', '') or m.get('milestone', '')
                    if not action:
                        continue
                    # Check if this action already exists as a child
                    existing = None
                    parent_node = self.tree.get(current_parent)
                    if parent_node:
                        for cid in parent_node.get('children', []):
                            child = self.tree.get(cid)
                            if child and child.get('action') == action and child.get('status') == 'active':
                                existing = cid
                                break
                    if existing:
                        current_parent = existing
                    else:
                        node = self._create_node(current_parent, action,
                                                 state_summary=m.get('milestone', ''))
                        if first_node_id is None:
                            first_node_id = node['id']
                        current_parent = node['id']

                if first_node_id:
                    desc = op_entry.get('description', milestones[0].get('milestone', ''))
                    print(f"[ActionTree] Added branch from [{parent_id}] starting at [{first_node_id}] '{desc}' — {reason}")

            elif op == 'add_child':
                parent_id = op_entry.get('parent_id', 'root')
                action = op_entry.get('action', '') or op_entry.get('description', '')
                if not action or parent_id not in self.tree:
                    continue
                # Dedup check
                parent_node = self.tree.get(parent_id)
                if parent_node:
                    existing = False
                    for cid in parent_node.get('children', []):
                        child = self.tree.get(cid)
                        if child and child.get('action') == action and child.get('status') == 'active':
                            existing = True
                            break
                    if existing:
                        continue
                node = self._create_node(parent_id, action,
                                         state_summary=op_entry.get('state_summary', ''))
                print(f"[ActionTree] Added child [{node['id']}] '{action}' under [{parent_id}] — {reason}")

            elif op == 'update_node':
                node_id = op_entry.get('node_id', '') or op_entry.get('strategy_id', '')
                if node_id in self.tree:
                    node = self.tree[node_id]
                    if op_entry.get('action'):
                        node['action'] = op_entry['action']
                    if op_entry.get('state_summary'):
                        node['state_summary'] = op_entry['state_summary']
                    print(f"[ActionTree] Updated [{node_id}] — {reason}")

    def save(self, path: str):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            data = {
                'version': 2,
                'space_type': 'action_tree',
                'nodes': self.tree,
                'next_node_id': self.next_node_id,
                'total_episodes': self.total_episodes,
            }
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"[ActionTree] Saved tree with {self.active_count()} active nodes")
        except Exception as e:
            print(f"[ActionTree] Error saving: {e}")

    def load(self, path: str):
        if not os.path.exists(path):
            print(f"[ActionTree] No existing space found (first run)")
            self._ensure_root()
            return

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if data.get('space_type') == 'action_tree':
                self.tree = data.get('nodes', {})
                self.next_node_id = data.get('next_node_id', 1)
                self.total_episodes = data.get('total_episodes', 0)
                self._ensure_root()
                print(f"[ActionTree] Loaded tree with {self.active_count()} active nodes")
            else:
                print(f"[ActionTree] Incompatible space type, starting fresh")
                self._ensure_root()
        except Exception as e:
            print(f"[ActionTree] Error loading: {e}")
            self.tree = {}
            self._ensure_root()

    def format_for_reflection(self) -> str:
        if not self.has_strategies():
            return "No actions explored yet. The action tree is empty."
        lines = self._format_recursive('root', indent=0)
        return "\n".join(lines)

    def format_tree_display(self) -> str:
        header = f"Action Tree ({self.active_count()} active nodes)"
        sep = "=" * 60
        lines = [f"\n{sep}", f"  {header}", sep]
        lines.extend(self._format_recursive('root', indent=1))
        lines.append("")
        return "\n".join(lines)

    def active_count(self) -> int:
        return sum(1 for n in self.tree.values()
                   if n.get('status') == 'active' and n['id'] != 'root')

    def get_milestone_list(self) -> List[str]:
        """Return list of action node descriptions for tree-aware summaries."""
        return [f"[{n['id']}] {n.get('action', '')}" for n in self.tree.values()
                if n.get('status') == 'active' and n['id'] != 'root']

    def _format_recursive(self, node_id: str, indent: int = 0) -> List[str]:
        lines = []
        node = self.tree.get(node_id)
        if not node:
            return lines

        prefix = "  " * indent
        visits = node.get('visits', 0)
        avg_r = node['total_reward'] / visits if visits > 0 else 0.0

        if node_id == 'root':
            lines.append(f"{prefix}[root] visits={visits} avg={avg_r:.1f}")
        else:
            action = node.get('action', '?')
            lines.append(f"{prefix}[{node_id}] {action}  visits={visits} avg={avg_r:.1f}")

        for child_id in node.get('children', []):
            child = self.tree.get(child_id)
            if child and child.get('status') == 'active':
                lines.extend(self._format_recursive(child_id, indent + 1))

        return lines
