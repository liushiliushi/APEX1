"""Milestone × DAG strategy space.

Replaces the strict parent-child tree with a DAG (directed acyclic graph)
where nodes declare explicit dependencies via `deps` lists.
All deps are hard locks — the milestone cannot be pursued until its
deps are completed.

Independent milestones have deps=[] and can be pursued in any order.

Selection walks through dependency layers using UCB at each level.
Backpropagation uses γ-discounted returns along the selected path,
so dependency nodes inherit value from downstream milestones.
"""
import json
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from .base import StrategySpace


class MilestoneDAGSpace(StrategySpace):

    @property
    def space_type(self) -> str:
        return 'dag'

    def __init__(self, backprop_gamma: float = 0.6,
                 backprop_method: str = 'linear',
                 backprop_unreached_discount: float = 0.3,
                 **_kwargs):
        self.nodes: Dict[str, Dict] = {}
        self.next_node_id = 1
        self.total_episodes = 0
        self.backprop_gamma = backprop_gamma
        self.backprop_method = backprop_method
        self.backprop_unreached_discount = backprop_unreached_discount
        self.global_lessons: List[Dict] = []
        self._ensure_root()

    # ------------------------------------------------------------------ #
    #  Node helpers
    # ------------------------------------------------------------------ #

    def _ensure_root(self):
        if 'root' not in self.nodes:
            self.nodes['root'] = {
                'id': 'root',
                'milestone': None,
                'key_action': None,
                'deps': [],
                'visits': 0,
                'total_reward': 0.0,
                'reward_sq_sum': 0.0,
                'status': 'active',
            }

    @staticmethod
    def _sanitize_key_action(key_action: str) -> str:
        """Clean up key_action formatting. Preserves all actions including navigation."""
        if not key_action:
            return ''
        # Strip [from ...] prefix (navigation to destination is handled by BFS routing)
        key_action = re.sub(r'\[from\s+.+?\]\s*', '', key_action, flags=re.IGNORECASE).strip()
        # Strip trailing → explore
        key_action = re.sub(r'[→\->]+\s*explore\s*$', '', key_action, flags=re.IGNORECASE).strip()
        return key_action

    def _create_node(self, milestone: str, key_action: str = '',
                     deps: List = None) -> dict:
        node_id = f"node_{self.next_node_id:03d}"
        self.next_node_id += 1

        node = {
            'id': node_id,
            'milestone': milestone,
            'key_action': self._sanitize_key_action(key_action),
            'destination': -1,
            'deps': self._normalize_deps(deps),
            'visits': 0,
            'total_reward': 0.0,
            'reward_sq_sum': 0.0,
            'max_reward': 0.0,
            'status': 'active',
        }
        self.nodes[node_id] = node
        return node

    def add_attempt_note(self, node_id: str, note: str, max_notes: int = 5):
        """Append a brief attempt note to a node's history. Keeps last `max_notes`."""
        node = self.nodes.get(node_id)
        if not node:
            return
        if 'attempt_notes' not in node:
            node['attempt_notes'] = []
        node['attempt_notes'].append(note)
        if len(node['attempt_notes']) > max_notes:
            node['attempt_notes'] = node['attempt_notes'][-max_notes:]

    def _prune_node(self, node_id: str):
        """Prune a node. Dependents are NOT cascade-deleted — instead,
        the pruned node is removed from their deps so they stay active."""
        node = self.nodes.get(node_id)
        if not node:
            return
        node['status'] = 'pruned'
        # Remove this node from dependents' deps (un-block them)
        for nid, n in list(self.nodes.items()):
            if nid == 'root' or n['status'] != 'active':
                continue
            dep_ids = self._normalize_deps(n.get('deps', []))
            if node_id in dep_ids:
                n['deps'] = [d for d in n.get('deps', [])
                             if self._resolve_node_id(
                                 d.get('node', d) if isinstance(d, dict) else d
                             ) != node_id]
                print(f"[MilestoneDAG] Removed pruned dep {node_id} from {nid}")

    def _find_similar_node(self, milestone: str, deps: List = None,
                           key_action: str = '', destination: int = -1) -> Optional[str]:
        """Find a similar active node. Returns node ID if found, None otherwise.

        Duplicate if ANY of these conditions match:
        1. Same milestone text (exact) + same deps
        2. Milestone substring containment + same deps
        3. Same key_action + same destination (structural match, ignores deps)
        4. Same key_action + same deps (structural match, ignores destination)
        """
        if not milestone:
            return None
        milestone_lower = milestone.lower().strip()
        check_dep_ids = sorted(self._normalize_deps(deps)) if deps is not None else None
        new_key_action = self._sanitize_key_action(key_action).lower().strip()

        for node in self.nodes.values():
            if node['id'] == 'root' or node['status'] != 'active':
                continue
            existing = (node.get('milestone') or '').lower().strip()
            node_dep_ids = sorted(self._normalize_deps(node.get('deps', [])))

            # --- Text match (original logic): requires same deps ---
            if existing:
                same_deps = (check_dep_ids is not None and node_dep_ids == check_dep_ids)
                if check_dep_ids is None or same_deps:
                    if existing == milestone_lower:
                        return node['id']
                    if len(existing) > 10 and len(milestone_lower) > 10:
                        if existing in milestone_lower or milestone_lower in existing:
                            return node['id']

            # --- Structural match: key_action + destination ---
            if new_key_action:
                existing_ka = self._sanitize_key_action(node.get('key_action', '')).lower().strip()
                if existing_ka == new_key_action:
                    # Same action from same room → duplicate regardless of milestone text or deps
                    if destination > 0 and node.get('destination', -1) == destination:
                        return node['id']
                    # Same action with same deps → duplicate regardless of destination
                    if check_dep_ids is not None and node_dep_ids == check_dep_ids:
                        return node['id']

        return None

    def _is_abandoned(self, node_id: str) -> bool:
        """Check if a node meets abandon criteria (v>=threshold, avg=0, no rewarding descendants).

        Exploration nodes get a higher threshold (5 vs 3) since they need more
        attempts to discover new areas.
        """
        node = self.nodes.get(node_id)
        if not node or node['id'] == 'root' or node.get('status') != 'active':
            return False
        visits = node.get('visits', 0)
        # Exploration nodes need more patience
        milestone = node.get('milestone', '')
        key_action = node.get('key_action', '')
        is_explore = ('explore' in milestone.lower()
                      or '→ explore' in key_action or '-> explore' in key_action)
        threshold = 5 if is_explore else 3
        if visits < threshold:
            return False
        if node.get('total_reward', 0) > 0:
            return False
        return not self._has_rewarding_descendants(node_id)

    def _resolve_node_id(self, raw_id: str) -> Optional[str]:
        """Resolve a raw node reference (id or milestone text) to a node id."""
        if not raw_id:
            return 'root'
        if raw_id in self.nodes:
            return raw_id
        raw_lower = raw_id.lower().strip()
        for nid, node in self.nodes.items():
            if (node.get('milestone') or '').lower().strip() == raw_lower:
                print(f"[MilestoneDAG] Resolved '{raw_id}' -> {nid} (exact match)")
                return nid
        for nid, node in self.nodes.items():
            ms = (node.get('milestone') or '').lower().strip()
            if ms and (raw_lower in ms or ms in raw_lower):
                print(f"[MilestoneDAG] Resolved '{raw_id}' -> {nid} (partial match)")
                return nid
        return None

    def _deps_for_parent(self, parent_id: str) -> List[str]:
        """Convert a tree-style parent_id into a DAG deps list.

        - parent_id == 'root' or None → deps=[] (independent node)
        - parent_id == some node → deps=[parent_id]
        """
        if not parent_id or parent_id == 'root':
            return []
        return [parent_id]

    @staticmethod
    def _normalize_deps(deps_list) -> List[str]:
        """Convert deps to standard List[str] of node IDs.

        Handles: None, str, list of str, list of dicts with 'node' key (legacy).
        """
        if not deps_list:
            return []
        if isinstance(deps_list, str):
            return [deps_list]
        result = []
        for d in deps_list:
            if isinstance(d, str):
                result.append(d)
            elif isinstance(d, dict) and 'node' in d:
                result.append(d['node'])
        return result

    @staticmethod
    def _avg_reward(node: Dict) -> float:
        visits = node.get('visits', 0)
        return node.get('total_reward', 0.0) / visits if visits > 0 else 0.0

    @staticmethod
    def _reset_node_stats(node: Dict):
        node['visits'] = 0
        node['total_reward'] = 0.0
        node['reward_sq_sum'] = 0.0
        node['max_reward'] = 0.0

    @staticmethod
    def _parse_destination(raw) -> int:
        if raw is None or raw == '' or raw == -1:
            return -1
        try:
            return int(raw)
        except (ValueError, TypeError):
            return -1

    @staticmethod
    def _set_danger_warning(node: Dict, op_entry: Dict):
        dw = op_entry.get('danger_warning')
        if dw and dw != 'null':
            node['danger_warning'] = dw

    @staticmethod
    def _apply_estimated_reward(node: Dict, raw_reward):
        """Seed a new node with an estimated reward from TreeUpdate (source='tree_update')."""
        if raw_reward is None:
            return
        try:
            reward_est = float(raw_reward)
        except (ValueError, TypeError):
            return
        node['visits'] = 1
        node['total_reward'] = reward_est
        node['reward_sq_sum'] = reward_est ** 2
        node['max_reward'] = reward_est

    # ------------------------------------------------------------------ #
    #  StrategySpace interface
    # ------------------------------------------------------------------ #

    def has_strategies(self) -> bool:
        return any(
            n.get('status') == 'active' and n['id'] != 'root'
            for n in self.nodes.values()
        )

    def get_candidates(self, node_id: str = None) -> List[Dict[str, Any]]:
        """Return active nodes whose deps are all satisfied.

        For compatibility, node_id is ignored in DAG mode — use
        get_available_candidates(satisfied_deps) for the real logic.
        """
        # Return top-level independent nodes by default
        return [
            n for n in self.nodes.values()
            if n['id'] != 'root' and n['status'] == 'active' and not n.get('deps')
        ]

    def select_path(self, exploration_method) -> List[str]:
        """Select a path through the DAG using UCB to decide both order and inclusion.

        Nodes with visits=0 (unexplored) are always included.
        For explored nodes, UCB score must beat a minimum threshold to be included,
        ensuring the path stays focused on high-value or under-explored milestones.
        """
        self._ensure_root()
        path = ['root']
        satisfied_deps = set()
        parent_visits = max(self.nodes['root']['visits'], 1)

        # Exclude zero-reward high-visit nodes (abandon mechanism)
        abandoned = self._get_abandoned_node_ids()
        satisfied_deps.update(abandoned)  # treat as pre-satisfied so dependents unlock

        if abandoned:
            info = [f"{nid}(v={self.nodes[nid].get('visits',0)})" for nid in sorted(abandoned)]
            print(f"[MilestoneDAG] Abandoned nodes (excluded from path): {', '.join(info)}")

        def _collect_available():
            """Collect all nodes whose deps are satisfied and not yet in path."""
            available = []
            for node in self.nodes.values():
                if node['id'] == 'root' or node['status'] != 'active':
                    continue
                if node['id'] in satisfied_deps:
                    continue
                dep_ids = self._normalize_deps(node.get('deps', []))
                if all(d in satisfied_deps for d in dep_ids):
                    available.append(node)
            return available

        while True:
            candidates = _collect_available()
            if not candidates:
                break

            eligible_visits = max(sum(c.get('visits', 0) for c in candidates), 1)
            idx = exploration_method.select(candidates, eligible_visits)
            selected = candidates[idx]

            path.append(selected['id'])
            satisfied_deps.add(selected['id'])

        return path

    def path_to_strategy(self, path: List[str]) -> Optional[Dict[str, Any]]:
        if not path or path == ['root']:
            return None

        high_level_steps = []
        description_parts = []

        # Build node_id → step_index mapping for deps_indices
        node_to_idx = {}
        for i, node_id in enumerate(path[1:]):
            node_to_idx[node_id] = i

        for node_id in path[1:]:
            node = self.nodes.get(node_id, {})
            milestone = node.get('milestone', '')
            if milestone:
                dep_ids = self._normalize_deps(node.get('deps', []))
                deps_indices = [node_to_idx[d] for d in dep_ids if d in node_to_idx]
                high_level_steps.append({
                    'step': milestone,
                    'key_action': node.get('key_action', ''),
                    'destination': node.get('destination', -1),
                    'deps_indices': deps_indices,
                    'attempt_notes': node.get('attempt_notes', []),
                    'visits': node.get('visits', 0),
                    'total_reward': node.get('total_reward', 0.0),
                    'is_depended_on': False,
                    'diagnostic': node.get('diagnostic'),
                })
                description_parts.append(milestone)

        # Mark nodes that are depended on by other nodes in the path
        for hl in high_level_steps:
            for dep_idx in hl.get('deps_indices', []):
                if 0 <= dep_idx < len(high_level_steps):
                    high_level_steps[dep_idx]['is_depended_on'] = True

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
            'is_dag': True,
        }

    def get_downstream_nodes(self, node_id: str) -> List[Dict]:
        """Return active nodes that directly depend on this node."""
        result = []
        for nid, n in self.nodes.items():
            if n.get('status') != 'active' or nid == 'root':
                continue
            dep_ids = self._normalize_deps(n.get('deps', []))
            if node_id in dep_ids:
                result.append({
                    'node_id': nid,
                    'milestone': n.get('milestone', ''),
                    'avg_reward': round(self._avg_reward(n), 2),
                })
        return result

    def _has_rewarding_descendants(self, node_id: str, _visited: set = None) -> bool:
        """Check if any active descendant has positive avg reward (instrumental protection)."""
        if _visited is None:
            _visited = set()
        if node_id in _visited:
            return False
        _visited.add(node_id)

        for nid, node in self.nodes.items():
            if node.get('status') != 'active' or nid == 'root':
                continue
            dep_ids = self._normalize_deps(node.get('deps', []))
            if node_id in dep_ids:
                visits = node.get('visits', 0)
                if visits > 0 and node.get('total_reward', 0) > 0:
                    return True
                if self._has_rewarding_descendants(nid, _visited):
                    return True
        return False

    def _get_abandoned_node_ids(self) -> set:
        """Return node IDs that should be excluded from path selection.

        Criterion: visits >= 3 AND total_reward <= 0 AND no rewarding descendants.
        """
        abandoned = set()
        for nid in self.nodes:
            if self._is_abandoned(nid):
                abandoned.add(nid)
        return abandoned

    def get_abandoned_summary(self) -> str:
        """Return a concise summary of abandoned nodes for exploration modules."""
        abandoned_ids = self._get_abandoned_node_ids()
        if not abandoned_ids:
            return ""
        lines = []
        for nid in sorted(abandoned_ids):
            node = self.nodes.get(nid, {})
            milestone = node.get('milestone', '?')
            key_action = node.get('key_action', '')
            visits = node.get('visits', 0)
            ka_short = f" — key: {key_action[:60]}" if key_action else ""
            lines.append(f"  [{nid}] {milestone} (v={visits}, avg=0){ka_short}")
        return "\n".join(lines)

    def backpropagate(self, path: List[str], score: float,
                      milestone_rewards: Dict[int, float],
                      not_attempted: Optional[set] = None, **kwargs):
        if self.backprop_method == 'dag':
            self._backpropagate_dag(path, score, milestone_rewards,
                                    not_attempted=not_attempted)
        else:  # 'linear' (default)
            self._backpropagate_per_node(path, score, milestone_rewards,
                                         not_attempted=not_attempted)

    def apply_operations(self, operations: List[Dict[str, Any]],
                         source: str = 'tree_update'):
        """Apply operations from TreeUpdate / DPM.

        Maps tree-style ops to DAG:
        - add_child(parent_id, milestone) → node with deps=[parent_id] (or deps=[] if root)
        - add_branch(parent_id, milestones) → chain of nodes, each depending on previous
        - prune(node_id) → mark pruned, cascade to dependents
        - update_node → same as tree
        """
        self._ensure_root()

        # Two-phase batch ID mapping: LLM may use temp IDs like "node_0", "node_1"
        # to reference other nodes created in the same batch. We track these so deps
        # can be resolved after all nodes are created.
        batch_id_map = {}  # temp_id -> real_node_id
        add_count = 0  # count of add_child ops seen so far

        def resolve_with_batch(raw_id: str) -> Optional[str]:
            """Resolve node ID: prefer real nodes, then batch refs, then fuzzy match."""
            if raw_id in self.nodes:
                return raw_id
            if raw_id in batch_id_map:
                real = batch_id_map[raw_id]
                print(f"[MilestoneDAG] Resolved batch ref '{raw_id}' -> {real}")
                return real
            return self._resolve_node_id(raw_id)

        for op_entry in operations:
            if not isinstance(op_entry, dict):
                continue
            op = op_entry.get('op', '')

            if op == 'add_child':
                add_count = self._apply_add_child(
                    op_entry, resolve_with_batch, source, batch_id_map, add_count)
            elif op == 'add_branch':
                add_count = self._apply_add_branch(
                    op_entry, resolve_with_batch, source, batch_id_map, add_count)
            elif op == 'prune':
                self._apply_prune(op_entry)
            elif op == 'update_node':
                self._apply_update_node(op_entry, resolve_with_batch)

        self._validate_dag()

    # ------------------------------------------------------------------ #
    #  apply_operations helpers
    # ------------------------------------------------------------------ #

    def _apply_add_child(self, op_entry, resolve_with_batch, source,
                         batch_id_map, add_count) -> int:
        """Handle add_child operation. Returns updated add_count."""
        reason = op_entry.get('reason', '')
        milestone = op_entry.get('milestone', '')
        key_action = re.sub(r'\s*\[[\+\-]?\d+\]', '', op_entry.get('key_action', ''))

        if not milestone:
            print(f"[MilestoneDAG] Skipping add_child: empty milestone")
            return add_count

        # Prefer direct 'deps' field (DAG-native); fall back to parent_id (tree compat)
        if 'deps' in op_entry:
            raw_deps = op_entry['deps']
            if not isinstance(raw_deps, list):
                raw_deps = [raw_deps] if raw_deps else []
            deps = []
            for rd in raw_deps:
                raw_id = rd.get('node', rd) if isinstance(rd, dict) else rd
                resolved = resolve_with_batch(raw_id)
                if resolved and resolved != 'root':
                    deps.append(resolved)
                elif resolved is None:
                    print(f"[MilestoneDAG] Warning: dep '{raw_id}' not found, skipping it")
        else:
            raw_parent = op_entry.get('parent_id') or 'root'
            parent_id = resolve_with_batch(raw_parent)
            if parent_id is None:
                print(f"[MilestoneDAG] Skipping add_child: parent '{raw_parent}' not found")
                return add_count
            deps = self._deps_for_parent(parent_id)

        raw_destination = op_entry.get('destination', -1)
        try:
            check_dest = int(raw_destination) if raw_destination not in (None, '', -1) else -1
        except (ValueError, TypeError):
            check_dest = -1
        similar_id = self._find_similar_node(milestone, deps, key_action, check_dest)
        if similar_id:
            similar_node = self.nodes[similar_id]
            # If the similar node is abandoned and new key_action differs, revive it
            if self._is_abandoned(similar_id) and key_action:
                old_ka = self._sanitize_key_action(similar_node.get('key_action', ''))
                new_ka = self._sanitize_key_action(key_action)
                if old_ka.lower().strip() != new_ka.lower().strip():
                    similar_node['key_action'] = new_ka
                    self._reset_node_stats(similar_node)
                    print(f"[MilestoneDAG] Revived abandoned [{similar_id}] '{similar_node.get('milestone','')}' "
                          f"with new key_action='{new_ka}' (old='{old_ka}')")
                    return add_count
            print(f"[MilestoneDAG] Skipping add_child: similar node exists "
                  f"(milestone='{milestone}', key='{key_action}', dest={check_dest}, deps={deps})")
            return add_count

        node = self._create_node(milestone, key_action, deps)
        dest = self._parse_destination(op_entry.get('destination', -1))
        if dest >= 0:
            node['destination'] = dest
        self._set_danger_warning(node, op_entry)
        if source == 'tree_update':
            self._apply_estimated_reward(node, op_entry.get('estimated_milestone_reward'))
        # Register batch ID mapping with multiple formats
        real_id = node['id']
        batch_id_map[f"node_{add_count}"] = real_id
        batch_id_map[f"node_{add_count:02d}"] = real_id
        batch_id_map[f"node_{add_count:03d}"] = real_id
        add_count += 1
        # Revive any abandoned deps — if this new node needs them, they deserve another chance
        for dep_ref in (deps or []):
            dep_id = self._resolve_node_id(str(dep_ref))
            if dep_id and self._is_abandoned(dep_id):
                dep_node = self.nodes[dep_id]
                self._reset_node_stats(dep_node)
                print(f"[MilestoneDAG] Revived abandoned dep [{dep_id}] '{dep_node.get('milestone','')}' "
                      f"(needed by new node [{node['id']}])")
        print(f"[MilestoneDAG] Added [{node['id']}] '{milestone}' deps={deps} — {reason}")
        return add_count

    def _apply_add_branch(self, op_entry, resolve_with_batch, source,
                          batch_id_map, add_count) -> int:
        """Handle add_branch operation. Returns updated add_count."""
        reason = op_entry.get('reason', '')
        raw_parent = op_entry.get('parent_id') or 'root'
        parent_id = resolve_with_batch(raw_parent)
        milestones = op_entry.get('milestones', [])

        if not milestones:
            single = op_entry.get('milestone', '')
            if single:
                milestones = [{'milestone': single, 'key_action': op_entry.get('key_action', ''),
                               'estimated_milestone_reward': op_entry.get('estimated_milestone_reward', 0)}]
            else:
                print(f"[MilestoneDAG] Skipping add_branch: no milestones")
                return add_count
        if parent_id is None:
            print(f"[MilestoneDAG] Skipping add_branch: parent '{raw_parent}' not found")
            return add_count

        # Normalize milestones
        normalized = []
        for m in milestones:
            if isinstance(m, str):
                normalized.append({'milestone': m, 'key_action': ''})
            elif isinstance(m, dict):
                normalized.append(m)
        milestones = normalized

        first_milestone = milestones[0].get('milestone', '') if milestones else ''
        first_deps = self._deps_for_parent(parent_id)
        first_ka = self._sanitize_key_action(milestones[0].get('key_action', '')) if milestones else ''
        first_dest_raw = milestones[0].get('destination', -1) if milestones else -1
        try:
            first_dest = int(first_dest_raw) if first_dest_raw not in (None, '', -1) else -1
        except (ValueError, TypeError):
            first_dest = -1
        if first_milestone and self._find_similar_node(first_milestone, first_deps, first_ka, first_dest):
            print(f"[MilestoneDAG] Skipping add_branch: first milestone similar to existing node")
            return add_count

        # Chain: each node depends on the previous one
        current_dep_source = parent_id
        created_nodes = []
        for m in milestones:
            ms = m.get('milestone', '')
            if not ms:
                continue
            ka = re.sub(r'\s*\[[\+\-]?\d+\]', '', m.get('key_action', ''))
            deps = self._deps_for_parent(current_dep_source)
            node = self._create_node(ms, ka, deps)
            dest = self._parse_destination(m.get('destination', -1))
            if dest >= 0:
                node['destination'] = dest
            if source == 'tree_update':
                self._apply_estimated_reward(node, m.get('estimated_milestone_reward'))
            created_nodes.append(node)
            current_dep_source = node['id']

        created_ids = [n['id'] for n in created_nodes]
        for n in created_nodes:
            batch_id_map[f"node_{add_count}"] = n['id']
            add_count += 1
        if created_ids:
            print(f"[MilestoneDAG] Added branch [{' -> '.join(created_ids)}] — {reason}")
        return add_count

    def _apply_prune(self, op_entry):
        """Handle prune operation."""
        reason = op_entry.get('reason', '')
        node_id = op_entry.get('node_id', '')
        if not node_id or node_id == 'root':
            print(f"[MilestoneDAG] Skipping prune: cannot prune root or empty node_id")
            return
        if node_id not in self.nodes:
            print(f"[MilestoneDAG] Skipping prune: {node_id} not found")
            return

        node = self.nodes[node_id]
        milestone = node.get('milestone', '?')
        self._prune_node(node_id)
        print(f"[MilestoneDAG] Pruned [{node_id}] '{milestone}' (dependents preserved) — {reason}")

    def _apply_update_node(self, op_entry, resolve_with_batch):
        """Handle update_node operation."""
        reason = op_entry.get('reason', '')
        node_id = op_entry.get('node_id', '')
        if not node_id or node_id not in self.nodes:
            print(f"[MilestoneDAG] Skipping update_node: {node_id} not found")
            return

        node = self.nodes[node_id]
        # Penalty nodes verified by auto-set should not be modified by reflection
        if node.get('danger_action'):
            print(f"[MilestoneDAG] Skipping update_node [{node_id}]: penalty node already verified by auto-set")
            return
        old_milestone = node.get('milestone', '?')
        new_milestone = op_entry.get('milestone', '')
        if new_milestone:
            # Log rename but NEVER reset stats — accumulated experience is too
            # valuable to discard.  If the goal truly changed, new episodes will
            # naturally shift the running averages.  Resetting wipes proven
            # knowledge and forces the system to re-learn from scratch.
            if new_milestone != old_milestone:
                print(f"[MilestoneDAG] Milestone renamed [{node_id}]: '{old_milestone}' -> '{new_milestone}' (stats preserved: v={node.get('visits', 0)}, avg={node.get('total_reward', 0) / max(node.get('visits', 1), 1):.1f})")
            node['milestone'] = new_milestone
        new_key_action = op_entry.get('key_action')
        if new_key_action is not None:
            cleaned = re.sub(r'\s*\[[\+\-]?\d+\]', '', new_key_action)
            node['key_action'] = self._sanitize_key_action(cleaned)
        new_destination = op_entry.get('destination')
        if new_destination is not None:
            try:
                node['destination'] = int(new_destination)
            except (ValueError, TypeError):
                pass  # keep existing
        new_deps = op_entry.get('deps')
        if new_deps and isinstance(new_deps, list):
            resolved_deps = []
            for d in new_deps:
                raw_id = d.get('node', d) if isinstance(d, dict) else d
                rd = resolve_with_batch(raw_id) if resolve_with_batch else self._resolve_node_id(raw_id)
                if rd and rd != node_id:
                    resolved_deps.append(rd)
            node['deps'] = resolved_deps
            # Revive any abandoned deps — if this node needs them, they deserve another chance
            for dep_id in resolved_deps:
                if dep_id and self._is_abandoned(dep_id):
                    dep_node = self.nodes[dep_id]
                    self._reset_node_stats(dep_node)
                    print(f"[MilestoneDAG] Revived abandoned dep [{dep_id}] '{dep_node.get('milestone','')}' "
                          f"(needed by updated node [{node_id}])")
        self._set_danger_warning(node, op_entry)
        print(f"[MilestoneDAG] Updated [{node_id}] '{old_milestone}' -> '{node.get('milestone', '?')}' deps={node.get('deps', [])} — {reason}")

    def _validate_dag(self):
        """Post-operation DAG validation: fix broken deps, detect cycles, warn on issues."""
        # 1. Fix broken deps — remove refs to pruned/nonexistent nodes
        for nid, node in self.nodes.items():
            if nid == 'root' or node.get('status') != 'active':
                continue
            dep_ids = self._normalize_deps(node.get('deps', []))
            valid_deps = []
            for d in dep_ids:
                dep_node = self.nodes.get(d)
                if dep_node and dep_node.get('status') == 'active':
                    valid_deps.append(d)
                else:
                    print(f"[MilestoneDAG] Validate: removed broken dep '{d}' from [{nid}]")
            if len(valid_deps) != len(dep_ids):
                node['deps'] = valid_deps

        # 2. Detect and break cycles via DFS
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {nid: WHITE for nid in self.nodes if self.nodes[nid].get('status') == 'active'}

        def _dfs(nid):
            color[nid] = GRAY
            node = self.nodes.get(nid)
            if not node:
                color[nid] = BLACK
                return
            for dep in list(node.get('deps', [])):
                if dep not in color:
                    continue
                if color[dep] == GRAY:
                    # Cycle found: remove the dep that causes it
                    node['deps'] = [d for d in node['deps'] if d != dep]
                    print(f"[MilestoneDAG] Validate: broke cycle by removing dep '{dep}' from [{nid}]")
                elif color[dep] == WHITE:
                    _dfs(dep)
            color[nid] = BLACK

        for nid in list(color.keys()):
            if color.get(nid) == WHITE:
                _dfs(nid)

        # 3. Warn on visited nodes with empty key_action
        for nid, node in self.nodes.items():
            if nid == 'root' or node.get('status') != 'active':
                continue
            if node.get('visits', 0) > 0 and not node.get('key_action'):
                print(f"[MilestoneDAG] Validate: [{nid}] has {node['visits']} visits but empty key_action")

    def save(self, path: str):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            data = {
                'version': 5,
                'space_type': 'milestone_dag',
                'nodes': self.nodes,
                'next_node_id': self.next_node_id,
                'total_episodes': self.total_episodes,
                'global_lessons': self.global_lessons,
            }
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"[MilestoneDAG] Saved DAG with {self.active_count()} active nodes")
        except Exception as e:
            print(f"[MilestoneDAG] Error saving: {e}")

    def load(self, path: str):
        if not os.path.exists(path):
            print(f"[MilestoneDAG] No existing space found (first run)")
            self._ensure_root()
            return

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            version = data.get('version', 1)
            space_type = data.get('space_type', '')

            if space_type == 'milestone_dag' and version >= 3:
                # Native DAG format
                self.nodes = data.get('nodes', {})
                self.next_node_id = data.get('next_node_id', 1)
                self.total_episodes = data.get('total_episodes', 0)
                self._ensure_root()

                for node in self.nodes.values():
                    if 'reward_sq_sum' not in node:
                        node['reward_sq_sum'] = 0.0
                    if 'max_reward' not in node:
                        # Backfill: best estimate is avg_reward for old data
                        v = node.get('visits', 0)
                        node['max_reward'] = node['total_reward'] / v if v > 0 else 0.0
                    if 'destination' not in node:
                        node['destination'] = -1
                    else:
                        # Migrate old string destinations to -1
                        dest = node['destination']
                        if isinstance(dest, str):
                            node['destination'] = -1
                        elif not isinstance(dest, int):
                            node['destination'] = -1
                    # Normalize deps to new format (backward compat)
                    node['deps'] = self._normalize_deps(node.get('deps', []))
                self.global_lessons = data.get('global_lessons', [])

                print(f"[MilestoneDAG] Loaded DAG with {self.active_count()} active nodes")

            elif version >= 2:
                # v2 milestone_tree format → migrate
                print(f"[MilestoneDAG] Detected v2 milestone_tree format, migrating to DAG...")
                self._migrate_tree_to_dag(data)
                print(f"[MilestoneDAG] Migration complete: {self.active_count()} active nodes")

            else:
                print(f"[MilestoneDAG] Unknown format v{version}, starting fresh")
                self.nodes = {}
                self._ensure_root()

        except Exception as e:
            print(f"[MilestoneDAG] Error loading: {e}")
            self.nodes = {}
            self._ensure_root()

    def format_for_reflection(self) -> str:
        if not self.has_strategies():
            return "No milestones discovered yet. The DAG is empty (root only)."
        lines = self._format_dag_lines()
        return "\n".join(lines)

    def format_tree_display(self) -> str:
        header = f"Milestone DAG ({self.active_count()} active nodes)"
        sep = "=" * 60
        lines = [f"\n{sep}", f"  {header}", sep]
        root = self.nodes.get('root', {})
        visits = root.get('visits', 0)
        avg_r = root['total_reward'] / visits if visits > 0 else 0.0
        lines.append(f"  [root] visits={visits} avg_reward={avg_r:.1f}")
        lines.extend(self._format_dag_lines(indent=2))
        # Total estimated score from all non-abandoned active nodes
        abandoned_ids = self._get_abandoned_node_ids()
        total_avg = 0.0
        total_max = 0.0
        for n in self.nodes.values():
            if n['id'] == 'root' or n.get('status') != 'active' or n['id'] in abandoned_ids:
                continue
            avg = self._avg_reward(n)
            if avg > 0:
                total_avg += avg
            mr = n.get('max_reward', 0.0)
            if mr > 0:
                total_max += mr
        lines.append(f"  --- Total score potential (active, non-abandoned): avg={total_avg:.1f} max={total_max:.1f} ---")
        lines.append("")
        return "\n".join(lines)

    def active_count(self) -> int:
        return sum(1 for n in self.nodes.values()
                   if n.get('status') == 'active' and n['id'] != 'root')

    def evolve(self, episode_summaries, llm_model: str, args):
        """Milestone DAG evolution — method controlled by args.evolution_method."""
        method = getattr(args, 'evolution_method', 'decision_point_mining')

        if method == 'none':
            print("[MilestoneDAG] Evolution disabled (--evolution_method none)")
            return []
        elif method == 'free_reflection':
            from ..evolution.free_reflection import FreeReflection
            evolver = FreeReflection()
        else:  # default: decision_point_mining
            from ..evolution.decision_point_mining import DecisionPointMining
            evolver = DecisionPointMining()

        result = evolver.reflect(episode_summaries, self, llm_model, args)
        if result:
            operations = result.get('operations', [])
            if operations:
                source = 'free_reflection' if method == 'free_reflection' else 'dpm'
                self.apply_operations(operations, source=source)
            return operations
        return []

    def get_milestone_list(self) -> List[str]:
        milestones = []
        for node in self.nodes.values():
            if node.get('status') == 'active' and node['id'] != 'root':
                milestone = node.get('milestone', '')
                if milestone:
                    key_action = node.get('key_action', '')
                    ka_hint = f" (key: {key_action[:60]})" if key_action else ""
                    milestones.append(f"[{node['id']}] {milestone}{ka_hint}")
        return milestones

    def sync_root_visits(self):
        """Ensure root visits >= max of all node visits."""
        root = self.nodes.get('root', {})
        max_visits = max(
            (n.get('visits', 0) for n in self.nodes.values() if n['id'] != 'root'),
            default=0,
        )
        if max_visits > root.get('visits', 0):
            root['visits'] = max_visits

    # ------------------------------------------------------------------ #
    #  Backpropagation
    # ------------------------------------------------------------------ #

    def _backpropagate_per_node(self, path: List[str], score: float,
                                milestone_rewards: Dict[int, float],
                                not_attempted: Optional[set] = None):
        not_attempted = not_attempted or set()

        # Free exploration
        if path == ['root']:
            node = self.nodes.get('root')
            if node:
                node['visits'] += 1
                node['total_reward'] += score
                node['reward_sq_sum'] = node.get('reward_sq_sum', 0.0) + score ** 2
            return

        # Per-node attribution with gamma-discounted backpropagation:
        # Each node gets its own milestone reward PLUS a discounted share
        # of downstream rewards, so dependency nodes that enable high-reward
        # successors accumulate strategic value.
        #   value[t] = sum_{u=t}^{k} gamma^{u-t} * r[u]
        # Computed via backward recursion: value[i] = r[i] + gamma * value[i+1]
        num_nodes = len(path) - 1
        raw_rewards = [milestone_rewards.get(i, 0.0) for i in range(num_nodes)]

        # Compute gamma-discounted values (backward pass)
        gamma = self.backprop_gamma
        node_values = [0.0] * num_nodes
        node_values[-1] = raw_rewards[-1]
        for i in range(num_nodes - 2, -1, -1):
            node_values[i] = raw_rewards[i] + gamma * node_values[i + 1]

        # Root gets sum of all raw milestone rewards (not discounted)
        root_value = sum(raw_rewards)
        root_node = self.nodes.get('root')
        if root_node:
            root_node['visits'] += 1
            root_node['total_reward'] += root_value
            root_node['reward_sq_sum'] = root_node.get('reward_sq_sum', 0.0) + root_value ** 2
            if root_value > root_node.get('max_reward', 0.0):
                root_node['max_reward'] = root_value

        for i in range(num_nodes):
            node_id = path[i + 1]
            node = self.nodes.get(node_id)
            if node:
                # Skip visit counting for NOT_ATTEMPTED nodes — agent never
                # reached this milestone, so 0 reward reflects navigation
                # failure, not the node being worthless.
                if node_id in not_attempted:
                    continue
                node['visits'] += 1
                node['total_reward'] += node_values[i]
                node['reward_sq_sum'] = node.get('reward_sq_sum', 0.0) + node_values[i] ** 2
                if node_values[i] > node.get('max_reward', 0.0):
                    node['max_reward'] = node_values[i]

    def _backpropagate_dag(self, path: List[str], score: float,
                           milestone_rewards: Dict[int, float],
                           not_attempted: Optional[set] = None):
        """DAG dependency-based backpropagation.

        Credit flows backwards along actual dep edges, not path order.
        A node only gets discounted credit from nodes that explicitly depend on it.
        Independent nodes in the same path do not share credit.

          value[v] = r[v] + gamma * sum(value[u] for u where v in u.deps and u in path)

        Computed via reverse topological pass (path is already in topo order
        since select_path only adds a node after its deps are satisfied).
        """
        not_attempted = not_attempted or set()

        if path == ['root']:
            node = self.nodes.get('root')
            if node:
                node['visits'] += 1
                node['total_reward'] += score
                node['reward_sq_sum'] = node.get('reward_sq_sum', 0.0) + score ** 2
            return

        num_nodes = len(path) - 1
        path_ids = path[1:]
        raw_rewards = [milestone_rewards.get(i, 0.0) for i in range(num_nodes)]
        idx_map = {nid: i for i, nid in enumerate(path_ids)}

        # For each node index i, collect indices of path nodes that depend on i
        # (i.e., i is in their deps → i "enabled" them → i deserves their credit)
        dependents: List[List[int]] = [[] for _ in range(num_nodes)]
        for j, nid in enumerate(path_ids):
            node = self.nodes.get(nid, {})
            for dep_id in self._normalize_deps(node.get('deps', [])):
                if dep_id in idx_map:
                    dependents[idx_map[dep_id]].append(j)

        # Reverse topological pass: compute value[i] = r[i] + gamma * sum(value[j] for j in dependents[i])
        # Path is in topo order (deps before dependents), so reversing gives dependents before deps.
        node_values = list(raw_rewards)
        for i in range(num_nodes - 1, -1, -1):
            for j in dependents[i]:
                node_values[i] += self.backprop_gamma * node_values[j]

        # Root gets undiscounted sum of raw rewards
        root_value = sum(raw_rewards)
        root_node = self.nodes.get('root')
        if root_node:
            root_node['visits'] += 1
            root_node['total_reward'] += root_value
            root_node['reward_sq_sum'] = root_node.get('reward_sq_sum', 0.0) + root_value ** 2
            if root_value > root_node.get('max_reward', 0.0):
                root_node['max_reward'] = root_value

        for i, node_id in enumerate(path_ids):
            if node_id in not_attempted:
                continue
            node = self.nodes.get(node_id)
            if node:
                node['visits'] += 1
                node['total_reward'] += node_values[i]
                node['reward_sq_sum'] = node.get('reward_sq_sum', 0.0) + node_values[i] ** 2
                if node_values[i] > node.get('max_reward', 0.0):
                    node['max_reward'] = node_values[i]

    # ------------------------------------------------------------------ #
    #  Score gap analysis
    # ------------------------------------------------------------------ #

    def get_score_gap_analysis(self, path: List[str],
                               milestone_rewards: Dict[int, float],
                               episode_score: float) -> Optional[str]:
        """Compare episode score against DAG theoretical max.

        Returns a text feedback string if there's a gap, or None if the
        agent achieved the theoretical max.
        """
        # Calculate theoretical max from active non-abandoned nodes
        theoretical_max = 0.0
        active_nodes = []
        for n in self.nodes.values():
            if n['id'] == 'root' or n.get('status') != 'active':
                continue
            if self._is_abandoned(n['id']):
                continue
            max_r = n.get('max_reward', 0.0)
            if max_r > 0:
                theoretical_max += max_r
                active_nodes.append(n)

        if theoretical_max <= 0:
            return None

        # Which nodes on the path got rewards this episode?
        achieved_ids = set()
        for i, reward in milestone_rewards.items():
            if reward > 0 and i + 1 < len(path):
                achieved_ids.add(path[i + 1])

        # Find missed milestones (active, non-abandoned, has max_reward > 0, not achieved)
        missed = []
        for n in active_nodes:
            if n['id'] not in achieved_ids:
                missed.append(n)

        gap = theoretical_max - episode_score
        if gap <= 0:
            return None  # achieved or exceeded theoretical max

        lines = [
            f"SCORE GAP: You scored {episode_score:.0f} but known milestones suggest "
            f"{theoretical_max:.0f} is achievable (gap: {gap:.0f}).",
            f"Missed milestones ({len(missed)}):"
        ]
        for n in sorted(missed, key=lambda x: x.get('max_reward', 0), reverse=True):
            max_r = n.get('max_reward', 0.0)
            milestone = n.get('milestone', '')
            key_action = n.get('key_action', '')
            line = f"  - {milestone} (up to {max_r:.0f} pts)"
            if key_action:
                line += f" — key action: {key_action}"
            lines.append(line)

        lines.append(
            "Focus on the highest-value missed milestones first. "
            "If a milestone was missed due to a prerequisite, solve the prerequisite."
        )
        return '\n'.join(lines)

    # ------------------------------------------------------------------ #
    #  Display helpers
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

    def _format_dag_lines(self, indent: int = 0) -> List[str]:
        """Format the DAG as a layered graph.

        Nodes are grouped by topological layer (depth from roots).
        Each node shows all its deps explicitly, so multi-parent
        relationships are visible.
        """
        lines = []
        prefix = "  " * indent

        # Collect all active non-root nodes
        active_nodes = {
            nid: n for nid, n in self.nodes.items()
            if nid != 'root' and n.get('status') == 'active'
        }

        if not active_nodes:
            return lines

        # Compute topological layer for each node
        layers: Dict[str, int] = {}
        _visiting: set = set()  # cycle detection
        def _get_layer(nid: str) -> int:
            if nid in layers:
                return layers[nid]
            if nid in _visiting:
                # Cycle detected — treat as independent to break it
                layers[nid] = 0
                return 0
            node = active_nodes.get(nid)
            if not node:
                return 0
            _visiting.add(nid)
            dep_ids = self._normalize_deps(node.get('deps', []))
            # Only consider deps that are active
            active_deps = [d for d in dep_ids if d in active_nodes]
            if not active_deps:
                layers[nid] = 0
            else:
                layers[nid] = max(_get_layer(d) for d in active_deps) + 1
            _visiting.discard(nid)
            return layers[nid]

        for nid in active_nodes:
            _get_layer(nid)

        # Group by layer
        abandoned_ids = self._get_abandoned_node_ids()
        max_layer = max(layers.values()) if layers else 0
        for layer in range(max_layer + 1):
            layer_nodes = sorted(
                [nid for nid, l in layers.items() if l == layer])
            if not layer_nodes:
                continue
            lines.append(f"{prefix}--- Layer {layer} {'(independent)' if layer == 0 else ''} ---")
            for nid in layer_nodes:
                node = active_nodes[nid]
                avg_r = self._avg_reward(node)
                visits = node.get('visits', 0)
                milestone = node.get('milestone', '?')
                key_action = node.get('key_action', '')
                ka_short = key_action[:60] + '...' if len(key_action) > 60 else key_action
                dep_ids = self._normalize_deps(node.get('deps', []))
                deps_text = f"  ← {', '.join(dep_ids)}" if dep_ids else ""
                abandoned_tag = " [ABANDONED]" if nid in abandoned_ids else ""
                max_r = node.get('max_reward', 0.0)
                lines.append(
                    f"{prefix}  [{nid}] {milestone}  "
                    f"v={visits} avg={avg_r:.1f} max={max_r:.1f}{deps_text}{abandoned_tag}")
                if ka_short:
                    lines.append(f"{prefix}    key: {ka_short}")

        return lines

    # ------------------------------------------------------------------ #
    #  Migration
    # ------------------------------------------------------------------ #

    def _migrate_tree_to_dag(self, data: dict):
        """Migrate v2 milestone_tree data to DAG format."""
        self.nodes = {}
        self._ensure_root()

        tree_nodes = data.get('nodes', {})
        self.next_node_id = data.get('next_node_id', 1)
        self.total_episodes = data.get('total_episodes', 0)

        # Copy root stats
        old_root = tree_nodes.get('root', {})
        self.nodes['root']['visits'] = old_root.get('visits', 0)
        self.nodes['root']['total_reward'] = old_root.get('total_reward', 0.0)
        self.nodes['root']['reward_sq_sum'] = old_root.get('reward_sq_sum', 0.0)

        # Convert each non-root node: parent → deps
        for nid, old_node in tree_nodes.items():
            if nid == 'root':
                continue
            parent_id = old_node.get('parent', 'root')
            deps = [] if parent_id == 'root' else [parent_id]
            self.nodes[nid] = {
                'id': nid,
                'milestone': old_node.get('milestone'),
                'key_action': old_node.get('key_action', ''),
                'deps': deps,
                'visits': old_node.get('visits', 0),
                'total_reward': old_node.get('total_reward', 0.0),
                'reward_sq_sum': old_node.get('reward_sq_sum', 0.0),
                'status': old_node.get('status', 'active'),
            }
