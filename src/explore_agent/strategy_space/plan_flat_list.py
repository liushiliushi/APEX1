"""Plan × Flat List strategy space (Config A).

A flat list of independent strategies (plans). Each strategy is a complete
plan with milestones. Selection is MAB-style: exploration_method.select()
is called once over all strategies.
"""
import json
import os
import re
from typing import Any, Dict, List, Optional

from .base import StrategySpace


class PlanFlatListSpace(StrategySpace):
    """Flat list of plan strategies, selected via multi-armed bandit."""

    def __init__(self, max_strategies: int = 15, min_strategies: int = 3):
        self.strategies: List[Dict[str, Any]] = []
        self.max_strategies = max_strategies
        self.min_strategies = min_strategies
        self.total_episodes = 0

    def has_strategies(self) -> bool:
        return len([s for s in self.strategies if s.get('status') == 'active']) > 0

    def get_candidates(self, node_id: str = None) -> List[Dict[str, Any]]:
        return [s for s in self.strategies if s.get('status') == 'active']

    def select_path(self, exploration_method) -> List[str]:
        """Select one strategy from the flat list. Returns [strategy_id]."""
        candidates = self.get_candidates()
        if not candidates:
            return []

        parent_visits = sum(c.get('visits', 0) for c in candidates)
        idx = exploration_method.select(candidates, max(parent_visits, 1))
        selected = candidates[idx]
        return [selected['id']]

    def path_to_strategy(self, path: List[str]) -> Optional[Dict[str, Any]]:
        if not path:
            return None
        strategy_id = path[0]
        for s in self.strategies:
            if s['id'] == strategy_id and s.get('status') == 'active':
                return s
        return None

    def backpropagate(self, path: List[str], score: float,
                      milestone_rewards: Dict[int, float], **kwargs):
        if not path:
            return
        strategy_id = path[0]
        for s in self.strategies:
            if s['id'] == strategy_id:
                s['visits'] = s.get('visits', 0) + 1
                s['total_reward'] = s.get('total_reward', 0.0) + score
                s['reward_sq_sum'] = s.get('reward_sq_sum', 0.0) + score ** 2
                break

    def apply_operations(self, operations: List[Dict[str, Any]], **kwargs):
        """Apply operations. Supported: add_strategy, prune, update_strategy."""
        for op_entry in operations:
            if not isinstance(op_entry, dict):
                continue
            op = op_entry.get('op', '')
            reason = op_entry.get('reason', '')

            if op == 'add_strategy' or op == 'add_branch':
                description = op_entry.get('description', '')
                milestones = op_entry.get('milestones', [])
                if not description and not milestones:
                    continue

                strategy_id = f"strategy_{len(self.strategies) + 1:03d}"
                # Normalize milestones: handle both dict and string entries
                normalized = []
                for m in milestones:
                    if isinstance(m, str):
                        normalized.append({'milestone': m, 'key_action': ''})
                    elif isinstance(m, dict):
                        normalized.append(m)
                milestones = normalized
                high_level_steps = []
                for m in milestones:
                    high_level_steps.append({
                        'step': m.get('milestone', ''),
                        'key_action': re.sub(r'\s*\[[\+\-]?\d+\]', '', m.get('key_action', '')),
                    })

                steps = []
                for i, hl in enumerate(high_level_steps, 1):
                    step_name = hl.get('step', '')
                    key_action = hl.get('key_action', '')
                    if step_name:
                        if key_action:
                            steps.append(f"{i}. {step_name} — key action: {key_action}")
                        else:
                            steps.append(f"{i}. {step_name}")

                if not description and high_level_steps:
                    parts = [hl['step'] for hl in high_level_steps if hl.get('step')]
                    description = parts[0] if len(parts) == 1 else f"{parts[0]} -> ... -> {parts[-1]}"

                strategy = {
                    'id': strategy_id,
                    'description': description,
                    'steps': steps,
                    'high_level_steps': high_level_steps,
                    'visits': 0,
                    'total_reward': 0.0,
                    'reward_sq_sum': 0.0,
                    'status': 'active',
                }
                self.strategies.append(strategy)
                print(f"[PlanFlatList] Added [{strategy_id}] '{description}' — {reason}")

            elif op == 'prune':
                node_id = op_entry.get('node_id', '') or op_entry.get('strategy_id', '')
                for s in self.strategies:
                    if s['id'] == node_id:
                        s['status'] = 'pruned'
                        print(f"[PlanFlatList] Pruned [{node_id}] — {reason}")
                        break

            elif op == 'update_strategy' or op == 'update_node':
                node_id = op_entry.get('node_id', '') or op_entry.get('strategy_id', '')
                for s in self.strategies:
                    if s['id'] == node_id:
                        if op_entry.get('description'):
                            s['description'] = op_entry['description']
                        # Update milestones if provided
                        milestones = op_entry.get('milestones', [])
                        if milestones:
                            normalized = []
                            for m in milestones:
                                if isinstance(m, str):
                                    normalized.append({'milestone': m, 'key_action': ''})
                                elif isinstance(m, dict):
                                    normalized.append(m)
                            high_level_steps = []
                            steps = []
                            for i, m in enumerate(normalized, 1):
                                step_name = m.get('milestone', '')
                                key_action = re.sub(r'\s*\[[\+\-]?\d+\]', '', m.get('key_action', ''))
                                high_level_steps.append({'step': step_name, 'key_action': key_action})
                                if key_action:
                                    steps.append(f"{i}. {step_name} — key action: {key_action}")
                                else:
                                    steps.append(f"{i}. {step_name}")
                            s['high_level_steps'] = high_level_steps
                            s['steps'] = steps
                            if not op_entry.get('description') and high_level_steps:
                                parts = [hl['step'] for hl in high_level_steps if hl.get('step')]
                                s['description'] = parts[0] if len(parts) == 1 else f"{parts[0]} -> ... -> {parts[-1]}"
                        print(f"[PlanFlatList] Updated [{node_id}] — {reason}")
                        break

    def save(self, path: str):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            data = {
                'version': 2,
                'space_type': 'plan_flat_list',
                'strategies': self.strategies,
                'total_episodes': self.total_episodes,
            }
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"[PlanFlatList] Saved {self.active_count()} active strategies")
        except Exception as e:
            print(f"[PlanFlatList] Error saving: {e}")

    def load(self, path: str):
        if not os.path.exists(path):
            print(f"[PlanFlatList] No existing space found (first run)")
            return

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.strategies = data.get('strategies', [])
            self.total_episodes = data.get('total_episodes', 0)
            # Ensure reward_sq_sum
            for s in self.strategies:
                if 'reward_sq_sum' not in s:
                    s['reward_sq_sum'] = 0.0
            print(f"[PlanFlatList] Loaded {self.active_count()} active strategies")
        except Exception as e:
            print(f"[PlanFlatList] Error loading: {e}")

    def format_for_reflection(self) -> str:
        active = self.get_candidates()
        if not active:
            return "No strategies discovered yet. The list is empty."
        lines = []
        for s in active:
            visits = s.get('visits', 0)
            avg_r = s['total_reward'] / visits if visits > 0 else 0.0
            lines.append(f"[{s['id']}] {s['description']}  visits={visits} avg={avg_r:.1f}")
            for step in s.get('steps', []):
                lines.append(f"  {step}")
        return "\n".join(lines)

    def format_tree_display(self) -> str:
        active = self.get_candidates()
        header = f"Strategy List ({len(active)} active strategies)"
        sep = "=" * 60
        lines = [f"\n{sep}", f"  {header}", sep]
        for s in active:
            visits = s.get('visits', 0)
            avg_r = s['total_reward'] / visits if visits > 0 else 0.0
            lines.append(f"  [{s['id']}] {s['description']}  visits={visits} avg={avg_r:.1f}")
        lines.append("")
        return "\n".join(lines)

    def active_count(self) -> int:
        return len([s for s in self.strategies if s.get('status') == 'active'])

    def evolve(self, episode_summaries, llm_model: str, args):
        """Plan List uses Free Reflection for evolution."""
        from ..evolution.free_reflection import FreeReflection
        evolver = FreeReflection()
        result = evolver.reflect(episode_summaries, self, llm_model, args)
        if result:
            operations = result.get('operations', [])
            if operations:
                self.apply_operations(operations, source='free_reflection')
            return operations
        return []

    def get_milestone_list(self) -> List[str]:
        """Return list of strategy descriptions for tree-aware summaries."""
        return [f"[{s['id']}] {s['description']}" for s in self.strategies
                if s.get('status') == 'active']
