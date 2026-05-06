"""ExploreAgent — modular orchestrator for exploration strategies.

Composes three pluggable modules:
  M0: StrategySpace    (plan_flat_list / milestone_tree / milestone_dag)
      Each space bundles its own evolution method:
        plan_flat_list → Free Reflection
        milestone_tree → Decision Point Mining
        milestone_dag  → Decision Point Mining
  M1: GuidanceMode     (full_plan / step_by_step / hierarchical)
  M2: ExplorationMethod (ucb / thompson / epsilon_greedy)
"""
import json
import os
import random
import time
from typing import Any, Dict, List, Optional, Set

from .openai_helpers_proxy import chat_completion, parse_json_response
from ..openai_helpers import chat_completion_with_retries
from ..utils import generate_detailed_episode_summary

from .strategy_space import STRATEGY_SPACES
from .exploration import EXPLORATION_METHODS
from .guidance import GUIDANCE_MODES
from .evolution.tree_update import TreeUpdate


class ExploreAgent:
    """
    Modular exploration agent that composes three pluggable modules.

    Exposes the same interface as ReflectiveExplorationAgent:
      - start_episode()
      - generate_action(state_node, info)
      - update_game_history_reward(reward, score)
      - end_episode(state, score)
    """

    def __init__(self, args, guiding_prompt: str = None):
        self.args = args
        self.guiding_prompt = guiding_prompt or "Explore systematically and examine objects to make progress."

        # Per-episode state
        self.memory = []
        self.game_history = []
        self.current_strategy = None
        self.current_path = []
        self.current_milestone_idx = 0
        self.milestone_rewards = {}
        self._action_tree_current_node = None  # tracks position in action_tree during gameplay

        # Cross-episode state
        self.episodes_since_reflection = 0
        self.episode_summaries = []
        self._pending_backprop = []  # deferred backprop data, flushed after MapRefinement
        self.total_episodes = 0
        self.reflect_interval = getattr(args, 'reflect_interval', 5)

        # Game identification
        self.game_name = getattr(args, 'game_name', 'unknown_game')
        self._python_game = self.game_name.startswith('catnip') or self.game_name.startswith('maze')
        self._scienceworld = self.game_name.startswith('sw:') or getattr(args, 'env_type', '') == 'scienceworld'

        # Persistence paths (set by evaluation.py or main.py)
        output_path = getattr(args, 'output_path', 'output')
        model_slug = getattr(args, 'llm_model', 'model').replace('/', '_').replace('\\', '_')
        agent_type = getattr(args, 'agent_type', 'explore')
        self.search_space_dir = os.path.join(output_path, self.game_name, agent_type, model_slug)
        self.search_space_path = os.path.join(self.search_space_dir, 'search_space.json')

        # ---- Instantiate modules ----

        # M2: Exploration method (stateless, needed by M0)
        exploration_name = getattr(args, 'exploration_method', 'ucb')
        exploration_cls = EXPLORATION_METHODS[exploration_name]
        exploration_params = {
            'ucb': {'c': getattr(args, 'ucb_exploration_c', 1.414)},
            'thompson': {'prior_std': getattr(args, 'thompson_prior_std', 100.0)},
            'epsilon_greedy': {'epsilon': getattr(args, 'epsilon', 0.1)},
        }
        self.exploration_method = exploration_cls(**exploration_params.get(exploration_name, {}))

        # M0: Strategy space
        space_name = getattr(args, 'strategy_space', 'milestone_tree')
        space_cls = STRATEGY_SPACES[space_name]
        space_params: Dict[str, Any] = {}
        if space_name in ('milestone_tree', 'milestone_dag'):
            space_params = dict(
                backprop_gamma=getattr(args, 'backprop_gamma', 0.6),
                backprop_method=getattr(args, 'backprop_method', 'linear'),
                backprop_unreached_discount=getattr(args, 'backprop_unreached_discount', 0.3),
            )
        elif space_name == 'plan_flat_list':
            space_params = dict(
                max_strategies=getattr(args, 'max_strategies', 15),
                min_strategies=getattr(args, 'min_strategies', 3),
            )
        self.strategy_space = space_cls(**space_params)

        # M1: Guidance mode
        guidance_name = getattr(args, 'guidance_mode', 'full_plan')
        self.guidance_mode = GUIDANCE_MODES[guidance_name]()

        # Tree update (always runs before space evolution)
        self.tree_update = TreeUpdate()

        # Reflection sub-modules (instantiated once, reused each reflection cycle)
        from .evolution.stuck_node_diagnosis import StuckNodeDiagnosis
        from .evolution.global_memory import GlobalMemoryUpdate
        self.stuck_node_diagnosis = StuckNodeDiagnosis()
        self.global_memory_updater = GlobalMemoryUpdate()

        # Module names for logging
        self._module_names = {
            'strategy_space': space_name,
            'exploration_method': exploration_name,
            'guidance_mode': guidance_name,
        }

    # ------------------------------------------------------------------ #
    #  Episode lifecycle
    # ------------------------------------------------------------------ #

    def start_episode(self, initial_score=0):
        """Reset per-episode state, load space, select path."""
        self.memory = []
        self.game_history = []
        self._initial_score = initial_score  # score awarded by game before any agent action
        self.current_strategy = None
        self.current_path = []
        self.current_milestone_idx = 0
        self.milestone_rewards = {}
        self._action_tree_current_node = None
        self._episode_feedback = None  # score gap analysis from last episode
        self._steps_on_current_milestone: int = 0
        # Room tracking (Jericho only)
        self._current_room_id: int = -1
        self._tracked_milestone_idx: Optional[int] = None  # which milestone we're counting steps for

        if hasattr(self.guidance_mode, 'reset'):
            self.guidance_mode.reset()

        self._load_state()
        self.total_episodes += 1

        is_action_tree = self._module_names.get('strategy_space') == 'action_tree'

        if self.strategy_space.has_strategies():
            # After freeze, switch to greedy (epsilon=0) to stabilize execution order
            freeze_ep = getattr(self.args, 'exploration_freeze_episode', 0)
            frozen = freeze_ep > 0 and self.total_episodes > freeze_ep
            if frozen and not hasattr(self, '_greedy_method'):
                from .exploration import EpsilonGreedy
                self._greedy_method = EpsilonGreedy(epsilon=0.0)
            method = self._greedy_method if frozen and hasattr(self, '_greedy_method') else self.exploration_method
            self.current_path = self.strategy_space.select_path(method)
            self.current_strategy = self.strategy_space.path_to_strategy(self.current_path)
            if self.current_strategy:
                print(f"[ExploreAgent] Episode {self.total_episodes}: "
                      f"path: {' -> '.join(self.current_path[1:])}")
                self._print_episode_strategy()
                # Inject score gap analysis from last episode
                if hasattr(self, '_last_gap_analysis') and self._last_gap_analysis:
                    self._episode_feedback = self._last_gap_analysis
                    self._last_gap_analysis = None
                    print(f"[ExploreAgent] Injecting gap analysis from EP{self._episode_feedback['episode']}")
            else:
                print(f"[ExploreAgent] Episode {self.total_episodes}: "
                      f"selected root only, exploring freely")
        else:
            print(f"[ExploreAgent] Episode {self.total_episodes}: "
                  f"no strategies yet, exploring freely")

        # For action_tree: initialize position tracking
        if is_action_tree:
            if self.current_path:
                # Start at the leaf of the selected path
                self._action_tree_current_node = self.current_path[-1]
            else:
                self._action_tree_current_node = 'root'
                self.current_path = ['root']

    def end_episode(self, state, score):
        """Summarize, backpropagate, reflect if due, persist."""
        print(f"[ExploreAgent] Ending episode {self.total_episodes} with score: {score}")

        # 1. Build action sequence with reward annotations
        action_sequence = []
        for entry in self.game_history:
            if entry.get('action'):
                action = entry['action']
                reward = entry.get('reward', 0)
                if reward and reward != 0:
                    action = f"{action} [+{reward}]" if reward > 0 else f"{action} [{reward}]"
                action_sequence.append(action)

        # 2. Generate episode summary
        #    --skip_summary: use raw game history directly (no LLM call, deterministic reward attribution)
        #    default: LLM-generated milestone-centric summary
        skip_summary = getattr(self.args, 'skip_summary', False)
        if self._python_game or skip_summary:
            summary_text = self._format_raw_game_history(score)
        else:
            milestone_list = self.strategy_space.get_milestone_list() if self.strategy_space.has_strategies() else None
            summary_text = generate_detailed_episode_summary(
                self.game_history, score,
                llm_model=self.args.llm_model,
                temperature=0.3,
                milestones=milestone_list if milestone_list else None,
                game_name=self.game_name,
            )

        initial = getattr(self, '_initial_score', 0)
        episode_summary = {
            'episode': self.total_episodes,
            'score': score,
            'initial_score': initial,
            'total_steps': len(self.game_history),
            'summary': summary_text,
            'actions': action_sequence,
            'strategy_id': ' -> '.join(self.current_path[1:]) if self.current_path and len(self.current_path) > 1 else None,
            'strategy_description': self.current_strategy['description'] if self.current_strategy else None,
        }
        self.episode_summaries.append(episode_summary)
        self.episodes_since_reflection += 1
        print(f"[ExploreAgent] Episode {self.total_episodes} Summary:\n{summary_text}")

        # 3. Backpropagate (use earned score, excluding initial game-start points)
        initial = getattr(self, '_initial_score', 0)
        earned_score = score - initial
        milestone_rewards = {}
        if self.current_path:
            if self._python_game or skip_summary:
                # Code-level reward attribution from game_history (deterministic)
                milestone_rewards, not_attempted = self._compute_rewards_from_history()
            else:
                # LLM summary-based reward attribution
                milestone_rewards = self._parse_summary_rewards(summary_text)
                not_attempted = self._parse_not_attempted_nodes(summary_text)
            is_collect_all = getattr(self.exploration_method, 'collect_all', False)
            self._pending_backprop.append({
                'path': list(self.current_path),
                'earned_score': earned_score,
                'milestone_rewards': dict(milestone_rewards),
                'not_attempted': set(not_attempted) if not_attempted else set(),
                'collect_all': is_collect_all,
                'episode': self.total_episodes,
            })
            print(f"[ExploreAgent] Deferred backprop for episode {self.total_episodes} "
                  f"(path len={len(self.current_path)}, score={earned_score})")

        # 3b. Store success game_history on nodes that earned positive reward
        if hasattr(self.strategy_space, 'nodes'):
            success_node_ids = set()
            if self.current_path:
                for idx, reward in milestone_rewards.items():
                    if reward > 0 and idx + 1 < len(self.current_path):
                        success_node_ids.add(self.current_path[idx + 1])
            # In summary mode, also parse [node_xxx] (+N) from summary text
            # This handles BOTH in-path and off-path reward attribution
            path_node_set = set(self.current_path[1:]) if self.current_path else set()
            if not skip_summary and not self._python_game:
                import re
                for match in re.finditer(r'\[(node_\d+)\].*?\(\+(\d+)', summary_text):
                    node_id = match.group(1)
                    if node_id in self.strategy_space.nodes:
                        success_node_ids.add(node_id)
                        # Off-path reward attribution: update reward stats for nodes
                        # that earned rewards outside the planned path.
                        # Don't increment visits (wasn't deliberately selected),
                        # but update total_reward/max_reward so the node won't be
                        # wrongly abandoned and UCB/Thompson can see its true value.
                        if node_id not in path_node_set:
                            reward = float(match.group(2))
                            node = self.strategy_space.nodes[node_id]
                            node['total_reward'] = node.get('total_reward', 0.0) + reward
                            node['max_reward'] = max(node.get('max_reward', 0.0), reward)
                            node['reward_sq_sum'] = node.get('reward_sq_sum', 0.0) + reward ** 2
                            print(f"[OffPathReward] [{node_id}] '{node.get('milestone','')}' "
                                  f"earned +{reward} off-path (total={node['total_reward']:.0f}, "
                                  f"max={node['max_reward']:.0f})")
            # Save condensed episode history to file (always when score>0, needed for backfill after reflection)
            if score > 0:
                condensed_history = [
                    {k: v for k, v in entry.items() if k != 'full_response'}
                    for entry in self.game_history
                ]
                self._save_episode_history(self.total_episodes, condensed_history)
            # Tag successful nodes with episode number
            for node_id in success_node_ids:
                node = self.strategy_space.nodes.get(node_id)
                if node:
                    node['success_episode'] = self.total_episodes

            # 3c. Attempt notes are extracted during the reflection cycle (every N episodes)

            # 4. Retroactive dep discovery — compare failed milestones against success history
            if hasattr(self.strategy_space, 'nodes') and milestone_rewards is not None:
                from .evolution.dep_discovery import DepDiscovery
                if not hasattr(self, '_dep_discovery'):
                    self._dep_discovery = DepDiscovery()
                dep_ops = self._dep_discovery.run(
                    self.current_path, milestone_rewards, self.strategy_space,
                    self.game_history, self.total_episodes, self.args.llm_model,
                    history_loader=self._load_episode_history)
                if dep_ops:
                    self.strategy_space.apply_operations(dep_ops, source='dep_discovery')
                    print(f"[ExploreAgent] DepDiscovery applied {len(dep_ops)} update(s)")

            # 5. Score gap analysis — compare episode score vs DAG theoretical max
            if hasattr(self.strategy_space, 'get_score_gap_analysis'):
                gap_text = self.strategy_space.get_score_gap_analysis(
                    self.current_path, milestone_rewards, score)
                if gap_text:
                    self._last_gap_analysis = {
                        'text': gap_text,
                        'score': score,
                        'episode': self.total_episodes,
                    }
                    print(f"[ExploreAgent] {gap_text.split(chr(10))[0]}")

        # 7. Cross-episode reflection
        if self.episodes_since_reflection >= self.reflect_interval:
            self._run_reflection()
            self.episodes_since_reflection = 0
            self.episode_summaries = []

            # Backfill success_episode for nodes created by reflection with max_reward>0
            if hasattr(self.strategy_space, 'nodes'):
                for node in self.strategy_space.nodes.values():
                    if node.get('max_reward', 0.0) > 0 and 'success_episode' not in node and not node.get('success_game_history'):
                        node['success_episode'] = self.total_episodes

        # 6b. Auto-set destination & accumulate danger_actions for penalty nodes
        #     For each penalty step this episode, match to penalty nodes by:
        #       1) key_action text match, OR  2) destination room match
        #     This catches action variants like "hit thief with sword" vs "with blade"
        #     (Jericho only — python games have no room IDs)
        if not self._python_game and hasattr(self.strategy_space, 'nodes'):
            penalty_steps = []
            for entry in self.game_history:
                if entry.get('reward', 0) < 0 and entry.get('room_id', -1) >= 0:
                    penalty_steps.append((entry['room_id'], entry['action']))
            if penalty_steps:
                for nid, node in self.strategy_space.nodes.items():
                    if nid == 'root':
                        continue
                    visits = node.get('visits', 0)
                    if visits > 0 and node.get('total_reward', 0) / visits < 0:
                        ka = node.get('key_action', '')
                        first_action = ka.split(',')[0].strip()
                        first_action = re.sub(r'\[from:\s*[^\]]+\]\s*', '', first_action).strip().lower()
                        node_dest = node.get('destination', -1)
                        for room_id, action in penalty_steps:
                            # Match by key_action text OR by destination room
                            matched = (action.lower().strip() == first_action
                                       or (node_dest >= 0 and room_id == node_dest))
                            if matched:
                                node['danger_action'] = action
                                # Accumulate all observed penalty action variants
                                danger_actions = node.setdefault('danger_actions', [])
                                if action not in danger_actions:
                                    danger_actions.append(action)
                                    print(f"[DangerZone] [{nid}] added danger variant: \"{action}\" "
                                          f"(now {len(danger_actions)} variants)")

        # 7. Persist
        self._save_state()

    # ------------------------------------------------------------------ #
    #  Action generation
    # ------------------------------------------------------------------ #

    def generate_action(self, state_node, info=None):
        """Generate next action via LLM call."""
        sys_prompt, user_prompt = self._build_prompts(state_node, info)
        response_format = self.guidance_mode.build_response_schema()

        max_retries = 5
        for attempt in range(max_retries):
            res_obj = chat_completion_with_retries(
                model=self.args.llm_model,
                sys_prompt=sys_prompt,
                prompt=user_prompt,
                max_tokens=2000,
                temperature=self.args.llm_temperature,
                response_format=response_format,
            )

            try:
                if not res_obj or not hasattr(res_obj, 'choices') or not res_obj.choices:
                    raise AttributeError("Empty API response")
                full_response = res_obj.choices[0].message.content
                if full_response is None:
                    raise AttributeError("API returned content=None")
                json_response = json.loads(full_response.strip())
                break
            except (json.JSONDecodeError, AttributeError, TypeError, IndexError) as e:
                if attempt < max_retries - 1:
                    print(f"[ExploreAgent] Parse failed (attempt {attempt + 1}/{max_retries}): {e}")
                else:
                    raise ValueError(f"[ExploreAgent] Failed after {max_retries} attempts: {e}")

        action_text = json_response["action"]

        # Update milestone progress
        if self.current_strategy and self.current_strategy.get('high_level_steps'):
            if self.current_strategy.get('is_dag'):
                # DAG mode: LLM reports only whether the CURRENT TASK is complete
                completed_current = json_response.get('current_milestone_completed', False)
                if completed_current:
                    target_idx = getattr(self.guidance_mode, '_last_target_idx', None)
                    if target_idx is not None and target_idx not in self.guidance_mode.completed_milestones:
                        self.guidance_mode.completed_milestones.add(target_idx)
                        # Auto-set destination on milestone completion
                        cur_rid = info.get('player_location', -1) if info else -1
                        if cur_rid >= 0 and self.current_path:
                            if target_idx + 1 < len(self.current_path):
                                node_id = self.current_path[target_idx + 1]
                                node = self.strategy_space.nodes.get(node_id)
                                if node and node.get('destination', -1) == -1:
                                    node['destination'] = cur_rid
                    self._steps_on_current_milestone = 0
                    self._tracked_milestone_idx = None

                # Auto-skip: if stuck on same milestone for too many steps, skip it
                current_target = getattr(self.guidance_mode, '_last_target_idx', None)
                if current_target is not None:
                    if current_target == self._tracked_milestone_idx:
                        self._steps_on_current_milestone += 1
                    else:
                        self._steps_on_current_milestone = 1
                        self._tracked_milestone_idx = current_target

                    # Exploration nodes get more patience — they need steps to discover new areas
                    node_id_for_skip = self.current_path[current_target + 1] if current_target + 1 < len(self.current_path) else '?'
                    is_explore_node = False
                    node_name = ''
                    if hasattr(self.strategy_space, 'nodes') and node_id_for_skip in self.strategy_space.nodes:
                        node_name = self.strategy_space.nodes[node_id_for_skip].get('milestone', '')
                        key_action = self.strategy_space.nodes[node_id_for_skip].get('key_action', '')
                        is_explore_node = ('explore' in node_name.lower()
                                           or '→ explore' in key_action or '-> explore' in key_action)
                    max_steps_per_milestone = 40 if is_explore_node else 20
                    if self._steps_on_current_milestone >= max_steps_per_milestone and current_target not in self.guidance_mode.completed_milestones:
                        node_id = node_id_for_skip
                        print(f"[ExploreAgent] Auto-skip: milestone {current_target + 1} [{node_id}] '{node_name}' after {max_steps_per_milestone} steps — moving on")
                        self.guidance_mode.completed_milestones.add(current_target)
                        self._steps_on_current_milestone = 0
                        self._tracked_milestone_idx = None

                self.current_milestone_idx = len(self.guidance_mode.completed_milestones)  # backprop compat
            else:
                # Linear mode: single milestone index
                reported = json_response.get('current_milestone', 1)
                if isinstance(reported, int) and reported > 0:
                    new_idx = reported - 1
                    num_milestones = len(self.current_strategy['high_level_steps'])
                    new_idx = min(new_idx, num_milestones)
                    if new_idx > self.current_milestone_idx:
                        old_idx = self.current_milestone_idx
                        self.current_milestone_idx = new_idx
                        # Notify guidance mode of advancement
                        self.guidance_mode.on_milestone_advance(
                            old_idx, new_idx, self.current_strategy)

        # --- Track current room (Jericho only) ---
        room_name = ''
        if not self._python_game:
            self._current_room_id = info.get('player_location', -1) if info else -1
            room_name = info.get('player_location_name', '') if info else ''

        valid_actions = info.get('valid', []) if info else []

        self._add_to_game_history(state_node.state, action_text, full_response,
                                  valid_actions=valid_actions,
                                  room_name=room_name,
                                  room_id=self._current_room_id)

        # For action_tree: expand tree with the action taken
        if self._action_tree_current_node is not None and hasattr(self.strategy_space, 'expand'):
            state_summary = state_node.state[:100] if state_node.state else ''
            new_node_id = self.strategy_space.expand(
                self._action_tree_current_node, action_text.strip(), state_summary)
            if new_node_id not in self.current_path:
                self.current_path.append(new_node_id)
            self._action_tree_current_node = new_node_id

        return action_text.strip(), full_response

    def update_game_history_reward(self, reward, score):
        """Update last game history entry with reward/score."""
        if self.game_history:
            self.game_history[-1]["reward"] = reward
            self.game_history[-1]["score"] = score

    def _compute_rewards_from_history(self):
        """Compute per-milestone rewards from game_history via exact action matching.

        For each step with reward != 0, check if the action exactly matches
        any path node's key_action. No fuzzy matching — if key_action is wrong,
        TreeUpdate will fix it from the raw data in the next reflection.

        Returns:
            (milestone_rewards, not_attempted): milestone_rewards maps path index
            to reward value; not_attempted is always empty (no guessing).
        """
        milestone_rewards = {}

        if not self.current_path or len(self.current_path) <= 1:
            return milestone_rewards, set()

        # Build action -> path_idx lookup from all nodes' key_actions
        # Support both DAG (nodes) and Tree (tree) strategy spaces
        _node_store = getattr(self.strategy_space, 'nodes',
                              getattr(self.strategy_space, 'tree', {}))
        action_to_idx = {}
        for i, node_id in enumerate(self.current_path[1:]):
            node = _node_store.get(node_id, {})
            key_action = node.get('key_action', '')
            for part in key_action.split(','):
                part = part.strip().lower()
                if part:
                    action_to_idx[part] = i

        # Attribute rewards by exact action match
        for entry in self.game_history:
            reward = entry.get('reward', 0)
            if reward == 0:
                continue
            action = entry.get('action', '').strip().lower()
            if action in action_to_idx:
                idx = action_to_idx[action]
                milestone_rewards[idx] = milestone_rewards.get(idx, 0.0) + reward

        return milestone_rewards, set()

    def _parse_summary_rewards(self, summary_text: str) -> Dict[int, float]:
        """Parse episode summary to extract per-milestone rewards.

        Matches patterns like [node_005] ... (+10) in the summary text
        and maps them to path indices for accurate reward attribution.
        """
        import re
        milestone_rewards = {}

        if not self.current_path or len(self.current_path) <= 1:
            return milestone_rewards

        # Build node_id -> path_index mapping (skip root at index 0)
        node_to_idx = {}
        for i, node_id in enumerate(self.current_path[1:]):
            node_to_idx[node_id] = i

        # Match [node_xxx] ... (+N) or (-N) with optional trailing text before ')'
        # Handles: [node_005] ... (+10), [node_001] ... (+10 reward), [node_005] ... (+-10)
        pattern = r'\[(node_\d+)\].*?\(([+-]*\d+)[^)]*\)'
        for match in re.finditer(pattern, summary_text):
            node_id = match.group(1)
            try:
                reward = float(match.group(2).lstrip('+'))
            except ValueError:
                continue
            if node_id in node_to_idx:
                idx = node_to_idx[node_id]
                milestone_rewards[idx] = milestone_rewards.get(idx, 0.0) + reward

        return milestone_rewards

    def _parse_not_attempted_nodes(self, summary_text: str) -> Set[str]:
        """Parse which nodes were NOT_ATTEMPTED from the episode summary.

        Returns a set of node_ids that the agent never reached/attempted.
        These nodes should not have their visits incremented during backprop,
        since the agent failing to navigate there is different from the node
        being truly worthless.
        """
        import re
        not_attempted = set()
        if not self.current_path:
            return not_attempted

        path_nodes = set(self.current_path[1:])
        pattern = r'\[(node_\d+)\][^\n]*\n\s*Status:\s*NOT_ATTEMPTED'
        for match in re.finditer(pattern, summary_text):
            node_id = match.group(1)
            if node_id in path_nodes:
                not_attempted.add(node_id)

        return not_attempted

    def _extract_attempt_notes(self, summary_text: str, path_nodes: set, ep: int):
        """Extract per-node attempt notes from episode summary and store them.

        Parses two patterns from the summary:
        1. Achieved: [node_xxx] Milestone Name (+10) — achieved at Step N
        2. Not achieved: [node_xxx] Milestone Name
             Status: ATTEMPTED/NOT_ATTEMPTED
             Failure reason: ...
        """
        import re
        if not path_nodes or not hasattr(self.strategy_space, 'add_attempt_note'):
            return

        # Pattern 1: Achieved milestones — [node_xxx] ... (+N) — achieved at Step N
        achieved_pattern = r'\[(node_\d+)\][^\n]*\(([+-]*\d+)[^)]*\)[^\n]*'
        for match in re.finditer(achieved_pattern, summary_text):
            node_id = match.group(1)
            if node_id not in path_nodes:
                continue
            try:
                reward = float(match.group(2).lstrip('+'))
            except ValueError:
                reward = 0
            # Extract key steps if present
            line_end = match.end()
            rest = summary_text[line_end:line_end + 200]
            steps_match = re.search(r'Key steps:\s*(.+?)(?:\n|$)', rest)
            steps = steps_match.group(1).strip() if steps_match else ''
            note = f"EP{ep}: achieved ({reward:+.0f})"
            if steps:
                note += f" via {steps[:80]}"
            self.strategy_space.add_attempt_note(node_id, note)

        # Pattern 2: Not achieved milestones — [node_xxx] ... Status: ...
        not_achieved_pattern = (
            r'\[(node_\d+)\][^\n]*\n'
            r'\s*Status:\s*(.*?)(?:\n|$)'
            r'(?:\s*Failure reason:\s*(.*?)(?:\n|$))?'
        )
        for match in re.finditer(not_achieved_pattern, summary_text):
            node_id = match.group(1)
            if node_id not in path_nodes:
                continue
            status = match.group(2).strip() if match.group(2) else ''
            reason = match.group(3).strip() if match.group(3) else ''
            # Include the node's key_action so the agent knows what was tried
            node_data = self.strategy_space.nodes.get(node_id, {})
            key_action = node_data.get('key_action', '')
            if 'NOT_ATTEMPTED' in status:
                note = f"EP{ep}: not attempted (key action: {key_action})" if key_action else f"EP{ep}: not attempted"
                if reason:
                    note += f" — {reason[:80]}"
            elif 'ATTEMPTED' in status:
                note = f"EP{ep}: tried '{key_action}' but failed" if key_action else f"EP{ep}: attempted but failed"
                if reason:
                    note += f" — {reason[:80]}"
            else:
                continue
            self.strategy_space.add_attempt_note(node_id, note)

    # ------------------------------------------------------------------ #
    #  Valid action normalization
    # ------------------------------------------------------------------ #

    # Jericho sometimes exposes navigation as "take into <X>" instead of
    # the canonical short form.  Normalize so the LLM sees familiar commands.
    _ACTION_ALIASES = {
        'take into floor': 'down',
    }

    def _normalize_valid_actions(self, valid_actions):
        """Replace Jericho-specific action aliases with canonical forms."""
        if not valid_actions:
            return valid_actions
        normalized = []
        for a in valid_actions:
            replacement = self._ACTION_ALIASES.get(a)
            if replacement:
                print(f"[ACTION-NORM] '{a}' → '{replacement}'")
                normalized.append(replacement)
            else:
                normalized.append(a)
        return normalized

    # ------------------------------------------------------------------ #
    #  Prompt construction
    # ------------------------------------------------------------------ #

    def _build_prompts(self, state_node, info=None):
        """Build system and user prompts using the guidance mode module."""
        # Some games have actions that Jericho's valid-action detection cannot
        # enumerate (spells, item interactions, examine commands, etc.).
        # Forcing the agent to pick only from valid_actions would lock it out
        # of key commands, so we disable the constraint for these games.
        if self.game_name in ('balances', 'ludicorp', 'deephome'):
            valid_actions = None
        else:
            valid_actions = info.get('valid', []) if (info and self.args.use_valid_actions) else None
        current_inventory = info.get('inv', None) if info else None

        # === System prompt base ===
        if self._scienceworld:
            sys_prompt = """You are an expert scientist completing tasks in a text-based science experiment environment. Points are given for completing sub-goals of the experiment. Type commands to interact with objects and navigate rooms. Pay close attention to the environment's responses to learn which command formats work.
"""
        elif self._python_game:
            sys_prompt = """You are an expert player aiming to maximize your score in a strategy game. Points are given for making progress. Select the best actions based on the game state and memory of past interactions.
"""
        else:
            sys_prompt = """You are an expert player aiming to complete a text-based adventure game. Points are given for making progress in the game. Select promising actions based on the game state and memory of past interactions.
"""
        # Global lessons injection
        if hasattr(self.strategy_space, 'global_lessons') and self.strategy_space.global_lessons:
            lessons = self.strategy_space.global_lessons
            sys_prompt += "\nLESSONS FROM PAST EXPERIENCE (follow these strictly):\n"
            for l in lessons:
                sys_prompt += f"- [{l['category']}] {l['lesson']}\n"
            sys_prompt += "\n"

        # Strategy injection via guidance mode
        if self.current_strategy:
            step_limit = getattr(self.args, 'env_step_limit', 50)
            cur_state = state_node.state if state_node else ''
            room_id = info.get('player_location', -1) if info else self._current_room_id
            # Build danger zones from penalty nodes in the DAG
            danger_zones = []
            if hasattr(self.strategy_space, 'nodes'):
                for nid, node in self.strategy_space.nodes.items():
                    if nid == 'root':
                        continue
                    visits = node.get('visits', 0)
                    if visits > 0:
                        avg = node.get('total_reward', 0) / visits
                        if avg < 0 and node.get('danger_warning') and node.get('destination', -1) >= 0:
                            danger_zones.append({
                                'destination': node['destination'],
                                'danger_warning': node['danger_warning'],
                                'danger_action': node.get('danger_action', ''),
                            })
            strategy_prompt = self.guidance_mode.build_strategy_prompt(
                self.current_strategy, self.current_milestone_idx, step_limit,
                current_step=len(self.game_history), navigation_graph={},
                current_state=cur_state, current_room_id=room_id,
                room_names={},
                danger_zones=danger_zones,
                blocked_edges=set())
            sys_prompt += strategy_prompt
            print(f"[Guidance] Step {len(self.game_history)}:\n{strategy_prompt.strip()}")

        # Inject score gap analysis from last episode (only at episode start, step 0)
        if self._episode_feedback and len(self.game_history) == 0:
            fb = self._episode_feedback
            sys_prompt += f"\n\n{fb['text']}"

        if valid_actions:
            sys_prompt += """\n\n**CRITICAL CONSTRAINT**: When REFERENCE ACTIONS are provided, you MUST ONLY choose actions from that list. Any action not in the REFERENCE ACTIONS list is INVALID and will fail. Do NOT create custom actions. The list is unordered - position doesn't indicate quality."""

        # === Recent history ===
        recent_history = ""
        if self.game_history:
            recent_game_history = self.game_history[-20:]
            start_index = max(0, len(self.game_history) - 20)
            for idx, entry in enumerate(recent_game_history):
                actual_step = start_index + idx
                recent_history += f"Step {actual_step}:\n"
                recent_history += f"State: {entry.get('state', '')}\n"
                recent_history += f"Action: {entry.get('action', '')}\n"
                if entry.get('reward') is not None:
                    recent_history += f"Reward: {entry.get('reward', 0)}\n"
                recent_history += "\n"

        # === Valid actions (with normalization) ===
        valid_actions_text = ""
        if valid_actions:
            valid_actions = self._normalize_valid_actions(valid_actions[:])
            rng = random.Random(time.time_ns())
            rng.shuffle(valid_actions)
            valid_actions_text = f"\nREFERENCE ACTIONS (ONLY VALID ACTIONS):\n{valid_actions}\n\n**STRICT REQUIREMENT**: These are the ONLY valid actions for the current state. You MUST select your actions EXCLUSIVELY from this list. Any action not in this list is INVALID and will be rejected by the game. Do NOT create, modify, or suggest any custom actions.**\n"

        # === Inventory ===
        inventory_text = ""
        if current_inventory:
            inventory_text = f"\nINVENTORY: {current_inventory}\n"

        # === Progress context ===
        current_step = len(self.game_history)
        step_limit = getattr(self.args, 'env_step_limit', 50)
        steps_remaining = step_limit - current_step

        steps_since_last_reward = 0
        for e in reversed(self.game_history):
            if e.get('reward', 0) > 0:
                break
            steps_since_last_reward += 1

        current_score = self.game_history[-1].get('score', 0) if self.game_history else 0

        progress_context = f"\nPROGRESS: Step {current_step}/{step_limit} ({steps_remaining} remaining)"
        progress_context += f" | Score: {current_score}"
        if steps_since_last_reward >= 5:
            progress_context += f" | No points in last {steps_since_last_reward} steps — consider changing approach"

        # === Key action check rule (varies by whether valid actions are available) ===
        if valid_actions:
            key_action_rule = "- ACTION CHECK: Before executing a KEY ACTION or NAVIGATION step from the guidance, verify it exists in the REFERENCE ACTIONS list. If it does NOT appear, a precondition is unmet (e.g., carrying too much to fit through a passage, missing a required tool, door is locked, or wrong game state). Resolve the obstacle first — drop items you don't need, find the required item, or try an alternative route. Do NOT just go back the way you came — that leads to loops."
        else:
            key_action_rule = "- KEY ACTION CHECK: Before executing a KEY ACTION from your current milestone, consider whether preconditions are met (e.g., you have the right items, you are in the right place). If the game rejects your command, a precondition is unmet — resolve the obstacle first"

        # === User prompt ===
        user_prompt = f"""
RECENT STEPS:
{recent_history}
CURRENT STATE: {state_node.state}
{inventory_text}
{progress_context}

TASK:
1. Analyze your progress: What have you achieved? What's your next objective?
2. Propose your best next action with reasoning

RESPONSE FORMAT (JSON):
{self._response_format_example()}

KEY RULES:
- Pay attention to game hints and clues in state descriptions
- If you haven't scored points recently, you are likely stuck — try a fundamentally different approach instead of repeating similar movements
- Avoid revisiting the same locations repeatedly without a clear new action to try there
- LOOP DETECTION: Look at the LAST ACTIONS trail above. If you see yourself alternating between the same 2-3 locations (e.g. north → south → north → south), you are STUCK IN A LOOP. Immediately do something completely different: interact with an object, use an item from inventory, try a command you haven't used, or navigate to a different area entirely
{key_action_rule}

{valid_actions_text}
"""
        return sys_prompt, user_prompt

    def _response_format_example(self) -> str:
        """Return the JSON example for the user prompt, varying by DAG vs linear."""
        if self.current_strategy and self.current_strategy.get('is_dag'):
            return (
                '{\n'
                '    "progress_analysis": "What you\'ve achieved and current challenges",\n'
                '    "current_milestone_completed": false,\n'
                '    "next_objective": "Your next goal",\n'
                '    "reasoning": "Why this action makes sense",\n'
                '    "action": "action command"\n'
                '}'
            )
        return (
            '{\n'
            '    "progress_analysis": "What you\'ve achieved and current challenges",\n'
            '    "current_milestone": "<int, which milestone number you will work on NEXT (1-based)>",\n'
            '    "next_objective": "Your next goal",\n'
            '    "reasoning": "Why this action makes sense",\n'
            '    "action": "action command"\n'
            '}'
        )

    # ------------------------------------------------------------------ #
    #  Reflection orchestration
    # ------------------------------------------------------------------ #

    def _run_reflection(self):
        """Orchestrate cross-episode reflection: tree update then space evolution."""

        # Check if exploration is frozen (no new nodes after freeze episode)
        freeze_ep = getattr(self.args, 'exploration_freeze_episode', 0)
        frozen = freeze_ep > 0 and self.total_episodes > freeze_ep
        if frozen:
            print(f"[ExploreAgent] Exploration frozen (episode {self.total_episodes} > freeze at {freeze_ep}): skipping new node creation")

        # Step 1: Tree Update — encode observed facts
        # Skip TreeUpdate for flat list strategy spaces (prompt assumes tree/DAG structure)
        skip_tree_update = self._module_names.get('strategy_space') == 'plan_flat_list'
        if skip_tree_update:
            print(f"[ExploreAgent] Skipping TreeUpdate for plan_flat_list (not tree-structured)")
            update_result = None
        else:
            update_result = self.tree_update.update(
                self.episode_summaries,
                self.strategy_space,
                self.args.llm_model,
                self.args,
            )

        if update_result:
            update_ops = update_result.get('operations', [])
            if update_ops:
                # TreeUpdate is never frozen — it encodes observed facts (including
                # new reward nodes discovered during free exploration after freeze).
                if update_ops:
                    self.strategy_space.apply_operations(update_ops)

            # Off-path reward attribution is now handled by the deferred
            # ReturnPropagation step below (after map is refined).

        print(f"[ExploreAgent] After tree update: {self.strategy_space.active_count()} active nodes")
        self._display_strategy_space()

        # Step 1.5: ReturnPropagation — flush deferred backprop on the refined map
        for bp in self._pending_backprop:
            if bp['collect_all'] and hasattr(self.strategy_space, '_backpropagate_collect_all'):
                self.strategy_space.backpropagate(
                    bp['path'], bp['earned_score'], bp['milestone_rewards'],
                    collect_all=True, not_attempted=bp['not_attempted'])
            else:
                self.strategy_space.backpropagate(
                    bp['path'], bp['earned_score'], bp['milestone_rewards'],
                    not_attempted=bp['not_attempted'])
            rs = ", ".join(f"M{k}={v:.1f}" for k, v in sorted(bp['milestone_rewards'].items()))
            print(f"[ExploreAgent] ReturnPropagation ep{bp['episode']}: [{rs}]")
        self._pending_backprop.clear()

        # Step 1.6: Attempt notes — extract from episode summaries on the refined map
        skip_summary = getattr(self.args, 'skip_summary', False)
        if not skip_summary and not self._python_game:
            for ep_summary in self.episode_summaries:
                summary_text = ep_summary.get('summary', '')
                strategy_id = ep_summary.get('strategy_id', '')
                path_nodes = set(strategy_id.split(' -> ')) - {'root', ''} if strategy_id else set()
                ep_num = ep_summary.get('episode', 0)
                if summary_text and path_nodes:
                    self._extract_attempt_notes(summary_text, path_nodes, ep_num)

        # Step 2: Space Evolution — built into strategy space (DPM for tree/DAG, Free Reflection for flat list)
        if not frozen:
            self.strategy_space.evolve(
                self.episode_summaries,
                self.args.llm_model,
                self.args,
            )
        else:
            print(f"[ExploreAgent] Frozen: skipping space evolution")

        # Step 3: Stuck Node Diagnosis — analyze underperforming nodes
        self.stuck_node_diagnosis.diagnose(
            self.strategy_space, self.args.llm_model,
            self.total_episodes, self.reflect_interval)

        # Step 4: Global Memory — extract cross-episode lessons
        self.global_memory_updater.update(
            self.episode_summaries, self.strategy_space, self.args.llm_model)

        # Sync root visits
        if hasattr(self.strategy_space, 'sync_root_visits'):
            self.strategy_space.sync_root_visits()

        print(f"[ExploreAgent] After exploration: {self.strategy_space.active_count()} active nodes")
        self._display_strategy_space()

    def _display_strategy_space(self):
        try:
            print(self.strategy_space.format_tree_display())
        except Exception as e:
            print(f"[ExploreAgent] Warning: display failed: {e}")

    def _credit_off_path_rewards(self, analysis: list):
        """Credit rewards to DAG nodes achieved off-path.

        During an episode, rewards are attributed to whichever path milestone
        the agent is currently working on. If the agent achieves a milestone
        that exists in the DAG but wasn't in the selected path, the reward
        goes to the wrong node. This method fixes that using the TreeUpdate
        analysis which correctly maps reward events to existing nodes.
        """
        if not analysis or not hasattr(self.strategy_space, 'nodes'):
            return

        for ep_analysis in analysis:
            ep_num = ep_analysis.get('episode')
            # Find which path was used in this episode
            ep_summary = next(
                (s for s in self.episode_summaries if s.get('episode') == ep_num),
                None)
            if not ep_summary:
                continue

            path_str = ep_summary.get('strategy_id', '')
            path_nodes = set(path_str.split(' -> ')) if path_str else set()

            for m in ep_analysis.get('milestones_achieved', []):
                node_id = m.get('existing_node')
                if not node_id or node_id == 'none':
                    continue
                if node_id in path_nodes:
                    continue  # already credited by normal backprop

                reward = m.get('reward', 0)
                if not reward:
                    continue

                node = self.strategy_space.nodes.get(node_id)
                if not node or node.get('status') != 'active':
                    continue

                node['visits'] = node.get('visits', 0) + 1
                node['total_reward'] = node.get('total_reward', 0.0) + reward
                node['reward_sq_sum'] = node.get('reward_sq_sum', 0.0) + reward ** 2
                print(f"[MilestoneDAG] Off-path credit: [{node_id}] += {reward}"
                      f" (from ep{ep_num})")

    # ------------------------------------------------------------------ #
    #  Persistence
    # ------------------------------------------------------------------ #

    def _load_state(self):
        """Load strategy space and cross-episode state from disk."""
        self.strategy_space.load(self.search_space_path)

        # Load cross-episode metadata
        meta_path = self._meta_path()
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                self.total_episodes = meta.get('total_episodes', 0)
                self.episodes_since_reflection = meta.get('episodes_since_reflection', 0)
                self.episode_summaries = meta.get('episode_summaries', [])
            except Exception as e:
                print(f"[ExploreAgent] Error loading metadata: {e}")

    def _save_state(self):
        """Persist strategy space and cross-episode state."""
        os.makedirs(self.search_space_dir, exist_ok=True)
        self.strategy_space.save(self.search_space_path)

        meta = {
            'total_episodes': self.total_episodes,
            'episodes_since_reflection': self.episodes_since_reflection,
            'episode_summaries': self.episode_summaries,
        }
        try:
            with open(self._meta_path(), 'w') as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[ExploreAgent] Error saving metadata: {e}")

    def _meta_path(self) -> str:
        return os.path.join(self.search_space_dir, 'explore_meta.json')

    def _episode_histories_dir(self) -> str:
        return os.path.join(self.search_space_dir, 'episode_histories')

    def _save_episode_history(self, episode_num: int, condensed_history: List[Dict]):
        """Save condensed game history for an episode to disk."""
        hist_dir = self._episode_histories_dir()
        os.makedirs(hist_dir, exist_ok=True)
        path = os.path.join(hist_dir, f'{episode_num}.json')
        try:
            with open(path, 'w') as f:
                json.dump(condensed_history, f, ensure_ascii=False)
        except Exception as e:
            print(f"[ExploreAgent] Error saving episode history {episode_num}: {e}")

    def _load_episode_history(self, episode_num: int) -> Optional[List[Dict]]:
        """Load condensed game history for an episode from disk."""
        path = os.path.join(self._episode_histories_dir(), f'{episode_num}.json')
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return None
        except Exception as e:
            print(f"[ExploreAgent] Error loading episode history {episode_num}: {e}")
            return None

    # ------------------------------------------------------------------ #
    #  Memory helpers
    # ------------------------------------------------------------------ #

    def add_to_memory(self, state, response):
        self.memory.append({"state": state, "response": response})
        if len(self.memory) > self.args.max_memory:
            self.memory.pop(0)

    def _add_to_game_history(self, state, action, full_response, reward=None, score=None, valid_actions=None, room_name=None, room_id=None):
        entry = {
            "state": state,
            "action": action,
            "full_response": full_response,
            "reward": reward,
            "score": score,
            "valid_actions": valid_actions or [],
        }
        if room_name:
            entry["room_name"] = room_name
        if room_id is not None and room_id >= 0:
            entry["room_id"] = room_id
        self.game_history.append(entry)

    def _format_raw_game_history(self, final_score: int) -> str:
        """Format raw game history as episode summary for python games.

        Skips the LLM summary step entirely — passes full game state, actions,
        rewards, and valid actions directly to reflection. Zero information loss.
        """
        lines = []
        initial = getattr(self, '_initial_score', 0)
        earned = final_score - initial
        lines.append(f"Final score: {final_score} (initial: {initial}, earned by actions: {earned}), Total steps: {len(self.game_history)}")
        if initial > 0:
            lines.append(f"NOTE: The game awards {initial} point(s) automatically at the start BEFORE any player action. Do NOT attribute these points to any action.")
        lines.append("")
        for i, entry in enumerate(self.game_history):
            action = entry.get('action', '')
            state = entry.get('state', '')
            reward = entry.get('reward', 0)
            valid = entry.get('valid_actions', [])

            reward_str = ""
            if reward and reward > 0:
                reward_str = f" **REWARD: +{reward}**"
            elif reward and reward < 0:
                reward_str = f" **REWARD: {reward}**"

            valid_str = f"\nValid actions: {valid}" if valid else ""

            lines.append(f"--- Step {i+1} ---")
            lines.append(f"Action: {action}{reward_str}")
            lines.append(f"Game state:\n{state}")
            if valid_str:
                lines.append(valid_str)
            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  Display helpers
    # ------------------------------------------------------------------ #

    def _print_episode_strategy(self):
        s = self.current_strategy
        if not s:
            return
        print("\n" + "=" * 60)
        print(f"  Episode {self.total_episodes} — Selected Path")
        print("=" * 60)

        # Print path with node stats
        node_store = getattr(self.strategy_space, 'nodes', None) or getattr(self.strategy_space, 'tree', None)
        if node_store and self.current_path:
            for i, node_id in enumerate(self.current_path):
                node = node_store.get(node_id, {})
                milestone = node.get('milestone') or '(root)'
                visits = node.get('visits', 0)
                avg_r = node['total_reward'] / visits if visits > 0 else 0
                indent = "  " * i
                if i == 0:
                    print(f"  {indent}[root] visits={visits} avg_reward={avg_r:.1f}")
                else:
                    max_r = node.get('max_reward', 0.0)
                    print(f"  {indent}-> [{node_id}] {milestone}  visits={visits} avg_reward={avg_r:.1f} max={max_r:.0f}")

        steps = s.get('steps', [])
        if steps:
            print(f"\n  Milestones:")
            for step in steps:
                print(f"    {step}")

        # Print estimated score (sum of max_reward for each node in path, skip root)
        if node_store and self.current_path:
            est_score = 0.0
            for node_id in self.current_path:
                if node_id == 'root':
                    continue
                node = node_store.get(node_id, {})
                mr = node.get('max_reward', 0.0)
                if mr > 0:
                    est_score += mr
            print(f"\n  Estimated score: {est_score:.0f}")

        print("=" * 60 + "\n")
