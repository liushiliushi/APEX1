"""Hierarchical guidance mode (Mode C).

Shows a brief overview of all milestones plus focused detailed instructions
for the current milestone only.

DAG mode: brief progress summary + focused current task with navigation
path and highlighted critical actions.

Linear mode: overview with markers + current milestone detail.
"""
import re
from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple

from .base import GuidanceMode


class HierarchicalGuidance(GuidanceMode):
    """High-level overview + focused current milestone detail."""

    def __init__(self):
        self.completed_milestones: Set[int] = set()
        self._is_dag: bool = False
        self._last_target_idx: Optional[int] = None

    def reset(self):
        self.completed_milestones = set()
        self._last_target_idx = None

    def build_strategy_prompt(self, strategy: Dict[str, Any],
                              milestone_idx: int,
                              step_limit: int,
                              current_step: int = 0,
                              navigation_graph: Dict = None,
                              current_state: str = '',
                              current_room_id: int = -1,
                              room_names: Dict[int, str] = None,
                              **kwargs) -> str:
        hl_steps = strategy.get('high_level_steps', [])
        self._is_dag = strategy.get('is_dag', False)

        if not hl_steps:
            return (
                f"\n\n**CURRENT STRATEGY**: {strategy.get('description', '')}\n"
                f"Choose actions that work toward this goal."
            )

        if navigation_graph is None:
            navigation_graph = {}
        if room_names is None:
            room_names = {}

        if self._is_dag:
            danger_zones = kwargs.get('danger_zones', [])
            blocked_edges = kwargs.get('blocked_edges', set())
            return self._build_dag_prompt(hl_steps, step_limit, current_step,
                                          navigation_graph, current_room_id,
                                          room_names, danger_zones=danger_zones,
                                          blocked_edges=blocked_edges)
        else:
            return self._build_linear_prompt(hl_steps, milestone_idx,
                                             step_limit, current_step)

    def _build_linear_prompt(self, hl_steps: List[Dict], milestone_idx: int,
                             step_limit: int, current_step: int) -> str:
        """Original linear hierarchical logic."""
        cur = milestone_idx
        remaining = max(step_limit - current_step, 0)

        # High-level overview (always visible, brief)
        text = "\n\nSTRATEGY OVERVIEW:\n"
        for i, hl in enumerate(hl_steps):
            step_name = hl.get('step', '')
            if i < cur:
                text += f"  \u2713 {i+1}. {step_name}\n"
            elif i == cur:
                text += f"  \u25b6 {i+1}. {step_name}  \u2190 CURRENT\n"
            else:
                text += f"  \u25cb {i+1}. {step_name}\n"

        if cur >= len(hl_steps):
            text += (
                "\nAll milestones completed! Use remaining steps to explore new areas "
                "and score additional points."
            )
            return text

        # Detailed instructions for current milestone
        hl = hl_steps[cur]
        key_action = hl.get('key_action', '')
        text += f"\n--- CURRENT TASK (Milestone {cur + 1}) ---\n"
        text += f"Goal: {hl.get('step', '')}\n"
        if key_action:
            text += f"Key action sequence: {key_action}\n"
            text += f"Follow this sequence step by step. Do NOT deviate.\n"
        text += (
            f"\nWhen complete, report the updated milestone number in your response."
        )

        return text

    def _build_dag_prompt(self, hl_steps: List[Dict], step_limit: int,
                          current_step: int, nav_graph: Dict[int, Dict[str, int]] = None,
                          current_room_id: int = -1,
                          room_names: Dict[int, str] = None,
                          danger_zones: List[Dict] = None,
                          blocked_edges: set = None) -> str:
        """DAG mode: brief overview + focused current task with navigation."""
        if nav_graph is None:
            nav_graph = {}
        if room_names is None:
            room_names = {}
        completed = self.completed_milestones
        num = len(hl_steps)

        # --- Classify milestones ---
        completed_count = len(completed)
        available = []
        locked_count = 0
        for i in range(num):
            deps_info = hl_steps[i].get('deps_indices', [])
            if i in completed:
                continue
            dep_indices = []
            for d in deps_info:
                if isinstance(d, dict):
                    dep_indices.append(d['idx'])
                else:
                    dep_indices.append(d)
            if all(d in completed for d in dep_indices):
                available.append(i)
            else:
                locked_count += 1

        text = "\n\n"

        if not available:
            if completed_count >= num:
                text += "\nAll milestones completed! Explore new areas for bonus points.\n"
            else:
                text += "\nNo available milestones (all locked). Explore freely.\n"
            return text

        text += f"\nProgress: {completed_count} completed, {len(available)} available, {locked_count} locked\n"

        # --- Focused Current Task ---
        target_idx = self._pick_closest_available(
            available, hl_steps, nav_graph, current_room_id,
            room_names=room_names, previous_target=self._last_target_idx,
            blocked_edges=blocked_edges)
        self._last_target_idx = target_idx
        target = hl_steps[target_idx]
        target_name = target.get('step', '')
        key_action = target.get('key_action', '')

        text += f"\n{'='*40}\n"
        text += f"CURRENT TASK: Milestone {target_idx + 1} \u2014 {target_name}\n"
        text += f"{'='*40}\n"

        # Show current location first, then destination and navigation
        # Disambiguate same-named rooms (e.g. multiple "Maze" rooms)
        display_names = self._disambiguate_room_names(room_names) if room_names else {}
        if current_room_id >= 0:
            cur_name = display_names.get(current_room_id, room_names.get(current_room_id, f'room_{current_room_id}'))
            text += f"\nYour current location: {cur_name}\n"

        # Location-aware danger warnings
        if danger_zones and current_room_id >= 0:
            for dz in danger_zones:
                if dz['destination'] == current_room_id:
                    action = dz.get('danger_action', '')
                    if action:
                        text += f"\n⚠️ WARNING — do NOT execute \"{action}\" here: {dz['danger_warning']}\n"
                    else:
                        text += f"\n⚠️ WARNING: {dz['danger_warning']}\n"

        dest_id = target.get('destination', -1)
        # Check if agent is mid-sequence: current room matches a [from: Room]
        # in the key_action, meaning agent has left destination and is executing
        # the key_action steps. Don't navigate back to destination in that case.
        #
        # IMPORTANT: Only match when the room name is unambiguous. Many games have
        # multiple rooms with the same name (e.g. "Outside", "Maze"). Matching by
        # name alone can cause the agent to execute key_actions in the wrong room.
        mid_sequence = False
        if current_room_id >= 0 and current_room_id != dest_id and key_action:
            # Build reverse lookup: name -> list of room_ids with that name
            name_to_ids: dict = {}
            all_names = display_names if display_names else (room_names if room_names else {})
            for rid, rname in all_names.items():
                name_to_ids.setdefault(rname.strip().lower(), []).append(rid)

            for from_match in re.finditer(r'\[from:\s*([^\]]+)\]', key_action):
                from_room_name = from_match.group(1).strip().lower()
                cur_name_lower = display_names.get(current_room_id, room_names.get(current_room_id, '')).lower()
                if from_room_name and cur_name_lower and from_room_name in cur_name_lower:
                    # Check if this name is ambiguous (multiple rooms share it)
                    matching_ids = name_to_ids.get(cur_name_lower, [])
                    if len(matching_ids) <= 1:
                        mid_sequence = True
                        break
                    # Ambiguous name: only match if nav_graph confirms current room
                    # is reachable from destination (i.e. agent actually left dest)
                    if nav_graph and dest_id >= 0:
                        adj = self._get_nav_adjacency(nav_graph, blocked_edges=blocked_edges)
                        # Check if dest -> current is a direct neighbor
                        for _, neighbor_id in adj.get(dest_id, []):
                            if neighbor_id == current_room_id:
                                mid_sequence = True
                                break
                    if mid_sequence:
                        break

        if isinstance(dest_id, int) and dest_id >= 0:
            dest_name = display_names.get(dest_id, room_names.get(dest_id, f'room_{dest_id}'))
            if mid_sequence:
                text += f"\nYou have left the destination and are executing key actions. Continue with the next key action below.\n"
            else:
                text += f"\nDESTINATION: {dest_name}\n"
                if nav_graph and current_room_id >= 0:
                    route = self._find_route(nav_graph, current_room_id, dest_id, room_names,
                                             blocked_edges=blocked_edges)
                    if route is not None:
                        if route:
                            text += "\nNAVIGATION (from your current area):\n"
                            for from_name, action, to_name in route:
                                text += f"  {from_name} \u2192 {action} \u2192 {to_name}\n"
                        else:
                            text += "You are already here.\n"

        # Detect exploration milestone
        is_explore = ('explore' in target_name.lower()
                      or '\u2192 explore' in key_action or '-> explore' in key_action)

        # Parse actions: split by comma, preserving original order
        all_actions = [a.strip() for a in key_action.split(',') if a.strip()] if key_action else []

        if is_explore:
            text += "\n\u26a0\ufe0f EXPLORATION TASK:\n"
            if all_actions:
                text += f"  Start by going: {', '.join(all_actions)}\n"
            text += (
                "  Thoroughly explore this area and its surroundings:\n"
                "  - Try ALL available exits to discover adjacent rooms\n"
                "  - When you find a new room, explore its exits too \u2014 go deeper\n"
                "  - Pick up any items you find\n"
                "  - Examine interesting objects\n"
                "  Keep exploring until you feel the area is fully covered and there is nothing new to discover.\n"
                "  Only THEN set current_milestone_completed to true.\n"
                "  If the area is dark or completely inaccessible, set current_milestone_completed to true and move on."
            )
        elif all_actions:
            # If agent is not yet at destination, remind to navigate first
            at_dest = (not isinstance(dest_id, int) or dest_id < 0
                       or current_room_id == dest_id or mid_sequence)
            if not at_dest:
                text += "\n>> FIRST follow the NAVIGATION above to reach the destination. Do NOT execute key actions until you arrive.\n"
            text += f"\n\u26a0\ufe0f KEY ACTIONS (execute ONLY after arriving at destination):\n"
            for j, ca in enumerate(all_actions, 1):
                text += f"  {j}. {ca}\n"
        elif key_action:
            text += f"\nKey actions: {key_action}\n"

        # Show attempt history for current task
        attempt_history = self._format_attempt_history(target)
        if attempt_history:
            text += attempt_history

        # Show diagnostic if available
        diagnostic = target.get('diagnostic')
        if diagnostic:
            text += f"\n💡 DIAGNOSTIC (based on {target.get('visits', 0)} previous visits):\n"
            text += f"  Why: {diagnostic['diagnosis']}\n"
            text += f"  Action: {diagnostic['guidance']}\n"
            if diagnostic.get('is_setup_node'):
                text += "  Note: This is a SETUP node — no direct reward expected, but needed for later milestones.\n"

        if not is_explore and all_actions:
            text += (
                f"\nOnly set current_milestone_completed to true AFTER you have executed ALL {len(all_actions)} "
                "key actions above. Do NOT set it to true while actions remain."
            )
        elif not is_explore:
            text += (
                "\nWhen this milestone is complete, set current_milestone_completed to true."
            )

        # Preview next milestone
        next_preview = self._build_next_milestone_preview(
            target_idx, target_name, available, completed, hl_steps,
            nav_graph, room_names, current_room_id=current_room_id,
            blocked_edges=blocked_edges)
        if next_preview:
            text += next_preview

        return text

    @staticmethod
    def _format_attempt_history(hl_step: Dict) -> str:
        """Format attempt notes for a milestone, if any exist."""
        notes = hl_step.get('attempt_notes', [])
        visits = hl_step.get('visits', 0)
        total_reward = hl_step.get('total_reward', 0)
        avg = total_reward / visits if visits > 0 else 0
        if not notes or visits == 0:
            return ''
        # Skip history for milestones that already succeed on average
        if avg > 0:
            return ''
        # Skip for dependency nodes (depended on by other milestones) —
        # showing "tried but failed" misleads the agent into abandoning
        # a correct key_action that is a prerequisite for downstream milestones.
        if hl_step.get('is_depended_on', False):
            return ''
        # Only show notes where the agent actually attempted something
        useful_notes = [n for n in notes[-5:] if 'not attempted' not in n.lower()]
        if not useful_notes:
            return ''
        text = f"\n\u26a0\ufe0f PREVIOUS ATTEMPTS ({visits} visits, avg reward: {avg:.1f}):\n"
        for note in useful_notes[-3:]:
            text += f"  - {note}\n"
        text += "  Try a different approach from what was attempted before.\n"
        return text

    @classmethod
    def _build_next_milestone_preview(cls, current_idx: int, current_name: str,
                                       available: list, completed: set,
                                       hl_steps: list,
                                       nav_graph: Dict[int, Dict[str, int]] = None,
                                       room_names: Dict[int, str] = None,
                                       current_room_id: int = -1,
                                       blocked_edges: set = None) -> Optional[str]:
        """Build a preview of the next milestone with route and key actions."""
        if nav_graph is None:
            nav_graph = {}
        if room_names is None:
            room_names = {}
        num = len(hl_steps)
        future_completed = completed | {current_idx}
        candidates = []
        for i in range(num):
            if i == current_idx or i in future_completed:
                continue
            deps_info = hl_steps[i].get('deps_indices', [])
            dep_indices = []
            for d in deps_info:
                if isinstance(d, dict):
                    dep_indices.append(d['idx'])
                else:
                    dep_indices.append(d)
            if all(d in future_completed for d in dep_indices):
                candidates.append(i)
        if not candidates:
            return None

        # Use agent's actual room for NEXT TASK routing (agent will be here after current task)
        current_dest_id = hl_steps[current_idx].get('destination', -1)
        pick_from_id = current_room_id if current_room_id >= 0 else (
            current_dest_id if isinstance(current_dest_id, int) and current_dest_id >= 0 else -1)
        route_from_id = pick_from_id
        next_idx = cls._pick_closest_available(
            candidates, hl_steps, nav_graph,
            pick_from_id,
            room_names=room_names,
            blocked_edges=blocked_edges)
        next_step = hl_steps[next_idx]
        next_name = next_step.get('step', '')
        next_key_action = next_step.get('key_action', '')

        text = f"\n\nNEXT TASK (after completing this one): Milestone {next_idx + 1} \u2014 {next_name}\n"

        # Route from agent's actual location to next milestone's destination
        if nav_graph and route_from_id >= 0:
            next_dest_id = next_step.get('destination', -1)
            if isinstance(next_dest_id, int) and next_dest_id >= 0:
                if route_from_id == next_dest_id:
                    text += "  Route: You will already be here. Do NOT leave.\n"
                else:
                    route = cls._find_route(nav_graph, route_from_id, next_dest_id, room_names,
                                             blocked_edges=blocked_edges)
                    if route:
                        text += "  Route: "
                        text += " \u2192 ".join(f"{action}" for _, action, _ in route)
                        disp = cls._disambiguate_room_names(room_names) if room_names else {}
                        next_dest_name = disp.get(next_dest_id, room_names.get(next_dest_id, f'room_{next_dest_id}'))
                        text += f" \u2192 {next_dest_name}\n"

        is_next_explore = 'explore' in next_name.lower()
        if is_next_explore:
            text += "  Task: Explore the area (try exits, take items, examine objects)\n"
        elif next_key_action:
            text += f"  Key actions: {next_key_action}\n"

        # Show attempt history for next task
        next_history = cls._format_attempt_history(next_step)
        if next_history:
            # Indent for next task context
            text += next_history

        return text

    @staticmethod
    def _get_nav_adjacency(nav_graph: Dict[int, Dict[str, int]],
                           blocked_edges: set = None) -> Dict[int, List[Tuple[str, int]]]:
        """Convert navigation graph into adjacency list.

        Args:
            nav_graph: {room_id: {action: room_id, ...}, ...}
            blocked_edges: set of (room_id, action) tuples to exclude

        Returns:
            adj[room_id] = [(action, room_id), ...]
        """
        adj: Dict[int, List[Tuple[str, int]]] = {}
        for from_id, edges in nav_graph.items():
            for action, to_id in edges.items():
                if blocked_edges and (from_id, action) in blocked_edges:
                    continue
                adj.setdefault(from_id, []).append((action, to_id))
        return adj

    @staticmethod
    def _disambiguate_room_names(room_names: Dict[int, str]) -> Dict[int, str]:
        """Add room_id suffix to names that appear more than once."""
        if not room_names:
            return {}
        # Count how many room_ids share each display name
        from collections import Counter
        name_counts = Counter(room_names.values())
        result = {}
        for rid, name in room_names.items():
            if name_counts[name] > 1:
                result[rid] = f"{name} (#{rid})"
            else:
                result[rid] = name
        return result

    @classmethod
    def _find_route(cls, nav_graph: Dict[int, Dict[str, int]],
                    from_id: int, to_id: int,
                    room_names: Dict[int, str] = None,
                    blocked_edges: set = None) -> Optional[List[Tuple[str, str, str]]]:
        """BFS route between room IDs. Returns (from_name, action, to_name) tuples."""
        if not nav_graph or from_id < 0 or to_id < 0:
            return None
        if room_names is None:
            room_names = {}
        if from_id == to_id:
            return []

        # Disambiguate same-named rooms by appending room_id
        display_names = cls._disambiguate_room_names(room_names)

        adj = cls._get_nav_adjacency(nav_graph, blocked_edges=blocked_edges)
        queue = deque([(from_id, [])])
        visited = {from_id}
        while queue:
            loc, path = queue.popleft()
            for action, next_id in adj.get(loc, []):
                if next_id in visited:
                    continue
                from_name = display_names.get(loc, f'room_{loc}')
                to_name = display_names.get(next_id, f'room_{next_id}')
                new_path = path + [(from_name, action, to_name)]
                if next_id == to_id:
                    return new_path
                visited.add(next_id)
                queue.append((next_id, new_path))

        return None

    @classmethod
    def _pick_closest_available(cls, available: List[int], hl_steps: List[Dict],
                                nav_graph: Dict[int, Dict[str, int]],
                                current_room_id: int,
                                room_names: Dict[int, str] = None,
                                previous_target: Optional[int] = None,
                                blocked_edges: set = None) -> int:
        """Pick the available milestone that minimises total route cost
        (look-ahead greedy nearest-neighbour TSP).

        Same-room priority: when multiple milestones share the agent's current
        room, complete non-movement (pickup/interaction) milestones first before
        doing movement milestones that leave the room.  This avoids leaving a
        room before collecting everything needed (e.g. taking the sword before
        descending through a one-way trap door).
        """
        if len(available) <= 1:
            return available[0]

        if not nav_graph or current_room_id < 0:
            return available[0]

        adj = cls._get_nav_adjacency(nav_graph, blocked_edges=blocked_edges)
        if not adj:
            return available[0]

        # 1. Collect all relevant rooms (current + each milestone's destination)
        dests: Dict[int, int] = {}  # milestone_idx -> room_id
        all_rooms = {current_room_id}
        for idx in available:
            dest_id = hl_steps[idx].get('destination', -1)
            if isinstance(dest_id, int) and dest_id >= 0:
                dests[idx] = dest_id
                all_rooms.add(dest_id)

        # --- Same-room priority ---
        # If multiple milestones target the agent's current room, prefer
        # non-movement ones (pickup / interaction) over movement ones that
        # would leave the room.
        same_room = [idx for idx in available
                     if dests.get(idx, -1) == current_room_id]
        if len(same_room) >= 2:
            non_movement = [idx for idx in same_room
                            if not cls._key_action_has_movement(hl_steps[idx])]
            if non_movement and len(non_movement) < len(same_room):
                # There are both movement and non-movement milestones here;
                # pick a non-movement one first.
                if len(non_movement) == 1:
                    return non_movement[0]
                # Multiple non-movement: fall through to TSP among them
                available = non_movement

        # 2. BFS from each relevant room to compute pairwise distances
        pair_dist: Dict[tuple, int] = {}
        for src in all_rooms:
            d = {src: 0}
            q = deque([src])
            while q:
                loc = q.popleft()
                for _, nxt in adj.get(loc, []):
                    if nxt not in d:
                        d[nxt] = d[loc] + 1
                        q.append(nxt)
            for dst in all_rooms:
                pair_dist[(src, dst)] = d.get(dst, 999)

        # 3. For each candidate start, greedy nearest-neighbor to estimate total cost
        #    Give the previous target a stickiness bonus to avoid ping-pong switching
        STICKINESS = 3  # previous target gets this many steps of cost discount
        best_start = available[0]
        best_cost = float('inf')
        for start_idx in available:
            cost = pair_dist.get((current_room_id, dests.get(start_idx, -1)), 999)
            cur_room = dests.get(start_idx, -1)
            remaining = [i for i in available if i != start_idx]
            while remaining:
                next_idx = min(remaining,
                               key=lambda i: pair_dist.get((cur_room, dests.get(i, -1)), 999))
                cost += pair_dist.get((cur_room, dests.get(next_idx, -1)), 999)
                cur_room = dests.get(next_idx, -1)
                remaining.remove(next_idx)
            # Stickiness: discount the previous target to prevent ping-pong
            if previous_target is not None and start_idx == previous_target:
                cost -= STICKINESS
            if cost < best_cost:
                best_cost = cost
                best_start = start_idx

        return best_start

    # Direction words used by _key_action_has_movement
    _DIRECTION_WORDS = frozenset({
        'north', 'south', 'east', 'west', 'up', 'down',
        'northeast', 'northwest', 'southeast', 'southwest',
        'ne', 'nw', 'se', 'sw', 'in', 'out',
    })

    @classmethod
    def _key_action_has_movement(cls, hl_step: Dict) -> bool:
        """Return True if the milestone's key_action contains a movement command."""
        ka = hl_step.get('key_action', '')
        if not ka:
            return False
        if '[from:' in ka or '[from :' in ka:
            return True
        # Check each comma-separated action for bare direction words
        for action in ka.split(','):
            action = action.strip().lower()
            if action in cls._DIRECTION_WORDS:
                return True
        return False

    def build_response_schema(self) -> Dict[str, Any]:
        if self._is_dag:
            properties = {
                "progress_analysis": {"type": "string"},
                "current_milestone_completed": {"type": "boolean"},
                "next_objective": {"type": "string"},
                "reasoning": {"type": "string"},
                "action": {"type": "string"},
            }
            required = ["progress_analysis", "current_milestone_completed", "next_objective", "reasoning", "action"]
        else:
            properties = {
                "progress_analysis": {"type": "string"},
                "current_milestone": {"type": "integer"},
                "next_objective": {"type": "string"},
                "reasoning": {"type": "string"},
                "action": {"type": "string"},
            }
            required = ["progress_analysis", "current_milestone", "next_objective", "reasoning", "action"]

        return {
            "type": "json_schema",
            "json_schema": {
                "name": "game_action",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False
                }
            }
        }

    def on_milestone_advance(self, old_idx: int, new_idx: int,
                             strategy: Dict[str, Any]) -> Optional[str]:
        return None
