from jericho import *
import sys
import random

try:
    from .python_games import load_game as _load_python_game
except ImportError:
    _load_python_game = None

try:
    from scienceworld import ScienceWorldEnv as _ScienceWorldEnv
except ImportError:
    _ScienceWorldEnv = None


_SYSTEM_RANDOM = random.SystemRandom()


def _shuffle_valid_actions(actions):
    """Return a freshly shuffled copy of valid actions."""
    if not actions:
        return actions
    shuffled = list(actions)
    _SYSTEM_RANDOM.shuffle(shuffled)
    return shuffled

class JerichoEnv:
    ''' Returns valid actions at each step of the game. '''

    def __init__(self, rom_path, seed, step_limit=None, get_valid=True, cache = None):
        self.rom_path = rom_path
        self.env = FrotzEnv(rom_path)
        self.bindings = self.env.bindings
        self.seed = seed
        self.steps = 0
        self.end_scores = []
        self.step_limit = step_limit
        self.get_valid = get_valid
        self.max_score = 0
        self.cache = cache
        
        # 检测是否在调试模式下，如果是则禁用jericho的并行处理
        self.use_parallel = not self._is_debugging()
        

    def get_objects(self):
        desc2objs = self.env._identify_interactive_objects(use_object_tree=False)
        obj_set = set()
        for objs in desc2objs.values():
            for obj, pos, _ in objs:
                if pos == 'ADJ': continue
                obj_set.add(obj)
        return list(obj_set)

    def _is_debugging(self):
        """检测是否在调试模式下运行"""
        debugger_indicators = [
            'pydevd',  # PyCharm/VSCode debugger
            'debugpy',  # VSCode Python debugger
            'pdb',     # Python debugger
            'ipdb',    # IPython debugger
        ]
        
        # 检查是否有调试器模块被导入
        for indicator in debugger_indicators:
            if any(indicator in module_name for module_name in sys.modules):
                return True
        return False

    def _get_state_hash(self, ob):
        return self.env.get_world_state_hash()
    

    def step(self, action):
        ob, reward, done, info = self.env.step(action)

        # Initialize with default values
        info['look'] = 'unknown'
        info['inv'] = 'unknown'
        info['valid'] = ['wait', 'yes', 'no']
        info['player_location'] = -1
        info['player_location_name'] = ''
        if not done:
            try:
                loc = self.env.get_player_location()
                info['player_location'] = loc.num if loc else -1
                info['player_location_name'] = loc.name if loc else ''
            except:
                info['player_location'] = -1
                info['player_location_name'] = ''
            save = self.env.get_state()
            hash_save = self._get_state_hash(ob)
            if self.cache is not None and hash_save in self.cache:
                info['look'], info['inv'], info['valid'] = self.cache[hash_save]
            else:
                look, _, _, _ = self.env.step('look')
                info['look'] = look.lower()
                self.env.set_state(save)
                inv, _, _, _ = self.env.step('inventory')
                info['inv'] = inv.lower()
                self.env.set_state(save)
                # Get the valid actions for this state
                if self.get_valid:
                    valid = self.env.get_valid_actions(use_parallel=self.use_parallel)
                    if len(valid) == 0:
                        valid = ['wait', 'yes', 'no']
                    info['valid'] = _shuffle_valid_actions(valid)
                if self.cache is not None:
                    self.cache[hash_save] = info['look'], info['inv'], info['valid']
        self.steps += 1
        if self.step_limit and self.steps >= self.step_limit:
            done = True
        self.max_score = max(self.max_score, info['score'])
        if done: 
            self.end_scores.append(info['score'])
        return ob.lower(), reward, done, info

    def reset(self):
        initial_ob, info = self.env.reset()
        try:
            loc = self.env.get_player_location()
            info['player_location'] = loc.num if loc else -1
            info['player_location_name'] = loc.name if loc else ''
        except:
            info['player_location'] = -1
            info['player_location_name'] = ''
        save = self.env.get_state()
        look, _, _, _ = self.env.step('look')
        info['look'] = look
        self.env.set_state(save)
        inv, _, _, _ = self.env.step('inventory')
        info['inv'] = inv
        self.env.set_state(save)
        valid = self.env.get_valid_actions(use_parallel=self.use_parallel)
        info['valid'] = _shuffle_valid_actions(valid)
        self.steps = 0
        self.max_score = 0
        return initial_ob, info


    def copy(self):
        copy_env = JerichoEnv(self.rom_path, self.seed)
        copy_env.env = self.env.copy()
        copy_env.cache = self.cache
        return copy_env
    

    def close(self):
        self.env.close()


class PythonGameEnv:
    """Wrapper for pure-Python text games (e.g. Catnip Singularity)."""

    def __init__(self, game_name, seed, step_limit=None, get_valid=True):
        if _load_python_game is None:
            raise ImportError("python_games module not found.")
        self.game_name = game_name
        self.game = _load_python_game(game_name, seed=seed)
        self.seed = seed
        self.steps = 0
        self.end_scores = []
        self.step_limit = step_limit
        self.get_valid = get_valid
        self.max_score = 0
        self.score = 0

    def _safe_call(self, method_name, default_value):
        method = getattr(self.game, method_name, None)
        if not callable(method):
            return default_value
        try:
            return method()
        except Exception:
            return default_value

    def _normalize_info(self, info, ob, reward):
        if info is None:
            info = {}
        if "score" in info:
            self.score = info["score"]
        else:
            self.score += reward
            info["score"] = self.score
        if "look" not in info:
            info["look"] = self._safe_call("get_look", ob)
        if "inv" not in info:
            info["inv"] = self._safe_call("get_inventory", "")
        if "valid" not in info:
            if self.get_valid:
                info["valid"] = self._safe_call("get_valid_actions", [])
            else:
                info["valid"] = []
        info["valid"] = _shuffle_valid_actions(info["valid"])
        # Provide defaults for fields that Jericho normally supplies
        info.setdefault("player_location", -1)
        info.setdefault("player_location_name", "")
        return info

    def step(self, action):
        ob, reward, done, info = self.game.step(action)
        info = self._normalize_info(info, ob, reward)
        self.steps += 1
        if self.step_limit and self.steps >= self.step_limit:
            done = True
        self.max_score = max(self.max_score, info["score"])
        if done:
            self.end_scores.append(info["score"])
        return ob, reward, done, info

    def reset(self):
        ob, info = self.game.reset()
        self.steps = 0
        self.score = 0
        info = self._normalize_info(info, ob, reward=0)
        self.max_score = max(self.max_score, info["score"])
        return ob, info

    def close(self):
        if hasattr(self.game, "close"):
            self.game.close()


class ScienceWorldGameEnv:
    """Wrapper for ScienceWorld text-based science experiment environment.

    Maps ScienceWorld's API to the same interface as JerichoEnv/PythonGameEnv:
      reset() -> (observation, info)
      step(action) -> (observation, reward, done, info)
      close()

    Args:
        task_name: Task name or ID (e.g., 'boil', '1-1', 'melt').
        variation: Task variation index. Use -1 for random train variation each episode.
        simplification_str: Comma-separated simplifications (e.g., 'easy').
        step_limit: Max steps per episode.
        seed: Not used by ScienceWorld directly, kept for interface compatibility.
    """

    # All 30 ScienceWorld tasks for reference
    TASK_NAMES = None  # Populated on first instantiation

    def __init__(self, task_name, variation=0, simplification_str='easy',
                 step_limit=100, seed=0):
        if _ScienceWorldEnv is None:
            raise ImportError(
                "scienceworld package not found. Install with: pip install scienceworld"
            )
        self.env = _ScienceWorldEnv("", envStepLimit=step_limit)
        self.task_name = task_name
        self.variation = variation
        self.simplification_str = simplification_str
        self.step_limit = step_limit
        self.seed = seed
        self.steps = 0
        self.end_scores = []
        self.max_score = 0

        # Resolve task name (support both name and numeric ID like '1-1')
        task_names = self.env.get_task_names()
        if ScienceWorldGameEnv.TASK_NAMES is None:
            ScienceWorldGameEnv.TASK_NAMES = task_names

        # If task_name is numeric index, convert
        if task_name.isdigit():
            idx = int(task_name)
            if 0 <= idx < len(task_names):
                self.task_name = task_names[idx]
        elif task_name not in task_names:
            # Try partial match
            matches = [t for t in task_names if task_name.lower() in t.lower()]
            if matches:
                self.task_name = matches[0]
            else:
                raise ValueError(
                    f"Task '{task_name}' not found. Available: {task_names}"
                )

        # Load task
        var_idx = self.variation
        if var_idx < 0:
            self.env.load(self.task_name, 0, self.simplification_str)
            var_idx = self.env.get_random_variation_train()
        self.env.load(self.task_name, var_idx, self.simplification_str)
        self._current_variation = var_idx

    def _get_valid_actions(self):
        """Return empty list — ScienceWorld has 500-1000+ valid actions per step,
        too many to include in prompts. Let the agent discover commands through exploration."""
        return []

    def _get_look(self):
        try:
            return self.env.look()
        except Exception:
            return ''

    def _get_inventory(self):
        try:
            return self.env.inventory()
        except Exception:
            return ''

    def _build_info(self, score):
        return {
            'score': score,
            'look': self._get_look(),
            'inv': self._get_inventory(),
            'valid': self._get_valid_actions(),
            'player_location': -1,
            'player_location_name': '',
            'task_description': self.env.get_task_description(),
            'goal_progress': '',
        }

    def reset(self):
        # Re-load with possibly new variation for new episode
        var_idx = self.variation
        if var_idx < 0:
            var_idx = self.env.get_random_variation_train()
        self.env.load(self.task_name, var_idx, self.simplification_str)
        self._current_variation = var_idx

        ob, _ = self.env.reset()
        self.steps = 0
        self.max_score = 0

        info = self._build_info(score=0)
        # Prepend task description to initial observation
        task_desc = self.env.get_task_description()
        ob = f"Task: {task_desc}\n\n{ob}"
        return ob, info

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        score = info.get('score', 0)

        self.steps += 1
        if self.step_limit and self.steps >= self.step_limit:
            done = True

        self.max_score = max(self.max_score, score)

        # Build standardized info
        std_info = self._build_info(score=score)

        if done:
            self.end_scores.append(score)
            try:
                std_info['goal_progress'] = self.env.get_goal_progress()
            except Exception:
                pass

        return ob, reward, done, std_info

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass
