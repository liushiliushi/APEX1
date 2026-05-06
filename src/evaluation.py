import os
import json
import time
import random
import numpy as np
import statistics
from typing import Tuple, List, Dict, Any
import src.utils as utils

from .memory_agent import MemoryAgent
from .naive_agent import NaiveAgent
from .reflexion_agent import ReflexionAgent
from .ace_agent import ACEAgent
from .explore_agent import ExploreAgent
from .env import JerichoEnv, PythonGameEnv, ScienceWorldGameEnv

class StateNode:
    def __init__(self, state, reward=0.0):
        self.state = state
        self.reward = reward 
        self.response = ""

class GameEvaluator:
    
    # Python game names that use PythonGameEnv instead of Jericho
    PYTHON_GAMES = {'catnip', 'catnip_singularity', 'catnip_easy', 'catnip_singularity_easy',
                    'catnip_hard', 'catnip_singularity_hard'}

    def __init__(self, args):
        self.args = args
        self.base_output_path = os.path.join(self.args.output_path, self.args.game_name)
        os.makedirs(self.base_output_path, exist_ok=True)
        if self.args.seed is None:
            self.args.seed = random.randint(0, 100000)
        np.random.seed(self.args.seed)
        random.seed(self.args.seed)

        # Determine environment type
        env_type = getattr(self.args, 'env_type', 'auto')
        if env_type == 'auto':
            if self.args.game_name in self.PYTHON_GAMES:
                env_type = 'python'
            elif self.args.game_name.startswith('sw:') or env_type == 'scienceworld':
                env_type = 'scienceworld'
            else:
                env_type = 'jericho'
        self.env_type = env_type

        if self.env_type == 'jericho':
            self.args.rom_path = os.path.join(self.args.rom_path, utils.game_file(self.args.game_name))
    
    def run_evaluation(self) -> Dict[str, Any]:

        print(f"\n--- STARTING EVALUATION ---")
        print(f"Game: {self.args.game_name}, Agent LLM Model: {self.args.llm_model}")
        print(f"Agent Type: {self.args.agent_type}, Runs for statistics: {self.args.eval_runs}")
        print(f"Base Seed for evaluation session: {self.args.seed}")
        
        model_slug = self.args.llm_model.replace('/', '_').replace('\\', '_')
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.log_dir = os.path.join(self.base_output_path, self.args.agent_type, model_slug, timestamp)
        os.makedirs(self.log_dir, exist_ok=True)
                
        print(f"Episode logs and summary will be in: {self.log_dir}")

        if self.args.agent_type == 'memory':
            agent = MemoryAgent(self.args, guiding_prompt="")
        elif self.args.agent_type == 'naive':
            agent = NaiveAgent(self.args, guiding_prompt="")
        elif self.args.agent_type == 'explore':
            agent = ExploreAgent(self.args, guiding_prompt="")
            agent.search_space_dir = self.log_dir
            agent.search_space_path = os.path.join(self.log_dir, 'search_space.json')
        elif self.args.agent_type == 'reflexion':
            agent = ReflexionAgent(self.args, guiding_prompt="")
        elif self.args.agent_type == 'ace':
            agent = ACEAgent(self.args, guiding_prompt="")
        else:
            raise ValueError(f"Unknown agent type: {self.args.agent_type}.")

        # Resume: copy search_space and metadata from previous run
        if getattr(self.args, 'resume_from', None):
            import shutil
            resume_dir = self.args.resume_from
            for fname in ['search_space.json', 'explore_meta.json']:
                src = os.path.join(resume_dir, fname)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(self.log_dir, fname))
                    print(f"[Resume] Copied {fname} from {resume_dir}")
            # Copy episode_histories directory (needed by dep_discovery)
            hist_src = os.path.join(resume_dir, 'episode_histories')
            if os.path.isdir(hist_src):
                hist_dst = os.path.join(self.log_dir, 'episode_histories')
                shutil.copytree(hist_src, hist_dst)
                print(f"[Resume] Copied episode_histories/ from {resume_dir}")

        scores = []
        strategies_used = []
        strategy_ids_used = []
        for i in range(self.args.eval_runs):
            print(f"\nStarting game run {i+1}/{self.args.eval_runs}...")
            current_score, _ = self.run_game_episode(agent, episode_num=i)
            scores.append(current_score)
            # Record which strategy was used for this episode
            if hasattr(agent, 'current_strategy') and agent.current_strategy:
                strategies_used.append(agent.current_strategy.get('description', 'unknown'))
                # For tree-based agent, record MCTS path; otherwise fall back to strategy id
                if hasattr(agent, 'current_path') and agent.current_path and len(agent.current_path) > 1:
                    strategy_ids_used.append(' → '.join(agent.current_path[1:]))
                else:
                    strategy_ids_used.append(agent.current_strategy.get('id', 'unknown'))
            else:
                strategies_used.append('free exploration')
                strategy_ids_used.append('none')
            print(f"Run {i+1}/{self.args.eval_runs} finished. Score: {current_score}")

            average_score = statistics.mean(scores)
            stdev_score = statistics.stdev(scores) if len(scores) > 1 else 0

            print(f"\n--- EVALUATION SUMMARY ({self.args.agent_type} method) ---")
            print(f"Agent LLM Model: {self.args.llm_model}")
            print(f"Number of final evaluation runs: {self.args.eval_runs}")
            print(f"Individual Scores: {scores}")
            print(f"Strategy IDs Used: {strategy_ids_used}")
            print(f"Strategies Used:   {strategies_used}")
            print(f"Average Score: {average_score:.2f}")
            print(f"Standard Deviation: {stdev_score:.2f}")
            
            model_slug = self.args.llm_model.replace('/', '_').replace('\\', '_')
            summary_file_name = f"evaluation_summary_{self.args.agent_type}_{model_slug}_{timestamp}.json"
            summary_path = os.path.join(self.log_dir, summary_file_name)
            
            summary_data = {
                "individual_scores": scores,
                "strategy_ids_used": strategy_ids_used,
                "strategies_used": strategies_used,
                "average_score": average_score,
                "stdev_score": stdev_score,
                "timestamp_evaluation_session_start": timestamp,
                "timestamp_summary_created": time.strftime("%Y%m%d-%H%M%S"),
                "all_args": vars(self.args),
                "success": True
            }
            
            with open(summary_path, 'w') as f:
                json.dump(summary_data, f, indent=4)

        return summary_data


    def run_game_episode(self, agent, episode_num: int = 0):
        self.args.llm_model_name_for_log = self.args.llm_model.replace('/', '_') # for logging
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_file_name = f"episode_{episode_num:03d}_{timestamp}.txt"
        log_file_path = os.path.join(self.log_dir, log_file_name)
        os.makedirs(self.log_dir, exist_ok=True)

        episode_specific_seed = self.args.seed + episode_num if self.args.seed is not None else random.randint(0, 1000000)

        if self.env_type == 'python':
            env = PythonGameEnv(game_name=self.args.game_name,
                                seed=episode_specific_seed,
                                step_limit=self.args.env_step_limit)
        elif self.env_type == 'scienceworld':
            # game_name format: "sw:task_name" or just task name with --env_type scienceworld
            task_name = self.args.game_name
            if task_name.startswith('sw:'):
                task_name = task_name[3:]
            env = ScienceWorldGameEnv(
                task_name=task_name,
                variation=getattr(self.args, 'sw_variation', 0),
                simplification_str=getattr(self.args, 'sw_simplification', 'easy'),
                step_limit=self.args.env_step_limit,
                seed=episode_specific_seed)
        else:
            env = JerichoEnv(rom_path=self.args.rom_path,
                            seed=episode_specific_seed,
                            step_limit=self.args.env_step_limit)

        ob, info = env.reset()
        score = info['score']

        with open(log_file_path, 'w', encoding='utf-8') as output:
            output.write(f"Game: {self.args.game_name}, Model: {self.args.llm_model}, Episode: {episode_num}, Seed for episode: {episode_specific_seed}\n")
            # Pass initial score to agents that support it (e.g. ExploreAgent)
            import inspect
            if 'initial_score' in inspect.signature(agent.start_episode).parameters:
                agent.start_episode(initial_score=score)
            else:
                agent.start_episode()
            output.write(f"Guiding prompt: {agent.guiding_prompt}\n")

            for step in range(self.args.env_step_limit):
                state_node = StateNode(state=ob)

                # Jericho-specific: add missing action for technician interaction
                if self.env_type == 'jericho' and 'technician' in ob.lower() and 'copier technician' not in ob.lower():
                    if 'ask technician about gate' not in info['valid']:
                        info['valid'].append('ask technician about gate')

                action, raw_llm_output = agent.generate_action(state_node, info=info)

                output.write(f"==========\n")
                output.write(f"[STEP] {step}\n")
                output.write(f"----------\n")
                output.write(f"[OBS] {ob}\n")
                output.write(f"----------\n")
                output.write(f"[INV] {info.get('inv', 'N/A')}\n")
                valid_actions = info.get('valid', []) if self.args.use_valid_actions else None
                if valid_actions:
                    output.write(f"[VALID_ACTIONS] {valid_actions}\n")
                output.write(f"[RAW_LLM_OUTPUT] {raw_llm_output}\n")
                output.write(f"[CHOSEN_ACTION] {action}\n")
                output.write(f"----------\n")
                prev_ob = ob
                ob, reward, done, info = env.step(action)
                score = info['score']
                output.write(f"[REWARD] {reward}\n")
                output.write(f"[CUM_REWARD] {score}\n")  
                output.write(f"----------\n")

                # Feed reward and score back into the agent for cross-episode memory
                try:
                    agent.update_game_history_reward(reward, score)
                except Exception:
                    pass

                prev_ob_short = prev_ob[:70].replace('\n', ' ').ljust(70)
                print(f"[STEP {step}] | {prev_ob_short} || Action: {action.ljust(25)} || Reward: {reward}, Score: {score}")
                
                if done:
                    output.write(f"\nGame finished at step {step}. Final score: {score}\n")
                    break
            if not done:
                output.write(f"\nStep limit ({self.args.env_step_limit}) reached. Final score: {score}\n")
            agent.end_episode(state=ob, score=score)
            
        return score, log_file_path
