import os
import sys
import json
import argparse
from datetime import datetime
from dotenv import load_dotenv
from src.evaluation import GameEvaluator

load_dotenv(override=True)


class TeeStream:
    """Write to both terminal and log file simultaneously."""
    def __init__(self, log_file, original_stream):
        self.log_file = log_file
        self.original_stream = original_stream

    def write(self, data):
        self.original_stream.write(data)
        self.original_stream.flush()
        self.log_file.write(data)
        self.log_file.flush()

    def flush(self):
        self.original_stream.flush()
        self.log_file.flush()

    def fileno(self):
        return self.original_stream.fileno()

    def isatty(self):
        return self.original_stream.isatty()




try:
    from debug_fix import setup_multiprocessing_for_debug
except ImportError:
    def setup_multiprocessing_for_debug():
        pass  # No-op if debug_fix module is not available


def parse_args():
    parser = argparse.ArgumentParser()
    
    # Game
    parser.add_argument('--rom_path', default='jericho-games/', type=str, help="Path to the directory containing game ROMs.")
    # parser.add_argument('--game_name', default='library', type=str, help="Name of the game to play (e.g., 'zork1', 'library').")
    parser.add_argument('--game_name', default='library', type=str, help="Name of the game to play (e.g., 'zork1', 'library').")
    parser.add_argument('--env_type', default='auto', choices=['auto', 'jericho', 'python', 'scienceworld'], type=str,
                    help="Environment type. 'auto' detects from game_name (sw:* → scienceworld, catnip* → python, else jericho).")

    parser.add_argument('--output_path', default='output', type=str, help="Base directory for all output, logs, and outputs.") 
    parser.add_argument('--env_step_limit', default=20, type=int, help="Maximum number of steps per game episode.")
    parser.add_argument('--seed', default=0, type=int, help="Random seed for reproducibility. If None, a random seed is used.")

    # LLM
    # parser.add_argument('--llm_model', default='anthropic/claude-4-sonnet-20250522', type=str, help="LLM model for the game-playing agent.")
    # parser.add_argument('--llm_model', default='openai/gpt-4o', type=str, help="LLM model for the game-playing agent.")
    # parser.add_argument('--llm_model', default='openai/gpt-4o-mini', type=str, help="LLM model for the game-playing agent.")
    # parser.add_argument('--llm_model', default='openai/gpt-4o', type=str, help="LLM model for the game-playing agent.")
    parser.add_argument('--llm_model', default='google/gemini-3-flash-preview', type=str, help="LLM model for the game-playing agent.")
    # parser.add_argument('--llm_model', default='openai/gpt-5-mini', type=str, help="LLM model for the game-playing agent.")
    # parser.add_argument('--llm_model', default='openai/gpt-oss-120b', type=str, help="LLM model for the game-playing agent.")

    parser.add_argument('--top_actions', default=3, type=int, help="Number of potential action.")
    parser.add_argument('--llm_temperature', default=0.8, type=float, help="Temperature for the agent's LLM.")
    parser.add_argument('--max_memory', default=30, type=int, help="Maximum number of past states to keep in memory for the agent.")
    parser.add_argument('--gamma', default=0.5, type=float, help="Discount factor for computing returns in cross-episode memory.")
    parser.add_argument('--max_trajectory_window', default=5, type=int, help="Maximum window size for trajectory comparison (uses sliding window for longer trajectories).")
    parser.add_argument('--exploration_rate', default=0.65, type=float, help="Exploration rate parameter for adjusting confidence calculation in action selection.")
    parser.add_argument('--use_history_prompt', default=True, action=argparse.BooleanOptionalAction, help="Use history-based prompt generation that compares with top-scoring episodes (recommended).")

    # Debug options
    parser.add_argument('--debug_info', default=False, action=argparse.BooleanOptionalAction, help='Print detailed info updates during game episodes.')
    parser.add_argument('--track_valid_changes', default=False, action=argparse.BooleanOptionalAction, help='Track valid action changes (if applicable).')

    # Evaluation parameters
    parser.add_argument('--agent_type', type=str, default='explore', choices=['naive', 'memory', 'reflexion', 'ace', 'explore'], help='Method to evaluate.')

    parser.add_argument('--eval_runs', type=int, default=50, help='Number of episodes to run for statistical evaluation.')
    parser.add_argument('--evol_temperature', default=0.8, type=float, help="Temperature for the evolutionary's LLM.")

    # Summary agent parameters
    # parser.add_argument('--summary_llm_model', type=str, help='LLM model for summarization (defaults to game LLM if not specified).')
    parser.add_argument('--summary_temperature', type=float, default=0.8, help='Temperature for the summarization LLM.')
    parser.add_argument('--summary_max_tokens', type=int, default=300, help='Maximum tokens for summarization response.')

    # RAG agent parameters
    parser.add_argument('--retrieval_top_k', type=int, default=10, help='Number of top-k most similar trajectories to retrieve from cross-episode memory.')
    parser.add_argument('--retrieval_threshold', type=float, default=0.95, help='Similarity threshold for retrieving relevant history entries.')
    parser.add_argument('--use_vector_similarity', default=False, action=argparse.BooleanOptionalAction,
                    help='Use vector similarity (0.25*hist + 0.75*state) instead of Jaccard n-gram similarity for retrieval ranking.')
    # parser.add_argument('--embedding_model', type=str, help='LLM model for RAG enhancement (defaults to game LLM if not specified).')
    # parser.add_argument('--embedding_api_key', type=str, help='API key for embedding API (if different from main LLM API key).')
    # parser.add_argument('--rag_temperature', type=float, default=0.4, help='Temperature for the RAG enhancement LLM.')
    # parser.add_argument('--rag_max_tokens', type=int, default=400, help='Maximum tokens for RAG enhancement response.')

    # Evolutionary parameters (used by EvolutionaryPrompter and for 'evolved' evaluation)
    parser.add_argument('--evolution_llm_model', default='google/gemini-3-flash-preview', type=str, help='LLM model for the evolutionary operator.')
    parser.add_argument('--initial_prompts_file', default='initial_prompts.json', type=str, help='JSON file with initial prompts to seed the pool (relative to project root or absolute).')
    parser.add_argument('--exploration_constant', default=1.0, type=float, help='Exploration constant for UCB calculation in tree-based agent.')
    parser.add_argument('--depth_constant', default=0.8, type=float, help='Decay factor of exploration term in tree-based agent.')

    parser.add_argument('--freeze_on_win', default=True, action=argparse.BooleanOptionalAction,
                    help='Once any node reaches win_freeze_threshold, stop exploration and reuse the best prompt thereafter.')
    parser.add_argument('--win_freeze_threshold', type=int, default=0,
                    help='Score threshold to freeze on win (e.g., 310 for detective). 0 disables freezing.')
    parser.add_argument('--force_best_after_drop', default=True, action=argparse.BooleanOptionalAction,
                    help='If the last episode score drops far below best, force exploiting the best prompt next episode.')
    parser.add_argument('--drop_threshold', type=int, default=50,
                    help='Score drop margin vs best to trigger forced exploit.')

    # Cross-episode memory toggle (few-shot positives + negative contrast during evolution)
    parser.add_argument('--enable_cross_mem', default=True, action=argparse.BooleanOptionalAction,
                    help='Enable cross-episode memory: store successful/failed snippets across episodes, few-shot retrieval, and negative-contrast evolution.')

    # Guiding prompt update control
    parser.add_argument('--update_guiding_prompt', default=False, action=argparse.BooleanOptionalAction,
                    help='Enable automatic guiding prompt updates at the end of each episode based on performance.')

    # Valid actions control
    parser.add_argument('--use_valid_actions', default=True, action=argparse.BooleanOptionalAction,
                    help='Provide the list of valid actions from the game environment to the agent.')

    # Confidence mode control
    parser.add_argument('--confidence_mode', type=str, default='verbalized', choices=['logit', 'verbalized'],
                    help='Mode for confidence calculation: "logit" extracts logprobs (OpenAI only), "verbalized" asks model for explicit confidence percentages (works with all models).')

    # Evaluation LLM model (for step scoring)
    parser.add_argument('--eval_llm_model', type=str, default='google/gemini-3-flash-preview',
                    help='LLM model for evaluating step scores in cross-episode memory.')

    # Reflective exploration agent parameters
    parser.add_argument('--reflect_interval', type=int, default=5, help='Number of episodes between cross-episode reflections for the reflective agent.')
    parser.add_argument('--ucb_exploration_c', type=float, default=1.414, help='Exploration constant for UCB strategy selection in reflective agent.')
    parser.add_argument('--max_strategies', type=int, default=15, help='Maximum number of strategies in the reflective agent search space.')
    parser.add_argument('--min_strategies', type=int, default=3, help='Minimum number of strategies to maintain in the reflective agent search space.')
    parser.add_argument('--backprop_unreached_discount', type=float, default=0.3, help='Discount factor for unreached/skipped milestones during backpropagation.')
    parser.add_argument('--backprop_method', type=str, default='linear', choices=['linear', 'dag'],
                    help='Backprop method: linear (gamma-discount along path order) or dag (credit flows along dep edges only).')
    parser.add_argument('--backprop_gamma', type=float, default=0.6,
                    help='Gamma discount factor for backpropagation.')

    # Resume from previous run
    parser.add_argument('--resume_from', type=str, default=None,
                    help='Path to a previous run directory to resume from (copies search_space.json and metadata).')

    # ExploreAgent modular parameters
    parser.add_argument('--strategy_space', type=str, default='milestone_tree',
                    choices=['plan_flat_list', 'milestone_tree', 'milestone_dag', 'action_tree'],
                    help='Strategy space module (M0).')
    parser.add_argument('--guidance_mode', type=str, default='full_plan',
                    choices=['full_plan', 'step_by_step', 'hierarchical', 'none'],
                    help='Guidance mode module (M1). "none" shows strategy as context only without forcing execution order.')
    parser.add_argument('--exploration_method', type=str, default='ucb',
                    choices=['ucb', 'thompson', 'epsilon_greedy'],
                    help='Exploration method module (M2).')
    parser.add_argument('--evolution_method', type=str, default='decision_point_mining',
                    choices=['decision_point_mining', 'free_reflection', 'none'],
                    help='Evolution method for strategy space. "none" disables evolution (static map).')
    parser.add_argument('--exploration_freeze_episode', type=int, default=30,
                    help='Stop generating new DAG nodes after this many episodes. 0 = never freeze.')
    parser.add_argument('--epsilon', type=float, default=0.1,
                    help='Epsilon for epsilon-greedy exploration.')
    parser.add_argument('--thompson_prior_std', type=float, default=100.0,
                    help='Prior standard deviation for Thompson Sampling.')
    parser.add_argument('--skip_summary', action='store_true', default=False,
                    help='Skip LLM episode summary; use raw game history for reflection and code-level reward attribution.')
    parser.add_argument('--abandon_threshold', type=int, default=3,
                    help='Number of zero-reward visits before a node is abandoned (default: 3).')

    # ScienceWorld parameters
    parser.add_argument('--sw_variation', type=int, default=0,
                    help='ScienceWorld task variation index. Use -1 for random train variation.')
    parser.add_argument('--sw_simplification', type=str, default='easy',
                    help='ScienceWorld simplification preset (e.g., "easy", or comma-separated like "teleportAction,openDoors").')

    return parser.parse_args()


if __name__ == "__main__":
    # 应用调试器与多进程的兼容性修复
    setup_multiprocessing_for_debug()

    args = parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    # Setup log file: output/{game}/{agent}/{model}/{timestamp}/run.log
    model_name = args.llm_model.replace("/", "_")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(args.output_path, args.game_name, args.agent_type, model_name, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "run.log")

    log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = TeeStream(log_file, sys.__stdout__)
    sys.stderr = TeeStream(log_file, sys.__stderr__)
    print(f"[Log] Saving output to {log_path}")

    from src.openai_helpers import reset_token_stats, get_token_stats
    reset_token_stats()

    evaluator = GameEvaluator(args)
    results = evaluator.run_evaluation()

    # Log token usage stats
    token_stats = get_token_stats()
    print(f"\n[Token Stats] LLM calls: {token_stats['calls']}, "
          f"prompt_tokens: {token_stats['prompt_tokens']}, "
          f"completion_tokens: {token_stats['completion_tokens']}, "
          f"total_tokens: {token_stats['total_tokens']}")
    token_stats_path = os.path.join(log_dir, "token_stats.json")
    with open(token_stats_path, "w") as f:
        json.dump(token_stats, f, indent=2)
    print(f"[Token Stats] Saved to {token_stats_path}")

    # Exit with appropriate code
    if results.get("success", False):
        print("Evaluation completed successfully!")
        log_file.close()
        exit(0)
    else:
        print("Evaluation failed!")
        log_file.close()
        exit(1)