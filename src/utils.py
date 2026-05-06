import numpy as np
import json
from .openai_helpers import chat_completion_with_retries, claude_completion_with_retries

def softmax(a, T=1):
    a = np.array(a) / T
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


def game_file(game_name):
    rom_dict = {'zork1': 'zork1.z5', 
                'zork3': 'zork3.z5', 
                'spellbrkr' : 'spellbrkr.z3',
                'advent': 'advent.z5',                 
                'detective': 'detective.z5', 
                'pentari': 'pentari.z5',
                'enchanter': 'enchanter.z3',
                'library' : 'library.z5',
                'balances' : 'balances.z5',
                'ztuu' : 'ztuu.z5',
                'ludicorp' : 'ludicorp.z5',
                'deephome' : 'deephome.z5',
                'temple' : 'temple.z5',
                'anchor' : 'anchor.z8',
                'awaken' : 'awaken.z5',
                'zenon' : 'zenon.z5'
                }
                
    if game_name not in rom_dict:
        return None  # Python games don't have ROM files
    return rom_dict[game_name]

# Placeholder for initial prompt loading utilities
# These should be implemented based on the actual structure of initial_prompts.json
# and how generic prompts are defined.

def load_initial_prompts(file_path):
    """
    Loads initial prompts from a JSON file.
    Expects a JSON file with a "prompts" key, which is a list of strings.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        if "prompts" not in data or not isinstance(data["prompts"], list):
            print(f"Warning: Prompt file {file_path} is missing 'prompts' list or it's not a list. Returning empty list.")
            return []
        return data["prompts"]
    except FileNotFoundError:
        print(f"Warning: Initial prompts file {file_path} not found. Returning empty list.")
        return []
    except json.JSONDecodeError:
        print(f"Warning: Error decoding JSON from {file_path}. Returning empty list.")
        return []

def get_generic_initial_prompts(num_prompts=1):
    """
    Returns a list of generic initial prompts.
    """
    # This is a basic implementation. You might want to make these more sophisticated.
    generic_prompts = [
        "Analyze the game state and choose the best action to maximize the score.",
        "Think step-by-step to understand the current situation and decide the next move.",
        "Your goal is to achieve the highest score. Observe, think, and act.",
        "Explore the environment, interact with objects, and solve puzzles to progress.",
        "Be methodical. Consider all available actions and their potential outcomes."
    ]
    return generic_prompts[:num_prompts]


def generate_detailed_episode_summary(game_history, score, llm_model, temperature=0.3, max_tokens=2000, milestones=None, game_name=None):
    """
    Generate a milestone-centric LLM summary of an episode.

    Analyzes the trajectory to extract milestones (reward-earning achievements),
    their key action sequences, and warnings about pitfalls.

    Args:
        game_history: List of history entries with 'state', 'action', 'reward', etc.
        score: Final score of the episode.
        llm_model: The LLM model to use.
        temperature: Temperature for LLM generation.
        max_tokens: Maximum tokens for the summary.
        milestones: Optional list of milestone strings (e.g. "[node_003] Take sword in Living Room").
                    When provided, the summary maps achievements to known milestones.

    Returns:
        str: A milestone-centric structured summary, or fallback text if generation fails.
    """
    if not game_history:
        return f"Empty episode with score {score}."

    total_steps = len(game_history)

    # Format trajectory with full details
    import re
    # Pattern to remove delayed score notifications from game text (e.g. Inform engine games)
    # These appear in the NEXT step's state after a reward, misleading LLM into double-attribution
    _score_noise_re = re.compile(r'\[your score has just gone up[^\]]*\]\.?\s*', re.IGNORECASE)

    lines = []
    for i, entry in enumerate(game_history):
        action = entry.get('action', '')
        state_preview = entry.get('state', '')[:200]
        state_preview = _score_noise_re.sub('', state_preview)
        reward = entry.get('reward', 0)
        if reward > 0:
            reward_str = f" | REWARD: +{reward}"
        elif reward < 0:
            reward_str = f" | REWARD: {reward}"
        else:
            reward_str = ""
        valid = entry.get('valid_actions', [])
        valid_str = f" | VALID: {valid}" if valid else ""
        room = entry.get('room_name', '')
        room_str = f" | ROOM: {room}" if room else ""
        lines.append(f"Step {i+1}:{room_str} ACTION: {action} | STATE: {state_preview}{reward_str}{valid_str}")
    formatted = "\n".join(lines)

    # Python games (catnip etc.) use a separate pipeline with full state details
    if game_name and game_name.startswith('catnip'):
        return _generate_python_game_episode_summary(game_history, score, total_steps, llm_model, temperature, max_tokens, milestones)

    sys_prompt = "You are an expert at analyzing text adventure game trajectories. Your job is to extract milestones — the key achievements that earned rewards — and analyze the path to each one."

    if milestones:
        milestone_list_str = "\n".join(f"  - {m}" for m in milestones)
        user_prompt = f"""Analyze this episode trajectory against the KNOWN MILESTONES below.

Final score: {score}, Total steps: {total_steps}

KNOWN MILESTONES:
{milestone_list_str}

EPISODE TRAJECTORY:
{formatted}

Write the summary in EXACTLY this format:

**Milestones Achieved:**
IMPORTANT: A reward is ONLY indicated by the "REWARD: +N" marker in the trajectory. Lines WITHOUT this marker earned NO reward, even if the game text mentions "score has gone up" — that text is a delayed message from a PREVIOUS step. Do NOT treat it as a reward for the current action.

For each line that has "REWARD: +N", follow this process:
1. IDENTIFY: Which exact action command is on that REWARD line?
2. MATCH: Does this action match a known milestone's key_action? Only match if the rewarding action IS or CLOSELY RESEMBLES the milestone's key_action.
3. DECIDE: If matched → use that [node_id]. If NOT matched → use [NEW].

Format:
- [node_id or NEW] milestone text (+reward) — achieved at Step X by action "exact action"
  Key steps: exact action1 → exact action2 → exact action3
  Warning: description of wasted steps or pitfalls encountered on the way

COMMON MISTAKE: A reward belongs to the action on the REWARD line, NOT to a preceding action.
CRITICAL: Do NOT match a reward to a milestone just because they are in the same room. The rewarding ACTION must match the milestone's key_action. Use [NEW] for any reward whose action doesn't match.

**Penalties Received:**
For each NEGATIVE reward (score decrease), record the exact event:
- [node_id or NEW] milestone text (reward) — at Step X
  What happened: exact description of what caused the penalty and how to avoid it

**Milestones NOT Achieved:**
For each known milestone NOT completed, provide structured analysis:
- [node_id] milestone text
  Status: ATTEMPTED at Step X / NOT_ATTEMPTED
  Failure reason: specific reason (e.g. "no light source", "got lost for 30 steps", "trap door closed")
  Missing prerequisite: which milestone or item would fix this

**Unexplored Opportunities:**
List actions that were available or observable but never tried — these are candidates for new milestones:
- Step X: valid actions not taken (e.g. chose "go north" but also had "open chest", "examine altar")
- Step X: game mentioned [item/path/object] that was never interacted with
- Step X: agent observed [event/change] but did not follow up

CRITICAL: Use EXACT action commands from the trajectory. Do NOT paraphrase.
CRITICAL: Use EXACT node_ids from the known milestone list. Do NOT rephrase milestone descriptions."""
    else:
        user_prompt = f"""Analyze this episode trajectory. Focus on extracting MILESTONES — every action that earned a reward is a milestone worth recording.

Final score: {score}, Total steps: {total_steps}

EPISODE TRAJECTORY:
{formatted}

Write the summary in EXACTLY this format:

**Milestones Achieved:**
For EVERY action that earned a POSITIVE reward (marked with REWARD in the trajectory), create a milestone entry:
- [NEW] <concise WHERE + WHAT description> (+reward amount) — achieved at Step X by action "exact action"
  Key steps: exact action1 → exact action2 → ... → reward action
  Warning: <wasted steps or pitfalls on the way. Write "none" if path was clean.>

**Penalties Received:**
For EVERY action that earned a NEGATIVE reward (score decrease), record it:
- [NEW] <concise description> (reward amount) — at Step X
  What happened: exact description of what caused the penalty and how to avoid it

**Milestones NOT Achieved:**
(Leave blank if this is the first episode with no known milestones yet.)

**Unexplored Opportunities:**
List actions that were available or observable but never tried — these are candidates for new milestones:
- Step X: valid actions not taken (e.g. chose "go north" but also had "open chest", "examine altar")
- Step X: game mentioned [item/path/object] that was never interacted with
- Step X: agent observed [event/change] but did not follow up

CRITICAL: Use EXACT action commands from the trajectory. Do NOT paraphrase or translate.
CRITICAL: Every REWARD action MUST appear as a milestone. Do not skip any."""

    try:
        response = chat_completion_with_retries(
            model=llm_model,
            sys_prompt=sys_prompt,
            prompt=user_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if response and hasattr(response, 'choices') and response.choices and response.choices[0].message:
            return response.choices[0].message.content.strip()
        else:
            return f"Episode with {total_steps} steps, final score {score}. (Summary generation failed.)"
    except Exception as e:
        print(f"[Warning] Error generating detailed episode summary: {e}")
        return f"Episode with {total_steps} steps, final score {score}. (Summary generation failed.)"


def _generate_python_game_episode_summary(game_history, score, total_steps, llm_model, temperature, max_tokens, milestones):
    """Episode summary for python games (catnip etc.).

    Preserves FULL game state each step (stats, influence, suspicion, warnings etc.)
    so the reflection LLM can discover state-action-reward relationships on its own.
    """
    # Format trajectory with full state (not truncated like jericho path)
    lines = []
    for i, entry in enumerate(game_history):
        action = entry.get('action', '')
        # Keep full state — this contains stats, influence, suspicion, warnings etc.
        state = entry.get('state', '')
        reward = entry.get('reward', 0)
        if reward > 0:
            reward_str = f" | REWARD: +{reward}"
        elif reward < 0:
            reward_str = f" | REWARD: {reward}"
        else:
            reward_str = ""
        valid = entry.get('valid_actions', [])
        valid_str = f" | VALID: {valid}" if valid else ""
        lines.append(f"Step {i+1}: ACTION: {action}{reward_str}{valid_str}\nGAME STATE:\n{state}\n")
    formatted = "\n".join(lines)

    sys_prompt = "You are an expert at analyzing strategy game trajectories. Your job is to extract milestones, analyze how game state (stats, influence, suspicion, stress, items) changed across actions, and identify what caused high or low rewards."

    if milestones:
        milestone_list_str = "\n".join(f"  - {m}" for m in milestones)
        user_prompt = f"""Analyze this episode trajectory against the KNOWN MILESTONES below.

Final score: {score}, Total steps: {total_steps}

KNOWN MILESTONES:
{milestone_list_str}

EPISODE TRAJECTORY:
{formatted}

Write the summary in this format:

**Milestones Achieved:**
For each line with "REWARD: +N", identify which milestone it matches:
- [node_id or NEW] milestone text (+reward) — achieved at Step X by action "exact action"
  Key steps: exact action1 → exact action2 → exact action3
  State context: what were the key stats/influence/suspicion levels when this reward was earned

**Milestones NOT Achieved:**
For each known milestone NOT completed:
- [node_id] milestone text
  Status: ATTEMPTED / NOT_ATTEMPTED
  Failure reason: specific reason based on game state (e.g. "suspicion was dangerous", "energy depleted", "influence too low to convert at HIGH tier")

**State Pattern Analysis:**
Analyze what game state conditions led to high vs low rewards:
- Which actions increased influence most effectively for each target?
- What suspicion/stress levels were present during successful vs failed conversions?
- How did item usage affect the outcomes?
- What warnings appeared and were they heeded?

**New Discoveries (not in any known milestone):**
- (+reward) "exact action" at Step X — what this achieved and the game state context

**Untried Valid Actions:**
At key decision points, list valid actions the agent did NOT try:
- Step X: chose "action A", but also had valid: ["action B", "action C"]

CRITICAL: Use EXACT action commands from the trajectory. Do NOT paraphrase."""
    else:
        user_prompt = f"""Analyze this episode trajectory. Extract MILESTONES and analyze state-action-reward patterns.

Final score: {score}, Total steps: {total_steps}

EPISODE TRAJECTORY:
{formatted}

Write the summary in this format:

**Milestones Achieved:**
For EVERY action with a POSITIVE reward:
- Milestone: <description> (+reward amount)
  Key steps: exact action1 → exact action2 → ... → reward action
  State context: what were the key stats/influence/suspicion levels when this reward was earned (e.g. "Influence: dominant, Suspicion: faint → HIGH tier conversion")

**Penalties Received:**
For EVERY action with a NEGATIVE reward:
- Penalty: <description> (reward amount)
  What happened: exact description including game state

**State Pattern Analysis:**
Analyze what game state conditions led to high vs low rewards:
- Which actions increased influence most effectively for each target?
- What suspicion/stress levels were present during successful conversions?
- How did breed choice and item usage affect the outcomes?
- What warnings appeared and were they heeded?
- What conversion tier (LOW/MED/HIGH) was achieved for each target and why?

**Untried Valid Actions:**
At key decision points, list valid actions the agent did NOT try:
- Step X: chose "action A", but also had valid: ["action B", "action C"]

CRITICAL: Use EXACT action commands from the trajectory."""

    try:
        response = chat_completion_with_retries(
            model=llm_model,
            sys_prompt=sys_prompt,
            prompt=user_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if response and hasattr(response, 'choices') and response.choices and response.choices[0].message:
            return response.choices[0].message.content.strip()
        else:
            return f"Episode with {total_steps} steps, final score {score}. (Summary generation failed.)"
    except Exception as e:
        print(f"[Warning] Error generating python game episode summary: {e}")
        return f"Episode with {total_steps} steps, final score {score}. (Summary generation failed.)"


def generate_trajectory_summary(game_history, llm_model="gpt-5", temperature=0.8, max_tokens=500):
    """
    Generates a concise summary of a game trajectory using LLM.
    
    Args:
        game_history: List of game history entries, each containing 'state', 'action', 
                     and optionally 'score', 'reward'
        llm_model: The LLM model to use for generation
        temperature: Temperature for LLM generation (lower = more consistent)
        max_tokens: Maximum tokens for the summary
        
    Returns:
        str: A concise summary of the trajectory, or fallback text if generation fails
    """
    if not game_history:
        return ""
    
    # Prepare the game history for LLM summarization
    history_text = ""
    for i, entry in enumerate(game_history):
        history_text += f"Step {i}:\n"
        history_text += f"State: {entry['state']}\n"
        if i != len(game_history) - 1:
            history_text += f"Action: {entry['action']}\n"
        history_text += "\n"
    # print("******************************history text******************************")
    # print(history_text)
    # Use LLM to generate a concise summary
    sys_prompt = """You are an expert at summarizing game trajectories. Extract and format key information from each step in a structured way. No need to provide the overall summary. And No need to record the score change.  IMPORTANT: For the LAST step (including when there's only ONE step total), only output the state, DO NOT include the action"""
    
    user_prompt = f"""Analyze the following game trajectory and provide a structured summary with keywords for each step.

Game Trajectory:
{history_text}

Please provide a formatted summary with the following structure:
1. For each step, extract ALL key nouns/objects from the current state IN THE ORDER they appear in the text and 1-2 action keywords
2. Format as: "Step X: [State: noun1, noun2, noun3, noun4, ...] [Action: keyword1, keyword2]"
3. IMPORTANT: For the LAST step (or if there's only ONE step), only output [State: ...], no [Action: ...]
4. Focus on extracting ALL important objects, characters, locations, and items mentioned in the state
5. Include both concrete objects (door, key, chest) and abstract concepts (darkness, danger, mystery)

Example format (multi):
Step 0: [State: dark room, stone walls, locked door, rusty key, torch, shadows, dust] [Action: take torch]
Step 1: [State: illuminated room, wooden chest, golden lock, stone floor, spider webs, treasure map] [Action: open chest]
Step 2: [State: treasure room, gold coins, silver jewelry, exit door, bright light, victory banner]

Example format (single step):
Step 0: [State: starting room, wooden door, lit candle, stone table, mysterious book, cold air]

Provide the structured summary:"""
    
    def validate_summary_format(summary, num_steps):
        """Validate that the summary follows the required format"""
        lines = [line.strip() for line in summary.split('\n') if line.strip() and line.strip().startswith('Step')]

        if len(lines) != num_steps:
            return False

        for i, line in enumerate(lines):
            # Check if it's the last step
            is_last_step = (i == len(lines) - 1)

            if is_last_step:
                # Last step should only have [State: ...], no [Action: ...]
                if '[Action:' in line:
                    return False
                if '[State:' not in line:
                    return False
            else:
                # Non-last steps should have both [State: ...] and [Action: ...]
                if '[State:' not in line or '[Action:' not in line:
                    return False

        return True

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = chat_completion_with_retries(
                model=llm_model,
                sys_prompt=sys_prompt,
                prompt=user_prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )

            if response and hasattr(response, 'choices') and response.choices:
                summary = response.choices[0].message.content.strip()

                # Validate the format
                if validate_summary_format(summary, len(game_history)):
                    return summary
                else:
                    print(f"[Warning] Summary format validation failed on attempt {attempt + 1}, retrying...")
                    continue
            else:
                print(f"[Warning] No valid response on attempt {attempt + 1}")
                continue

        except Exception as e:
            print(f"[Warning] Failed to generate LLM summary on attempt {attempt + 1}: {e}")
            continue

    # If all attempts failed, fallback to simple format
    print("[Warning] All summary generation attempts failed, using fallback")
    return generate_trajectory_summary_fallback(game_history)


def generate_trajectory_summary_fallback(game_history):
    """
    Fallback method for trajectory summarization if LLM fails.
    
    Args:
        game_history: List of game history entries
        
    Returns:
        str: A simple fallback summary
    """
    if not game_history:
        return ""
    
    # Return last few states and actions in simple format
    summary_text = "Recent actions: "
    recent_history = game_history[-3:] if len(game_history) > 3 else game_history
    
    actions = [entry['action'] for entry in recent_history if entry.get('action')]
    if actions:
        summary_text += ", ".join(actions) + ". "
    
    # Add final state and score info
    last_entry = game_history[-1]
    if last_entry.get('score') is not None:
        summary_text += f"Current score: {last_entry['score']}. "
    
    # Add brief state description (first 100 chars of last state)
    if last_entry.get('state'):
        state_snippet = last_entry['state'][:100].replace('\n', ' ').strip()
        summary_text += f"State: {state_snippet}..."
    
    return summary_text


def calculate_summary_similarity(summary1: str, summary2: str, llm_model: str = "gpt-5", temperature: float = 0.8, max_tokens: int = 50) -> float:
    """
    Calculate similarity score between two summaries using LLM.
    
    Args:
        summary1: First trajectory summary
        summary2: Second trajectory summary  
        llm_model: The LLM model to use for scoring
        temperature: Temperature for LLM generation (lower = more consistent)
        max_tokens: Maximum tokens for the response
        
    Returns:
        float: Similarity score between 0.0 and 1.0
    """
    if not summary1 or not summary2:
        return 0.0
    
    sys_prompt = """You are an expert at evaluating the semantic similarity between game trajectory summaries. 
You must output ONLY a single decimal number between 0.0 and 1.0 representing the similarity score.
BE STRICT in your evaluation:
- 0.0-0.5: Not similar trajectories.
- 0.5-0.79: Similar trajectories.
- 0.8+: Nearly identical trajectories
CRITICAL: The LAST STATE in each summary represents the current game situation and is more important."""
    
    user_prompt = f"""Compare these two game trajectory summaries and output ONLY a similarity score between 0.0 and 1.0.

Summary 1:
{summary1}

Summary 2:
{summary2}

Output only the numerical score (e.g., 0.65):"""
    
    try:
        response = chat_completion_with_retries(
            model=llm_model,
            sys_prompt=sys_prompt,
            prompt=user_prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        if response and hasattr(response, 'choices') and response.choices:
            score_text = response.choices[0].message.content.strip()
            # Extract numerical value from response
            try:
                score = float(score_text)
                # Ensure score is within valid range
                return max(0.0, min(1.0, score))
            except ValueError:
                print(f"[Warning] Failed to parse similarity score: {score_text}")
                return 0.0
        else:
            print("[Warning] No response from LLM for similarity calculation")
            return 0.0
    except Exception as e:
        print(f"[Warning] Failed to calculate LLM similarity: {e}")
        return 0.0


def detect_loops_in_trajectory(game_history, window_size=3):
    """
    检测轨迹中的循环模式。

    Args:
        game_history: 游戏历史记录列表
        window_size: 检测循环的窗口大小

    Returns:
        list: 每一步是否处于循环状态的布尔值列表
    """
    if not game_history or len(game_history) < window_size * 2:
        return [False] * len(game_history)

    loop_flags = [False] * len(game_history)

    # 为每一步检查是否在循环中
    for i in range(len(game_history)):
        # 获取当前步的状态和动作
        current_step = game_history[i]
        current_state = current_step.get('state', '')
        current_action = current_step.get('action', '')

        # 检查之前的步骤是否有相同的状态-动作组合
        loop_count = 0
        for j in range(max(0, i - 10), i):  # 在前10步中检查
            prev_step = game_history[j]
            prev_state = prev_step.get('state', '')
            prev_action = prev_step.get('action', '')

            # 检查状态相似度（简单的子串匹配）
            if (current_state and prev_state and
                len(current_state) > 20 and len(prev_state) > 20):
                # 计算状态相似度
                shorter = min(len(current_state), len(prev_state))
                common_chars = sum(c1 == c2 for c1, c2 in zip(current_state[:shorter], prev_state[:shorter]))
                similarity = common_chars / shorter if shorter > 0 else 0

                if similarity > 0.8 and current_action == prev_action:
                    loop_count += 1

        # 如果发现多次重复，标记为循环
        if loop_count >= 2:
            loop_flags[i] = True

    return loop_flags


def evaluate_step_scores_with_llm(game_history, state, final_score, success, llm_model="google/gemini-2.5-flash-preview-09-2025", temperature=0.8):
    """
    使用LLM为轨迹中的每一步分配评分，考虑到最终结果和循环惩罚。

    Args:
        game_history: 游戏历史记录列表
        final_score: 最终得分
        success: 是否成功
        llm_model: 使用的LLM模型
        temperature: LLM温度参数

    Returns:
        list: 每一步的评分列表
    """
    if not game_history:
        return []

    # 如果轨迹太长，直接返回全0分数以避免token限制
    # if len(game_history) > 80:
    #     return [0] * len(game_history)

    # 构建轨迹描述
    trajectory_text = ""
    for i, entry in enumerate(game_history):
        trajectory_text += f"Step {i}:\n"
        trajectory_text += f"State: {entry.get('state', '')}\n"  # 限制状态长度
        if entry.get('action'):
            trajectory_text += f"Action: {entry.get('action')}\n"
        score = entry.get('score', 0)
        reward = entry.get('reward', 0)
        trajectory_text += f"Reward: {reward}\n"
        trajectory_text += "\n"
    trajectory_text += f"Step: {len(game_history)}:\nState: {state}\n"
    sys_prompt = """You are scoring game actions to build training data for future gameplay.

PURPOSE: Rate each action based on its overall impact - positive scores for actions worth repeating, negative for actions to avoid.

SCORING RULES:
- Positive: Action led to progress or useful discoveries
- Negative: Action wasted time, caused loops, or had no benefit
- Magnitude: Match the game's typical reward scale (calibrate based on rewards shown in trajectory)
- Evaluation: Judge by full consequence chain, not just immediate result

ANALYSIS: For each action, explain what happened and why future players should repeat or avoid it.
"""

    user_prompt = f"""Score each action in this game session.

Final Result: {"SUCCESS" if success else "FAILURE"}, Final Score: {final_score}

Trajectory:
{trajectory_text}

JSON FORMAT:
{{
  "step_analysis": [
    {{
      "step": 0,
      "action": "exact action taken",
      "detailed_reasoning": "What happened after this action and its consequences? Why should this be repeated or avoided? (80+ words)",
      "score": 5,
}}
  ],
  "overall_assessment": "Key lessons: what worked and what didn't"
}}

Provide complete JSON response:"""

    response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "step_analysis_scores",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "step_analysis": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "step": {"type": "integer"},
                                        "action": {"type": "string"},
                                        "detailed_reasoning": {"type": "string"},
                                        "score": {
                                            "type": "number",
                                            "minimum": -10,
                                            "maximum": 10
                                        },
                                    },
                                    "required": ["step", "action", "detailed_reasoning", "score"],
                                    "additionalProperties": False
                                }
                            },
                            "overall_assessment": {"type": "string"}
                        },
                        "required": ["step_analysis", "overall_assessment"],
                        "additionalProperties": False
                    }
                }
            }

    try:
        response = chat_completion_with_retries(
                    model=llm_model,
                    sys_prompt=sys_prompt,
                    prompt=user_prompt,
                    max_tokens=20000,
                    temperature=temperature,
                    response_format=response_format
                )
    except Exception as e:
        print(f"[Error] LLM call failed: {e}")
        print("[Warning] Falling back to reward-based scoring")
        return [entry.get('reward', 0) for entry in game_history], ["No reasoning available" for _ in game_history]

    if response and hasattr(response, 'choices') and response.choices:
        scores_text = response.choices[0].message.content.strip()

                # 清理可能的markdown代码块包装
        if scores_text.startswith('```json'):
            scores_text = scores_text.replace('```json', '').replace('```', '').strip()
        elif scores_text.startswith('```'):
            scores_text = scores_text.replace('```', '').strip()

        try:
            response_data = json.loads(scores_text)
        except json.JSONDecodeError as e:
            print(f"[Warning] JSON decode error: {e}")
            print(f"[Warning] Problematic JSON text (first 500 chars): {scores_text[:500]}")
            print(f"[Warning] Falling back to reward-based scoring")
            return [entry.get('reward', 0) for entry in game_history], ["No reasoning available" for _ in game_history]

                # 处理新的带分析的JSON格式
        if isinstance(response_data, dict):
            if 'step_analysis' in response_data:
                # 从step_analysis中提取每一步的分数和reasoning
                step_analyses = response_data['step_analysis']

                # 创建字典存储每个步骤的分析结果
                analysis_dict = {}
                for analysis in step_analyses:
                    step_num = analysis.get('step', -1)
                    if step_num >= 0:
                        analysis_dict[step_num] = analysis

                scores = []
                reasonings = []
                missing_steps = []

                print("\n=== LLM Step Analysis ===")
                # 按顺序处理每个步骤
                for i in range(len(game_history)):
                    if i in analysis_dict:
                        analysis = analysis_dict[i]
                        step_score = analysis.get('score', 0)
                        reasoning = analysis.get('detailed_reasoning', 'No reasoning')
                        scores.append(step_score)
                        reasonings.append(reasoning)
                        print(f"Step {i}: {analysis.get('action', 'Unknown')}")
                        print(f"  Reasoning: {reasoning}")
                        print(f"  Score: {step_score}")
                        print()
                    else:
                        # LLM遗漏了这个步骤，用0分填充
                        scores.append(0)
                        reasonings.append('Missing from LLM analysis - filled with score 0')
                        missing_steps.append(i)
                        print(f"Step {i}: [MISSING FROM LLM ANALYSIS - FILLED WITH 0]")
                        print(f"  Action: {game_history[i].get('action', 'Unknown')}")
                        print(f"  Score: 0 (auto-filled)")
                        print()

                if missing_steps:
                    print(f"[Warning] LLM missed analyzing steps: {missing_steps}")
                    print(f"[Info] Filled missing steps with score 0")

                if 'overall_assessment' in response_data:
                    print(f"Overall Assessment: {response_data['overall_assessment']}")
                    print("=" * 50)
            else:
                print(f"[Warning] No step_analysis found in JSON format: {response_data}")
                return [entry.get('reward', 0) for entry in game_history], ["No reasoning available" for _ in game_history]
        elif isinstance(response_data, list):
            scores = response_data
            reasonings = ["Legacy format - no reasoning available" for _ in game_history]
        else:
            print(f"[Warning] Unexpected JSON format: {response_data}")
            return [entry.get('reward', 0) for entry in game_history], ["No reasoning available" for _ in game_history]


        return scores, reasonings
    else:
        print("[Warning] No valid response from LLM for step scoring")
        return [entry.get('reward', 0) for entry in game_history], ["No reasoning available" for _ in game_history]

def generate_history_summary(game_history, current_state=None, llm_model="gpt-5o", temperature=0.8, max_tokens=1000, current_inventory=None):
    """
    Generate structured LLM-summarized history context for vector embedding.
    Summarizes the past states and optionally includes the current state.

    Args:
        game_history: List of game history entries containing state, action, score, reward, etc.
        current_state: The current state to include in the summary (optional)
        llm_model: The LLM model to use for generation
        temperature: Temperature for LLM generation
        max_tokens: Maximum tokens for the summary
        current_inventory: The current inventory to include in the summary (optional)

    Returns:
        tuple: (summary_text, structured_summary)
            - summary_text: Natural language summary under [SUMMARY]
            - structured_summary: Structured sections [PROGRESS], [LOCATION], [NEXT_OBJECTIVE], [INVENTORY]
    """
    if not game_history and not current_state:
        return "", ""

    # Extract states and actions from game_history
    states = [entry.get('state', '') for entry in game_history]
    actions = [entry.get('action', '') for entry in game_history]
    current_step = len(game_history)  # Current step index

    # Build trajectory text for all past steps
    earlier_trajectory_text = ""
    for i in range(current_step):
        earlier_trajectory_text += f"Step {i}:\n"
        if i < len(states):
            earlier_trajectory_text += f"State: {states[i]}\n"
        if i < len(actions) and actions[i]:
            earlier_trajectory_text += f"Action: {actions[i]}\n"
        earlier_trajectory_text += "\n"

    # Add current state if provided
    if current_state:
        earlier_trajectory_text += f"Step {current_step} (Current):\n"
        earlier_trajectory_text += f"State: {current_state}\n"

    sys_prompt = """You are an expert at analyzing game trajectories and creating highly distinctive summaries.
Your milestones must be CONCISE and DISTINCTIVE - use specific keywords that clearly differentiate different game states.

Output Format Requirements:
[SUMMARY]: Provide a natural language summary that describes:
   - The game's objective or goal (inferred from the trajectory)
   - Current progress toward that goal
   - Key accomplishments so far
   - Current situation/status

[PROGRESS]: List milestones in format "✓ M#: <action>→<key object/result>"
   - If NO steps have score increases, output ONLY "No Progress" (no milestones)

[LOCATION]: Track location changes in format "Location: A→B→C"
   - Extract location names from state descriptions
   - Only record when location actually changes
   - CRITICAL: IGNORE unproductive loops - if the player goes back and forth between locations WITHOUT score increases, skip those redundant location movements
   - Focus on the MEANINGFUL location trajectory that led to progress or new discoveries
   - Example: If player went "room A→room B→room A→room B→room A→room C" with no score increase for the A↔B movements, record as "room A→room C"

Milestone Format Rules:
- Use format: <verb>→<critical object/state>
- Focus on STATE CHANGES, not descriptions
- Use specific nouns not generic terms
- Each milestone should capture ONE concrete action or discovery
- CRITICAL: Only record steps where the score increased compared to the previous step

Examples:
BAD SUMMARY: "The player has been exploring."
GOOD SUMMARY: "The game's objective is to escape the haunted mansion. So far, the player has found a key in the library and unlocked the basement door, earning 15 points. Currently in the dark basement, the player needs to find a light source to continue exploring."

BAD:  "✓ Milestone 1: Entered the library and proceeded to the ground floor stacks"
GOOD: "✓ M1: enter→library ground floor"

Location Examples:
BAD:  "Location: You are in a library→You moved to another area→You are somewhere else"
GOOD: "Location: entrance→library→north corridor→atrium"
BAD:  "Location: room A→room B→room A→room B→room A" (unproductive loop)
GOOD: "Location: room A" (ignore the meaningless back-and-forth)
"""

    user_prompt = f"""Analyze the game trajectory and generate a natural language summary followed by structured sections.

Game History:
{earlier_trajectory_text}

Generate summary in this format:

[SUMMARY]
<Natural language paragraph (2-4 sentences) describing: game objective, current progress, key accomplishments, current situation>

[PROGRESS]
✓ M1: <verb>→<result>
✓ M2: <verb>→<result>
...

[LOCATION]
Location: A→B→C→...

Requirements:
- SUMMARY: Write a natural language paragraph explaining the game's goal and current status
- Use specific object names
- Focus on concrete state changes and discoveries
- Avoid repetitive phrasing between milestones
- CRITICAL: ONLY record progress where the score increased from the previous step
- IMPORTANT: If there are NO steps with score increases, output ONLY "No Progress" under [PROGRESS] section (do not create any milestone items)
- For locations: Extract actual location names from state descriptions and track when they change
- Location format must be: "Location: " followed by location names separated by →
- CRITICAL FOR LOCATION: Detect and REMOVE unproductive loops from the location trajectory
  * Look at the score/reward at each step
  * If the player moves back and forth between locations WITHOUT any score increase, those movements are loops - SKIP them
  * Only include location changes that either: (1) led to score increases, OR (2) represent meaningful forward exploration
  * Example: If steps show "A→B (no reward)→A (no reward)→B (no reward)→A (no reward)→C (reward +5)", output "Location: A→C"

Generate the complete summary:"""

    response = chat_completion_with_retries(
            model=llm_model,
            sys_prompt=sys_prompt,
            prompt=user_prompt,
            max_tokens=max_tokens,
            temperature=temperature
    )

    if response and hasattr(response, 'choices') and response.choices:
        content = response.choices[0].message.content
        if content is None:
            return "", ""
        full_response = content.strip()

        # Extract [SUMMARY] section
        summary_text = ""
        if "[SUMMARY]" in full_response:
            summary_start = full_response.find("[SUMMARY]") + len("[SUMMARY]")
            # Find the next section marker
            next_section = full_response.find("[PROGRESS]", summary_start)
            if next_section == -1:
                next_section = full_response.find("[LOCATION]", summary_start)

            if next_section != -1:
                summary_text = full_response[summary_start:next_section].strip()
            else:
                summary_text = full_response[summary_start:].strip()

        # Extract structured sections (everything except [SUMMARY])
        structured_summary = full_response
        if "[SUMMARY]" in full_response:
            # Remove [SUMMARY] section from structured summary
            progress_start = full_response.find("[PROGRESS]")
            if progress_start != -1:
                structured_summary = full_response[progress_start:]

        # Add inventory section to structured summary if available
        if current_inventory:
            structured_summary += f"\n\n[INVENTORY]\n{current_inventory}"

        return summary_text, structured_summary
    else:
        print("[Warning] No response from LLM for history summary generation")
        # Return empty summary and inventory only if available
        summary_text = ""
        structured_summary = ""
        if current_inventory:
            structured_summary = f"[INVENTORY]\n{current_inventory}"
        return summary_text, structured_summary


def generate_progress_analysis_with_recommendation(game_history, current_state, llm_model="gpt-5o", temperature=0.8, max_tokens=1500):
    """
    Generate an analysis of game progress and recommended next actions.
    Uses both the game history and the current state to provide strategic guidance.

    Args:
        game_history: List of game history entries containing state, action, score, reward, etc.
        current_state: The current state of the game
        llm_model: The LLM model to use for generation
        temperature: Temperature for LLM generation
        max_tokens: Maximum tokens for the analysis

    Returns:
        str: Analysis with progress assessment and recommended actions
    """
    if not game_history and not current_state:
        return ""

    # Build trajectory text from game history
    trajectory_text = ""
    for i, entry in enumerate(game_history):
        trajectory_text += f"Step {i}:\n"
        if entry.get('state'):
            trajectory_text += f"State: {entry['state']}\n"
        if entry.get('action'):
            trajectory_text += f"Action: {entry['action']}\n"
        if entry.get('score') is not None:
            trajectory_text += f"Score: {entry['score']}\n"
        trajectory_text += "\n"

    # Add current state
    current_step = len(game_history)
    trajectory_text += f"Step {current_step} (Current):\n"
    trajectory_text += f"State: {current_state}\n"

    sys_prompt = """You are an expert game strategist. Analyze the player's progress through the game and provide actionable recommendations.

Your analysis should:
1. Assess what progress has been made toward the game objective
2. Identify what the player has accomplished and what remains to be done
3. Recommend specific next actions based on the current situation
4. Be concise and actionable

Output Format:
[PROGRESS ANALYSIS]
<Brief assessment of what has been accomplished and current situation>

[NEXT STEPS]
1. <Specific recommended action with reasoning>
2. <Alternative action if applicable>
..."""

    user_prompt = f"""Analyze the game progress and recommend what the player should do next.

Game Trajectory:
{trajectory_text}

Provide your analysis in this format:

[PROGRESS ANALYSIS]
<What has been accomplished? What is the current situation? What challenges or obstacles remain?>

[NEXT STEPS]
1. <Primary recommended action and why>
2. <Secondary option or backup plan>
...

Be specific and actionable in your recommendations:"""

    try:
        response = chat_completion_with_retries(
            model=llm_model,
            sys_prompt=sys_prompt,
            prompt=user_prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )

        if response and hasattr(response, 'choices') and response.choices:
            analysis = response.choices[0].message.content.strip()
            return analysis
        else:
            print("[Warning] No response from LLM for progress analysis")
            return ""
    except Exception as e:
        print(f"[Warning] Failed to generate progress analysis: {e}")
        return ""




def generate_trajectory_context_for_vector(states, actions, earlier_summary,current_step, episode_number, llm_model="gpt-5", temperature=0.8, max_tokens=1000, info=None):

    if info:
        look_info = info.get('look', '')
        current_state = states[current_step] if current_step < len(states) else ''

            # Combine look_info and current_state (only add both if they're different)
        if look_info and current_state:
            if look_info.strip() != current_state.strip():
                current_env_info2 = f"{look_info}\n\n{current_state}"
            else:
                current_env_info2 = current_state
        elif look_info:
            current_env_info2 = look_info
        elif current_state:
            current_env_info2 = current_state
        current_state_text = current_env_info2
    else:
        detailed_current_state = find_detailed_environment_info(states, actions, current_step, episode_number, llm_model, temperature)
        current_state_text = f"step {current_step}: State: {detailed_current_state}\n{states[current_step]}"

    if earlier_summary:
        full_summary = earlier_summary + "\n" # + current_state_text + states[current_step]
    else:
        full_summary = current_state_text
    return full_summary, current_state_text




def generate_trajectory_context_fallback(states, actions, current_step):
    """
    Fallback method for trajectory context generation if LLM fails.

    Args:
        states: List of all states in the episode
        actions: List of all actions in the episode
        current_step: Current step index

    Returns:
        str: Simple fallback trajectory context
    """
    if not states or current_step < 0 or current_step >= len(states):
        return ""

    # Simple fallback: concatenate states and actions
    context_parts = []
    for i in range(current_step + 1):
        if i < len(states):
            context_parts.append(f"State {i}: {states[i]}")
        if i < len(actions) and actions[i] and i < current_step:
            context_parts.append(f"Action {i}: {actions[i]}")

    return "\n".join(context_parts)


def find_detailed_environment_info(states, actions, current_step, episode_number, llm_model="gpt-5", temperature=0.8, max_tokens=800, max_history_steps=10):
    """
    Find complete environment information based on current_state and game trajectory using LLM.
    Returns empty string if current state has sufficient info, otherwise finds detailed info from history.
    Only considers the most recent max_history_steps states for environment information.

    Args:
        states: List of all states in the episode
        actions: List of all actions in the episode
        current_step: Current step index
        llm_model: LLM model to use
        temperature: LLM temperature parameter
        max_tokens: Maximum tokens
        max_history_steps: Maximum number of recent steps to consider (default: 10)

    Returns:
        str: Complete environment information from history, or empty string if current state is sufficient
    """
    if not states or current_step < 0 or current_step >= len(states):
        return ""
    elif current_step == 0:
        return ""

    current_state = states[current_step]

    # Limit to most recent max_history_steps states for environment search
    start_step = max(0, current_step - max_history_steps + 1)  # Include current step, so look at max_history_steps steps total

    # Build game history text for context (only from recent max_history_steps steps)
    history_text = ""
    for i in range(start_step, current_step):
        history_text += f"Step {i}:\n"
        if i < len(states):
            history_text += f"State: {states[i]}\n"
        if i < len(actions):
            history_text += f"Action: {actions[i]}\n"
        history_text += "\n"

    sys_prompt = """You are a game environment analysis expert. Your task is to analyze the current state and determine if it contains sufficient environment information.

A state has sufficient environment information if it includes location description (room, area, place details) or spatial relationships or layout details.
If the currentstate contains the marker "<<location>>", it ALWAYS has sufficient environment information and you should return "None"

If the current state has sufficient environment information, return "None".
If the current state lacks sufficient information (like "you can't see anything" or other simple prompts), find the most relevant detailed environment description from the game history and return it EXACTLY as it appears in the history, without any modification or reasoning.

PRIORITY RULE: When searching for environment information in the history, ALWAYS prioritize states that contain the "<<location>>" marker, as these contain the most detailed and accurate location information."""

    user_prompt = f"""Analyze the current state and game history:

Game History:
{history_text}

Current State:
{current_state}

Task:
1. First determine if the current state contains sufficient environment information
2. If YES: Return "None"
3. If NO: Find the most relevant detailed environment description from the game history and copy it EXACTLY as it appears, without any changes
4. PRIORITY: When searching the history, give highest priority to states containing "<<location>>" markers as they have the most detailed environment information

Return either "None" or the exact text from history:"""

    try:
        response = chat_completion_with_retries(
            model=llm_model,
            sys_prompt=sys_prompt,
            prompt=user_prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )

        if response and hasattr(response, 'choices') and response.choices:
            result = response.choices[0].message.content.strip()

            # Remove triple backticks if present
            if result.startswith('```'):
                result = result.replace('```', '').strip()

            # If result is empty string or very short, current state was sufficient
            if not result or result == 'None' or len(result) < 10:
                print("[Info] Current state has sufficient environment info")
                return ""
            else:

                print(f"[Info] Found detailed environment info from history:\n{result}")
                return result
        else:
            print("[Warning] No response from LLM for environment info analysis")
            return find_environment_info_fallback(states, current_step)

    except Exception as e:
        print(f"[Warning] Failed to analyze environment info with LLM: {e}")
        return find_environment_info_fallback(states, current_step)


def find_environment_info_fallback(states, current_step, max_search_steps=10):
    """
    Fallback method when LLM call fails, finds environment info through simple matching.
    Only considers the most recent max_search_steps states for environment information.

    Args:
        states: List of all states in the episode
        current_step: Current step index
        max_search_steps: Maximum number of recent steps to search (default: 10)

    Returns:
        str: Found environment information or original current_state
    """
    if not states or current_step < 0 or current_step >= len(states):
        return ""

    current_state = states[current_step]

    # If current_state is already detailed (length > 50), return directly
    if len(current_state) > 50:
        return current_state

    # Limit search to most recent max_search_steps states
    start_step = max(0, current_step - max_search_steps + 1)  # Include current step, so look at max_search_steps steps total

    # Search for most recent detailed state description from recent history
    for i in range(current_step, start_step - 1, -1):
        state = states[i]
        if len(state) > 50:  # Consider states with length > 50 as detailed
            # Check if contains location-related information
            location_keywords = ['room', 'hallway', 'corridor', 'chamber', 'area', 'space', 'place']
            if any(keyword in state.lower() for keyword in location_keywords):
                return state

    # If no suitable detailed description found, return last non-empty state from recent history
    for i in range(current_step, start_step - 1, -1):
        state = states[i]
        if state.strip():
            return state

    return current_state


def generate_prompt_recommendation(game_history, final_score, success, current_prompt, llm_model="gpt-5", temperature=0.8, max_tokens=3000, use_claude=True):
    """
    Generate a recommendation for updating the guiding prompt based on the complete trajectory.
    Model first reasons about what worked/didn't work, then generates a step-by-step strategy prompt.

    Args:
        game_history: Complete game history from the episode
        final_score: Final score achieved
        success: Whether the episode was successful
        current_prompt: The current guiding prompt being used
        llm_model: LLM model to use (default for non-Claude, or Claude model name if use_claude=True)
        temperature: LLM temperature parameter
        max_tokens: Maximum tokens for the response
        use_claude: If True, use Claude model for generation (recommended)

    Returns:
        dict: Contains 'recommended_prompt' (str), 'reasoning' (str), and 'key_insights' (list)
    """
    if not game_history:
        return {
            'recommended_prompt': current_prompt,
            'reasoning': 'No game history available for analysis',
            'key_insights': []
        }

    # Build trajectory summary
    trajectory_text = ""
    for i, entry in enumerate(game_history):
        trajectory_text += f"Step {i}:\n"
        trajectory_text += f"State: {entry.get('state', '')[:200]}...\n"  # Limit state length
        if entry.get('action'):
            trajectory_text += f"Action: {entry.get('action')}\n"
        if entry.get('reward') is not None:
            trajectory_text += f"Reward: {entry.get('reward')}\n"
        trajectory_text += "\n"

    sys_prompt = """You are an expert game strategy analyst. Your task is to analyze game trajectories and create step-by-step strategic guidance.

IMPORTANT: You must FIRST do deep reasoning about what worked and what didn't, THEN create a strategic prompt based on that reasoning.

Your output should have two parts:
1. REASONING: Analyze what actions/strategies were useful vs useless (verified by outcomes)
2. STRATEGIC PROMPT: A step-by-step guide telling the agent what to do first, second, third, etc.

The strategic prompt should BALANCE two strategies:
- EXPLOIT: Use VERIFIED successful patterns from the trajectory (actions that led to progress)
- EXPLORE: If the game was NOT successful, encourage exploring NEW areas/objects that were NOT tried before

**CRITICAL FOR FAILURE CASES:**
When the game was NOT successful, you MUST conduct a thorough failure analysis:
1. **Exploration Gaps**: Identify SPECIFIC locations, objects, directions, or NPCs that were mentioned in game states but NEVER explored or interacted with
2. **Wrong Directions**: Identify patterns where the agent kept trying the SAME failed approach repeatedly instead of changing strategy
3. **Missed Opportunities**: Identify hints or clues in the game text that the agent IGNORED or didn't follow up on
4. **Concrete Improvements**: Provide SPECIFIC, ACTIONABLE changes for the next attempt (e.g., "explore the northern passage that was mentioned but never visited" instead of generic "explore more")

The strategic prompt should:
- Provide a SEQUENCE of steps (do X first, then Y, then Z)
- Include both exploitation (repeat what worked) AND exploration (try new things)
- Tell the agent what to AVOID (verified failures/loops)
- Be specific and actionable (not vague advice)
- Be concise (2-4 sentences max)
- For FAILURES: Explicitly mention unexplored areas and alternative approaches

Example good prompt formats:
SUCCESS case: "First, examine all objects in the starting area to gather information. Next, focus on [specific verified action] which led to progress. Avoid [specific verified useless action] as it wastes time."

FAILURE case: "First, repeat [verified useful action] which showed some progress. CRITICAL: You failed because you never explored [specific unexplored area/object]. Next, immediately investigate [specific unexplored location/item/direction] that was mentioned in state descriptions. Avoid [verified loop/failure pattern] as it led nowhere."
"""

    user_prompt = f"""Analyze this game episode and create improved strategic guidance.

Current Guiding Prompt:
"{current_prompt}"

Episode Result:
- Final Score: {final_score}
- Success: {"Yes" if success else "No"}
- Total Steps: {len(game_history)}

Game Trajectory:
{trajectory_text}

Task: First REASON deeply about this episode, then create a STRATEGIC PROMPT.

In your reasoning, analyze:
1. What actions were USEFUL (led to score increases or clear progress)?
2. What actions were USELESS (wasted time, loops, no effect)?
3. What was the successful SEQUENCE of actions (if any)?
4. What patterns should be AVOIDED?
5. What areas/objects/directions/NPCs were MENTIONED in state descriptions but NEVER explored or interacted with?
6. What should the agent do FIRST, SECOND, THIRD to make progress?

**CRITICAL for FAILURE cases (Success = No):**
You MUST perform deep failure analysis and answer these questions:
A. **Exploration Insufficiency**:
   - Which SPECIFIC areas/locations were mentioned in the game text but the agent never visited?
   - Which objects were visible/mentioned but never examined or interacted with?
   - Which directions were available but never tried?
   - Which NPCs or interactive elements were present but never engaged with?

B. **Wrong Direction Analysis**:
   - Did the agent get stuck in loops, repeating the same failed actions?
   - Did the agent pursue a dead-end strategy for too long without trying alternatives?
   - Were there obvious hints in the game text that the agent ignored?

C. **Specific Improvements for Next Attempt**:
   - List 2-3 CONCRETE actions the agent should take next time (e.g., "examine the brass lantern", "go north to the dark passage", "ask the wizard about the crystal")
   - Explain WHY these actions might lead to progress
   - Identify which failed approaches to AVOID completely

Then create a strategic prompt with:
- Step 1: What to do first (exploit: repeat verified useful action OR explore if stuck)
- Step 2: What to do next (EXPLORE: try SPECIFIC unexplored areas/objects if game failed - name them explicitly!)
- Step 3: What to do after that (continue systematic exploration or follow up on clues)
- What to avoid (verified loops/failures with specific examples)

Respond in JSON format:
{{
    "reasoning": {{
        "useful_actions": ["action 1 that helped", "action 2 that helped"],
        "useless_actions": ["action 1 that wasted time", "action 2 that failed"],
        "successful_sequence": "Describe the sequence of actions that led to progress",
        "patterns_to_avoid": ["pattern 1", "pattern 2"],
        "unexplored_areas": ["location/object 1 mentioned but not explored", "location/object 2 mentioned but not explored"],
        "failure_analysis": {{
            "exploration_gaps": ["specific location/object 1 that was mentioned but never tried", "specific location/object 2 that was mentioned but never tried"],
            "wrong_directions": ["describe loop pattern 1", "describe dead-end strategy that was pursued too long"],
            "missed_hints": ["hint 1 that was ignored", "hint 2 that was not followed up"],
            "concrete_improvements": ["specific action 1 to try next time with reason", "specific action 2 to try next time with reason", "specific action 3 to try next time with reason"]
        }},
        "strategic_insights": "What strategy should guide future attempts? Balance exploit vs explore. For failures: explain why the agent got stuck and what needs to change."
    }},
    "recommended_prompt": "First, [do X]. Next, [do Y - for FAILURES include SPECIFIC unexplored areas by name]. Then [do Z]. Avoid [doing A] as it [specific reason from trajectory].",
    "key_insights": ["insight 1", "insight 2", "insight 3"]
}}

IMPORTANT:
- Your recommended_prompt must provide a SEQUENCE (first, next, then)
- Base exploitation on VERIFIED outcomes from this trajectory
- If episode FAILED: Your recommended_prompt MUST explicitly name the unexplored areas/objects to try (e.g., "explore the dark cellar" not "explore more")
- If episode FAILED: Fill out the failure_analysis section with SPECIFIC details from the trajectory
- Be SPECIFIC (not generic advice like "explore more" or "try different things")
- Keep recommended_prompt CONCISE (2-4 sentences)
"""

    try:
        # Use Claude for prompt generation (better reasoning and analysis)
        if use_claude:
            # Use Claude 3.5 Sonnet (latest) through OpenRouter
            claude_model = "anthropic/claude-3.5-sonnet"
            print(f"[Info] Using Claude model for prompt generation: {claude_model}")

            result_text = claude_completion_with_retries(
                model=claude_model,
                sys_prompt=sys_prompt,
                prompt=user_prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )

            if not result_text:
                print("[Warning] Failed to get Claude response, falling back to current prompt")
                return {
                    'recommended_prompt': current_prompt,
                    'reasoning': 'Failed to get Claude response',
                    'key_insights': []
                }
        else:
            # Use GPT model with structured output
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "strategic_prompt_recommendation",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "reasoning": {
                                "type": "object",
                                "properties": {
                                    "useful_actions": {
                                        "type": "array",
                                        "items": {"type": "string"}
                                    },
                                    "useless_actions": {
                                        "type": "array",
                                        "items": {"type": "string"}
                                    },
                                    "successful_sequence": {"type": "string"},
                                    "patterns_to_avoid": {
                                        "type": "array",
                                        "items": {"type": "string"}
                                    },
                                    "unexplored_areas": {
                                        "type": "array",
                                        "items": {"type": "string"}
                                    },
                                    "failure_analysis": {
                                        "type": "object",
                                        "properties": {
                                            "exploration_gaps": {
                                                "type": "array",
                                                "items": {"type": "string"}
                                            },
                                            "wrong_directions": {
                                                "type": "array",
                                                "items": {"type": "string"}
                                            },
                                            "missed_hints": {
                                                "type": "array",
                                                "items": {"type": "string"}
                                            },
                                            "concrete_improvements": {
                                                "type": "array",
                                                "items": {"type": "string"}
                                            }
                                        },
                                        "required": ["exploration_gaps", "wrong_directions", "missed_hints", "concrete_improvements"],
                                        "additionalProperties": False
                                    },
                                    "strategic_insights": {"type": "string"}
                                },
                                "required": ["useful_actions", "useless_actions", "successful_sequence", "patterns_to_avoid", "unexplored_areas", "failure_analysis", "strategic_insights"],
                                "additionalProperties": False
                            },
                            "recommended_prompt": {"type": "string"},
                            "key_insights": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        },
                        "required": ["reasoning", "recommended_prompt", "key_insights"],
                        "additionalProperties": False
                    }
                }
            }

            response = chat_completion_with_retries(
                model=llm_model,
                sys_prompt=sys_prompt,
                prompt=user_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                response_format=response_format
            )

            if not response or not hasattr(response, 'choices') or not response.choices:
                print("[Warning] Failed to get prompt recommendation")
                return {
                    'recommended_prompt': current_prompt,
                    'reasoning': 'Failed to generate recommendation',
                    'key_insights': []
                }

            result_text = response.choices[0].message.content.strip()

        # Parse JSON response
        try:
            # Clean up potential markdown code blocks
            if result_text.startswith('```json'):
                result_text = result_text.replace('```json', '').replace('```', '').strip()
            elif result_text.startswith('```'):
                result_text = result_text.replace('```', '').strip()

            result = json.loads(result_text)

            # Print the reasoning for transparency
            print("\n" + "="*60)
            print("TRAJECTORY REASONING")
            print("="*60)
            reasoning = result.get('reasoning', {})
            print(f"Useful actions: {reasoning.get('useful_actions', [])}")
            print(f"Useless actions: {reasoning.get('useless_actions', [])}")
            print(f"Successful sequence: {reasoning.get('successful_sequence', 'None identified')}")
            print(f"Patterns to avoid: {reasoning.get('patterns_to_avoid', [])}")
            print(f"Unexplored areas: {reasoning.get('unexplored_areas', [])}")

            # Print failure analysis if game was not successful
            failure_analysis = reasoning.get('failure_analysis', {})
            if failure_analysis:
                print("\n" + "-"*60)
                print("FAILURE ANALYSIS (Deep Dive)")
                print("-"*60)
                print(f"Exploration gaps: {failure_analysis.get('exploration_gaps', [])}")
                print(f"Wrong directions taken: {failure_analysis.get('wrong_directions', [])}")
                print(f"Missed hints: {failure_analysis.get('missed_hints', [])}")
                print(f"Concrete improvements for next time: {failure_analysis.get('concrete_improvements', [])}")
                print("-"*60 + "\n")

            print(f"Strategic insights: {reasoning.get('strategic_insights', 'None')}")
            print("="*60 + "\n")

            # Format the return to include reasoning text
            return {
                'recommended_prompt': result.get('recommended_prompt', current_prompt),
                'reasoning': reasoning.get('strategic_insights', 'No reasoning provided'),
                'key_insights': result.get('key_insights', [])
            }
        except json.JSONDecodeError as e:
            print(f"[Warning] Failed to parse recommendation JSON: {e}")
            print(f"[Debug] Raw response (first 500 chars): {result_text[:500]}")
            return {
                'recommended_prompt': current_prompt,
                'reasoning': 'Failed to parse response',
                'key_insights': []
            }

    except Exception as e:
        print(f"[Warning] Failed to generate prompt recommendation: {e}")
        return {
            'recommended_prompt': current_prompt,
            'reasoning': f'Error: {str(e)}',
            'key_insights': []
        }