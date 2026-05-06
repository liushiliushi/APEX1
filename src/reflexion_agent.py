"""
Reflexion agent (Shinn et al., 2023): after each episode, generates a
structured textual self-reflection that critiques performance and guides
the next attempt. No exploration mechanism — purely memory + reflection.

Adapted from Reflexion (Shinn et al., 2023).
"""
from .openai_helpers import chat_completion_with_retries


class ReflexionAgent:

    def __init__(self, args, guiding_prompt: str = None):
        self.guiding_prompt = guiding_prompt or "Explore systematically and examine objects to make progress."
        self.args = args

        # Intra-episode sliding-window memory
        self.memory = []

        # Cross-episode: list of textual reflections
        self.reflections = []
        self.max_reflections = 10

        # Game history for current episode
        self.game_history = []

    # -------------------- Intra-episode memory --------------------

    def add_to_memory(self, state, response):
        self.memory.append({"state": state, "response": response})
        if len(self.memory) > self.args.max_memory:
            self.memory.pop(0)

    def _format_memory_for_prompt(self):
        if not self.memory:
            return ""
        parts = ["MEMORY (Recent states and agent's responses):"]
        for i, entry in enumerate(self.memory):
            parts.append(f"Memory {i+1}:")
            parts.append(f"STATE: {entry['state']}")
            if entry['response']:
                parts.append(f"AGENT'S RESPONSE: {entry['response']}")
        return "\n".join(parts)

    # -------------------- Episode lifecycle --------------------

    def start_episode(self):
        self.memory = []
        self.game_history = []
        if self.reflections:
            print(f"[Reflexion] Injecting {len(self.reflections)} reflection(s) from prior episodes:")
            for i, r in enumerate(self.reflections):
                print(f"[Reflexion Memory {i+1}]\n{r}\n[/Reflexion Memory {i+1}]")
        else:
            print(f"[Reflexion] First episode, no prior reflections.")

    def end_episode(self, state, score):
        self.game_history.append({"state": state, "action": "(end)", "reward": None, "score": score})

        reflection = self._reflect(score)
        if reflection:
            self.reflections.append(reflection)
            if len(self.reflections) > self.max_reflections:
                self.reflections.pop(0)
            print(f"[Reflexion] Generated reflection (score={score}):\n{reflection}\n[/Reflexion]")
        else:
            print(f"[Reflexion] Failed to generate reflection for episode with score={score}.")

    # -------------------- Action generation --------------------

    def generate_action(self, state_node, info=None):
        sys_prompt, user_prompt = self.get_prompts(state_node, info)

        res_obj = chat_completion_with_retries(
            model=self.args.llm_model,
            sys_prompt=sys_prompt,
            prompt=user_prompt,
            max_tokens=400,
            temperature=getattr(self.args, 'llm_temperature', 1.0),
        )

        if res_obj and hasattr(res_obj, 'choices') and res_obj.choices and res_obj.choices[0].message:
            full_response = res_obj.choices[0].message.content
            action_text = self._parse_llm_response(full_response)
        else:
            full_response = ""
            action_text = "look"

        self.add_to_memory(state_node.state, full_response)
        self.game_history.append({"state": state_node.state, "action": action_text, "reward": None, "score": None})

        return action_text.strip(), full_response

    def update_game_history_reward(self, reward, score):
        if self.game_history:
            self.game_history[-1]["reward"] = reward
            self.game_history[-1]["score"] = score

    def get_prompts(self, state_node, info=None):
        memory_text = self._format_memory_for_prompt()
        reflections_text = self._format_reflections()

        # Check if this is a maze game (only direction actions)
        is_maze = info and info.get('valid') and all(
            a in ('north', 'south', 'east', 'west', 'look', 'inventory')
            for a in info.get('valid', [])
        )

        if is_maze:
            return self._get_maze_prompts(state_node, info, memory_text, reflections_text)

        sys_prompt = (
            "You are an expert player aiming to complete a text-based adventure game. "
            "Points are given for making progress in the game. "
            "Select promising actions based on the game state and memory of past interactions."
        )
        if self.guiding_prompt:
            sys_prompt += f"\n\nFollow this guide: {self.guiding_prompt}"

        if reflections_text:
            sys_prompt += f"\n\n{reflections_text}"

        # Add valid actions if available
        valid_actions_text = ""
        if info and info.get('valid'):
            valid = info['valid']
            valid_actions_text = f"\nAvailable actions: {', '.join(valid)}\n"

        user_prompt = f"""{memory_text}

Your current state is: {state_node.state}
{valid_actions_text}
Type your next action as if you were playing the game directly. It should be a short command that can be understood by the game parser. Common actions include: look, inventory, directions (north, northeast, up, etc.), examine X, say X, drop X, get X, open X, enter X, ask X about Y, look in X, give X to Y, and other context-specific commands. When stuck, explore all rooms and objects mentioned in room descriptions systematically. *DO NOT REPEAT* the same failed action multiple times. Do not use the "help" command.

Your response MUST strictly follow this format and include nothing else:
REASONING: [A short, concise explanation of your choice, 1-2 sentences]
ACTION: [short word or phrase for text command to execute]

For example:
REASONING: I should examine the book to learn more about it.
ACTION: examine book
"""
        return sys_prompt, user_prompt

    def _get_maze_prompts(self, state_node, info, memory_text, reflections_text):
        """Simplified prompts for maze games."""
        sys_prompt = "You are navigating a maze to find treasures. Respond with ONLY a direction: north, south, east, or west."

        if reflections_text:
            sys_prompt += f"\n\n{reflections_text}"

        dirs = [a for a in info.get('valid', []) if a in ('north', 'south', 'east', 'west')]

        user_prompt = f"""{memory_text}

{state_node.state}

Valid directions: {', '.join(dirs)}
Action:"""
        return sys_prompt, user_prompt

    # -------------------- Reflection --------------------

    def _reflect(self, score):
        history_str = self._format_game_history()

        sys_prompt = (
            "You are an expert analyst for text adventure games. "
            "Your job is to analyze a completed game episode and produce a concise self-reflection "
            "that will help the agent perform better in the next attempt."
        )

        user_prompt = f"""The agent just completed an episode with a final score of {score}.

Here is the full game history:
--- GAME HISTORY START ---
{history_str}
--- GAME HISTORY END ---

Generate a self-reflection that includes:
1. For each reward found, the EXACT step-by-step path from the starting room to reach it (e.g., "From Dungeon Entrance: east, east, north, north → Torch Crossing (+5)").
2. Dead ends encountered and how to avoid them.
3. Unexplored directions that were skipped and should be tried next time.
4. A recommended opening sequence for the next attempt — the first 10-15 actions to take.

Focus on CONCRETE NAVIGATION INSTRUCTIONS, not general advice."""

        for attempt in range(3):
            try:
                res_obj = chat_completion_with_retries(
                    model=self.args.llm_model,
                    sys_prompt=sys_prompt,
                    prompt=user_prompt,
                    max_tokens=500,
                    temperature=0.5,
                )
                if res_obj and hasattr(res_obj, 'choices') and res_obj.choices and res_obj.choices[0].message:
                    content = res_obj.choices[0].message.content
                    if content:
                        return content.strip()
                print(f"[Reflexion] Reflection attempt {attempt+1}/3 returned empty, retrying...")
            except Exception as e:
                print(f"[Reflexion] Reflection attempt {attempt+1}/3 failed: {e}")
        return None

    def _format_reflections(self):
        if not self.reflections:
            return ""
        latest = self.reflections[-1]
        return f"REFLECTION FROM LAST EPISODE (follow this strategy):\n{latest}"

    def _format_game_history(self):
        parts = []
        for i, entry in enumerate(self.game_history):
            parts.append(f"Step {i+1}:")
            parts.append(f"  STATE: {entry['state'][:500]}")
            parts.append(f"  ACTION: {entry['action']}")
            if entry['reward'] is not None:
                parts.append(f"  REWARD: {entry['reward']}, SCORE: {entry['score']}")
            parts.append("---")
        return "\n".join(parts)

    # -------------------- Helpers --------------------

    def _parse_llm_response(self, full_response: str):
        action_text = "look"
        if not full_response or not isinstance(full_response, str):
            return action_text

        lines = full_response.strip().split('\n')
        try:
            for line in lines:
                if line.upper().startswith("ACTION:"):
                    action_text = line.split(":", 1)[1].strip()
                    return action_text
        except Exception as e:
            print(f"Error parsing LLM response: {e}. Response was: '{full_response}'")

        # Fallback: check if response is a bare direction
        for word in full_response.strip().lower().split():
            if word in ('north', 'south', 'east', 'west'):
                return word

        return action_text
