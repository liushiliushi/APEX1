import random
from .openai_helpers import chat_completion_with_retries



class NaiveAgent:
    """
    Stateless baseline agent: has intra-episode sliding-window memory
    (same as Reflexion) but NO cross-episode memory or reflection.
    Each episode starts completely fresh.
    """
    def __init__(self, args, guiding_prompt: str = None):
        self.guiding_prompt = guiding_prompt or "Explore systematically and examine objects to make progress."
        self.args = args
        self.memory = []
        self.game_history = []

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

    def start_episode(self):
        self.memory = []
        self.game_history = []
        print(f"[Naive] Starting fresh episode (no cross-episode memory).")

    def end_episode(self, state, score):
        print(f"[Naive] Ending episode with score: {score}.")

    def update_game_history_reward(self, reward, score):
        if self.game_history:
            self.game_history[-1]["reward"] = reward
            self.game_history[-1]["score"] = score

    def get_prompts(self, state_node, info=None):
        memory_text = self._format_memory_for_prompt()

        # Check if this is a maze game
        is_maze = info and info.get('valid') and all(
            a in ('north', 'south', 'east', 'west', 'look', 'inventory')
            for a in info.get('valid', [])
        )

        if is_maze:
            sys_prompt = "You are navigating a maze to find treasures. Respond with ONLY a direction: north, south, east, or west."
            dirs = [a for a in info.get('valid', []) if a in ('north', 'south', 'east', 'west')]
            user_prompt = f"""{memory_text}

{state_node.state}

Valid directions: {', '.join(dirs)}
Action:"""
            return sys_prompt, user_prompt

        sys_prompt = (
            "You are an expert player aiming to complete a text-based adventure game. "
            "Points are given for making progress in the game. "
            "Select promising actions based on the game state and memory of past interactions."
        )
        if self.guiding_prompt:
            sys_prompt += f"\n\nFollow this guide: {self.guiding_prompt}"

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

    def _parse_llm_response(self, full_response: str):
        action_text = "look"
        if not full_response or not isinstance(full_response, str):
            return action_text

        # Try ACTION: format first
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
