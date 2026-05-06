"""
ACE Agent for Text Adventure Games
===================================
Adapted from: https://github.com/ace-agent/ace.git
Paper: "Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models"
       (Zhang et al., 2025, arXiv:2510.04618)

Three-role architecture adapted to sequential game-playing:
  - Generator  : at each step, uses playbook + memory to select an action
  - Reflector  : after each episode, tags playbook bullets as helpful/harmful
  - Curator    : updates the playbook with new ADD operations from reflections
"""

import os
import re
import json
from .openai_helpers import chat_completion_with_retries


# ---------------------------------------------------------------------------
# Playbook utilities (adapted from ace/playbook_utils.py)
# ---------------------------------------------------------------------------

SECTION_SLUG_MAP = {
    'strategies_and_insights': 'str',
    'common_actions': 'act',
    'common_mistakes_to_avoid': 'mis',
    'progression_notes': 'pro',
    'others': 'oth',
}

EMPTY_PLAYBOOK = """\
## STRATEGIES & INSIGHTS

## COMMON ACTIONS

## COMMON MISTAKES TO AVOID

## PROGRESSION NOTES

## OTHERS"""


def _parse_playbook_line(line: str):
    pattern = r'\[([^\]]+)\]\s*helpful=(\d+)\s*harmful=(\d+)\s*::\s*(.*)'
    match = re.match(pattern, line.strip())
    if match:
        return {
            'id': match.group(1),
            'helpful': int(match.group(2)),
            'harmful': int(match.group(3)),
            'content': match.group(4),
        }
    return None


def _get_section_slug(section: str) -> str:
    normalized = section.lower().replace(' ', '_').replace('&', 'and')
    for key, slug in SECTION_SLUG_MAP.items():
        if key in normalized:
            return slug
    return 'oth'


def _get_next_id(playbook: str) -> int:
    max_id = 0
    for line in playbook.split('\n'):
        parsed = _parse_playbook_line(line)
        if parsed:
            m = re.search(r'-(\d+)$', parsed['id'])
            if m:
                max_id = max(max_id, int(m.group(1)))
    return max_id + 1


def _update_bullet_counts(playbook: str, bullet_tags: list) -> str:
    tag_map = {}
    for tag in bullet_tags:
        if isinstance(tag, dict):
            bid = tag.get('id') or tag.get('bullet', '')
            tv = tag.get('tag', 'neutral')
            if bid:
                tag_map[bid] = tv

    if not tag_map:
        return playbook

    lines = playbook.split('\n')
    new_lines = []
    for line in lines:
        parsed = _parse_playbook_line(line)
        if parsed and parsed['id'] in tag_map:
            tag = tag_map[parsed['id']]
            if tag == 'helpful':
                parsed['helpful'] += 1
            elif tag == 'harmful':
                parsed['harmful'] += 1
            new_lines.append(
                f"[{parsed['id']}] helpful={parsed['helpful']} harmful={parsed['harmful']} :: {parsed['content']}"
            )
        else:
            new_lines.append(line)
    return '\n'.join(new_lines)


def _apply_add_operations(playbook: str, operations: list, next_id: int):
    lines = playbook.split('\n')
    bullets_to_add = []

    for op in operations:
        if op.get('type') != 'ADD':
            continue
        section_raw = op.get('section', 'others')
        section = section_raw.lower().replace(' ', '_').replace('&', 'and')
        slug = _get_section_slug(section)
        bullet_id = f"{slug}-{next_id:05d}"
        next_id += 1
        content = op.get('content', '').strip()
        bullets_to_add.append((section, f"[{bullet_id}] helpful=0 harmful=0 :: {content}"))
        print(f"[ACEAgent] Curator ADD {bullet_id} → {section}: {content[:80]}")

    # Insert bullets after their section headers
    final_lines = []
    current_section = None
    pending = list(bullets_to_add)

    for line in lines:
        if line.strip().startswith('##'):
            # Flush bullets for outgoing section
            if current_section is not None:
                for sec, bullet in list(pending):
                    if sec == current_section:
                        final_lines.append(bullet)
                pending = [(s, b) for s, b in pending if s != current_section]
            header = line.strip()[2:].strip()
            current_section = header.lower().replace(' ', '_').replace('&', 'and')
        final_lines.append(line)

    # Flush last section
    if current_section is not None:
        for sec, bullet in list(pending):
            if sec == current_section:
                final_lines.append(bullet)
        pending = [(s, b) for s, b in pending if s != current_section]

    # Remaining (unmatched sections) → OTHERS
    if pending:
        others_bullets = [b for _, b in pending]
        others_idx = next((i for i, l in enumerate(final_lines) if l.strip() == '## OTHERS'), -1)
        if others_idx >= 0:
            for i, bullet in enumerate(others_bullets):
                final_lines.insert(others_idx + 1 + i, bullet)
        else:
            final_lines.extend(others_bullets)

    return '\n'.join(final_lines), next_id


def _get_playbook_stats(playbook: str) -> dict:
    stats = {'total_bullets': 0, 'high_performing': 0, 'problematic': 0, 'unused': 0}
    for line in playbook.split('\n'):
        p = _parse_playbook_line(line)
        if p:
            stats['total_bullets'] += 1
            if p['helpful'] > 5 and p['harmful'] < 2:
                stats['high_performing'] += 1
            elif p['harmful'] >= p['helpful'] and p['harmful'] > 0:
                stats['problematic'] += 1
            elif p['helpful'] + p['harmful'] == 0:
                stats['unused'] += 1
    return stats


def _extract_bullet_ids(text: str) -> list:
    return re.findall(r'\[([a-z]{2,5}-\d{5})\]', text)


def _extract_bullets_by_ids(playbook: str, ids: list) -> str:
    found = []
    for line in playbook.split('\n'):
        p = _parse_playbook_line(line)
        if p and p['id'] in ids:
            found.append(f"[{p['id']}] helpful={p['helpful']} harmful={p['harmful']} :: {p['content']}")
    return '\n'.join(found) if found else "(No relevant bullets found)"


def _extract_json(text: str) -> dict:
    """Try to extract a JSON object from LLM response."""
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    # Try ```json blocks
    for m in re.finditer(r'```json\s*(.*?)\s*```', text, re.DOTALL | re.IGNORECASE):
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
    # Brace-counting fallback
    for m in re.finditer(r'\{', text):
        start = m.start()
        depth = 0
        for i in range(start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
    return None


# ---------------------------------------------------------------------------
# Prompts (adapted for text adventure games)
# ---------------------------------------------------------------------------

GENERATOR_SYS = """\
You are an expert player of text-based adventure games. \
Your goal is to maximise the game score by choosing effective actions. \
You have access to a playbook of accumulated strategies and a short-term memory of recent steps.\
"""

GENERATOR_USER = """\
## PLAYBOOK (accumulated game strategies):
{playbook}

## RECENT MEMORY (last steps this episode):
{memory}

## CURRENT STATE:
{state}

## VALID ACTIONS (choose from these if provided):
{valid_actions}

Use the playbook and memory to pick the best next action. \
Reference any relevant playbook bullet IDs in your reasoning.

Respond in this exact JSON format:
{{
  "reasoning": "[Brief reasoning referencing playbook bullets if relevant]",
  "bullet_ids": ["str-00001", "act-00002"],
  "action": "[single game command, e.g. 'go north' or 'take lamp']"
}}
"""

REFLECTOR_SYS = """\
You are an expert analyst reviewing a text adventure game episode. \
Your job is to identify which playbook strategies helped or hurt performance, \
and extract new insights from the trajectory.\
"""

REFLECTOR_USER = """\
## PLAYBOOK (used during this episode):
{playbook}

## EPISODE TRAJECTORY:
{trajectory}

## FINAL SCORE: {score} (out of max possible)

Analyse the trajectory. For each playbook bullet that was relevant, tag it as 'helpful', 'harmful', or 'neutral'.
Then summarise what worked, what failed, and what new insights should be added to the playbook.

Respond in this exact JSON format:
{{
  "error_identification": "[What went wrong in this episode, if anything]",
  "correct_approach": "[What the agent should have done differently]",
  "key_insight": "[The single most important new lesson from this episode]",
  "bullet_tags": [
    {{"id": "str-00001", "tag": "helpful"}},
    {{"id": "act-00002", "tag": "harmful"}}
  ]
}}
"""

CURATOR_SYS = """\
You are a master curator of game-playing knowledge. \
You update a playbook of strategies for a text-based adventure game, \
adding concise, actionable insights that will help future episodes.\
"""

CURATOR_USER = """\
## CURRENT PLAYBOOK:
{playbook}

## PLAYBOOK STATS:
{stats}

## REFLECTION FROM LAST EPISODE:
{reflection}

## EPISODE CONTEXT (final score: {score}):
{trajectory_summary}

Review the reflection and identify ONLY new insights not already in the playbook. \
Be concise — prefer a few high-quality bullets over many redundant ones.

Respond in this exact JSON format:
{{
  "reasoning": "[Why these additions are needed]",
  "operations": [
    {{
      "type": "ADD",
      "section": "strategies_and_insights",
      "content": "[Concise, actionable strategy]"
    }}
  ]
}}

Available sections: strategies_and_insights, common_actions, common_mistakes_to_avoid, progression_notes, others.
If nothing new to add, return an empty operations list.
"""


# ---------------------------------------------------------------------------
# ACE Agent
# ---------------------------------------------------------------------------

class ACEAgent:
    """
    ACE (Agentic Context Engineering) Agent adapted for text adventure games.

    Maintains an evolving playbook of game strategies updated after each episode
    via a three-role pipeline: Generator → Reflector → Curator.
    """

    def __init__(self, args, guiding_prompt: str = None):
        self.guiding_prompt = guiding_prompt or "Explore systematically and try to maximise your score."
        self.args = args

        # Intra-episode state
        self.memory = []
        self.game_history = []

        # LLM settings
        self.llm_model = getattr(args, 'llm_model', 'google/gemini-2.5-flash-preview')
        self.temperature = getattr(args, 'llm_temperature', 0.8)
        self.reflect_model = getattr(args, 'evolution_llm_model', self.llm_model)

        # Playbook persistence: output/{game}/ace/{model}/playbook.txt
        output_path = getattr(args, 'output_path', 'output')
        game_name = getattr(args, 'game_name', 'game')
        agent_type = getattr(args, 'agent_type', 'ace')
        model_slug = self.llm_model.replace('/', '_').replace('\\', '_')
        self.playbook_dir = os.path.join(output_path, game_name, agent_type, model_slug)
        self.playbook_path = os.path.join(self.playbook_dir, 'playbook.txt')

        # Playbook state
        self.playbook = EMPTY_PLAYBOOK
        self.next_bullet_id = 1
        self.episode_count = 0

    # ------------------------------------------------------------------
    # Playbook persistence
    # ------------------------------------------------------------------

    def _load_playbook(self):
        if os.path.exists(self.playbook_path):
            try:
                with open(self.playbook_path, 'r', encoding='utf-8') as f:
                    self.playbook = f.read()
                self.next_bullet_id = _get_next_id(self.playbook)
                n = sum(1 for l in self.playbook.split('\n') if _parse_playbook_line(l))
                print(f"[ACEAgent] Loaded playbook ({n} bullets, next_id={self.next_bullet_id})")
            except Exception as e:
                print(f"[ACEAgent] Error loading playbook: {e}")
                self.playbook = EMPTY_PLAYBOOK
        else:
            print("[ACEAgent] No existing playbook (first run)")
            self.playbook = EMPTY_PLAYBOOK

    def _save_playbook(self):
        try:
            os.makedirs(self.playbook_dir, exist_ok=True)
            with open(self.playbook_path, 'w', encoding='utf-8') as f:
                f.write(self.playbook)
            n = sum(1 for l in self.playbook.split('\n') if _parse_playbook_line(l))
            print(f"[ACEAgent] Saved playbook ({n} bullets)")
        except Exception as e:
            print(f"[ACEAgent] Error saving playbook: {e}")

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def start_episode(self):
        self.memory = []
        self.game_history = []
        self.episode_count += 1
        self._load_playbook()
        print(f"[ACEAgent] Episode {self.episode_count} started")

    def end_episode(self, state: str, score: float):
        print(f"[ACEAgent] Episode {self.episode_count} ended — score: {score}")
        if self.game_history:
            self.game_history[-1]['final_state'] = state

        trajectory = self._format_trajectory()

        # --- Reflector ---
        reflection_text, bullet_tags = self._reflect(trajectory, score)

        # Update bullet helpful/harmful counts
        if bullet_tags:
            self.playbook = _update_bullet_counts(self.playbook, bullet_tags)

        # --- Curator ---
        self._curate(reflection_text, trajectory, score)

        self._save_playbook()

    # ------------------------------------------------------------------
    # Generator (intra-episode action selection)
    # ------------------------------------------------------------------

    def add_to_memory(self, state: str, response: str):
        self.memory.append({'state': state, 'response': response})
        if len(self.memory) > self.args.max_memory:
            self.memory.pop(0)

    def update_game_history_reward(self, reward, score):
        if self.game_history:
            self.game_history[-1]['reward'] = reward
            self.game_history[-1]['score'] = score

    def get_prompts(self, state_node, info=None):
        memory_text = self._format_memory()
        valid_actions = info.get('valid', []) if info else []
        valid_str = ', '.join(valid_actions) if valid_actions else "(not provided)"

        sys_prompt = GENERATOR_SYS
        if self.guiding_prompt:
            sys_prompt += f"\n\nGeneral guidance: {self.guiding_prompt}"

        user_prompt = GENERATOR_USER.format(
            playbook=self.playbook,
            memory=memory_text or "(no memory yet)",
            state=state_node.state,
            valid_actions=valid_str,
        )
        return sys_prompt, user_prompt

    def generate_action(self, state_node, valid_actions=None, info=None):
        sys_prompt, user_prompt = self.get_prompts(state_node, info=info)

        res_obj = chat_completion_with_retries(
            model=self.llm_model,
            sys_prompt=sys_prompt,
            prompt=user_prompt,
            max_tokens=400,
            temperature=self.temperature,
        )

        full_response = ""
        action = "look"

        if res_obj and hasattr(res_obj, 'choices') and res_obj.choices:
            full_response = res_obj.choices[0].message.content or ""
            parsed = _extract_json(full_response)
            if parsed and 'action' in parsed:
                action = str(parsed['action']).strip()
            else:
                # Fallback: try plain ACTION: line
                for line in full_response.split('\n'):
                    if line.upper().startswith('ACTION:'):
                        action = line.split(':', 1)[1].strip()
                        break

        self.add_to_memory(state_node.state, full_response)
        self.game_history.append({
            'state': state_node.state,
            'action': action,
            'full_response': full_response,
            'reward': None,
            'score': None,
        })

        return action.strip(), full_response

    # ------------------------------------------------------------------
    # Reflector
    # ------------------------------------------------------------------

    def _reflect(self, trajectory: str, score: float):
        """Call Reflector: returns (reflection_text, bullet_tags)."""
        sys_prompt = REFLECTOR_SYS
        user_prompt = REFLECTOR_USER.format(
            playbook=self.playbook,
            trajectory=trajectory,
            score=score,
        )

        res_obj = chat_completion_with_retries(
            model=self.reflect_model,
            sys_prompt=sys_prompt,
            prompt=user_prompt,
            max_tokens=1000,
            temperature=0.5,
        )

        reflection_text = ""
        bullet_tags = []

        if res_obj and hasattr(res_obj, 'choices') and res_obj.choices:
            reflection_text = res_obj.choices[0].message.content or ""
            parsed = _extract_json(reflection_text)
            if parsed:
                bullet_tags = parsed.get('bullet_tags', [])
                print(f"[ACEAgent] Reflector: {len(bullet_tags)} bullet tags")
            else:
                print("[ACEAgent] Reflector: could not parse JSON response")

        return reflection_text, bullet_tags

    # ------------------------------------------------------------------
    # Curator
    # ------------------------------------------------------------------

    def _curate(self, reflection_text: str, trajectory: str, score: float):
        """Call Curator: updates self.playbook with ADD operations."""
        stats = _get_playbook_stats(self.playbook)
        trajectory_summary = trajectory[:800] + '...' if len(trajectory) > 800 else trajectory

        sys_prompt = CURATOR_SYS
        user_prompt = CURATOR_USER.format(
            playbook=self.playbook,
            stats=json.dumps(stats, indent=2),
            reflection=reflection_text[:1500] if reflection_text else "(no reflection)",
            score=score,
            trajectory_summary=trajectory_summary,
        )

        res_obj = chat_completion_with_retries(
            model=self.reflect_model,
            sys_prompt=sys_prompt,
            prompt=user_prompt,
            max_tokens=800,
            temperature=0.5,
        )

        if not (res_obj and hasattr(res_obj, 'choices') and res_obj.choices):
            print("[ACEAgent] Curator: empty LLM response, skipping")
            return

        curator_text = res_obj.choices[0].message.content or ""
        parsed = _extract_json(curator_text)

        if not parsed:
            print("[ACEAgent] Curator: could not parse JSON response, skipping")
            return

        operations = parsed.get('operations', [])
        if not operations:
            print("[ACEAgent] Curator: no operations to apply")
            return

        self.playbook, self.next_bullet_id = _apply_add_operations(
            self.playbook, operations, self.next_bullet_id
        )
        print(f"[ACEAgent] Curator: applied {len(operations)} operation(s)")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _format_memory(self) -> str:
        if not self.memory:
            return ""
        parts = []
        for i, entry in enumerate(self.memory[-5:]):  # last 5 steps
            parts.append(f"Step {i+1}: STATE={entry['state'][:150]}")
            if entry['response']:
                parts.append(f"  RESPONSE={entry['response'][:80]}")
        return '\n'.join(parts)

    def _format_trajectory(self) -> str:
        if not self.game_history:
            return "No trajectory recorded."
        parts = []
        for i, entry in enumerate(self.game_history):
            parts.append(f"Step {i+1}:")
            parts.append(f"  STATE: {entry.get('state', '')[:300]}")
            parts.append(f"  ACTION: {entry.get('action', '')}")
            if entry.get('reward') is not None:
                parts.append(f"  REWARD: {entry['reward']}  SCORE: {entry.get('score', '')}")
        return '\n'.join(parts)
