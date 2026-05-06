"""Global Memory — extract cross-episode lessons from gameplay experience.

After reflection, analyzes recent episodes to extract reusable lessons about
penalties, navigation constraints, game mechanics, and prerequisites.
These lessons are injected into the agent's action prompt every step.
"""
from typing import Any, Dict, List

from ..openai_helpers_proxy import chat_completion, parse_json_response


class GlobalMemoryUpdate:
    """Extract and maintain cross-episode lessons."""

    MAX_LESSONS = 20

    def update(self, episode_summaries: List[Dict], strategy_space,
               llm_model: str) -> int:
        """Extract lessons from recent episodes, merge into strategy_space.global_lessons.

        Returns count of new lessons added.
        """
        if not episode_summaries or len(episode_summaries) < 2:
            print("[GlobalMemory] Need at least 2 episodes, skipping")
            return 0

        recent = episode_summaries[-2:]
        existing = getattr(strategy_space, 'global_lessons', [])

        sys_prompt, user_prompt = self._build_prompt(recent, existing)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                raw = chat_completion(
                    model=llm_model,
                    sys_prompt=sys_prompt,
                    prompt=user_prompt,
                    max_tokens=2000,
                    temperature=0.3,
                )

                if not raw:
                    print(f"[GlobalMemory] LLM call failed (attempt {attempt + 1}/{max_retries})")
                    continue

                result = parse_json_response(raw)
                if not result:
                    print(f"[GlobalMemory] JSON parse failed (attempt {attempt + 1}/{max_retries})")
                    continue

                new_lessons = result.get('new_lessons', [])
                remove_ids = result.get('remove_ids', [])

                count = self._apply(new_lessons, remove_ids, strategy_space,
                                    episode_summaries[-1].get('episode', 0))
                print(f"[GlobalMemory] Added {count} lessons, "
                      f"total {len(strategy_space.global_lessons)}")
                return count

            except Exception as e:
                print(f"[GlobalMemory] Error (attempt {attempt + 1}/{max_retries}): {e}")

        print(f"[GlobalMemory] Failed after {max_retries} attempts")
        return 0

    def _build_prompt(self, recent_episodes: List[Dict],
                      existing_lessons: List[Dict]) -> tuple:
        episodes_text = ""
        for ep in recent_episodes:
            ep_num = ep.get('episode', '?')
            episodes_text += f"\n=== Episode {ep_num} (score: {ep.get('score', 0)}) ===\n"
            episodes_text += ep.get('summary', ep.get('text', 'No summary available'))
            episodes_text += "\n"

        existing_text = ""
        if existing_lessons:
            existing_text = "\nEXISTING LESSONS (do not duplicate these):\n"
            for l in existing_lessons:
                existing_text += (f"  [id={l['id']}] [{l['category']}] {l['lesson']} "
                                  f"(confidence: {l['confidence']})\n")

        sys_prompt = """You are an expert at extracting reusable lessons from text adventure game episodes.

Analyze the provided episode summaries and extract GENERAL lessons the agent should remember across all future episodes. Focus on facts that would prevent repeated mistakes.

Three categories of lessons:
- PENALTY: Actions that caused score loss (e.g. "Never jump at Dome Room — causes -10 penalty")
- NAVIGATION: Movement constraints discovered (e.g. "Studio chimney blocked when carrying sword — drop it first")
- MECHANIC: Game mechanics learned (e.g. "Storing treasures in Trophy Case gives bonus points")

ACCURACY RULES:
- Only record facts actually observed in the episode data — do NOT speculate
- For conditional discoveries (e.g. a path blocked by an item), require evidence that the condition was tested (e.g. agent dropped item and path became available)
- Set confidence to "high" if observed multiple times or verified by contrast; "medium" if observed once

Respond with valid JSON only:
{
    "new_lessons": [
        {
            "category": "PENALTY|NAVIGATION|MECHANIC",
            "lesson": "Concise actionable text",
            "evidence": "EP/Step references",
            "confidence": "high|medium"
        }
    ],
    "remove_ids": []
}

Rules:
- Skip lessons already covered by EXISTING LESSONS
- If an existing lesson is factually wrong based on new evidence, include its id in remove_ids
- Keep lessons concise and actionable — the agent reads them every step
- Maximum 5 new lessons per call"""

        user_prompt = f"""Extract reusable lessons from these recent episodes:

{episodes_text}
{existing_text}

What general lessons should the agent remember for all future episodes?"""

        return sys_prompt, user_prompt

    def _apply(self, new_lessons: List[Dict], remove_ids: List[int],
               strategy_space, current_episode: int) -> int:
        """Merge new lessons into strategy_space.global_lessons. Returns count added."""
        if not hasattr(strategy_space, 'global_lessons'):
            strategy_space.global_lessons = []

        # Remove flagged lessons
        if remove_ids:
            before = len(strategy_space.global_lessons)
            strategy_space.global_lessons = [
                l for l in strategy_space.global_lessons if l['id'] not in remove_ids
            ]
            removed = before - len(strategy_space.global_lessons)
            if removed:
                print(f"[GlobalMemory] Removed {removed} outdated lessons")

        # Assign IDs and add new lessons
        max_id = max((l['id'] for l in strategy_space.global_lessons), default=0)
        count = 0
        for lesson in new_lessons:
            if not lesson.get('lesson') or not lesson.get('category'):
                continue
            max_id += 1
            strategy_space.global_lessons.append({
                'id': max_id,
                'category': lesson['category'],
                'lesson': lesson['lesson'],
                'evidence': lesson.get('evidence', ''),
                'confidence': lesson.get('confidence', 'medium'),
                'added_at_episode': current_episode,
            })
            print(f"  [+] [{lesson['category']}] {lesson['lesson']}")
            count += 1

        return count
