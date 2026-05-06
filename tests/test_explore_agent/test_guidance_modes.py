"""Tests for guidance modes (Module 1)."""
import pytest
from src.explore_agent.guidance.full_plan import FullPlanGuidance
from src.explore_agent.guidance.step_by_step import StepByStepGuidance
from src.explore_agent.guidance.hierarchical import HierarchicalGuidance


@pytest.fixture
def sample_strategy():
    return {
        'description': 'Get sword and slay dragon',
        'steps': ['1. Go north', '2. Take sword', '3. Slay dragon'],
        'high_level_steps': [
            {'step': 'Go to armory', 'key_action': 'go north'},
            {'step': 'Take the sword', 'key_action': 'take sword'},
            {'step': 'Slay the dragon', 'key_action': 'attack dragon with sword'},
        ],
        'pitfalls': [
            {'where': 'armory', 'loop_pattern': 'going back and forth', 'escape_action': 'take sword'},
            'Do not drop the sword',
        ],
    }


class TestFullPlanGuidance:
    def test_shows_all_milestones(self, sample_strategy):
        mode = FullPlanGuidance()
        prompt = mode.build_strategy_prompt(sample_strategy, milestone_idx=0, step_limit=50)
        assert 'Go to armory' in prompt
        assert 'Take the sword' in prompt
        assert 'Slay the dragon' in prompt

    def test_current_marker(self, sample_strategy):
        mode = FullPlanGuidance()
        prompt = mode.build_strategy_prompt(sample_strategy, milestone_idx=1, step_limit=50)
        assert 'CURRENT' in prompt
        # First should be completed
        assert '\u2713' in prompt  # checkmark

    def test_all_completed(self, sample_strategy):
        mode = FullPlanGuidance()
        prompt = mode.build_strategy_prompt(sample_strategy, milestone_idx=3, step_limit=50)
        assert 'ALL MILESTONES COMPLETED' in prompt

    def test_pitfall_prompt(self, sample_strategy):
        mode = FullPlanGuidance()
        prompt = mode.build_pitfall_prompt(sample_strategy['pitfalls'])
        assert 'armory' in prompt
        assert 'Do not drop the sword' in prompt

    def test_response_schema_with_pitfalls(self):
        mode = FullPlanGuidance()
        schema = mode.build_response_schema(has_pitfalls=True)
        props = schema['json_schema']['schema']['properties']
        assert 'pitfall_check' in props
        assert 'action' in props

    def test_response_schema_without_pitfalls(self):
        mode = FullPlanGuidance()
        schema = mode.build_response_schema(has_pitfalls=False)
        props = schema['json_schema']['schema']['properties']
        assert 'pitfall_check' not in props


class TestStepByStepGuidance:
    def test_shows_only_current(self, sample_strategy):
        mode = StepByStepGuidance()
        prompt = mode.build_strategy_prompt(sample_strategy, milestone_idx=0, step_limit=50)
        assert 'Go to armory' in prompt
        # Should NOT show future milestones in detail
        assert 'Slay the dragon' not in prompt or 'CURRENT' not in prompt.split('Slay')[0]

    def test_shows_completed_summary(self, sample_strategy):
        mode = StepByStepGuidance()
        prompt = mode.build_strategy_prompt(sample_strategy, milestone_idx=2, step_limit=50)
        # Current is "Slay the dragon"
        assert 'Slay the dragon' in prompt
        # Completed milestones shown briefly
        assert '\u2713' in prompt

    def test_all_completed(self, sample_strategy):
        mode = StepByStepGuidance()
        prompt = mode.build_strategy_prompt(sample_strategy, milestone_idx=3, step_limit=50)
        assert 'completed' in prompt.lower()


class TestHierarchicalGuidance:
    def test_shows_overview_and_detail(self, sample_strategy):
        mode = HierarchicalGuidance()
        prompt = mode.build_strategy_prompt(sample_strategy, milestone_idx=1, step_limit=50)
        # Overview should show all milestones
        assert 'OVERVIEW' in prompt
        assert 'Go to armory' in prompt
        assert 'Slay the dragon' in prompt
        # Detail for current
        assert 'CURRENT TASK' in prompt
        assert 'take sword' in prompt  # key_action of milestone 2

    def test_all_completed(self, sample_strategy):
        mode = HierarchicalGuidance()
        prompt = mode.build_strategy_prompt(sample_strategy, milestone_idx=3, step_limit=50)
        assert 'completed' in prompt.lower()
