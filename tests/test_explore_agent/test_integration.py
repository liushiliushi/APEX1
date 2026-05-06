"""Integration / smoke tests for ExploreAgent."""
import pytest
from src.explore_agent.strategy_space import STRATEGY_SPACES
from src.explore_agent.exploration import EXPLORATION_METHODS
from src.explore_agent.guidance import GUIDANCE_MODES
from src.explore_agent.evolution import EVOLUTION_METHODS


class TestModuleRegistries:
    """Verify all modules are properly registered."""

    def test_strategy_spaces_registered(self):
        assert 'milestone_tree' in STRATEGY_SPACES
        assert 'plan_flat_list' in STRATEGY_SPACES
        assert 'action_tree' in STRATEGY_SPACES

    def test_exploration_methods_registered(self):
        assert 'ucb' in EXPLORATION_METHODS
        assert 'thompson' in EXPLORATION_METHODS
        assert 'epsilon_greedy' in EXPLORATION_METHODS

    def test_guidance_modes_registered(self):
        assert 'full_plan' in GUIDANCE_MODES
        assert 'step_by_step' in GUIDANCE_MODES
        assert 'hierarchical' in GUIDANCE_MODES

    def test_evolution_methods_registered(self):
        assert 'free_reflection' in EVOLUTION_METHODS
        assert 'decision_point_mining' in EVOLUTION_METHODS


class TestAgentInstantiation:
    """Test that ExploreAgent can be instantiated with all module combinations."""

    @pytest.mark.parametrize("strategy_space", ['milestone_tree', 'plan_flat_list', 'action_tree'])
    @pytest.mark.parametrize("exploration_method", ['ucb', 'thompson', 'epsilon_greedy'])
    @pytest.mark.parametrize("guidance_mode", ['full_plan', 'step_by_step', 'hierarchical'])
    @pytest.mark.parametrize("space_evolution", ['free_reflection', 'decision_point_mining'])
    def test_instantiation(self, default_args, strategy_space, exploration_method,
                           guidance_mode, space_evolution):
        from src.explore_agent import ExploreAgent

        default_args.strategy_space = strategy_space
        default_args.exploration_method = exploration_method
        default_args.guidance_mode = guidance_mode
        default_args.space_evolution = space_evolution

        agent = ExploreAgent(default_args)
        assert agent._module_names['strategy_space'] == strategy_space
        assert agent._module_names['exploration_method'] == exploration_method
        assert agent._module_names['guidance_mode'] == guidance_mode
        assert agent._module_names['space_evolution'] == space_evolution


class TestSpaceExplorationInteraction:
    """Test that strategy spaces correctly interact with exploration methods."""

    @pytest.mark.parametrize("exploration_name", ['ucb', 'thompson', 'epsilon_greedy'])
    def test_milestone_tree_with_exploration(self, sample_tree, exploration_name):
        method_cls = EXPLORATION_METHODS[exploration_name]
        if exploration_name == 'ucb':
            method = method_cls(c=1.414)
        elif exploration_name == 'thompson':
            method = method_cls(prior_std=100.0)
        else:
            method = method_cls(epsilon=0.1)

        path = sample_tree.select_path(method)
        assert path[0] == 'root'
        assert len(path) >= 2

        strategy = sample_tree.path_to_strategy(path)
        assert strategy is not None

    @pytest.mark.parametrize("exploration_name", ['ucb', 'thompson', 'epsilon_greedy'])
    def test_flat_list_with_exploration(self, exploration_name):
        from src.explore_agent.strategy_space.plan_flat_list import PlanFlatListSpace

        space = PlanFlatListSpace()
        # Add some strategies
        for i in range(3):
            ops = [{'op': 'add_branch',
                    'description': f'Strategy {i}',
                    'milestones': [{'milestone': f'Step {i}', 'key_action': ''}],
                    'reason': 'test'}]
            space.apply_operations(ops)
            space.strategies[-1]['visits'] = i + 1
            space.strategies[-1]['total_reward'] = float((i + 1) * 10)
            space.strategies[-1]['reward_sq_sum'] = float((i + 1) * 100)

        method_cls = EXPLORATION_METHODS[exploration_name]
        if exploration_name == 'ucb':
            method = method_cls(c=1.414)
        elif exploration_name == 'thompson':
            method = method_cls(prior_std=100.0)
        else:
            method = method_cls(epsilon=0.1)

        path = space.select_path(method)
        assert len(path) == 1
        strategy = space.path_to_strategy(path)
        assert strategy is not None
