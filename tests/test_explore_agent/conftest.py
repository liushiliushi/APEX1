"""Shared fixtures for explore_agent tests."""
import pytest
from argparse import Namespace


@pytest.fixture
def default_args():
    """Default args namespace mimicking main.py defaults."""
    return Namespace(
        game_name='library',
        rom_path='jericho-games/',
        output_path='output',
        env_step_limit=20,
        seed=0,
        llm_model='google/gemini-3-flash-preview',
        llm_temperature=0.8,
        max_memory=30,
        use_valid_actions=True,
        agent_type='explore',
        eval_runs=5,
        # Reflective/explore parameters
        reflect_interval=5,
        ucb_exploration_c=1.414,
        max_strategies=15,
        min_strategies=3,
        backprop_unreached_discount=0.3,
        backprop_method='per_node',
        backprop_gamma=0.6,
        # ExploreAgent module parameters
        strategy_space='milestone_tree',
        guidance_mode='full_plan',
        exploration_method='ucb',
        space_evolution='free_reflection',
        epsilon=0.1,
        thompson_prior_std=100.0,
    )


@pytest.fixture
def sample_tree():
    """A sample milestone tree with known structure for testing."""
    from src.explore_agent.strategy_space.milestone_tree import MilestoneTreeSpace
    space = MilestoneTreeSpace(backprop_gamma=0.6)

    # root -> A (visits=5, reward=50) -> C (visits=2, reward=30)
    # root -> B (visits=3, reward=15)
    a = space._create_node('root', 'Milestone A', 'go north')
    a['visits'] = 5
    a['total_reward'] = 50.0
    a['reward_sq_sum'] = 600.0

    b = space._create_node('root', 'Milestone B', 'go south')
    b['visits'] = 3
    b['total_reward'] = 15.0
    b['reward_sq_sum'] = 100.0

    c = space._create_node(a['id'], 'Milestone C', 'take sword')
    c['visits'] = 2
    c['total_reward'] = 30.0
    c['reward_sq_sum'] = 500.0

    space.tree['root']['visits'] = 8
    space.tree['root']['total_reward'] = 65.0

    return space


@pytest.fixture
def sample_candidates():
    """Sample candidate list for exploration method tests."""
    return [
        {'id': 'a', 'visits': 10, 'total_reward': 50.0, 'reward_sq_sum': 300.0},
        {'id': 'b', 'visits': 5, 'total_reward': 40.0, 'reward_sq_sum': 400.0},
        {'id': 'c', 'visits': 0, 'total_reward': 0.0, 'reward_sq_sum': 0.0},
    ]
