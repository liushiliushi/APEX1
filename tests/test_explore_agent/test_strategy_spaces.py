"""Tests for strategy spaces (Module 0)."""
import json
import os
import tempfile
import pytest
from src.explore_agent.strategy_space.milestone_tree import MilestoneTreeSpace
from src.explore_agent.strategy_space.plan_flat_list import PlanFlatListSpace
from src.explore_agent.strategy_space.action_tree import ActionTreeSpace
from src.explore_agent.exploration.ucb import UCBExploration


class TestMilestoneTree:
    def test_empty_tree_has_no_strategies(self):
        space = MilestoneTreeSpace()
        assert not space.has_strategies()
        assert space.active_count() == 0

    def test_create_nodes(self, sample_tree):
        assert sample_tree.has_strategies()
        assert sample_tree.active_count() == 3  # A, B, C

    def test_select_path(self, sample_tree):
        ucb = UCBExploration(c=1.414)
        path = sample_tree.select_path(ucb)
        assert path[0] == 'root'
        assert len(path) >= 2

    def test_path_to_strategy(self, sample_tree):
        path = ['root', 'node_001', 'node_003']
        strategy = sample_tree.path_to_strategy(path)
        assert strategy is not None
        assert strategy['description']
        assert len(strategy['high_level_steps']) == 2

    def test_backprop_per_node(self, sample_tree):
        path = ['root', 'node_001', 'node_003']
        old_visits_a = sample_tree.tree['node_001']['visits']
        old_visits_c = sample_tree.tree['node_003']['visits']

        sample_tree.backpropagate(path, score=20.0,
                                   milestone_rewards={0: 10.0, 1: 10.0})

        assert sample_tree.tree['node_001']['visits'] == old_visits_a + 1
        assert sample_tree.tree['node_003']['visits'] == old_visits_c + 1

    def test_backprop_not_followed(self, sample_tree):
        path = ['root', 'node_001']
        old_reward = sample_tree.tree['node_001']['total_reward']

        sample_tree.backpropagate(path, score=20.0,
                                   milestone_rewards={0: 20.0},
                                   compliance_result={'compliance': 'not_followed'})

        # Reward should not change
        assert sample_tree.tree['node_001']['total_reward'] == old_reward
        # But visits should increase
        assert sample_tree.tree['node_001']['visits'] == 6

    def test_apply_add_child(self, sample_tree):
        ops = [{'op': 'add_child', 'parent_id': 'root',
                'milestone': 'New Milestone D', 'key_action': 'go east',
                'pitfalls': [], 'reason': 'test'}]
        sample_tree.apply_operations(ops)
        assert sample_tree.active_count() == 4

    def test_apply_prune(self, sample_tree):
        ops = [{'op': 'prune', 'node_id': 'node_001', 'reason': 'test'}]
        sample_tree.apply_operations(ops)
        # A and its child C should be pruned
        assert sample_tree.tree['node_001']['status'] == 'pruned'
        assert sample_tree.tree['node_003']['status'] == 'pruned'

    def test_save_load(self, sample_tree, tmp_path):
        save_path = str(tmp_path / 'test_space.json')
        sample_tree.save(save_path)

        loaded = MilestoneTreeSpace()
        loaded.load(save_path)
        assert loaded.active_count() == sample_tree.active_count()
        assert loaded.tree['node_001']['visits'] == 5

    def test_dedup_sibling(self, sample_tree):
        # Adding same milestone under root should be skipped
        ops = [{'op': 'add_child', 'parent_id': 'root',
                'milestone': 'Milestone A', 'key_action': '',
                'pitfalls': [], 'reason': 'duplicate test'}]
        old_count = sample_tree.active_count()
        sample_tree.apply_operations(ops)
        assert sample_tree.active_count() == old_count

    def test_reward_sq_sum_tracked(self, sample_tree):
        path = ['root', 'node_001']
        sample_tree.backpropagate(path, score=20.0,
                                   milestone_rewards={0: 20.0})
        node = sample_tree.tree['node_001']
        assert node.get('reward_sq_sum', 0) > 0


class TestPlanFlatList:
    def test_empty_has_no_strategies(self):
        space = PlanFlatListSpace()
        assert not space.has_strategies()

    def test_add_and_select(self):
        space = PlanFlatListSpace()
        ops = [{
            'op': 'add_branch',
            'description': 'Test strategy',
            'milestones': [
                {'milestone': 'Step 1', 'key_action': 'go north'},
                {'milestone': 'Step 2', 'key_action': 'take sword'},
            ],
            'reason': 'test',
        }]
        space.apply_operations(ops)
        assert space.active_count() == 1

        ucb = UCBExploration(c=1.414)
        path = space.select_path(ucb)
        assert len(path) == 1

        strategy = space.path_to_strategy(path)
        assert strategy is not None
        assert strategy['description'] == 'Test strategy'

    def test_backprop(self):
        space = PlanFlatListSpace()
        ops = [{'op': 'add_branch', 'description': 'Test',
                'milestones': [{'milestone': 'A', 'key_action': ''}],
                'reason': 'test'}]
        space.apply_operations(ops)

        strategy_id = space.strategies[0]['id']
        space.backpropagate([strategy_id], score=25.0,
                           milestone_rewards={})
        assert space.strategies[0]['visits'] == 1
        assert space.strategies[0]['total_reward'] == 25.0

    def test_save_load(self, tmp_path):
        space = PlanFlatListSpace()
        ops = [{'op': 'add_branch', 'description': 'Test',
                'milestones': [{'milestone': 'A', 'key_action': ''}],
                'reason': 'test'}]
        space.apply_operations(ops)

        save_path = str(tmp_path / 'flat.json')
        space.save(save_path)

        loaded = PlanFlatListSpace()
        loaded.load(save_path)
        assert loaded.active_count() == 1


class TestActionTree:
    def test_empty_tree(self):
        space = ActionTreeSpace()
        assert not space.has_strategies()

    def test_expand_and_select(self):
        space = ActionTreeSpace()
        # Simulate gameplay expansion
        n1 = space.expand('root', 'go north', 'You are in a hallway')
        n2 = space.expand(n1, 'take lamp', 'You pick up a lamp')
        n3 = space.expand('root', 'go south', 'You are in a garden')

        assert space.active_count() == 3

        ucb = UCBExploration(c=1.414)
        path = space.select_path(ucb)
        assert path[0] == 'root'
        assert len(path) >= 2

    def test_expand_dedup(self):
        space = ActionTreeSpace()
        n1 = space.expand('root', 'go north')
        n2 = space.expand('root', 'go north')  # same action
        assert n1 == n2  # should return same node

    def test_backprop(self):
        space = ActionTreeSpace()
        n1 = space.expand('root', 'go north')
        path = ['root', n1]
        space.backpropagate(path, score=10.0, milestone_rewards={})
        assert space.tree[n1]['visits'] == 1
        assert space.tree[n1]['total_reward'] == 10.0
