"""Tests for exploration methods (Module 2)."""
import math
import pytest
from src.explore_agent.exploration.ucb import UCBExploration
from src.explore_agent.exploration.thompson import ThompsonSampling
from src.explore_agent.exploration.epsilon_greedy import EpsilonGreedy


class TestUCBExploration:
    def test_unvisited_gets_infinity(self):
        ucb = UCBExploration(c=1.414)
        candidate = {'visits': 0, 'total_reward': 0.0, 'reward_sq_sum': 0.0}
        assert ucb.score(candidate, 10) == float('inf')

    def test_ucb_formula(self):
        ucb = UCBExploration(c=1.414)
        candidate = {'visits': 10, 'total_reward': 50.0, 'reward_sq_sum': 300.0}
        parent_visits = 100

        expected_avg = 50.0 / 10
        expected_explore = 1.414 * math.sqrt(math.log(100) / 10)
        expected = expected_avg + expected_explore

        assert abs(ucb.score(candidate, parent_visits) - expected) < 1e-6

    def test_select_prioritizes_unvisited(self, sample_candidates):
        ucb = UCBExploration(c=1.414)
        # Candidate 'c' has 0 visits, should always be selected
        selected = ucb.select(sample_candidates, parent_visits=15)
        assert selected == 2  # index of 'c'

    def test_select_among_visited(self):
        ucb = UCBExploration(c=1.414)
        candidates = [
            {'id': 'a', 'visits': 10, 'total_reward': 50.0, 'reward_sq_sum': 300.0},
            {'id': 'b', 'visits': 10, 'total_reward': 100.0, 'reward_sq_sum': 1100.0},
        ]
        # b has higher avg reward, should be selected more often
        selections = [ucb.select(candidates, 20) for _ in range(100)]
        b_count = selections.count(1)
        assert b_count > 50  # b should be selected majority of times

    def test_select_empty_raises(self):
        ucb = UCBExploration(c=1.414)
        with pytest.raises(ValueError):
            ucb.select([], 10)


class TestThompsonSampling:
    def test_wide_prior_for_unvisited(self):
        ts = ThompsonSampling(prior_std=100.0)
        candidate = {'visits': 0, 'total_reward': 0.0, 'reward_sq_sum': 0.0}
        # Should produce a wide range of samples
        samples = [ts.score(candidate, 10) for _ in range(100)]
        assert max(samples) - min(samples) > 50  # wide spread

    def test_narrow_for_well_visited(self):
        ts = ThompsonSampling(prior_std=100.0)
        candidate = {'visits': 100, 'total_reward': 500.0, 'reward_sq_sum': 2600.0}
        # Mean=5, variance should be small
        samples = [ts.score(candidate, 200) for _ in range(100)]
        # Most samples should be close to mean=5
        within_range = sum(1 for s in samples if 3 < s < 7)
        assert within_range > 50

    def test_select(self, sample_candidates):
        ts = ThompsonSampling(prior_std=100.0)
        # Should not crash, returns valid index
        idx = ts.select(sample_candidates, 15)
        assert 0 <= idx < len(sample_candidates)


class TestEpsilonGreedy:
    def test_unvisited_priority(self, sample_candidates):
        eg = EpsilonGreedy(epsilon=0.0)  # pure greedy
        idx = eg.select(sample_candidates, 15)
        assert idx == 2  # unvisited candidate

    def test_pure_greedy(self):
        eg = EpsilonGreedy(epsilon=0.0)
        candidates = [
            {'id': 'a', 'visits': 10, 'total_reward': 50.0, 'reward_sq_sum': 300.0},
            {'id': 'b', 'visits': 10, 'total_reward': 100.0, 'reward_sq_sum': 1100.0},
        ]
        # With epsilon=0, should always pick b (highest avg)
        selections = [eg.select(candidates, 20) for _ in range(100)]
        assert all(s == 1 for s in selections)

    def test_exploration_with_high_epsilon(self):
        eg = EpsilonGreedy(epsilon=1.0)  # always random
        candidates = [
            {'id': 'a', 'visits': 10, 'total_reward': 50.0, 'reward_sq_sum': 300.0},
            {'id': 'b', 'visits': 10, 'total_reward': 100.0, 'reward_sq_sum': 1100.0},
        ]
        selections = [eg.select(candidates, 20) for _ in range(200)]
        a_count = selections.count(0)
        # Should be roughly 50/50 with full random
        assert 60 < a_count < 140

    def test_select_empty_raises(self):
        eg = EpsilonGreedy(epsilon=0.1)
        with pytest.raises(ValueError):
            eg.select([], 10)
