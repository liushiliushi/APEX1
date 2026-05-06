from .base import StrategySpace
from .milestone_tree import MilestoneTreeSpace
from .milestone_dag import MilestoneDAGSpace
from .plan_flat_list import PlanFlatListSpace
from .action_tree import ActionTreeSpace

STRATEGY_SPACES = {
    'milestone_tree': MilestoneTreeSpace,
    'milestone_dag': MilestoneDAGSpace,
    'plan_flat_list': PlanFlatListSpace,
    'action_tree': ActionTreeSpace,
}

__all__ = ['StrategySpace', 'MilestoneTreeSpace', 'MilestoneDAGSpace', 'PlanFlatListSpace', 'ActionTreeSpace', 'STRATEGY_SPACES']
