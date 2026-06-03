"""RL-Dynamic: 强化学习驱动动态因子权重策略。

StateBuilder → PPO PolicyNet → Factor Weights → Scorer → NDrop Selection
FactorPool → IC 追踪 + 因子筛选
"""
from rl_dynamic.factor_pool import FactorPool

__all__ = ["FactorPool"]
