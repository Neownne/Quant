"""RL-Dynamic v2: GRPO 驱动动态因子权重策略。

StateBuilder → GRPO DirichletPolicyNet → Factor Weights → Scorer → NDrop Selection
FactorPool → IC 追踪 + 因子筛选
ConceptBoardFeatures → 热点板块动量 + 宽度/轮动特征
"""
from rl_dynamic.factor_pool import FactorPool
from rl_dynamic.env import WeightLearningEnv
from rl_dynamic.trainer import _build_daily_data, walk_forward_train_rl_weights
from rl_dynamic.grpo_trainer import DirichletPolicyNet, train_grpo_weights, GRPOPredictor
from rl_dynamic.concept_features import ConceptBoardFeatures

__all__ = [
    "FactorPool", "WeightLearningEnv", "_build_daily_data",
    "walk_forward_train_rl_weights",
    "DirichletPolicyNet", "train_grpo_weights", "GRPOPredictor",
    "ConceptBoardFeatures",
]