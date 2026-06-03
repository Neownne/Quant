"""RL 因子权重策略网络。

FactorWeightNet: 市场状态 → 因子权重 (softmax归一化)
"""
import torch
import torch.nn as nn


class FactorWeightNet(nn.Module):
    """状态 → 因子权重。

    输入: state_dim 维市场状态向量
    输出: n_factors 维权重 (softmax 归一化，和为1)
    """

    def __init__(self, state_dim: int, n_factors: int, hidden: int = 128):
        super().__init__()
        self.state_dim = state_dim
        self.n_factors = n_factors

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, n_factors),
            nn.Softmax(dim=-1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            state: (batch, state_dim) 市场状态

        Returns:
            (batch, n_factors) 因子权重
        """
        return self.net(state)
