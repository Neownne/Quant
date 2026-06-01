"""RL 神经网络模型。

StockScoreNet — 点式股票评分网络
  输入：因子值 + 市场环境特征 (obs_dim 维)
  输出：[0, 1] 评分
"""
from __future__ import annotations

import torch
import torch.nn as nn


class StockScoreNet(nn.Module):
    """股票评分策略网络。

    架构：
      FactorEncoder: Linear(n_factors, 64) -> ReLU -> Linear(64, 32)
      ContextEncoder: Linear(n_context, 16) -> ReLU -> Linear(16, 8)
      Fusion: Concat(32, 8) -> Linear(40, 32) -> ReLU -> Linear(32, 1) -> Sigmoid
    """

    def __init__(
        self,
        n_factors: int = 15,
        n_context: int = 4,
        hidden_dim: int = 64,
    ):
        super().__init__()

        self.n_factors = n_factors
        self.n_context = n_context
        self.total_dim = n_factors + n_context

        # 因子编码器
        self.factor_encoder = nn.Sequential(
            nn.Linear(n_factors, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        # 环境编码器
        self.context_encoder = nn.Sequential(
            nn.Linear(n_context, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.ReLU(),
        )

        # 融合层
        fusion_input = (hidden_dim // 2) + (hidden_dim // 8)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        参数
        ----
        x : (batch, n_factors + n_context) 观测张量

        返回
        ----
        (batch, 1) 评分张量，值域 [0, 1]
        """
        factors = x[:, :self.n_factors]
        context = x[:, self.n_factors:]

        f_enc = self.factor_encoder(factors)
        c_enc = self.context_encoder(context)

        fused = torch.cat([f_enc, c_enc], dim=-1)
        return self.fusion(fused)
