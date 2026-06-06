"""GRPO 因子权重训练器 — Dirichlet 策略 + 组内标准化 advantage。

不依赖 stable-baselines3。复用 FactorWeightNet 架构，输出 Dirichlet 浓度参数。
每个 state 采 M=8 个 action，组内标准化消除市场噪声。

原理:
    π_θ: state → Dirichlet(α) → 因子权重向量 (和为1)
    reward = top-K 选股的平均 ret_1d
    advantage = (r_i - mean(r)) / (std(r) + 1e-8)  # 组内标准化
    loss = -mean(clip(ratio, 1-ε, 1+ε) * advantage) - β * entropy
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet
from loguru import logger


# ── 策略网络 ──────────────────────────────────────────────

class DirichletPolicyNet(nn.Module):
    """市场状态 → Dirichlet 浓度参数 → 因子权重分布。

    Input:  state_dim 维市场状态向量
    Output: n_factors 维浓度参数 (正数, 通过 softplus 保证 > 0)
    """

    def __init__(self, state_dim: int, n_factors: int, hidden: int = 128):
        super().__init__()
        self.state_dim = state_dim
        self.n_factors = n_factors

        self.body = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden // 2, n_factors)
        # 较大初始化：让各因子 logits 有初始差异，不等权
        nn.init.normal_(self.head.weight, mean=0, std=0.5)
        nn.init.zeros_(self.head.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """返回浓度参数 α (正数)."""
        x = self.body(state)
        logits = self.head(x)
        # softplus 保证正浓度。不加偏置，允许策略输出极端权重。
        return F.softplus(logits) + 0.01  # +ε 避免严格为 0 的浓度


# ── 辅助函数 ──────────────────────────────────────────────

def _compute_rewards(
    weights: torch.Tensor,  # (M, N)  weight vectors
    factor_matrix: np.ndarray,  # (S, N)  factor values for S stocks
    returns: np.ndarray,  # (S,)  ret_1d for S stocks
    top_k: int = 10,
) -> torch.Tensor:
    """为每个权重向量计算 Spearman Rank IC（得分 vs 前向收益）。

    Rank IC 比 top-K 平均收益的信噪比高得多，训练更快收敛。
    """
    factor_t = torch.from_numpy(factor_matrix).float()  # (S, N)
    ret_t = torch.from_numpy(returns).float()  # (S,)

    scores = factor_t @ weights.T  # (S, M)

    rewards = []
    for m in range(weights.shape[0]):
        s = scores[:, m]  # (S,)
        # Spearman rank IC
        valid_mask = ~(torch.isnan(s) | torch.isnan(ret_t))
        s_v = s[valid_mask]
        r_v = ret_t[valid_mask]
        if len(s_v) < 10:
            rewards.append(torch.tensor(0.0))
            continue
        # Rank the scores and returns
        s_rank = s_v.argsort().argsort().float()
        r_rank = r_v.argsort().argsort().float()
        # Pearson on ranks = Spearman
        s_c = s_rank - s_rank.mean()
        r_c = r_rank - r_rank.mean()
        ic = (s_c * r_c).sum() / ((s_c.norm() * r_c.norm()) + 1e-10)
        rewards.append(ic)
    return torch.stack(rewards)  # (M,)


def _group_advantage(rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """组内标准化: (r - mean) / (std + eps)."""
    mean_r = rewards.mean()
    std_r = rewards.std()
    if std_r < eps:
        return torch.zeros_like(rewards)
    return (rewards - mean_r) / (std_r + eps)


# ── 训练主循环 ────────────────────────────────────────────

def train_grpo_epoch(
    policy: DirichletPolicyNet,
    daily_data: dict,
    factor_cols: list[str],
    optimizer: torch.optim.Optimizer,
    M: int = 8,
    top_k: int = 10,
    epsilon: float = 0.2,
    entropy_coef: float = 0.01,
    device: str = "cpu",
) -> float:
    """单 epoch GRPO 训练，返回平均 loss。

    Args:
        policy: DirichletPolicyNet
        daily_data: {date_str: {"state": np.array, "factor_matrix": np.array, "returns": np.array}}
        factor_cols: factor column names (used for consistency check)
        optimizer: torch optimizer
        M: 每 state 采样数
        top_k: 选股数量
        epsilon: clip 范围
        entropy_coef: 熵正则系数
        device: "cpu" | "cuda"
    """
    policy.train()
    dates = list(daily_data.keys())
    total_loss = 0.0
    n_valid = 0

    for date_str in dates:
        data = daily_data[date_str]
        state = data["state"]
        factor_matrix = data["factor_matrix"]  # (S, N)
        returns = data["returns"]  # (S,)

        if len(factor_matrix) < 10:
            continue

        state_t = torch.from_numpy(state).float().unsqueeze(0).to(device)  # (1, D)

        with torch.no_grad():
            alpha_old = policy(state_t).squeeze(0)  # (N,)
            dist_old = Dirichlet(alpha_old)
            actions = dist_old.sample((M,))  # (M, N)
            old_log_probs = dist_old.log_prob(actions)  # (M,)
            rewards = _compute_rewards(actions, factor_matrix, returns, top_k=top_k)
            advantages = _group_advantage(rewards)

        # Compute new log probs + loss
        alpha_new = policy(state_t).squeeze(0)  # (N,)
        dist_new = Dirichlet(alpha_new)
        new_log_probs = dist_new.log_prob(actions)  # (M,)

        ratio = torch.exp(new_log_probs - old_log_probs)  # (M,)

        # PPO-style clipped loss
        clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
        policy_loss = -torch.min(ratio * advantages, clipped * advantages).mean()

        # Entropy regularization: 惩罚熵偏离目标区间的两端
        entropy = dist_new.entropy().mean()
        n_f = policy.n_factors
        target_entropy = np.log(n_f) * 0.2  # 目标: 适度集中 (最大熵的 20%)
        entropy_penalty = entropy_coef * (entropy - target_entropy).abs()

        loss = policy_loss + entropy_penalty

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_valid += 1

    return total_loss / max(n_valid, 1)


def train_grpo_weights(
    policy: DirichletPolicyNet,
    daily_data: dict,
    factor_cols: list[str],
    epochs: int = 30,
    M: int = 8,
    top_k: int = 10,
    lr: float = 3e-3,
    epsilon: float = 0.2,
    entropy_coef: float = 0.005,
    device: str = "cpu",
) -> DirichletPolicyNet:
    """完整 GRPO 训练流程。

    Args:
        policy: DirichletPolicyNet (预初始化或前窗口的参数)
        daily_data: {date_str: {"state": ..., "factor_matrix": ..., "returns": ...}}
        factor_cols: 因子列名
        epochs: 训练 epoch 数
        M: 每 state 采样数
        top_k: 选股数 (top-K 等权)
        lr: 学习率
        epsilon: clip 范围
        entropy_coef: 熵正则
        device: 设备

    Returns:
        训练好的 policy (in-place modified)
    """
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    n_dates = len(daily_data)

    logger.info(f"GRPO 训练: {n_dates}天, {epochs}epochs, M={M}, lr={lr}, ε={epsilon}")

    for epoch in range(epochs):
        avg_loss = train_grpo_epoch(
            policy, daily_data, factor_cols, optimizer,
            M=M, top_k=top_k, epsilon=epsilon,
            entropy_coef=entropy_coef, device=device,
        )

        if (epoch + 1) % 10 == 0:
            # 检查策略输出的均值（浓度参数的平均值，>1 表示探索充分）
            with torch.no_grad():
                # 用一个虚拟 state 检查
                mean_conc = 0.0
                sample_size = min(5, len(daily_data))
                for i, date_str in enumerate(list(daily_data.keys())[:sample_size]):
                    state = daily_data[date_str]["state"]
                    st = torch.from_numpy(state).float().unsqueeze(0).to(device)
                    alpha = policy(st).squeeze(0)  # (N,)
                    mean_conc += alpha.mean().item()
                mean_conc /= max(sample_size, 1)
            logger.debug(f"  epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, "
                         f"mean_conc={mean_conc:.1f}")

    return policy


# ── IC 加权基线 Predictor ─────────────────────────────────

class ICWeightedPredictor:
    """直接用因子滚动 IC 做权重（零训练基线）。

    predict() 时从 ic_map_by_date 查找当日 IC 值，
    正值 IC → 正向权重，负值 IC → 零权重，归一化。
    """

    def __init__(self, factor_names: list[str], builder, smooth: int = 20):
        self.factor_names = list(factor_names)
        self.builder = builder
        self.smooth = smooth

    def predict(self, factor_df, market_state=None, deterministic=True):
        import pandas as pd
        if factor_df.empty:
            return pd.DataFrame(columns=["code", "score", "rank"])

        cols = [f for f in self.factor_names if f in factor_df.columns]
        col_idx = [i for i, f in enumerate(self.factor_names) if f in cols]
        X = factor_df[cols].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)

        # 从 market_state (包含 IC 特征) 提取 IC 作为权重
        if market_state is not None:
            n_base = len(market_state) - len(self.factor_names)
            ic_vals = market_state[n_base:]  # IC 在 state 末尾
            weights = np.maximum(ic_vals, 0)  # 只用正 IC
        else:
            weights = np.ones(len(self.factor_names))

        w = np.array([weights[i] for i in col_idx], dtype=np.float32)
        w = w / max(w.sum(), 1e-10)
        scores = X @ w

        result = pd.DataFrame({"code": factor_df["code"].values, "score": scores})
        result = result.sort_values("score", ascending=False).reset_index(drop=True)
        result["rank"] = range(1, len(result) + 1)
        return result


# ── GRPO Predictor 封装 ───────────────────────────────────

class GRPOPredictor:
    """GRPO 训练后的因子权重预测器。

    实现与 RLDynamicPredictor 相同的 predict() 接口，
    供 run_rl_backtest.py 直接替换使用。
    """

    def __init__(self, policy: DirichletPolicyNet, factor_names: list[str],
                 builder, device: str = "cpu"):
        self.policy = policy
        self.factor_names = list(factor_names)
        self.builder = builder
        self.device = device

    def predict(self, factor_df, market_state=None, deterministic=True):
        """用策略均值（确定性模式）或采样（随机模式）生成因子权重 → 打分。

        deterministic=True: 使用 Dirichlet 的期望值 (alpha / sum(alpha))
        deterministic=False: 从 Dirichlet 采样
        """
        import pandas as pd

        if factor_df.empty:
            return pd.DataFrame(columns=["code", "score", "rank"])

        if market_state is not None:
            state_t = torch.from_numpy(market_state).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                alpha = self.policy(state_t).squeeze(0)  # (N,)
                if deterministic:
                    # Dirichlet 期望: alpha_i / sum(alpha)
                    weights = (alpha / alpha.sum()).cpu().numpy()
                else:
                    dist = Dirichlet(alpha)
                    weights = dist.sample().cpu().numpy()
        else:
            weights = np.ones(len(self.factor_names)) / len(self.factor_names)

        cols = [f for f in self.factor_names if f in factor_df.columns]
        col_idx = [i for i, f in enumerate(self.factor_names) if f in cols]
        X = factor_df[cols].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
        w = np.array([weights[i] for i in col_idx], dtype=np.float32)
        w = w / max(w.sum(), 1e-10)
        scores = X @ w

        result = pd.DataFrame({"code": factor_df["code"].values, "score": scores})
        result = result.sort_values("score", ascending=False).reset_index(drop=True)
        result["rank"] = range(1, len(result) + 1)
        return result
