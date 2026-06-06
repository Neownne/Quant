"""RL 元控制器：根据市场状态 + 概念板块热度动态调整 ML 得分加成。

用法:
    controller = ConceptBoostController(state_dim, hidden=32)
    controller = train_boost_grpo(controller, daily_states, ml_scores, daily_returns, ...)
    boost = controller.predict(state)  # (2,) = [global_boost, rotation_sensitivity]
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from loguru import logger


# ── 策略网络 ──────────────────────────────────────────────

class ConceptBoostController(nn.Module):
    """市场状态 → 2 维 boost 参数: [boost_level, rotation_sensitivity]。

    boost_level:      概念板块热度加成的基准幅度 (0~1)
    rotation_sensitivity: 板块轮动速度的敏感度 (0.5~5)
                         低值 → 忽略轮动, 高值 → 轮动快时迅速减仓
    """

    def __init__(self, state_dim: int, hidden: int = 32):
        super().__init__()
        self.state_dim = state_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
        )
        # 输出 2 个均值 + 2 个 log_std
        self.mean_head = nn.Linear(hidden // 2, 2)
        self.log_std = nn.Parameter(torch.tensor([-1.0, -0.5]))  # std≈0.37, 0.61

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 (mean, std) for 2-dim Gaussian action."""
        x = self.net(state)
        raw = self.mean_head(x)  # (batch, 2)
        # Non-inplace transform
        mean = torch.cat([
            torch.sigmoid(raw[:, 0:1]),          # boost_level ∈ (0, 1)
            F.softplus(raw[:, 1:2]) + 0.5,       # rotation_sensitivity > 0.5
        ], dim=-1)
        std = self.log_std.exp().clamp(0.01, 2.0)
        return mean, std

    def sample(self, state: torch.Tensor, M: int = 1
               ) -> tuple[torch.Tensor, torch.Tensor]:
        """采样 M 个 action，返回 (actions: (M,2), log_probs: (M,))。"""
        mean, std = self.forward(state)  # (1, 2), (2,)
        dist = Normal(mean, std)
        actions = dist.sample((M,)).squeeze(1)  # (M, 2)
        log_probs = dist.log_prob(actions).sum(dim=-1)  # (M,)
        # Clamp actions (non-inplace to preserve autograd)
        actions = torch.stack([
            actions[:, 0].clamp(0.0, 1.0),
            actions[:, 1].clamp(0.5, 10.0),
        ], dim=-1)
        return actions, log_probs

    def predict(self, state: np.ndarray) -> np.ndarray:
        """确定性预测: 返回 [boost_level, rotation_sensitivity]."""
        st = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            mean, _ = self.forward(st)
        return mean.squeeze(0).cpu().numpy()


# ── IC 奖励计算 ───────────────────────────────────────────

def compute_boost_ic(
    ml_scores: np.ndarray,      # (S,)   ML 原始得分
    forward_returns: np.ndarray, # (S,)   ret_1d
    stock_ret_5d: np.ndarray,    # (S,)   股票近5日收益（热度的代理）
    boost_params: tuple[float, float],
) -> float:
    """计算概念加成后的得分 Rank IC。

    Args:
        ml_scores: ML 预测得分
        forward_returns: 次日实际收益
        stock_ret_5d: 股票近 5 日收益（z-score 后）
        boost_params: (boost_level, rotation_sensitivity)
    """
    boost_level, rot_sens = boost_params

    # 概念整体热度 = 股票近期收益的正向幅度
    hot_strength = float(np.mean(np.maximum(stock_ret_5d, 0)))

    # 板块轮动幅度 = 股票收益的截面标准差（代用 cb_rotation）
    effective_rotation = float(np.std(stock_ret_5d))

    # 加成公式
    rotation_penalty = max(0, 1.0 - effective_rotation * rot_sens)
    global_boost = hot_strength * rotation_penalty * boost_level

    if global_boost > 0.001 and np.std(stock_ret_5d) > 1e-8:
        stock_z = (stock_ret_5d - stock_ret_5d.mean()) / (stock_ret_5d.std() + 1e-10)
        adjusted = ml_scores * (1.0 + global_boost * np.clip(stock_z, -2, 3))
    else:
        adjusted = ml_scores

    # Spearman IC
    valid = ~(np.isnan(adjusted) | np.isnan(forward_returns))
    s, r = adjusted[valid], forward_returns[valid]
    if len(s) < 10:
        return 0.0
    s_rank = s.argsort().argsort().astype(float)
    r_rank = r.argsort().argsort().astype(float)
    s_c = s_rank - s_rank.mean()
    r_c = r_rank - r_rank.mean()
    ic = (s_c * r_c).sum() / ((np.linalg.norm(s_c) * np.linalg.norm(r_c)) + 1e-10)
    return float(ic) if not np.isnan(ic) else 0.0


# ── GRPO 训练 ────────────────────────────────────────────

def train_boost_grpo(
    controller: ConceptBoostController,
    daily_states: dict,  # {date_str: np.array(state)}
    ml_scores_map: dict,  # {date_str: np.array(scores_per_stock)}
    forward_rets: dict,   # {date_str: np.array(ret_1d_per_stock)}
    stock_rets_5d: dict,  # {date_str: np.array(5d_ret_per_stock)}
    epochs: int = 30,
    M: int = 8,
    lr: float = 5e-3,
    epsilon: float = 0.2,
    device: str = "cpu",
) -> ConceptBoostController:
    """GRPO 训练概念加成控制器。

    每个日期采样 M 个 boost 参数，IC 作为 reward，
    组内标准化 advantage。
    """
    optimizer = torch.optim.Adam(controller.parameters(), lr=lr)
    dates = list(daily_states.keys())

    logger.info(f"Boost GRPO: {len(dates)}天, {epochs}ep, M={M}, lr={lr}")

    for epoch in range(epochs):
        total_loss = 0.0
        n_valid = 0

        for date_str in dates:
            state = daily_states[date_str]
            ml_s = ml_scores_map[date_str]
            fwd_r = forward_rets[date_str]
            s5d = stock_rets_5d[date_str]

            if len(ml_s) < 10:
                continue

            st = torch.from_numpy(state).float().unsqueeze(0).to(device)

            with torch.no_grad():
                actions, old_lp = controller.sample(st, M=M)  # (M,2), (M,)
                rewards = []
                for m in range(M):
                    bp = (actions[m, 0].item(), actions[m, 1].item())
                    ic = compute_boost_ic(ml_s, fwd_r, s5d, bp)
                    rewards.append(ic)
                rewards_t = torch.tensor(rewards, device=device)
                mean_r, std_r = rewards_t.mean(), rewards_t.std()
                advantages = ((rewards_t - mean_r) / (std_r + 1e-8)) if std_r > 1e-8 else torch.zeros(M, device=device)

            # New policy
            _, new_lp = controller.sample(st, M=M)
            # Recompute log_probs for the same actions
            mean, std = controller.forward(st)
            dist = Normal(mean, std)
            new_lp = dist.log_prob(actions).sum(dim=-1)

            ratio = torch.exp(new_lp - old_lp)
            clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
            policy_loss = -torch.min(ratio * advantages, clipped * advantages).mean()

            loss = policy_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(controller.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_valid += 1

        if (epoch + 1) % 10 == 0:
            mean_ic = float(np.mean(rewards))
            logger.info(f"  epoch {epoch+1}/{epochs}: loss={total_loss/max(n_valid,1):.4f}, "
                        f"mean_IC={mean_ic:.4f}")

    return controller
