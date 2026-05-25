"""风控规则。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_atr(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 20,
) -> float:
    """计算 ATR(20) 最新值。

    参数
    ----
    high, low, close : 价格序列（按时间升序）
    period : ATR 周期

    返回
    ----
    float : 最新 ATR 值
    """
    if len(close) < 2:
        return 0.0
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(
        high - low,
        np.maximum(
            np.abs(high - prev_close),
            np.abs(low - prev_close),
        ),
    )
    if len(tr) < period:
        return float(tr.mean())
    atr = pd.Series(tr).ewm(span=period, adjust=False).mean()
    return float(atr.iloc[-1])


def apply_stop_loss(
    positions: pd.DataFrame,
    prices: dict[str, float],
    cost_basis: dict[str, float],
    stop_pct: float = 0.08,
) -> pd.DataFrame:
    """个股止损：-8% 触发卖出。

    返回需要平仓的 code 列表。
    """
    to_sell = []
    for _, row in positions.iterrows():
        code = row["code"]
        if code in prices and code in cost_basis and cost_basis[code] > 0:
            loss = (prices[code] - cost_basis[code]) / cost_basis[code]
            if loss <= -stop_pct:
                to_sell.append(code)
    return pd.DataFrame({"code": to_sell}) if to_sell else pd.DataFrame(columns=["code"])


def apply_atr_stop_loss(
    positions: pd.DataFrame,
    prices: dict[str, float],
    cost_basis: dict[str, float],
    atr_values: dict[str, float],
    stop_pct: float = 0.08,
    atr_multiplier: float = 1.5,
) -> pd.DataFrame:
    """个股止损：-8% 或 -1.5x ATR 触发卖出。

    满足任一条件即平仓。

    返回 DataFrame: [code, reason]
    """
    to_sell = []
    for _, row in positions.iterrows():
        code = row["code"]
        if code not in prices or code not in cost_basis or cost_basis[code] <= 0:
            continue
        loss = (prices[code] - cost_basis[code]) / cost_basis[code]
        reasons = []
        if loss <= -stop_pct:
            reasons.append("stop_pct")
        atr = atr_values.get(code, 0)
        if atr > 0 and cost_basis[code] - prices[code] >= atr_multiplier * atr:
            reasons.append("atr")
        if reasons:
            to_sell.append({"code": code, "reason": "+".join(reasons)})
    return pd.DataFrame(to_sell) if to_sell else pd.DataFrame(columns=["code", "reason"])


def portfolio_stop_reduce(
    positions: dict[str, dict],
    current_value: float,
    peak_value: float,
    threshold: float = 0.20,
    reduce_to: float = 0.50,
) -> tuple[bool, dict[str, dict] | None]:
    """组合回撤止损：回撤触及阈值时等比例减仓。

    返回 (should_reduce, reduced_positions)
    """
    if peak_value <= 0:
        return False, None
    drawdown = (peak_value - current_value) / peak_value
    if drawdown >= threshold:
        reduced = {}
        for code, pos in positions.items():
            reduced[code] = {
                **pos,
                "shares": int(pos["shares"] * reduce_to / 100) * 100,
            }
        return True, reduced
    return False, None


def check_drawdown_limit(current_value: float, peak_value: float, limit: float = 0.25) -> bool:
    """检查是否触发组合回撤上限。

    返回 True 表示应清仓/暂停。
    """
    if peak_value <= 0:
        return False
    drawdown = (peak_value - current_value) / peak_value
    return drawdown >= limit


def position_sizing(cash: float, risk_pct: float = 0.02, atr: float = 0) -> float:
    """基于风险的仓位计算。

    单个头寸风险 = cash × risk_pct
    止损距离 = 1.5 × ATR
    仓位 = 风险金额 / 止损距离
    """
    risk_amount = cash * risk_pct
    stop_distance = max(atr * 1.5, 0.01)
    return risk_amount / stop_distance
