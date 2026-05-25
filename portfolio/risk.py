"""风控规则。"""
import numpy as np
import pandas as pd


def apply_stop_loss(
    positions: pd.DataFrame,
    prices: dict[str, float],
    cost_basis: dict[str, float],
    stop_pct: float = 0.08,
) -> pd.DataFrame:
    """个股止损：-8% 或 -1.5x ATR 触发卖出。

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
