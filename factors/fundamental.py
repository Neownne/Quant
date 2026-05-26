"""基本面质量因子 —— 九项排雷系统适配。

将聚宽策略中的九项排雷逻辑改写为连续因子，供 ML 模型学习质量溢价。
所有因子遵循标准签名: (df: pd.DataFrame) -> pd.Series，df 为单只股票按 trade_date 排序的 DataFrame。

依赖数据（通过 FactorEngine 的 extra_data 参数传入）:
  财务列 (stock_financial, merge_asof by report_date):
    net_profit, cash_flow, roe, bps, net_margin, revenue, eps
    total_assets, total_liability, goodwill, holder_equity,
    operating_cash_flow, adjusted_profit
  日频列 (stock_pledge, left-merge by trade_date):
    pledge_ratio

对照聚宽九项排雷:

  聚宽检查项               数据需求              本项目适配
  ─────────────────────────────────────────────────────────────
  1. 年报迟发              pub_date              ❌ 无披露日期数据
  2. 业绩预告不良          STK_FIN_FORCAST       ❌ 无业绩预告数据
  3. 审计意见异常          STK_AUDIT_OPINION     ❌ 无审计意见数据
  4. 主业存疑 (扣非<0)     adjusted_profit       ✅ fin_audit_score (新增)
  5. 现金流异常            net_profit + cash_flow ✅ fin_cashflow_gap (增强)
  6. 商誉过高 (>30%)       goodwill / equity     ✅ fin_goodwill_ratio (新增)
  7. 资金链紧绷 (>70%)     total_liability/assets ✅ fin_debt_ratio (新增)
  8. 大股东高质押 (>80%)   pledge_ratio          ✅ fin_pledge_risk (新增)
  9. 监管立案              STK_INVESTIGATION     ❌ 无监管数据

最终输出 fin_audit_score 为 8 项可用检查的综合扣分（取负，越低越差）。
"""

import numpy as np
import pandas as pd


def _ffill(df: pd.DataFrame, col: str) -> pd.Series:
    """前向填充财务列，处理季度报告的稀疏性。"""
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return df[col].replace(0, np.nan).ffill()


# ══════════════════════════════════════════════════════════════════
# 单项质量因子
# ══════════════════════════════════════════════════════════════════

def fin_cashflow_gap(df: pd.DataFrame) -> pd.Series:
    """现金流缺口因子（对应排雷 #5 现金流异常）。

    优先使用 operating_cash_flow (总额) vs net_profit (总额)，
    不可用时回退到 cash_flow (每股) vs eps (每股)。

    净利润 > 0 但经营现金流 < 0 → -1（盈利质量最差）。
    经营现金流充裕 → +0.5。
    其余线性缩放至 [-1, +0.5]。
    """
    has_total = ("operating_cash_flow" in df.columns and
                 df["operating_cash_flow"].notna().any())

    if has_total:
        profit = _ffill(df, "net_profit")
        cf = _ffill(df, "operating_cash_flow")
    else:
        profit = _ffill(df, "eps")
        cf = _ffill(df, "cash_flow")

    gap = profit - cf
    profit_abs = profit.abs().replace(0, np.nan)
    norm_gap = gap / profit_abs

    result = -np.clip(norm_gap.abs() / 2.0, 0, 1)
    result[(profit > 0) & (cf < 0)] = -1.0
    result[(profit > 0) & (cf > profit)] = 0.5
    result[profit <= 0] = result[profit <= 0].clip(lower=-1.0)
    return result


def fin_roe_quality(df: pd.DataFrame) -> pd.Series:
    """ROE 质量因子。

    ROE 绝对水平高 + 趋势改善 → 正值；ROE 为负或恶化 → 负值。
    """
    roe = _ffill(df, "roe").clip(-2, 2)
    roe_chg = roe.diff(60)  # ~一个季度的交易日
    score = roe / 0.20 + roe_chg / 0.10
    return score.clip(-1, 1)


def fin_profit_cv(df: pd.DataFrame) -> pd.Series:
    """盈利稳定性因子（对应排雷 #4 主业存疑的代理）。

    用近 1 年净利润变异系数度量。高 CV → 盈利不可靠 → 负值。
    min_periods=3 因为财务数据按季度 ffill，252 个交易日仅含 ~4 个季度值。
    """
    profit = _ffill(df, "net_profit")
    roll_std = profit.rolling(252, min_periods=3).std()
    roll_mean = profit.rolling(252, min_periods=3).mean().abs().replace(0, np.nan)
    cv = roll_std / roll_mean
    return -np.clip(cv / 2.0, 0, 1)


def fin_net_margin(df: pd.DataFrame) -> pd.Series:
    """净利润率因子。

    净利率高 → 盈利能力强 → 正值。取 net_margin / 0.30 缩放至 [-1, 1]。
    """
    nm = _ffill(df, "net_margin")
    return (nm / 0.30).clip(-1, 1)


def fin_bps_growth(df: pd.DataFrame) -> pd.Series:
    """每股净资产增速因子。

    bps 持续增长 → 正值（价值积累）；下降 → 负值。
    """
    bps = _ffill(df, "bps")
    growth = bps.pct_change(60)
    return (growth / 0.50).clip(-1, 1)


def fin_revenue_stability(df: pd.DataFrame) -> pd.Series:
    """营收稳定性因子。

    营收持续增长且波动低 → 正值；营收萎缩或大起大落 → 负值。
    """
    rev = _ffill(df, "revenue")
    growth = rev.pct_change(60)
    growth_std = growth.rolling(252, min_periods=3).std()
    growth_mean = growth.rolling(252, min_periods=3).mean()

    trend = growth_mean.clip(-0.3, 0.3) / 0.30
    stability = -np.clip(growth_std / 0.30, 0, 1)
    score = (trend + stability) / 2
    return score.clip(-1, 1)


def fin_eps_growth(df: pd.DataFrame) -> pd.Series:
    """EPS 增长因子。

    EPS 季度环比增速。增长 → 正值，下降 → 负值。
    """
    eps = _ffill(df, "eps")
    growth = eps.pct_change(60)
    return (growth / 0.50).clip(-1, 1)


def fin_debt_ratio(df: pd.DataFrame) -> pd.Series:
    """资产负债率因子（对应排雷 #7 资金链紧绷）。

    total_liability / total_assets > 70% → 高风险。
    线性缩放：0% 负债 → +1，70% 负债 → 0，>70% → 负值。
    数据缺失时返回全 NaN。
    """
    if "total_assets" not in df.columns or "total_liability" not in df.columns:
        return pd.Series(np.nan, index=df.index)

    assets = _ffill(df, "total_assets")
    liability = _ffill(df, "total_liability")
    ratio = liability / assets.replace(0, np.nan)
    return (1.0 - ratio / 0.70).clip(-1, 1)


def fin_goodwill_ratio(df: pd.DataFrame) -> pd.Series:
    """商誉风险因子（对应排雷 #6 商誉过高）。

    goodwill / holder_equity > 30% → 高减值风险。
    线性缩放：0% → 0（无风险），30%+ → -1（高危）。
    数据缺失时返回全 NaN。
    """
    if "goodwill" not in df.columns or "holder_equity" not in df.columns:
        return pd.Series(np.nan, index=df.index)

    goodwill = _ffill(df, "goodwill")
    equity = _ffill(df, "holder_equity")
    ratio = goodwill / equity.abs().replace(0, np.nan)
    return -np.clip(ratio / 0.30, 0, 1)


def fin_pledge_risk(df: pd.DataFrame) -> pd.Series:
    """股权质押风险因子（对应排雷 #8 大股东高质押）。

    pledge_ratio > 80% → 爆仓/控制权转移风险。
    线性缩放：0% → 0，80%+ → -1。
    数据缺失时返回全 NaN。
    """
    if "pledge_ratio" not in df.columns:
        return pd.Series(np.nan, index=df.index)

    ratio = df["pledge_ratio"].replace(0, np.nan)
    return -np.clip(ratio / 0.80, 0, 1)


# ══════════════════════════════════════════════════════════════════
# 综合排雷评分
# ══════════════════════════════════════════════════════════════════

def fin_audit_score(df: pd.DataFrame) -> pd.Series:
    """排雷综合评分。

    检查 8 项可用信号，每触发一项扣 1 分，取负使"全通过 = 0、全踩雷 = -8"。
    财务数据缺失时该项不扣分（保守处理）。

    检查项:
    1. 主业存疑: adjusted_profit < 0（扣非净利润为负）
    2. 现金流异常: 净利润 > 0 但经营现金流 < 0
    3. 净利润为负
    4. ROE 为负
    5. 盈利高波动: 净利润 CV > 1
    6. 净利率下滑: 当期净利率 < 前一期
    7. 资金链紧绷: 资产负债率 > 70%
    8. 商誉过高: 商誉 / 股东权益 > 30%
    """
    score = pd.Series(0.0, index=df.index)

    fin_cols = ["net_profit", "cash_flow", "roe", "net_margin"]
    if not any(c in df.columns for c in fin_cols):
        return score

    # 1. 主业存疑: 扣非净利润 < 0
    if "adjusted_profit" in df.columns and df["adjusted_profit"].notna().any():
        adj = _ffill(df, "adjusted_profit")
        score -= (adj < 0).astype(float)

    # 2. 现金流异常: 有利润无现金
    profit = _ffill(df, "net_profit")
    if "operating_cash_flow" in df.columns and df["operating_cash_flow"].notna().any():
        cf = _ffill(df, "operating_cash_flow")
    else:
        cf = _ffill(df, "cash_flow")
    score -= ((profit > 0) & (cf < 0)).astype(float)

    # 3. 净利润为负
    score -= (profit < 0).astype(float)

    # 4. ROE 为负
    roe = _ffill(df, "roe")
    score -= (roe < 0).astype(float)

    # 5. 盈利高波动: CV > 1
    profit_std = profit.rolling(252, min_periods=3).std()
    profit_mean = profit.rolling(252, min_periods=3).mean().abs().replace(0, np.nan)
    cv = profit_std / profit_mean
    score -= (cv > 1.0).astype(float)

    # 6. 净利率下滑
    nm = _ffill(df, "net_margin")
    nm_prev = nm.shift(60)
    score -= ((nm < nm_prev) & nm_prev.notna()).astype(float)

    # 7. 资金链紧绷: 资产负债率 > 70%
    if ("total_assets" in df.columns and df["total_assets"].notna().any() and
            "total_liability" in df.columns and df["total_liability"].notna().any()):
        assets = _ffill(df, "total_assets")
        liability = _ffill(df, "total_liability")
        debt_ratio = liability / assets.replace(0, np.nan)
        score -= (debt_ratio > 0.70).astype(float)

    # 8. 商誉过高: 商誉 / 股东权益 > 30%
    if ("goodwill" in df.columns and df["goodwill"].notna().any() and
            "holder_equity" in df.columns and df["holder_equity"].notna().any()):
        gw = _ffill(df, "goodwill")
        equity = _ffill(df, "holder_equity")
        gw_ratio = gw / equity.abs().replace(0, np.nan)
        score -= (gw_ratio > 0.30).astype(float)

    return score


FUNDAMENTAL_FACTORS: dict = {
    "fin_cashflow_gap": fin_cashflow_gap,
    "fin_roe_quality": fin_roe_quality,
    "fin_profit_cv": fin_profit_cv,
    "fin_net_margin": fin_net_margin,
    "fin_bps_growth": fin_bps_growth,
    "fin_revenue_stability": fin_revenue_stability,
    "fin_eps_growth": fin_eps_growth,
    "fin_debt_ratio": fin_debt_ratio,
    "fin_goodwill_ratio": fin_goodwill_ratio,
    "fin_pledge_risk": fin_pledge_risk,
    "fin_audit_score": fin_audit_score,
}
