"""选股：模型打分 top-N + TopK+NDrop 增量调仓 + ST/停牌/涨跌停/次新过滤。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import date


def select_topk_ndrop(
    scores: pd.Series,
    current_holdings: set[str] | None = None,
    K: int = 20,
    N: int = 2,
    pnl_map: dict[str, float] | None = None,
    adaptive_n: bool = False,
    score_spread_threshold: float = 0.15,
    score_rank_threshold: float = 0.3,
    loss_tolerance: float = -0.08,
) -> tuple[set[str], set[str], set[str]]:
    """TopK + NDrop 增量调仓：盈亏感知替换（v2）。

    首次建仓时买入得分最高的 K 只。

    后续调仓逻辑：
    1. 自适应 N（可选）：基于分数 90-10 分位差动态调整替换数
    2. 保留持仓中得分最高的 K-N 只
    3. 底部 N 只做增强盈亏决策（五层判断，替代二元 pnl）
    4. 已不在候选池的强制清掉
    5. 用未持仓中得分最高的补足至 K 只

    参数
    ----
    scores : Series, index=code, values=预测得分（越高越好），已降序排列
    current_holdings : 当前持仓股票代码集合
    K : 目标持仓数
    N : 最大替换数（adaptive_n=True 时作为上限，实际由分数离散度决定）
    pnl_map : {code: pnl_pct}，None 则退化为纯分数排序
    adaptive_n : 启用自适应 N
    score_spread_threshold : 自适应 N 的分数离散度基准阈值
    score_rank_threshold : PnL 决策的分数排名百分位阈值 [0,1]，低于此强制卖出
    loss_tolerance : PnL 决策的亏损容忍线（负值），跌破即止损

    返回
    ----
    (new_holdings, to_buy, to_sell) : 新持仓集合、买入集合、卖出集合
    """
    if current_holdings is None:
        current_holdings = set()

    if not current_holdings:
        new = set(scores.head(K).index)
        return new, new, set()

    # ── v2: 自适应 N ──
    if adaptive_n and len(scores) >= 10:
        spread = scores.quantile(0.9) - scores.quantile(0.1)
        if spread > score_spread_threshold * 2:
            N = 4
        elif spread > score_spread_threshold:
            N = 3
        elif spread > score_spread_threshold * 0.5:
            N = 2
        else:
            N = 1
        N = min(N, len(current_holdings))

    # 已不在候选池中的持仓 → 必须清掉
    dropped = current_holdings - set(scores.index)

    # 仍在候选池中的持仓
    alive = current_holdings & set(scores.index)

    # ── v2: 分数百分位排名（供 PnL 决策使用）──
    score_rank = scores.rank(pct=True, ascending=True)

    # 持仓中得分最高的 K-N 只 → 无条件保留
    hold_scores = scores[scores.index.isin(alive)].head(K - N)
    keep = set(hold_scores.index)

    # ── v2: 底部 N 只，增强盈亏决策 ──
    bottom_n = [c for c in scores[scores.index.isin(alive)].index if c not in keep]
    to_sell_from_bottom = set()
    if pnl_map is not None:
        for code in bottom_n:
            rank = score_rank.get(code, 0.5)
            pnl = pnl_map.get(code, 0.0)
            if rank < score_rank_threshold:
                # 分数排名太低 → 不持有垃圾，不管盈亏都卖
                to_sell_from_bottom.add(code)
            elif pnl > loss_tolerance and rank > 0.5:
                # 轻微亏损或盈利 + 分数排名在中上 → 继续持有
                keep.add(code)
            elif pnl > 0 and rank < score_rank_threshold * 1.5:
                # 盈利但分数排名在衰退区 → 止盈
                to_sell_from_bottom.add(code)
            elif pnl < loss_tolerance:
                # 亏损超出容忍线 → 止损
                to_sell_from_bottom.add(code)
            else:
                to_sell_from_bottom.add(code)
    else:
        to_sell_from_bottom = set(bottom_n)  # 无 PnL 数据 → 纯分数

    to_sell = to_sell_from_bottom | dropped

    # 需要买入 = 从候选池中补足至 K 只
    slots = K - len(keep)
    candidates = scores[~scores.index.isin(alive | keep)]
    to_buy = set(candidates.head(slots).index)

    new_holdings = keep | to_buy
    return new_holdings, to_buy, to_sell


def select_top_n(scores: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """从排序结果中选前 N 只。"""
    return scores.sort_values("rank").head(n).reset_index(drop=True)


def filter_suspended(
    stocks: pd.DataFrame,
    ohlcv_lookup: dict[str, pd.DataFrame],
    ref_date: pd.Timestamp,
    lookback_days: int = 5,
) -> pd.DataFrame:
    """剔除停牌股票：近 N 日成交量全为零或收盘价完全不变。"""
    if stocks.empty:
        return stocks
    result = stocks.copy()
    valid_mask = pd.Series(True, index=result.index)
    for i, row in result.iterrows():
        code = row["code"]
        hist = ohlcv_lookup.get(code)
        if hist is None or hist.empty:
            continue
        hist = hist[hist["trade_date"] <= ref_date].tail(lookback_days)
        if len(hist) < lookback_days:
            continue
        if (hist["volume"] == 0).all() or hist["close"].nunique() == 1:
            valid_mask.iloc[i] = False
    return result[valid_mask].reset_index(drop=True)


def filter_limit_up_down(
    stocks: pd.DataFrame,
    prev_close_map: dict[str, float],
    limit_pct: float = 0.10,
) -> pd.DataFrame:
    """剔除涨停（无法买入）和跌停（无法卖出）股票。"""
    if stocks.empty:
        return stocks
    result = stocks.copy()
    valid_mask = pd.Series(True, index=result.index)
    for i, row in result.iterrows():
        code = row["code"]
        prev = prev_close_map.get(code)
        if prev is None or prev <= 0:
            continue
        current = row.get("close", row.get("price"))
        if current is None or pd.isna(current) or current <= 0:
            continue
        limit_up = prev * (1 + limit_pct) * 0.999
        limit_down = prev * (1 - limit_pct) * 1.001
        if current >= limit_up or current <= limit_down:
            valid_mask.iloc[i] = False
    return result[valid_mask].reset_index(drop=True)


def filter_stocks(
    stocks: pd.DataFrame,
    ref_date: pd.Timestamp | None = None,
    exclude_st: bool = True,
    min_list_days: int = 60,
    ohlcv_lookup: dict[str, pd.DataFrame] | None = None,
    prev_close_map: dict[str, float] | None = None,
    filter_suspended_flag: bool = False,
    filter_limit_flag: bool = False,
) -> pd.DataFrame:
    """过滤不可交易的股票。

    参数
    ----
    stocks : 至少含 code, name 列
    ref_date : 参考日期（默认今天）
    exclude_st : 排除 ST
    min_list_days : 最小上市天数
    ohlcv_lookup : {code: OHLCV DataFrame}，停牌过滤需要
    prev_close_map : {code: 前日收盘价}，涨跌停过滤需要
    filter_suspended_flag : 启用停牌过滤
    filter_limit_flag : 启用涨跌停过滤
    """
    result = stocks.copy()
    ref = ref_date or pd.Timestamp(date.today())

    if exclude_st and "name" in result.columns:
        result = result[~result["name"].str.contains("ST", na=False)]

    if "list_date" in result.columns:
        result["days_listed"] = (ref - pd.to_datetime(result["list_date"])).dt.days
        result = result[result["days_listed"] >= min_list_days]
        result = result.drop(columns=["days_listed"])

    if filter_suspended_flag and ohlcv_lookup:
        result = filter_suspended(result, ohlcv_lookup, ref)

    if filter_limit_flag and prev_close_map:
        result = filter_limit_up_down(result, prev_close_map)

    return result.reset_index(drop=True)
