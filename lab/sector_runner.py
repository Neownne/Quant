"""行业轮动策略管线 —— 从成分股合成行业指数 → 动量排序 → 龙头选股。

数据来源：
  - stock_daily: 个股OHLCV
  - stock_industry: 申万一级行业分类 (industry_sw1)
  - stock_daily_extra: 市值（用于龙头筛选）

管线：
  1. 按 industry_sw1 分组，每日计算等权/市值加权行业收益
  2. 多周期动量打分（5/10/20/60日）
  3. 选出 top-N 行业
  4. 在选中行业内，按涨停因子+市值+动量选龙头
  5. 回测：持有行业龙头组合，定期轮动
"""
from __future__ import annotations

import os, sys, json, time
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy import text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.db import get_engine
from data.loader import load_daily_data, load_mcap_data
from config.settings import TradingConfig


# ── 默认参数 ──
TOP_SECTORS = 3           # 选前 N 个行业
TOP_STOCKS_PER_SECTOR = 3  # 每个行业选前 N 只股票
TOTAL_POSITIONS = 9        # 总持仓数 = TOP_SECTORS × TOP_STOCKS_PER_SECTOR
REBALANCE_FREQ = 10        # 每 N 个交易日调仓
LOOKBACKS = [5, 10, 20, 60]  # 动量回看窗口
MCAP_MIN = 30              # 市值下限（亿）
MCAP_MAX = 500             # 市值上限（亿）


@dataclass
class SectorResult:
    """行业轮动回测结果。"""
    run_name: str
    n_sectors: int = 0
    n_stocks: int = 0
    start_date: str = ""
    end_date: str = ""
    sharpe: float = 0.0
    total_return: float = 0.0
    max_drawdown: float = 0.0
    n_trades: int = 0
    sector_hits: dict = field(default_factory=dict)  # 各行业被选中次数
    elapsed: float = 0.0
    error: str = ""

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    @property
    def verdict(self) -> str:
        if self.error:
            return "error"
        if self.sharpe < 0 or self.max_drawdown > 0.30:
            return "reject"
        if self.sharpe > 0.8:
            return "promising"
        return "baseline"


def build_sector_returns(daily, industry_map, weight="equal"):
    """从成分股合成行业日收益。

    Args:
        daily: 含 code, trade_date, close 的日线
        industry_map: {code: industry_sw1}
        weight: "equal" 等权 或 "mcap" 市值加权

    Returns:
        pd.DataFrame: [trade_date, industry, ret]
    """
    df = daily[["code", "trade_date", "close"]].copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["industry"] = df["code"].map(industry_map)
    df = df.dropna(subset=["industry"])

    # 按股票排序计算日收益
    df = df.sort_values(["code", "trade_date"])
    df["ret"] = df.groupby("code")["close"].pct_change()

    # 按行业+日期聚合
    if weight == "equal":
        sector_ret = df.groupby(["industry", "trade_date"])["ret"].mean().reset_index()
    else:
        sector_ret = df.groupby(["industry", "trade_date"])["ret"].mean().reset_index()

    sector_ret = sector_ret.dropna(subset=["ret"])
    return sector_ret


def compute_sector_momentum(sector_ret, lookbacks=LOOKBACKS):
    """计算每个行业在每个交易日的多周期动量得分。

    Returns:
        pd.DataFrame: [trade_date, industry, score]
    """
    df = sector_ret.copy()
    df = df.sort_values(["industry", "trade_date"])

    # 各回看窗口的累计收益
    for lb in lookbacks:
        col = f"mom_{lb}d"
        df[col] = df.groupby("industry")["ret"].transform(
            lambda x: x.rolling(lb, min_periods=max(5, lb // 2)).sum()
        )

    # 复合得分：加权平均（短期权重更高）
    weights = {5: 0.35, 10: 0.30, 20: 0.25, 60: 0.10}
    score_cols = [f"mom_{lb}d" for lb in lookbacks if f"mom_{lb}d" in df.columns]
    df["score"] = 0.0
    for lb, w in weights.items():
        col = f"mom_{lb}d"
        if col in df.columns:
            df["score"] += df[col].fillna(0) * w

    return df.dropna(subset=["score"])


def select_top_stocks_in_sector(trade_date, sector_codes, daily_snapshot,
                                mcap_snapshot, top_n=3):
    """在给定行业的股票池中，按动量+市值综合打分选龙头。

    评分 = 0.5 × 近20日收益 + 0.3 × 近5日收益 + 0.2 × log(1/市值排名)
    """
    candidates = daily_snapshot[daily_snapshot["code"].isin(sector_codes)].copy()
    if candidates.empty:
        return []

    # 合并市值
    if mcap_snapshot is not None and not mcap_snapshot.empty:
        candidates = candidates.merge(
            mcap_snapshot[["code", "market_cap"]], on="code", how="left"
        )
        candidates["market_cap"] = candidates["market_cap"].fillna(100)

    # 动量得分
    candidates["mom_20"] = candidates.groupby("code")["close"].transform(
        lambda x: x.pct_change(20) if len(x) > 20 else 0
    )
    candidates["mom_5"] = candidates.groupby("code")["close"].transform(
        lambda x: x.pct_change(5) if len(x) > 5 else 0
    )

    # 市值得分：市值越小分数越高
    if "market_cap" in candidates.columns:
        cap_rank = candidates["market_cap"].rank(pct=True)
        candidates["cap_score"] = 1 - cap_rank.fillna(0.5)
    else:
        candidates["cap_score"] = 0.5

    # 综合得分
    candidates["stock_score"] = (
        0.5 * candidates["mom_20"].fillna(0) +
        0.3 * candidates["mom_5"].fillna(0) +
        0.2 * candidates["cap_score"].fillna(0.5)
    )

    top = candidates.nlargest(min(top_n, len(candidates)), "stock_score")
    return list(zip(top["code"], top["close"], top["stock_score"]))


def run_sector_rotation(daily, industry_map, extra, start="2020-01-01",
                        end="2026-06-14", top_sectors=TOP_SECTORS,
                        top_per_sector=TOP_STOCKS_PER_SECTOR,
                        rebalance_freq=REBALANCE_FREQ):
    """执行行业轮动回测。

    流程:
      调仓日 → 计算行业动量 → 选top行业 → 各行业选龙头 → 持有到下次调仓
    """
    t0 = time.time()
    result = SectorResult(run_name=f"sector_{start}_{end}",
                          start_date=start, end_date=end)

    try:
        daily = daily.copy()
        daily["trade_date"] = pd.to_datetime(daily["trade_date"])

        # 过滤日期
        mask = (daily["trade_date"] >= start) & (daily["trade_date"] <= end)
        daily = daily[mask]

        if extra is not None:
            extra = extra.copy()
            extra["trade_date"] = pd.to_datetime(extra["trade_date"])
            extra = extra[(extra["trade_date"] >= start) & (extra["trade_date"] <= end)]

        # 1. 合成行业收益
        logger.info("合成行业日收益...")
        sector_ret = build_sector_returns(daily, industry_map)
        result.n_sectors = sector_ret["industry"].nunique()
        logger.info(f"  行业数: {result.n_sectors}")

        # 2. 计算行业动量
        sector_mom = compute_sector_momentum(sector_ret)
        trade_dates = sorted(sector_mom["trade_date"].unique())

        # 3. 调仓日循环
        all_trade_dates = sorted(daily["trade_date"].unique())
        trade_date_set = set(all_trade_dates)

        positions = {}  # {code: (entry_price, shares)}
        equity_curve = []
        cash = 1_000_000
        total_positions = top_sectors * top_per_sector

        # 预分组
        daily_by_date = {d: g.set_index("code")
                        for d, g in daily.groupby("trade_date")}

        trade_count = 0
        for i, td in enumerate(all_trade_dates):
            td_df = daily_by_date.get(td, pd.DataFrame())
            if td_df.empty:
                continue
            px_map = td_df["close"].to_dict()

            # ── 非调仓日：只估值 ──
            if i % rebalance_freq != 0 and positions:
                pos_val = sum(shares * px_map.get(code, entry_px)
                             for code, (entry_px, shares) in positions.items())
                total = cash + pos_val
                equity_curve.append({"date": str(td)[:10], "value": round(total, 2)})
                continue

            # ── 调仓日：清仓 → 选行业 → 买龙头 ──
            # 卖出
            sell_proceeds = 0
            for code, (entry_px, shares) in list(positions.items()):
                px = px_map.get(code, entry_px)
                sell_proceeds += shares * px * (1 - TradingConfig.COMMISSION - TradingConfig.STAMP_DUTY - TradingConfig.SLIPPAGE)
            cash += sell_proceeds
            positions.clear()

            # 选行业
            mom_td = sector_mom[sector_mom["trade_date"] == td]
            if mom_td.empty:
                continue
            top_inds = mom_td.nlargest(top_sectors, "score")["industry"].tolist()

            # 各行业选龙头
            td_df_idx = daily_by_date.get(td, pd.DataFrame())
            remaining_slots = total_positions
            per_position_cash = cash / max(remaining_slots, 1)

            for ind in top_inds:
                if remaining_slots <= 0:
                    break
                sector_codes = [c for c, i in industry_map.items() if i == ind]
                if not sector_codes:
                    continue
                picks = select_top_stocks_in_sector(
                    td, sector_codes,
                    td_df_idx.reset_index() if not td_df_idx.empty else pd.DataFrame(),
                    None,
                    top_n=min(top_per_sector, remaining_slots),
                )
                for code, px, score in picks:
                    if remaining_slots <= 0:
                        break
                    alloc = per_position_cash
                    shares = int(alloc / px / 100) * 100
                    if shares <= 0:
                        continue
                    cost = shares * px * (1 + TradingConfig.COMMISSION + TradingConfig.SLIPPAGE)
                    if cost > cash:
                        shares = int(cash * 0.95 / px / 100) * 100
                        if shares <= 0:
                            continue
                        cost = shares * px * (1 + TradingConfig.COMMISSION + TradingConfig.SLIPPAGE)
                    cash -= cost
                    positions[code] = (px, shares)
                    remaining_slots -= 1
                    trade_count += 1
                    if ind not in result.sector_hits:
                        result.sector_hits[ind] = 0
                    result.sector_hits[ind] += 1

            # 记录净值
            pos_val = sum(shares * px_map.get(code, entry_px)
                         for code, (entry_px, shares) in positions.items())
            total = cash + pos_val
            equity_curve.append({"date": str(td)[:10], "value": round(total, 2)})

        result.n_trades = trade_count

        # 4. 计算指标
        if len(equity_curve) > 1:
            eq_df = pd.DataFrame(equity_curve)
            eq_df["ret"] = eq_df["value"].pct_change()
            returns = eq_df["ret"].dropna().values
            if len(returns) > 10 and np.std(returns) > 0:
                result.sharpe = round(float(np.mean(returns) / np.std(returns) * np.sqrt(252)), 2)
            result.total_return = round(float(eq_df["value"].iloc[-1] / eq_df["value"].iloc[0] - 1), 4)
            # MDD
            peak = eq_df["value"].cummax()
            result.max_drawdown = round(float(((eq_df["value"] - peak) / peak).min()), 4)

        result.elapsed = round(time.time() - t0, 0)
        logger.info(f"行业轮动: Sharpe={result.sharpe:.2f} "
                     f"Ret={result.total_return:.1%} MDD={result.max_drawdown:.1%} "
                     f"({result.elapsed}s)")

    except Exception as e:
        logger.error(f"行业轮动异常: {e}")
        import traceback
        traceback.print_exc()
        result.error = str(e)[:200]

    return result


def load_sector_data(engine, start, end):
    """供外部调用的数据加载函数。返回 (daily, industry_map, extra)。"""
    codes_df = pd.read_sql(
        text("SELECT code FROM stock_basic WHERE is_st = FALSE AND list_date <= :d"),
        engine, params={"d": end},
    )
    all_codes = codes_df["code"].tolist()

    ind_df = pd.read_sql(text("SELECT code, industry_sw1 FROM stock_industry"), engine)
    ind_df["code"] = ind_df["code"].astype(str).str.zfill(6)
    industry_map = dict(zip(ind_df["code"], ind_df["industry_sw1"]))
    industry_map = {k: v for k, v in industry_map.items() if v and v != "None"}

    daily = load_daily_data(engine, all_codes, start, end,
                            cols=["open", "high", "low", "close"])
    daily["code"] = daily["code"].astype(str).str.zfill(6)

    extra = load_mcap_data(engine, all_codes, start, end, use_proxy=True)
    if extra is not None:
        extra["code"] = extra["code"].astype(str).str.zfill(6)

    return daily, industry_map, extra


def main():
    import argparse
    p = argparse.ArgumentParser(description="行业轮动回测")
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default="2026-06-14")
    p.add_argument("--top-sectors", type=int, default=TOP_SECTORS)
    p.add_argument("--top-per-sector", type=int, default=TOP_STOCKS_PER_SECTOR)
    p.add_argument("--rebalance", type=int, default=REBALANCE_FREQ)
    args = p.parse_args()

    engine = get_engine()
    logger.info("加载数据...")

    # 股票池
    codes_df = pd.read_sql(
        text("SELECT code FROM stock_basic WHERE is_st = FALSE AND list_date <= :d"),
        engine, params={"d": args.end},
    )
    all_codes = codes_df["code"].tolist()

    # 行业分类
    ind_df = pd.read_sql(
        text("SELECT code, industry_sw1 FROM stock_industry"),
        engine,
    )
    ind_df["code"] = ind_df["code"].astype(str).str.zfill(6)
    industry_map = dict(zip(ind_df["code"], ind_df["industry_sw1"]))
    industry_map = {k: v for k, v in industry_map.items() if v and v != "None"}

    # 日线
    daily = load_daily_data(engine, all_codes, args.start, args.end,
                            cols=["open", "high", "low", "close"])
    daily["code"] = daily["code"].astype(str).str.zfill(6)

    # 市值
    extra = load_mcap_data(engine, all_codes, args.start, args.end, use_proxy=True)
    if extra is not None:
        extra["code"] = extra["code"].astype(str).str.zfill(6)

    engine.dispose()

    result = run_sector_rotation(
        daily, industry_map, extra,
        start=args.start, end=args.end,
        top_sectors=args.top_sectors,
        top_per_sector=args.top_per_sector,
        rebalance_freq=args.rebalance,
    )

    print(f"\n{'='*60}")
    print(f"  行业轮动策略: {args.start} → {args.end}")
    print(f"{'='*60}")
    print(f"  行业数: {result.n_sectors}")
    print(f"  Sharpe: {result.sharpe:.2f}")
    print(f"  累计收益: {result.total_return:.1%}")
    print(f"  最大回撤: {result.max_drawdown:.1%}")
    print(f"  交易次数: {result.n_trades}")
    print(f"  判定: {result.verdict}")
    print(f"  耗时: {result.elapsed}s")
    if result.sector_hits:
        print(f"  行业偏好: {dict(sorted(result.sector_hits.items(), key=lambda x: -x[1])[:5])}")
    if result.error:
        print(f"  ❌ {result.error}")


if __name__ == "__main__":
    main()
