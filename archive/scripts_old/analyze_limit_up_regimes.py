#!/usr/bin/env python
"""打板策略分市场状态分析。

1. 加载上证指数日线 → 5 状态分类
2. 生成/加载涨停信号
3. 逐笔回测（正确交易逻辑：coc 收盘执行、封板跳过、止损、换仓）
4. 按市场状态 + 策略变体分段统计
5. 输出分析报告

用法:
    python scripts/analyze_limit_up_regimes.py --start 2020-01-01 --top-n 5
"""

from __future__ import annotations

import argparse, os, sys, time
from dataclasses import dataclass, field
from datetime import date, timedelta
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.db import get_engine
from data.loader import load_daily_data
from config.settings import TradingConfig
from factors.regime import detect_regime, REGIME_GROUPS
from sqlalchemy import text
from loguru import logger

# ── 常量 ──
INITIAL_CASH = 1_000_000
BUY_COST = 1.0 + TradingConfig.COMMISSION + TradingConfig.SLIPPAGE   # 1.00109
SELL_COST = 1.0 - TradingConfig.SLIPPAGE - TradingConfig.COMMISSION - TradingConfig.STAMP_DUTY  # 0.99841
# 涨停阈值（板别感知）
_LIMIT_MULT_MAP = {"688": 1.19899, "8": 1.29899, "4": 1.29899, "300": 1.19899, "301": 1.19899}
_DEFAULT_MULT_VAL = 1.09899

def _get_limit_mult(code: str) -> float:
    for prefix, m in _LIMIT_MULT_MAP.items():
        if str(code).startswith(prefix):
            return m
    return _DEFAULT_MULT_VAL

def _limit_up_price(prev_close: float, code: str) -> float:
    return round(prev_close * _get_limit_mult(code), 4)

def _limit_down_price(prev_close: float, code: str) -> float:
    return round(prev_close * (2 - _get_limit_mult(code)), 4)


def load_index_regime(engine, start: str, end: str) -> pd.DataFrame:
    """加载上证指数日线并做 5 状态分类。"""
    idx = pd.read_sql(
        text("SELECT trade_date, close FROM index_daily WHERE code='000001' AND trade_date BETWEEN :s AND :e ORDER BY trade_date"),
        engine, params={"s": start, "e": end})
    idx["trade_date"] = pd.to_datetime(idx["trade_date"])
    regime = detect_regime(idx, price_col="close")
    return regime


def generate_signals(engine, start: str, end: str, top_n: int = 30) -> pd.DataFrame:
    """调用 gen_limit_up_signals.py 生成信号。"""
    import subprocess, tempfile
    out = tempfile.mktemp(suffix='.csv', dir='/tmp')
    cmd = [
        sys.executable, "scripts/gen_limit_up_signals.py",
        "--start", start, "--end", end, "--top-n", str(top_n),
        "--out", out,
    ]
    logger.info(f"生成信号: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, capture_output=True)
    sig = pd.read_csv(out)
    sig["date"] = pd.to_datetime(sig["date"])
    sig["code"] = sig["code"].astype(str).str.zfill(6)
    os.unlink(out)
    return sig


@dataclass
class Position:
    code: str
    entry_date: pd.Timestamp
    entry_price: float
    shares: int
    cost: float
    peak_price: float = 0.0

    def __post_init__(self):
        self.peak_price = self.entry_price


@dataclass
class Trade:
    code: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    shares: int
    pnl: float
    pnl_pct: float
    regime_entry: str = ""
    regime_exit: str = ""


class LimitUpBacktest:
    """轻量级打板回测 — 匹配 bt_backtest.py 的交易逻辑。"""

    def __init__(self, daily: pd.DataFrame, signals: pd.DataFrame,
                 regime_map: dict, top_n: int = 5, stop_loss: float = 0.08,
                 trailing_stop: float = 0.0, cooling_days: int = 0,
                 require_positive: bool = False):
        self.daily = daily.sort_values(["code", "trade_date"])
        self.signals = signals.sort_values(["date", "score"], ascending=[True, False])
        self.regime_map = regime_map  # {date: "strong_bull"|...}
        self.top_n = top_n
        self.stop_loss = stop_loss
        self.trailing_stop = trailing_stop
        self.cooling_days = cooling_days
        self.require_positive = require_positive

        # 构建每只股票的日线查找表
        self.price_map = {}
        for code, grp in self.daily.groupby("code"):
            grp_sorted = grp.sort_values("trade_date")
            prev_close = grp_sorted["close"].shift(1)
            self.price_map[code] = dict(zip(
                grp_sorted["trade_date"],
                zip(grp_sorted["open"], grp_sorted["high"], grp_sorted["low"],
                    grp_sorted["close"], prev_close)
            ))

        self.trade_dates = sorted(
            set(self.daily["trade_date"].unique()) & set(self.regime_map.keys())
        )

    def _get_price(self, code: str, td) -> Optional[tuple]:
        return self.price_map.get(code, {}).get(td)

    def _is_limit_up(self, close: float, prev_close: float, code: str) -> bool:
        if pd.isna(prev_close) or prev_close <= 0:
            return False
        return close >= _limit_up_price(prev_close, code)

    def _is_limit_down(self, close: float, prev_close: float, code: str) -> bool:
        if pd.isna(prev_close) or prev_close <= 0:
            return False
        return close <= _limit_down_price(prev_close, code)

    def run(self) -> tuple[list[Trade], list[dict]]:
        """执行回测。返回 (trades, equity_curve)。"""
        trades: list[Trade] = []
        equity = []
        cash = INITIAL_CASH
        positions: dict[str, Position] = {}  # code → Position
        sold_history: dict[str, pd.Timestamp] = {}  # code → last sold date

        signal_dates = set(self.signals["date"].unique())

        for td in self.trade_dates:
            # ── 日终结算：检查止损 / 移动止盈 ──
            for code in list(positions.keys()):
                pos = positions[code]
                prices = self._get_price(code, td)
                if prices is None:
                    continue
                close = prices[3]
                prev_close = prices[4]

                # 止损
                hit_stop = close <= pos.entry_price * (1 - self.stop_loss)
                # 移动止盈
                pos.peak_price = max(pos.peak_price, close)
                hit_trail = (self.trailing_stop > 0 and
                             pos.peak_price > pos.entry_price * 1.05 and
                             close <= pos.peak_price * (1 - self.trailing_stop))

                if (hit_stop or hit_trail) and not self._is_limit_down(close, prev_close, code):
                    exit_type = "止损" if hit_stop else "移动止盈"
                    proceed = close * pos.shares * SELL_COST
                    pnl = proceed - pos.cost
                    trades.append(Trade(
                        code=code, entry_date=pos.entry_date, exit_date=td,
                        entry_price=pos.entry_price, exit_price=close,
                        shares=pos.shares, pnl=pnl, pnl_pct=pnl / pos.cost,
                        regime_entry=self.regime_map.get(pos.entry_date, ""),
                        regime_exit=self.regime_map.get(td, ""),
                    ))
                    cash += proceed
                    sold_history[code] = td
                    del positions[code]

            # ── 信号日：调仓 ──
            if td in signal_dates:
                today_signals = self.signals[self.signals["date"] == td]
                wanted = set(today_signals["code"].iloc[:self.top_n].tolist())

                # 卖出不在 Top-N 的持仓
                for code in list(positions.keys()):
                    if code not in wanted:
                        pos = positions[code]
                        prices = self._get_price(code, td)
                        if prices is None:
                            continue
                        close = prices[3]
                        prev_close = prices[4]
                        if self._is_limit_down(close, prev_close, code):
                            continue
                        proceed = close * pos.shares * SELL_COST
                        pnl = proceed - pos.cost
                        trades.append(Trade(
                            code=code, entry_date=pos.entry_date, exit_date=td,
                            entry_price=pos.entry_price, exit_price=close,
                            shares=pos.shares, pnl=pnl, pnl_pct=pnl / pos.cost,
                            regime_entry=self.regime_map.get(pos.entry_date, ""),
                            regime_exit=self.regime_map.get(td, ""),
                        ))
                        cash += proceed
                        sold_history[code] = td
                        del positions[code]

                # 买入新信号
                slots = self.top_n - len(positions)
                if slots > 0:
                    per_stock = cash * 0.95 / max(slots, 1)
                    for _, sig in today_signals.iterrows():
                        code = sig["code"]
                        if code in positions:
                            continue
                        if code in sold_history:
                            if self.cooling_days > 0:
                                days_since = (td - sold_history[code]).days
                                if days_since < self.cooling_days:
                                    continue

                        prices = self._get_price(code, td)
                        if prices is None:
                            continue
                        close = prices[3]
                        prev_close = prices[4]

                        # 封板跳过
                        if self._is_limit_up(close, prev_close, code):
                            continue

                        # 收阳确认
                        if self.require_positive and close <= prev_close:
                            continue

                        shares = int(per_stock / close / 100) * 100
                        if shares < 100:
                            continue
                        cost = shares * close * BUY_COST
                        if cost > cash * 0.98:
                            shares = int(cash * 0.98 / close / BUY_COST / 100) * 100
                            cost = shares * close * BUY_COST
                        if shares < 100:
                            continue

                        cash -= cost
                        positions[code] = Position(
                            code=code, entry_date=td, entry_price=close,
                            shares=shares, cost=cost
                        )
                        if len(positions) >= self.top_n:
                            break

            # ── 记录权益 ──
            pos_value = 0
            for code, pos in positions.items():
                prices = self._get_price(code, td)
                if prices is not None:
                    pos_value += pos.shares * prices[3] * SELL_COST
            equity.append({
                "date": td,
                "cash": cash,
                "position_value": pos_value,
                "total": cash + pos_value,
                "positions": len(positions),
                "regime": self.regime_map.get(td, ""),
            })

        return trades, equity


def analyze_regimes(trades: list[Trade], equity: list[dict],
                    variant_label: str) -> dict:
    """按市场状态汇总交易表现。"""
    df = pd.DataFrame(equity)
    df["date"] = pd.to_datetime(df["date"])
    df["return"] = df["total"].pct_change()

    results = {"variant": variant_label}

    for regime in ["strong_bull", "weak_bull", "sideways", "slow_bear", "fast_bear"]:
        regime_trades = [t for t in trades if t.regime_entry == regime]
        regime_equity = df[df["regime"] == regime]

        n = len(regime_trades)
        if n == 0:
            results[f"{regime}_trades"] = 0
            results[f"{regime}_win_rate"] = 0
            results[f"{regime}_avg_return"] = 0
            results[f"{regime}_total_pnl"] = 0
            results[f"{regime}_days"] = len(regime_equity)
            continue

        wins = [t for t in regime_trades if t.pnl > 0]
        win_rate = len(wins) / n * 100
        avg_return = np.mean([t.pnl_pct for t in regime_trades]) * 100
        total_pnl = sum(t.pnl for t in regime_trades)
        avg_hold = np.mean([(t.exit_date - t.entry_date).days for t in regime_trades])

        regime_ret = regime_equity["return"].dropna()
        if len(regime_ret) > 5:
            sharpe = (regime_ret.mean() / regime_ret.std() * np.sqrt(252)) if regime_ret.std() > 0 else 0
            max_dd = (regime_equity["total"] / regime_equity["total"].cummax() - 1).min() * 100
        else:
            sharpe = 0
            max_dd = 0

        results[f"{regime}_trades"] = n
        results[f"{regime}_win_rate"] = round(win_rate, 1)
        results[f"{regime}_avg_return"] = round(avg_return, 2)
        results[f"{regime}_total_pnl"] = round(total_pnl, 0)
        results[f"{regime}_avg_hold"] = round(avg_hold, 1)
        results[f"{regime}_sharpe"] = round(sharpe, 2)
        results[f"{regime}_max_dd"] = round(max_dd, 2)
        results[f"{regime}_days"] = len(regime_equity)

    # 总体
    all_wins = [t for t in trades if t.pnl > 0]
    results["total_trades"] = len(trades)
    results["total_win_rate"] = round(len(all_wins) / len(trades) * 100, 1) if trades else 0
    results["total_pnl"] = round(sum(t.pnl for t in trades), 0)
    results["total_return"] = round((df["total"].iloc[-1] / INITIAL_CASH - 1) * 100, 1)

    return results


def main():
    ap = argparse.ArgumentParser(description="打板策略分市场状态分析")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--top-n", type=int, default=5)
    ap.add_argument("--out-dir", default="data/arsenal")
    args = ap.parse_args()

    end = args.end or date.today().strftime("%Y-%m-%d")
    engine = get_engine()

    # ── 1. 市场状态 ──
    logger.info("═══ Step 1: 市场状态分类 ═══")
    regime_df = load_index_regime(engine, args.start, end)
    regime_map = dict(zip(regime_df["trade_date"], regime_df["regime"]))
    logger.info(f"  覆盖 {len(regime_map)} 个交易日")

    # 打印各状态分布
    for r in ["strong_bull", "weak_bull", "sideways", "slow_bear", "fast_bear"]:
        cnt = sum(1 for v in regime_map.values() if v == r)
        logger.info(f"  {r}: {cnt} 天 ({cnt/len(regime_map)*100:.1f}%)")

    # ── 2. 生成信号（一次，宽参数覆盖更多候选）──
    logger.info("═══ Step 2: 生成涨停信号 ═══")
    signals = generate_signals(engine, args.start, end, top_n=args.top_n * 4)
    logger.info(f"  信号: {len(signals)} 条, {signals['code'].nunique()} 只")

    # ── 3. 加载日线 ──
    logger.info("═══ Step 3: 加载日线数据 ═══")
    codes = signals["code"].unique().tolist()
    daily = load_daily_data(engine, codes, args.start, end,
                            cols=["open", "high", "low", "close"])
    daily["code"] = daily["code"].astype(str).str.zfill(6)
    daily["trade_date"] = pd.to_datetime(daily["trade_date"])
    engine.dispose()
    logger.info(f"  日线: {len(daily)} 行, {len(codes)} 只")

    # ── 4. 测试变体 ──
    logger.info("═══ Step 4: 运行变体回测 ═══")

    VARIANTS = [
        # (label, top_n, stop_loss, trailing_stop, cooling_days, require_positive)
        ("基线", 5, 0.08, 0.0, 0, False),
        ("紧止损(5%)", 5, 0.05, 0.0, 0, False),
        ("宽止损(12%)", 5, 0.12, 0.0, 0, False),
        ("移动止盈8%", 5, 0.08, 0.08, 0, False),
        ("移动止盈12%", 5, 0.08, 0.12, 0, False),
        ("移动止盈15%", 5, 0.08, 0.15, 0, False),
        ("冷却3天", 5, 0.08, 0.0, 3, False),
        ("冷却5天", 5, 0.08, 0.0, 5, False),
        ("收阳确认", 5, 0.08, 0.0, 0, True),
        ("冷却3天+收阳", 5, 0.08, 0.0, 3, True),
        ("持仓3只", 3, 0.08, 0.0, 0, False),
        ("持仓10只", 10, 0.08, 0.0, 0, False),
        ("紧止损+移动止盈10%", 5, 0.05, 0.10, 0, False),
        ("冷却5天+止盈12%", 5, 0.08, 0.12, 5, False),
    ]

    all_results = []
    for label, top_n, sl, ts, cool, rp in VARIANTS:
        bt = LimitUpBacktest(
            daily, signals, regime_map, top_n=top_n,
            stop_loss=sl, trailing_stop=ts,
            cooling_days=cool, require_positive=rp,
        )
        trades, equity = bt.run()
        res = analyze_regimes(trades, equity, label)
        all_results.append(res)
        logger.info(f"  {label}: {len(trades)}笔, 胜率{res['total_win_rate']}%, "
                     f"总收益{res['total_return']}%")

    # ── 5. 输出 ──
    os.makedirs(args.out_dir, exist_ok=True)
    tag = date.today().strftime("%Y%m%d")
    df_out = pd.DataFrame(all_results)

    # 按总收益排序
    df_out = df_out.sort_values("total_return", ascending=False)

    # ── 控制台报告 ──
    cols_5state = []
    for r in ["strong_bull", "weak_bull", "sideways", "slow_bear", "fast_bear"]:
        cols_5state.extend([f"{r}_win_rate", f"{r}_avg_return", f"{r}_trades"])

    print(f"\n{'='*120}")
    print(f"  打板策略分市场状态分析 | {args.start} → {end}")
    print(f"{'='*120}")
    print(f"\n{'变体':<22s} {'总收益':>7s} {'总胜率':>6s} {'交易':>5s} | "
          f"{'强牛胜率':>6s} {'强牛均':>6s} | {'弱牛胜率':>6s} {'弱牛均':>6s} | "
          f"{'震荡胜率':>6s} {'震荡均':>6s} | {'慢熊胜率':>6s} {'慢熊均':>6s} | {'快熊胜率':>6s} {'快熊均':>6s}")
    print("-" * 120)

    for _, r in df_out.iterrows():
        print(f'{r["variant"]:<22s} {r["total_return"]:>6.1f}% {r["total_win_rate"]:>5.1f}% {r["total_trades"]:>5.0f} | '
              f'{r["strong_bull_win_rate"]:>5.1f}% {r["strong_bull_avg_return"]:>5.1f}% | '
              f'{r["weak_bull_win_rate"]:>5.1f}% {r["weak_bull_avg_return"]:>5.1f}% | '
              f'{r["sideways_win_rate"]:>5.1f}% {r["sideways_avg_return"]:>5.1f}% | '
              f'{r["slow_bear_win_rate"]:>5.1f}% {r["slow_bear_avg_return"]:>5.1f}% | '
              f'{r["fast_bear_win_rate"]:>5.1f}% {r["fast_bear_avg_return"]:>5.1f}%')

    # CSV
    csv_path = f"{args.out_dir}/limit_up_regime_analysis_{tag}.csv"
    df_out.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n  输出: {csv_path}")

    # ── 市场状态分布 ──
    print(f"\n{'='*60}")
    print(f"  市场状态分布 ({args.start} → {end})")
    print(f"{'='*60}")
    for r in ["strong_bull", "weak_bull", "sideways", "slow_bear", "fast_bear"]:
        cnt = sum(1 for v in regime_map.values() if v == r)
        total = len(regime_map)
        print(f"  {r:<15s}: {cnt:>5d} 天 ({cnt/total*100:>5.1f}%)")


if __name__ == "__main__":
    main()
