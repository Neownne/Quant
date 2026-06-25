#!/usr/bin/env python
"""策略武器库 —— 多个策略独立运行，输出信号到统一面板。

用法:
    python scripts/run_arsenal.py              # 运行所有策略，输出今日信号
    python scripts/run_arsenal.py --recent 20   # 最近20日各策略表现
"""

from __future__ import annotations
import argparse, os, sys, json, time
from datetime import date, timedelta
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy import text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.db import get_engine
from config.settings import TradingConfig

OUT_DIR = "data/arsenal"

os.makedirs(OUT_DIR, exist_ok=True)


@dataclass
class Strategy:
    name: str
    description: str
    signal_file: str = ""
    category: str = ""  # 妖股/趋势/板块/反转
    status: str = "active"


STRATEGIES: list[Strategy] = [
    Strategy("yaogu", "妖股规则: 一字板+低振幅+缩量 评分≥6", "", "妖股"),
    Strategy("sector_mom", "板块动量: 行业20日动量排名", "", "板块"),
    Strategy("sector_lu", "板块涨停潮: 行业涨停家数/成分股比", "", "板块"),
    Strategy("small_cap_rev", "小市值反转: 市值最小100只+近期超跌", "data/backtest_trades/trades_sc_*.csv", "反转"),
]


def today_sector_heatmap():
    """生成今日行业热力图。"""
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(text("""
            SELECT b.industry, d.code, b.name, d.close, d.volume, d.turnover, d.high, d.low
            FROM stock_daily d
            JOIN stock_basic b ON d.code = b.code
            WHERE d.trade_date = (SELECT MAX(trade_date) FROM stock_daily)
              AND b.is_st = FALSE AND b.industry != ''
        """), conn)
        prev = pd.read_sql(text("""
            SELECT code, close as prev_close FROM stock_daily
            WHERE trade_date = (SELECT MAX(trade_date) FROM stock_daily WHERE trade_date < (SELECT MAX(trade_date) FROM stock_daily))
        """), conn)
    engine.dispose()

    df = df.merge(prev, on='code', how='left')
    df['ret'] = (df['close'] - df['prev_close']) / df['prev_close']
    # 板别感知涨停标记
    df['is_lu'] = df.apply(
        lambda r: 1 if pd.notna(r.get('prev_close')) and r['prev_close'] > 0
                  and TradingConfig.is_at_limit_up(r['close'], r['prev_close'], str(r['code']), tolerance=0.98)
                  else 0, axis=1
    )

    sector = df.groupby('industry').agg(
        n=('code', 'count'),
        avg_ret=('ret', 'mean'),
        lu_n=('is_lu', 'sum'),
        up_ratio=('ret', lambda x: (x > 0).mean()),
        volume_sum=('volume', 'sum'),
        turnover_avg=('turnover', 'mean'),
    ).reset_index()
    sector['score'] = sector['avg_ret'] * 0.4 + sector['up_ratio'] * 0.3 + (sector['lu_n']/sector['n']) * 0.3
    sector = sector.sort_values('score', ascending=False)
    return sector


def today_yaogu_signals(min_score=6):
    """用妖股规则扫描今日信号。"""
    # 直接从 stock_daily 实时计算
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(text("""
            SELECT d.code, b.name, d.close, d.open, d.high, d.low, d.volume, d.turnover,
                   d.trade_date, b.industry
            FROM stock_daily d
            JOIN stock_basic b ON d.code = b.code
            WHERE d.trade_date = (SELECT MAX(trade_date) FROM stock_daily)
              AND b.is_st = FALSE AND d.close > 0
        """), conn)
        # 拿前60天数据算因子
        lookback = pd.read_sql(text("""
            SELECT d.code, d.trade_date, d.close, d.open, d.high, d.low, d.volume, d.turnover
            FROM stock_daily d
            WHERE d.trade_date >= (SELECT MAX(trade_date) FROM stock_daily) - INTERVAL '90 days'
              AND d.code = ANY(:codes)
            ORDER BY d.code, d.trade_date
        """), conn, params={"codes": df['code'].tolist()})
    engine.dispose()

    lookback['code'] = lookback['code'].astype(str).str.zfill(6)
    lookback['trade_date'] = pd.to_datetime(lookback['trade_date'])
    lookback['ret'] = lookback.groupby('code')['close'].pct_change()
    lookback['prev_close'] = lookback.groupby('code')['close'].shift(1)

    today_date = df['trade_date'].iloc[0] if len(df) > 0 else None
    if today_date is None:
        return pd.DataFrame()

    # 简化因子计算
    codes = df['code'].tolist()
    signals = []
    for code in codes:
        cdata = lookback[lookback['code'] == code]
        if len(cdata) < 20:
            continue
        today_row = cdata[cdata['trade_date'] == today_date]
        if today_row.empty:
            continue
        r = today_row.iloc[-1]

        # 涨停检测（板别感知）
        is_lu = (pd.notna(r.get('prev_close')) and r['prev_close'] > 0
                 and TradingConfig.is_at_limit_up(r['close'], r['prev_close'], code, tolerance=0.98))

        if not is_lu:
            continue

        # 因子计算
        mult = TradingConfig.get_limit_multiplier(code)
        limit_pxs = (cdata['prev_close'] * mult).round(4)
        is_lu_series = cdata['close'] >= limit_pxs * 0.98
        streak = 0
        for v in is_lu_series[::-1]:
            if v: streak += 1
            else: break
        lu_count_5 = is_lu_series.tail(5).sum()

        hl = r['high'] - r['low']
        seal = r['close'] / r['high'] if r['high'] > 0 else 1
        vol_ratio = r['volume'] / cdata['volume'].tail(20).mean() if len(cdata) >= 20 else 1
        amplitude = hl / r['prev_close'] if r['prev_close'] > 0 else 0
        yiziban = 1 if abs(r['high'] - r['low']) < r['close'] * 0.001 else 0

        # 评分
        score = 0
        if yiziban: score += 3
        if pd.notna(amplitude) and amplitude < 0.08: score += 2
        if pd.notna(vol_ratio) and vol_ratio < 1.5: score += 1
        if streak >= 2: score += 1

        if score >= min_score:
            signals.append({
                'code': code, 'name': df[df['code']==code]['name'].iloc[0] if len(df[df['code']==code]) > 0 else '',
                'score': score, 'close': r['close'], 'ret': r['ret'],
                'yiziban': yiziban, 'streak': streak, 'seal': round(seal, 3),
                'amplitude': round(amplitude, 4) if pd.notna(amplitude) else 0,
                'vol_ratio': round(vol_ratio, 2) if pd.notna(vol_ratio) else 0,
            })

    return pd.DataFrame(signals).sort_values('score', ascending=False) if signals else pd.DataFrame()


def main():
    p = argparse.ArgumentParser(description="策略武器库")
    p.add_argument("--recent", type=int, default=0, help="显示最近N日各策略表现")
    args = p.parse_args()

    print("=" * 60)
    print("  策略武器库 —— 今日信号")
    print(f"  {date.today()}")
    print("=" * 60)

    # ── 行业热力图 ──
    print("\n## 行业热力图\n")
    sector = today_sector_heatmap()
    top5 = sector.head(5)
    for _, r in top5.iterrows():
        bar = '█' * int(r['up_ratio'] * 20)
        print(f"  {r['industry']:<20s} {r['n']:>4.0f}只 | "
              f"涨幅{r['avg_ret']:>+6.2%} | 涨停{r['lu_n']:>3.0f}只 | 上涨比{r['up_ratio']:.0%} {bar}")

    # ── 妖股信号 ──
    print("\n## 妖股信号 (今日涨停且满足规则)\n")
    yg = today_yaogu_signals()
    if len(yg) > 0:
        for _, r in yg.iterrows():
            tags = []
            if r['yiziban']: tags.append('一字板')
            if r['amplitude'] < 0.08: tags.append('低振幅')
            if r['vol_ratio'] < 1.5: tags.append('缩量')
            if r['streak'] >= 2: tags.append(f"连板{r['streak']}")
            print(f"  {r['code']} {r['name']:<8s} score={int(r['score'])} "
                  f"涨停{r['ret']:+.1%} | {' '.join(tags)}")
    else:
        print("  今日无高分妖股信号")

    # ── 板块轮动建议 ──
    print(f"\n## 板块轮动\n")
    print(f"  领涨行业: {', '.join(top5['industry'].head(3).tolist())}")
    print(f"  涨停潮行业 (>10只): ", end="")
    hot_sectors = sector[sector['lu_n'] >= 10]
    if len(hot_sectors) > 0:
        print(', '.join(f"{r['industry']}({int(r['lu_n'])}只)" for _, r in hot_sectors.iterrows()))
    else:
        print("无")

    # ── 保存 ──
    signal_json = os.path.join(OUT_DIR, f"signals_{date.today().strftime('%Y%m%d')}.json")
    output = {
        "date": str(date.today()),
        "sector_top5": top5[['industry', 'n', 'avg_ret', 'lu_n', 'up_ratio']].to_dict('records'),
        "yaogu_signals": yg.to_dict('records') if len(yg) > 0 else [],
        "hot_sectors": hot_sectors[['industry', 'lu_n']].to_dict('records') if len(hot_sectors) > 0 else [],
    }
    with open(signal_json, 'w') as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n信号已保存: {signal_json}")
    print("\n策略状态:")
    for s in STRATEGIES:
        print(f"  [{s.category}] {s.name}: {s.description} ({s.status})")


if __name__ == "__main__":
    main()
