#!/usr/bin/env python
"""入场价位分析 —— 基于技术指标的多维度评估。

对 stock_groups.csv 中的每只股票：
  1. 计算均线/布林带/ATR/RSI
  2. 评估当前估值分位（近120天）
  3. 推荐入场价格区间
  4. 评估趋势方向与风险等级

输出: data/arsenal/entry_analysis_YYYYMMDD.csv
"""

from __future__ import annotations

import argparse, os, sys, time
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.db import get_engine
from data.loader import load_daily_data
from sqlalchemy import text
from loguru import logger

LOOKBACK = 180  # 回看天数


def parse_groups_csv(path: str) -> pd.DataFrame:
    """解析 stock_groups.csv，处理中文逗号。"""
    raw = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or '分组名称' in line:
                continue
            # 整体替换中文逗号
            line = line.replace('，', ',').replace('　', '')
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 4:
                raw.append(parts[:4])

    df = pd.DataFrame(raw, columns=['sector', 'code', 'name', 'price_now'])
    df['code'] = df['code'].astype(str).str.replace(' ', '').str.zfill(6)
    df['price_now'] = pd.to_numeric(df['price_now'], errors='coerce')
    return df


def compute_indicators(daily: pd.DataFrame) -> pd.DataFrame:
    """计算技术指标。daily 需含 close, high, low, volume，按 code + trade_date 排序。"""
    df = daily.sort_values(['code', 'trade_date']).copy()
    grouped = df.groupby('code', group_keys=False)

    # ── 均线 ──
    for w in [5, 10, 20, 40, 60]:
        df[f'ma{w}'] = grouped['close'].transform(lambda x: x.rolling(w).mean())

    # ── 布林带 (20日) ──
    df['bb_mid'] = df['ma20']
    df['bb_std'] = grouped['close'].transform(lambda x: x.rolling(20).std())
    df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']

    # ── ATR (14日) ──
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - grouped['close'].shift(1)),
            abs(df['low'] - grouped['close'].shift(1))
        )
    )
    df['atr14'] = grouped['tr'].transform(lambda x: x.rolling(14).mean())

    # ── RSI (14日) ──
    delta = grouped['close'].diff()
    df['gain'] = delta.clip(lower=0)
    df['loss'] = (-delta).clip(lower=0)
    df['avg_gain'] = df.groupby('code')['gain'].transform(lambda x: x.rolling(14).mean())
    df['avg_loss'] = df.groupby('code')['loss'].transform(lambda x: x.rolling(14).mean())
    df['rs'] = df['avg_gain'] / df['avg_loss'].replace(0, np.nan)
    df['rsi14'] = 100 - (100 / (1 + df['rs']))

    # ── 成交量均线 + 量比 ──
    df['vol_ma20'] = grouped['volume'].transform(lambda x: x.rolling(20).mean())
    df['vol_ratio'] = df['volume'] / df['vol_ma20'].replace(0, np.nan)

    # ── 近期高低点 ──
    df['low_20d'] = grouped['low'].transform(lambda x: x.rolling(20).min())
    df['high_20d'] = grouped['high'].transform(lambda x: x.rolling(20).max())
    df['low_60d'] = grouped['low'].transform(lambda x: x.rolling(60).min())
    df['high_60d'] = grouped['high'].transform(lambda x: x.rolling(60).max())

    return df


def assess_stock(code: str, name: str, sector: str, price_now: float,
                 df_stock: pd.DataFrame) -> dict:
    """单只股票的综合评估。df_stock 按 trade_date 升序，至少 60 行。"""
    latest = df_stock.iloc[-1]
    close = float(latest['close'])

    # ── 趋势判断 ──
    # MA排列：多头 if ma5>ma10>ma20，空头 if ma5<ma10<ma20
    ma_order = 'bullish' if close > latest['ma5'] > latest['ma10'] > latest['ma20'] else \
               ('bearish' if close < latest['ma5'] < latest['ma10'] < latest['ma20'] else 'mixed')

    # ── 估值分位 ──
    rolling_120 = df_stock['close'].tail(120)
    pct = (close - rolling_120.min()) / (rolling_120.max() - rolling_120.min()) * 100 if \
        rolling_120.max() > rolling_120.min() else 50
    pct_60 = (close - rolling_120.tail(60).min()) / \
             (rolling_120.tail(60).max() - rolling_120.tail(60).min()) * 100 if \
        rolling_120.tail(60).max() > rolling_120.tail(60).min() else 50

    # ── 距均线 ──
    dist_ma20 = (close / latest['ma20'] - 1) * 100 if pd.notna(latest['ma20']) else 0
    dist_ma40 = (close / latest['ma40'] - 1) * 100 if pd.notna(latest['ma40']) else 0
    dist_ma60 = (close / latest['ma60'] - 1) * 100 if pd.notna(latest['ma60']) else 0

    # ── 布林带位置 ──
    bb_pos = (close - latest['bb_lower']) / (latest['bb_upper'] - latest['bb_lower']) * 100 \
        if pd.notna(latest['bb_upper']) and latest['bb_upper'] > latest['bb_lower'] else 50

    # ── ATR ──
    atr_pct = (latest['atr14'] / close * 100) if pd.notna(latest['atr14']) and close > 0 else 0

    # ── RSI ──
    rsi = float(latest['rsi14']) if pd.notna(latest['rsi14']) else 50

    # ── 量比 ──
    vol_ratio = float(latest['vol_ratio']) if pd.notna(latest['vol_ratio']) else 1

    # ── 距近期高低点 ──
    dist_high20 = (close / latest['high_20d'] - 1) * 100 if pd.notna(latest['high_20d']) else 0
    dist_low20 = (close / latest['low_20d'] - 1) * 100 if pd.notna(latest['low_20d']) else 0

    # ── 推荐入场价 ──
    # 保守入场: MA40 附近或布林下轨上方 10%
    support_levels = []
    for label, val in [('MA20', latest['ma20']), ('MA40', latest['ma40']),
                        ('MA60', latest['ma60']), ('BB下轨', latest['bb_lower']),
                        ('20日低', latest['low_20d']), ('60日低', latest['low_60d'])]:
        if pd.notna(val) and val > 0:
            support_levels.append((label, float(val)))

    # 取均线支撑
    ma_supports = [s for s in support_levels if 'MA' in s[0]]
    tec_supports = [s for s in support_levels if 'BB' in s[0] or '低' in s[0]]

    entry_conservative = None
    entry_aggressive = None

    if ma_supports:
        # 保守入场：最低的MA支撑位（通常MA60或MA40）
        conservative_ma = sorted(ma_supports, key=lambda x: x[1])[0]
        if conservative_ma[1] < close:
            entry_conservative = round(conservative_ma[1] * 1.02, 2)  # 略高于支撑
        else:
            entry_conservative = round(conservative_ma[1] * 0.98, 2)  # 等回调

    if tec_supports:
        # 激进入场：BB下轨附近
        bb_low = [s for s in tec_supports if 'BB' in s[0]]
        if bb_low and bb_low[0][1] < close * 0.95:
            entry_aggressive = round(bb_low[0][1] * 1.05, 2)
        elif ma_supports:
            entry_aggressive = round(sorted(ma_supports, key=lambda x: x[1])[0][1], 2)

    # ── 风险等级 ──
    risk = '低'
    risk_reasons = []
    if pct > 80:
        risk_reasons.append('近120日高位')
        risk = '高' if risk == '低' else risk
    if dist_ma20 > 10:
        risk_reasons.append('远离MA20')
        risk = '高'
    if rsi > 70:
        risk_reasons.append('RSI超买')
        risk = '高'
    elif rsi < 30:
        risk_reasons.append('RSI超卖→反弹机会')
        risk = '低' if risk == '高' else '关注'
    if dist_high20 > -1:
        risk_reasons.append('接近20日高点')
        if risk == '低':
            risk = '中'
    if vol_ratio < 0.5:
        risk_reasons.append('极度缩量')
    if atr_pct > 5:
        risk_reasons.append('高波动')
        risk = '高' if risk == '低' else risk

    # ── 趋势强度 ──
    trend_score = 0
    if close > latest['ma20']: trend_score += 1
    if close > latest['ma40']: trend_score += 1
    if latest['ma5'] > latest['ma20']: trend_score += 1
    if latest['ma20'] > latest['ma60']: trend_score += 1
    if 30 < rsi < 70: trend_score += 1
    trend_label = {5: '强上升', 4: '上升', 3: '震荡偏多', 2: '震荡偏空', 1: '下跌', 0: '强下跌'}.get(trend_score, '震荡')

    return {
        'sector': sector,
        'code': code,
        'name': name,
        'price_now': price_now,
        'close_latest': round(close, 2),
        'pct_120d': round(pct, 1),
        'pct_60d': round(pct_60, 1),
        'trend': trend_label,
        'ma_order': ma_order,
        'dist_ma20_pct': round(dist_ma20, 1),
        'dist_ma60_pct': round(dist_ma60, 1),
        'bb_pos_pct': round(bb_pos, 1),
        'rsi14': round(rsi, 1),
        'atr_pct': round(atr_pct, 2),
        'vol_ratio': round(vol_ratio, 2),
        'entry_low': entry_conservative,
        'entry_high': round(close * 0.97, 2) if entry_conservative else None,
        'entry_aggressive': entry_aggressive,
        'risk_level': risk,
        'risk_reasons': '; '.join(risk_reasons) if risk_reasons else '-',
        'supports': ' | '.join([f'{l}={v:.2f}' for l, v in support_levels[:4]]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='stock_groups.csv')
    ap.add_argument('--out-dir', default='data/arsenal')
    args = ap.parse_args()

    # ── 读取分组 ──
    groups = parse_groups_csv(args.csv)
    codes = groups['code'].unique().tolist()
    name_map = dict(zip(groups['code'], groups['name']))
    sector_map = dict(zip(groups['code'], groups['sector']))
    price_map = dict(zip(groups['code'], groups['price_now']))
    logger.info(f'股票数: {len(codes)} 只, {groups["sector"].nunique()} 个板块')

    # ── 加载数据 ──
    engine = get_engine()
    end = date.today()
    start = end - timedelta(days=LOOKBACK)

    daily = load_daily_data(engine, codes, start.strftime('%Y-%m-%d'),
                            end.strftime('%Y-%m-%d'),
                            cols=['open', 'high', 'low', 'close', 'volume'])
    daily['code'] = daily['code'].astype(str).str.zfill(6)
    daily['trade_date'] = pd.to_datetime(daily['trade_date'])
    engine.dispose()
    logger.info(f'日线数据: {len(daily)} 行')

    # ── 计算指标 ──
    daily = compute_indicators(daily)
    logger.info('指标计算完成')

    # ── 逐股评估 ──
    results = []
    skipped = 0
    for code in sorted(codes):
        stock_df = daily[daily['code'] == code].sort_values('trade_date')
        if len(stock_df) < 40:
            skipped += 1
            logger.warning(f'  {code} 数据不足 ({len(stock_df)} 天)')
            continue

        res = assess_stock(
            code, name_map.get(code, '?'), sector_map.get(code, '?'),
            price_map.get(code, 0), stock_df
        )
        results.append(res)

    logger.info(f'评估完成: {len(results)} 只, 跳过 {skipped}')

    # ── 输出 ──
    out_df = pd.DataFrame(results)

    # 按板块+入场价合理性排序
    out_df['entry_discount'] = ((out_df['price_now'] - out_df['entry_low']) / out_df['price_now'] * 100).round(1)
    out_df = out_df.sort_values(['sector', 'risk_level', 'pct_120d'])

    os.makedirs(args.out_dir, exist_ok=True)
    tag = date.today().strftime('%Y%m%d')
    csv_path = f'{args.out_dir}/entry_analysis_{tag}.csv'
    out_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    # ── 终端摘要 ──
    print(f'\n{"="*100}')
    print(f'  入场价位分析 · {date.today()}  (共 {len(results)} 只)')
    print(f'{"="*100}')
    print(f'\n{"板块":<16s} {"代码":<8s} {"名称":<8s} {"现价":>7s} {"分位":>6s} {"趋势":<8s} '
          f'{"RSI":>5s} {"风险":<4s} {"保守入场":>8s} {"折扣":>6s}  {"理由"}')
    print('-' * 110)

    for _, r in out_df.iterrows():
        discount = (1 - r['entry_low'] / r['price_now']) * 100 if r['entry_low'] and r['price_now'] > 0 else 0
        print(f'{r["sector"]:<16s} {r["code"]:<8s} {r["name"]:<8s} {r["price_now"]:>7.2f} '
              f'{r["pct_120d"]:>5.0f}% {r["trend"]:<8s} {r["rsi14"]:>5.1f} {r["risk_level"]:<4s} '
              f'{r["entry_low"] if r["entry_low"] else "N/A":>8} {discount:>5.1f}%  '
              f'{r["risk_reasons"][:60]}')

    print(f'\n  输出: {csv_path}')
    print(f'  解释: 保守入场 = MA支撑位附近; 分位 = 近120日价格位置(越低越安全)')
    print(f'        折扣 = 现价距保守入场价的跌幅; RSI<30 超卖/RSI>70 超买\n')

    # ── 板块汇总 ──
    print(f'{"板块":<16s} {"股数":>4s} {"高风险":>5s} {"中风险":>5s} {"低风险":>5s} {"平均分位":>7s}')
    print('-' * 50)
    for sector, grp in out_df.groupby('sector'):
        hi = (grp['risk_level'] == '高').sum()
        md = (grp['risk_level'] == '中').sum()
        lo = (grp['risk_level'] == '低').sum()
        avg_pct = grp['pct_120d'].mean()
        print(f'{sector:<16s} {len(grp):>4d} {hi:>5d} {md:>5d} {lo:>5d} {avg_pct:>6.0f}%')


if __name__ == '__main__':
    main()
