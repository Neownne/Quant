#!/usr/bin/env python
"""导出涨停池历史信号到 Excel —— 2026-01-01 至 2026-06-17 每个交易日。

用法:
    python scripts/export_limit_up_history.py
    python scripts/export_limit_up_history.py --start 2026-01-01 --end 2026-06-17
"""

from __future__ import annotations

import argparse, os, sys, time
import numpy as np
import pandas as pd
from datetime import date, timedelta
from loguru import logger
from sqlalchemy import text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.db import get_engine
from data.loader import load_daily_data, load_mcap_data

OUT_DIR = "data/arsenal"

# ── 涨停阈值 ──
_LIMIT_MULT = {"688": 1.19899, "8": 1.29899, "4": 1.29899, "300": 1.19899, "301": 1.19899}
_DEFAULT_MULT = 1.09899

def _get_limit(code: str) -> float:
    for prefix, limit in _LIMIT_MULT.items():
        if str(code).startswith(prefix):
            return limit
    return _DEFAULT_MULT


def export_history(start_date: str = "2026-01-01", end_date: str = "2026-06-17"):
    """导出涨停池历史信号。"""
    engine = get_engine()

    # ── 股票池 ──
    min_list = pd.Timestamp(start_date) - timedelta(days=252)
    with engine.connect() as conn:
        codes_df = pd.read_sql(
            text("SELECT code, name, industry FROM stock_basic "
                 "WHERE is_st=FALSE AND list_date <= :ld"),
            conn, params={"ld": pd.Timestamp(end_date).strftime("%Y-%m-%d")})
    codes_df['code'] = codes_df['code'].astype(str).str.zfill(6)
    name_map = dict(zip(codes_df['code'], codes_df['name']))
    ind_map = dict(zip(codes_df['code'], codes_df['industry'].fillna('其他')))
    codes = codes_df['code'].tolist()
    logger.info(f"股票池: {len(codes)} 只")

    # ── 加载数据（覆盖 120 天回溯期）──
    pre_start = (pd.Timestamp(start_date) - timedelta(days=120)).strftime("%Y-%m-%d")
    end_str = end_date

    logger.info(f"加载日线 {pre_start} ~ {end_str} ...")
    t0 = time.time()
    daily = load_daily_data(engine, codes, pre_start, end_str,
                            cols=['open', 'high', 'low', 'close', 'volume', 'turnover'])
    daily['code'] = daily['code'].astype(str).str.zfill(6)
    daily['trade_date'] = pd.to_datetime(daily['trade_date'])
    daily = daily.sort_values(['code', 'trade_date'])

    extra = load_mcap_data(engine, codes, pre_start, end_str, use_proxy=True)
    if not extra.empty:
        extra['code'] = extra['code'].astype(str).str.zfill(6)
        extra['trade_date'] = pd.to_datetime(extra['trade_date'])

    logger.info(f"日线: {len(daily)} 行 | 市值: {len(extra)} 行 ({time.time()-t0:.0f}s)")

    # ── 因子预计算 ──
    logger.info("预计算因子 ...")
    daily['ret'] = daily.groupby('code')['close'].pct_change()
    daily['prev_close'] = daily.groupby('code')['close'].shift(1)

    daily['is_lu'] = daily.apply(
        lambda r: 1 if pd.notna(r['ret']) and r['ret'] >= _get_limit(str(r['code'])) * 0.98 else 0,
        axis=1,
    )

    daily['ma5'] = daily.groupby('code')['close'].transform(
        lambda x: x.rolling(5, min_periods=3).mean())
    daily['ma10'] = daily.groupby('code')['close'].transform(
        lambda x: x.rolling(10, min_periods=5).mean())
    daily['lu_20d'] = daily.groupby('code')['is_lu'].transform(
        lambda x: x.rolling(20, min_periods=1).sum())
    logger.info(f"因子预计算完成 ({time.time()-t0:.0f}s)")

    # ── 按交易日遍历 ──
    all_dates = sorted(d for d in daily['trade_date'].unique()
                       if pd.Timestamp(start_date) <= d <= pd.Timestamp(end_date))

    # 过滤非交易日
    trading_dates = [d for d in all_dates if d.weekday() < 5]
    logger.info(f"交易日: {len(trading_dates)} 天")

    all_signals = []
    for td in trading_dates:
        td_mask = daily['trade_date'] == td
        today = daily[td_mask].set_index('code')

        if today.empty:
            continue

        # 市值
        if not extra.empty:
            ex_td = extra[extra['trade_date'] == td].set_index('code')
            today['mcap'] = ex_td.get('market_cap', np.nan) if not ex_td.empty else np.nan
        else:
            today['mcap'] = np.nan

        # 当日涨幅
        today['ret_calc'] = today['close'] / today['prev_close'] - 1

        # 4条件筛选
        mask = (
            (today['mcap'].between(30, 500)) &
            (today['close'].between(5, 100)) &
            (today['ma5'] > today['ma10']) &
            (today['lu_20d'] >= 2) & (today['lu_20d'] <= 4) &
            (today['close'] > 0)
        )
        sel = today[mask]

        if sel.empty:
            continue

        for code in sel.index:
            r = sel.loc[code]
            all_signals.append({
                '日期': td.strftime('%Y-%m-%d'),
                '代码': code,
                '名称': name_map.get(code, '?'),
                '行业': ind_map.get(code, '?'),
                '收盘价': round(float(r['close']), 2),
                '市值(亿)': round(float(r['mcap']), 1) if pd.notna(r['mcap']) else '',
                '今日涨幅': f"{r['ret_calc']*100:+.1f}%",
                '近20日涨停': int(r['lu_20d']),
                '今日涨停': '是' if r['is_lu'] == 1 else '否',
                'MA5': round(float(r['ma5']), 2) if pd.notna(r['ma5']) else '',
                'MA10': round(float(r['ma10']), 2) if pd.notna(r['ma10']) else '',
            })

    engine.dispose()

    result = pd.DataFrame(all_signals)
    if result.empty:
        logger.warning("无信号")
        return

    # 按日期降序、涨幅降序排列
    result = result.sort_values(['日期', '今日涨幅'], ascending=[False, False])

    # 输出
    os.makedirs(OUT_DIR, exist_ok=True)
    path = f"{OUT_DIR}/limit_up_history_{start_date.replace('-','')}_{end_date.replace('-','')}.xlsx"
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        result.to_excel(writer, sheet_name='涨停池信号', index=False)
        # 按日期统计
        daily_count = result.groupby('日期').size().reset_index(name='信号数')
        daily_count.to_excel(writer, sheet_name='每日信号数', index=False)
        # 按股票统计
        stock_count = result.groupby(['代码', '名称']).size().reset_index(name='出现次数')
        stock_count = stock_count.sort_values('出现次数', ascending=False)
        stock_count.to_excel(writer, sheet_name='个股频次', index=False)

    logger.success(f"导出 {len(result)} 条信号 → {path}")
    logger.info(f"日期跨度: {result['日期'].min()} ~ {result['日期'].max()}, "
                f"交易日: {result['日期'].nunique()} 天, "
                f"个股: {result['代码'].nunique()} 只")

    # 简单统计
    print(f"\n  总信号: {len(result)}")
    print(f"  涉及交易日: {result['日期'].nunique()} 天")
    print(f"  涉及个股: {result['代码'].nunique()} 只")
    print(f"  日均信号: {len(result)/result['日期'].nunique():.1f}")

    return result


def main():
    p = argparse.ArgumentParser(description="导出涨停池历史信号")
    p.add_argument("--start", default="2026-01-01", help="起始日期")
    p.add_argument("--end", default="2026-06-17", help="截止日期")
    args = p.parse_args()

    export_history(args.start, args.end)


if __name__ == "__main__":
    main()
