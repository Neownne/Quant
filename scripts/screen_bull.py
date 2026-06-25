#!/usr/bin/env python
"""牛股筛选器：缩量筑底的小市值股票。

条件:
  - 市值 5-50亿
  - 收盘 < MA40 (趋势偏弱)
  - 成交量 < 40日均量 (缩量)
  - 20日波动率 < 3% (低波动筑底)
  - 60日内涨停 < 2次 (非妖股)

输出每只股票的综合评分（0-100），基于各条件的偏离度。

用法:
    python scripts/screen_bull.py                    # 今日
    python scripts/screen_bull.py --date 2025-01-02  # 指定日期
"""

from __future__ import annotations
import argparse, os, sys
import numpy as np
import pandas as pd
from datetime import date, timedelta
from loguru import logger
from sqlalchemy import text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.db import get_engine
from data.loader import load_daily_data, load_mcap_data

# ── 涨停阈值 ──
_LIMIT_MULT = {"688": 1.19899, "8": 1.29899, "4": 1.29899, "300": 1.19899, "301": 1.19899}
_DEFAULT_MULT = 1.09899

def _get_limit(code: str) -> float:
    for prefix, limit in _LIMIT_MULT.items():
        if str(code).startswith(prefix):
            return limit
    return _DEFAULT_MULT


def screen(date_str=None, exclude_gem_star=False, daily_df=None, extra_df=None,
           name_map=None, ind_map=None):
    """主筛选函数，返回 DataFrame。

    支持两种模式:
    - 独立模式: 不传 daily_df/extra_df，自行加载数据
    - 集成模式: 传入预加载的 daily_df/extra_df（跳过加载，节省时间）
      同时传入 name_map/ind_map 用于显示
    """
    target_date = pd.Timestamp(date_str) if date_str else None

    if daily_df is not None and extra_df is not None:
        # ── 集成模式：使用预加载数据 ──
        daily = daily_df.copy()
        extra = extra_df.copy()
        # 确定目标日期
        if target_date is None:
            target_date = daily['trade_date'].max()
        engine = None
        if name_map is None:
            name_map = {}
        if ind_map is None:
            ind_map = {}
        codes_in_data = daily['code'].unique().tolist()
    else:
        # ── 独立模式：自行加载 ──
        engine = get_engine()

        # 股票池
        min_list = (target_date or pd.Timestamp(date.today())) - timedelta(days=252)
        with engine.connect() as conn:
            codes_df = pd.read_sql(
                text("SELECT code, name, industry FROM stock_basic "
                     "WHERE is_st=FALSE AND list_date <= :ld AND code !~ '^(300|301|688|[48])'"
                ), conn, params={"ld": min_list.strftime("%Y-%m-%d")}
            )
        codes_df['code'] = codes_df['code'].astype(str).str.zfill(6)
        codes = codes_df['code'].tolist()
        name_map = dict(zip(codes_df['code'], codes_df['name']))
        ind_map = dict(zip(codes_df['code'], codes_df['industry'].fillna('其他')))

        # 确定目标日期
        with engine.connect() as conn:
            r = conn.execute(text("SELECT MAX(trade_date) FROM stock_daily")).fetchone()
        latest = pd.Timestamp(r[0]) if r and r[0] else pd.Timestamp(date.today())
        if target_date is None or target_date > latest:
            target_date = latest

        # 加载数据
        pre_start = (target_date - timedelta(days=120)).strftime("%Y-%m-%d")
        end_str = target_date.strftime("%Y-%m-%d")

        daily = load_daily_data(engine, codes, pre_start, end_str,
                                cols=['open','high','low','close','volume','turnover'])
        daily['code'] = daily['code'].astype(str).str.zfill(6)
        daily['trade_date'] = pd.to_datetime(daily['trade_date'])
        daily = daily.sort_values(['code','trade_date'])
        daily['ret'] = daily.groupby('code')['close'].pct_change()

        extra = load_mcap_data(engine, codes, pre_start, end_str, use_proxy=True)
        extra['code'] = extra['code'].astype(str).str.zfill(6)
        extra['trade_date'] = pd.to_datetime(extra['trade_date'])
        engine.dispose()
        codes_in_data = codes

    # ── 因子计算（共用）──
    daily['ma40'] = daily.groupby('code')['close'].transform(lambda x: x.rolling(40,min_periods=20).mean())
    daily['vol_ma40'] = daily.groupby('code')['volume'].transform(lambda x: x.rolling(40,min_periods=20).mean())
    daily['ret_vol_20'] = daily.groupby('code')['ret'].transform(lambda x: x.rolling(20,min_periods=10).std())
    # 板别感知涨停标记
    daily['is_lu'] = daily.apply(
        lambda r: 1 if pd.notna(r['ret']) and r['ret'] >= _get_limit(str(r['code'])) * 0.98 else 0, axis=1
    )
    daily['lu_60d'] = daily.groupby('code')['is_lu'].transform(lambda x: x.rolling(60,min_periods=30).sum())
    daily['low60'] = daily.groupby('code')['low'].transform(lambda x: x.rolling(60,min_periods=30).min())

    # 取目标日截面
    td_data = daily[daily['trade_date'] == target_date].set_index('code')
    ex_td = extra[extra['trade_date'] == target_date].set_index('code')

    if td_data.empty:
        logger.error(f"无 {target_date.strftime('%Y-%m-%d')} 数据")
        return pd.DataFrame()

    td_data['mcap'] = ex_td['market_cap'] if not ex_td.empty else np.nan

    # ── 筛选 ──
    mask = (
        (td_data['mcap'].between(5, 50)) &
        (td_data['close'] < td_data['ma40']) &
        (td_data['volume'] < td_data['vol_ma40']) &
        (td_data['ret_vol_20'] < 0.03) &
        (td_data['lu_60d'].fillna(0) < 2) &
        (td_data['close'] > 0)
    )
    # 排除创业/科创板
    if exclude_gem_star:
        mask = mask & (~td_data.index.str.startswith(('300','301','688')))
    sel = td_data[mask].copy()

    if sel.empty:
        return pd.DataFrame()

    # ── 综合评分 (0-100) ──
    # 每个条件越好分越高
    scores = pd.DataFrame(index=sel.index)

    # 1. 市值越小越好 (5-50亿 → 线性映射到 0-20分)
    scores['s_mcap'] = (1 - (sel['mcap'] - 5) / 45).clip(0, 1) * 20

    # 2. 跌破MA40越深越好 (-30%→15分, 0%→0分)
    ma_dev = (sel['close'] / sel['ma40'] - 1).clip(-0.3, 0)
    scores['s_ma'] = (-ma_dev / 0.3 * 15).clip(0, 15)

    # 3. 缩量越狠越好 (量比 0.3→25分, 1.0→5分)
    vol_dev = (sel['volume'] / sel['vol_ma40']).clip(0.3, 1.0)
    scores['s_vol'] = (1 - (vol_dev - 0.3) / 0.7).clip(0, 1) * 25

    # 4. 低波动 (0%→20分, 3%→0分)
    scores['s_volat'] = (1 - sel['ret_vol_20'] / 0.03).clip(0, 1) * 20

    # 5. 无涨停加分 (0次→20分, 1次→10分)
    scores['s_lu'] = (2 - sel['lu_60d'].fillna(0).clip(0, 2)) / 2 * 20

    sel['bull_score'] = scores.sum(axis=1).round(1)
    sel['name'] = sel.index.map(name_map)
    sel['industry'] = sel.index.map(ind_map)
    sel['mcap_display'] = sel['mcap'].round(1)
    sel['ma_dev_pct'] = ((sel['close'] / sel['ma40'] - 1) * 100).round(1)
    sel['vol_ratio'] = (sel['volume'] / sel['vol_ma40']).round(2)
    sel['ret_vol_pct'] = (sel['ret_vol_20'] * 100).round(2)

    # 排序输出
    result = sel.nlargest(100, 'bull_score')[
        ['name', 'industry', 'close', 'mcap_display', 'bull_score',
         'ma_dev_pct', 'vol_ratio', 'ret_vol_pct', 'lu_60d']
    ].copy()
    result.columns = ['名称', '行业', '收盘价', '市值(亿)', '牛股评分',
                      'vsMA40(%)', '量比', '波动率(%)', '近期涨停次']
    result['代码'] = result.index

    return result


def main():
    p = argparse.ArgumentParser(description="牛股筛选器")
    p.add_argument("--date", default=None)
    p.add_argument("--top", type=int, default=30, help="输出Top-N")
    p.add_argument("--json", action="store_true", help="输出JSON")
    p.add_argument("--exclude-gem-star", action="store_true", help="排除创业/科创板(300/301/688)")
    p.add_argument("--ths", action="store_true", help="输出同花顺自选股导入格式")
    args = p.parse_args()

    df = screen(args.date, exclude_gem_star=args.exclude_gem_star)

    if df.empty:
        print("今日无符合条件的牛股候选")
        return

    print(f"\n{'='*80}")
    print(f"  牛股候选池 (缩量筑底) — {len(df)} 只" + (" [排除创业/科创]" if args.exclude_gem_star else ""))
    print(f"{'='*80}")
    print(df.head(args.top).to_string())

    # 同花顺自选股导入格式
    if args.ths:
        out_dir = "data/arsenal"
        os.makedirs(out_dir, exist_ok=True)
        date_tag = args.date or date.today().strftime("%Y%m%d")
        ths_path = f"{out_dir}/bull_ths_{date_tag}.txt"
        with open(ths_path, 'w') as f:
            for code in df.head(args.top).index:
                f.write(f"{code}\n")
        print(f"\n同花顺导入文件: {ths_path} (可直接导入自选股)")

    if args.json:
        import json
        out_dir = "data/arsenal"
        os.makedirs(out_dir, exist_ok=True)
        date_tag = args.date or date.today().strftime("%Y%m%d")
        path = f"{out_dir}/bull_signals_{date_tag}.json"
        df.head(args.top).to_json(path, orient='records', force_ascii=False, indent=2)
        print(f"\n已保存: {path}")


if __name__ == "__main__":
    main()
