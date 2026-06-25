#!/usr/bin/env python
"""涨停策略筛选脚本 —— 路径A：独立规则筛选，不依赖 ML 训练管线。

条件（5 选 4 即可）：
  1. 市值 50–300 亿（stock_daily_extra.market_cap）
  2. 股价 5–50 元（stock_daily.close）
  3. MA5 > MA10（均线短多）
  4. 近一个月涨停次数 > 1（日收益 ≥ 10%，统一阈值）
  5. 近 10 个交易日无跌停（日收益 > −10%）

用法:
    python scripts/scan_limit_up_strategy.py              # 默认参数
    python scripts/scan_limit_up_strategy.py --min-conditions 5  # 5条全满足
    python scripts/scan_limit_up_strategy.py --mcap-min 30 --mcap-max 200
    python scripts/scan_limit_up_strategy.py --limit-up-lookback 10 --limit-up-count 2
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import date, timedelta
from loguru import logger

from data.db import get_engine


def parse_args():
    p = argparse.ArgumentParser(description="涨停策略条件筛选")
    # 市值
    p.add_argument("--mcap-min", type=float, default=50.0, help="最小市值（亿元）")
    p.add_argument("--mcap-max", type=float, default=300.0, help="最大市值（亿元）")
    # 股价
    p.add_argument("--price-min", type=float, default=5.0, help="最低股价（元）")
    p.add_argument("--price-max", type=float, default=50.0, help="最高股价（元）")
    # 涨停
    p.add_argument("--limit-up-pct", type=float, default=0.099,
                   help="涨停收益阈值（默认 9.9%%，覆盖四舍五入误差）")
    p.add_argument("--limit-up-lookback", type=int, default=20,
                   help="涨停回看交易日数（≈一个月）")
    p.add_argument("--limit-up-count", type=int, default=1,
                   help="涨停次数 > 该值（默认 >1 即至少 2 次）")
    # 跌停
    p.add_argument("--limit-down-pct", type=float, default=-0.099,
                   help="跌停收益阈值（默认 −9.9%%）")
    p.add_argument("--limit-down-lookback", type=int, default=10,
                   help="跌停回看交易日数")
    # 放宽
    p.add_argument("--min-conditions", type=int, default=4,
                   help="最少满足条件数（默认 4，即 5 选 4）")
    # 其他
    p.add_argument("--min-listed-days", type=int, default=60,
                   help="最少上市天数（保证均线可算）")
    p.add_argument("--top-n", type=int, default=50,
                   help="输出前 N 只（按涨停次数降序）")
    return p.parse_args()


def main():
    args = parse_args()
    engine = get_engine()

    # ── 1. 取最新交易日 ──
    latest = pd.read_sql(
        "SELECT MAX(trade_date) AS d FROM stock_daily", engine
    ).iloc[0, 0]
    if latest is None:
        logger.error("stock_daily 无数据，请先同步。")
        return
    logger.info(f"最新交易日: {latest}")

    # ── 2. 圈定候选股票池（非 ST，上市 ≥ min_listed_days）──
    min_list_date = latest - timedelta(days=args.min_listed_days)
    codes_df = pd.read_sql(
        f"SELECT code, name FROM stock_basic "
        f"WHERE is_st = FALSE AND list_date <= '{min_list_date}'",
        engine,
    )
    if codes_df.empty:
        logger.error("无符合条件的股票。")
        return
    code_set = set(codes_df["code"].tolist())
    code_to_name = dict(zip(codes_df["code"], codes_df["name"]))
    logger.info(f"候选池: {len(code_set)} 只（排除 ST + 上市不足 {args.min_listed_days} 天）")

    # ── 3. 拉取日线（均线 + 涨跌停判定都从这里出）──
    data_start = latest - timedelta(days=max(args.limit_up_lookback, args.limit_down_lookback) + 30)
    code_tuple = tuple(code_set)
    daily = pd.read_sql(
        f"SELECT code, trade_date, close FROM stock_daily "
        f"WHERE code IN ({','.join(['%s']*len(code_tuple))}) "
        f"AND trade_date BETWEEN %s AND %s "
        f"ORDER BY code, trade_date",
        engine,
        params=(*code_tuple, data_start, latest),
    )
    logger.info(f"日线数据: {len(daily)} 行, {daily['code'].nunique()} 只")

    # 按 code + trade_date 排序后算收益率 & 均线
    daily = daily.sort_values(["code", "trade_date"]).reset_index(drop=True)

    # 日收益率
    daily["ret"] = daily.groupby("code")["close"].pct_change()

    # MA5 / MA10
    daily["ma5"] = daily.groupby("code")["close"].transform(lambda x: x.rolling(5, min_periods=5).mean())
    daily["ma10"] = daily.groupby("code")["close"].transform(lambda x: x.rolling(10, min_periods=10).mean())

    # ── 4. 取最新一天的基础数据 ──
    latest_data = daily[daily["trade_date"] == latest].copy()
    latest_data = latest_data.set_index("code")
    logger.info(f"最新日有交易: {len(latest_data)} 只")

    # ── 5. 市值数据（取最近可用日期，允许与 latest 有几日偏差）──
    extra_latest = pd.read_sql(
        "SELECT MAX(trade_date) FROM stock_daily_extra WHERE trade_date <= %s",
        engine, params=(latest,),
    ).iloc[0, 0]
    extra = pd.read_sql(
        "SELECT code, market_cap FROM stock_daily_extra WHERE trade_date = %s",
        engine, params=(extra_latest,),
    )
    extra = extra.set_index("code")
    logger.info(f"市值数据: {len(extra)} 只（日期: {extra_latest}，滞后 {(latest - extra_latest).days} 天）")

    # ── 6. 计算各条件 ──

    # 均线数据
    latest_ma = latest_data[["close", "ma5", "ma10"]].copy()

    # 回看期涨跌停统计
    lu_lb = args.limit_up_lookback
    ld_lb = args.limit_down_lookback
    lu_cutoff = latest - timedelta(days=lu_lb + 5)  # 宽松一点，SQL 过滤后再精确截断

    lookback = daily[daily["trade_date"] >= latest - timedelta(days=max(lu_lb, ld_lb) + 5)].copy()

    # 涨停次数（近 lookback 个交易日内）
    lu_mask = lookback["ret"] >= args.limit_up_pct
    limit_up_counts = lookback[lu_mask].groupby("code").size().rename("limit_up_n")

    # 跌停标记（近 lookback 个交易日内，只要有一次就算）
    ld_mask = lookback["ret"] <= args.limit_down_pct
    limit_down_codes = set(lookback[ld_mask]["code"].unique())

    # ── 7. 组装结果 & 判定条件 ──
    results = []
    for code in latest_ma.index:
        if code not in code_set:
            continue

        row = latest_ma.loc[code]
        close_p = row.get("close")
        ma5 = row.get("ma5")
        ma10 = row.get("ma10")

        if pd.isna(close_p) or close_p <= 0:
            continue

        # 市值
        mcap = extra.loc[code, "market_cap"] if code in extra.index else np.nan
        if pd.isna(mcap):
            mcap = np.nan

        # 条件判定
        conditions = []
        details = {}

        # 条件1: 市值 50-300 亿
        c1 = (not pd.isna(mcap)) and (args.mcap_min <= mcap <= args.mcap_max)
        conditions.append(c1)
        details["mcap"] = mcap

        # 条件2: 股价 5-50 元
        c2 = args.price_min <= close_p <= args.price_max
        conditions.append(c2)
        details["price"] = close_p

        # 条件3: MA5 > MA10
        c3 = (not pd.isna(ma5)) and (not pd.isna(ma10)) and (ma5 > ma10)
        conditions.append(c3)
        details["ma5"] = ma5
        details["ma10"] = ma10

        # 条件4: 近一个月涨停次数 > 1
        lu_n = int(limit_up_counts.get(code, 0))
        c4 = lu_n > args.limit_up_count
        conditions.append(c4)
        details["limit_up_n"] = lu_n

        # 条件5: 近10日无跌停
        c5 = code not in limit_down_codes
        conditions.append(c5)
        details["has_limit_down"] = not c5

        n_pass = sum(conditions)
        if n_pass >= args.min_conditions:
            name = code_to_name.get(code, "")
            results.append({
                "code": code,
                "name": name,
                "close": round(close_p, 2),
                "mcap": round(mcap, 1) if not pd.isna(mcap) else None,
                "ma5": round(ma5, 2) if not pd.isna(ma5) else None,
                "ma10": round(ma10, 2) if not pd.isna(ma10) else None,
                "limit_up_n": lu_n,
                "has_limit_down": not c5,
                "n_pass": n_pass,
                "cond_str": "".join(["✓" if c else "✗" for c in conditions]),
            })

    # ── 8. 排序 & 输出 ──
    df = pd.DataFrame(results)
    if df.empty:
        logger.warning("无股票满足条件，尝试放宽参数（如 --min-conditions 3）。")
        return

    # 按涨停次数降序，再按通过条件数降序
    df = df.sort_values(["n_pass", "limit_up_n"], ascending=[False, False]).head(args.top_n)

    print(f"\n{'='*90}")
    print(f"  涨停策略筛选结果（{latest}）")
    print(f"  条件: 市值{args.mcap_min}-{args.mcap_max}亿 | 股价{args.price_min}-{args.price_max}元")
    print(f"        MA5>MA10 | 近{lu_lb}日涨停>{args.limit_up_count}次 | 近{ld_lb}日无跌停")
    print(f"  满足 {args.min_conditions}/5 即可 | 共筛选出 {len(df)} 只")
    print(f"{'='*90}")
    print(f"{'代码':<8} {'名称':<8} {'收盘':>7} {'市值(亿)':>10} {'MA5':>7} {'MA10':>7} "
          f"{'涨停次':>5} {'跌停':>4} {'通过':>4}  {'条件明细'}")
    print(f"{'-'*90}")

    for _, r in df.iterrows():
        mcap_str = f"{r['mcap']:,.1f}" if r['mcap'] and not pd.isna(r['mcap']) else "N/A"
        print(f"{r['code']:<8} {r['name']:<8} {r['close']:>7.2f} {mcap_str:>10} "
              f"{r['ma5']:>7.2f} {r['ma10']:>7.2f} "
              f"{r['limit_up_n']:>5} {'是' if r['has_limit_down'] else '否':>4} "
              f"{r['n_pass']:>4}  [{r['cond_str']}]")

    print(f"{'='*90}")

    # 条件通过率统计
    cond_cols = ["市值50-300亿", "股价5-50元", "MA5>MA10", "涨停>1次", "无跌停"]
    for i, label in enumerate(cond_cols):
        passed = df["cond_str"].str[i].eq("✓").sum()
        print(f"  {label}: {passed}/{len(df)} ({passed/len(df)*100:.0f}%)")

    return df


if __name__ == "__main__":
    main()
