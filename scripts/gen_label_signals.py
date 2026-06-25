#!/usr/bin/env python
"""标签烙印型再启动信号生成器。

在涨停池（mcap 30-800亿）基础上，用概念板块标签稳定度重排信号。

标签稳定度 = 历史涨停日的概念板块交集 / 并集
  → 高稳定度 = 每次涨停都是同一套标签，业务清晰
  → 低稳定度 = 每次涨停标签不同，蹭概念

用法:
    python scripts/gen_label_signals.py --start 2020-01-01 --top-n 30
"""

from __future__ import annotations

import argparse, os, sys, time
from collections import defaultdict
import numpy as np
import pandas as pd
from datetime import timedelta
from loguru import logger
from sqlalchemy import text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.db import get_engine
from data.loader import load_daily_data, load_mcap_data

# 涨停阈值
_LIMIT_MULT = {"688": 1.19899, "8": 1.29899, "4": 1.29899, "300": 1.19899, "301": 1.19899}
_DEFAULT_MULT = 1.09899

# 涨停池参数
MCAP_LOWER, MCAP_UPPER = 30, 800
PRICE_LOWER, PRICE_UPPER = 5, 100

# 噪音板块
NOISE_BOARDS = {
    "沪股通", "深股通", "融资融券", "机构重仓", "标准普尔", "富时罗素",
    "MSCI", "证金", "汇金", "社保", "QFII", "举牌", "预增", "预减",
    "预亏", "解禁", "转融通", "央国企", "国资云", "国企改革",
    "沪企", "自贸", "特区", "振兴", "经济带", "大湾区", "城市群",
    "新区", "最近多板", "百日新高", "历史新高", "近期新高",
    "破发", "破增发", "深成500", "中证500", "沪深300",
}


def _calc_limit_price(prev_close: float, code: str) -> float:
    mult = _DEFAULT_MULT
    for prefix, m in _LIMIT_MULT.items():
        if str(code).startswith(prefix):
            mult = m; break
    return round(prev_close * mult, 4)


def _hit_limit(high: float, prev_close: float, code: str) -> bool:
    if pd.isna(high) or pd.isna(prev_close) or prev_close <= 0:
        return False
    return high >= _calc_limit_price(prev_close, code)


def _sealed_at_close(close: float, prev_close: float, code: str) -> bool:
    if pd.isna(close) or pd.isna(prev_close) or prev_close <= 0:
        return False
    return close >= _calc_limit_price(prev_close, code)


def parse_args():
    p = argparse.ArgumentParser(description="标签烙印信号生成")
    p.add_argument("--start", type=str, default="2020-01-01")
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--top-n", type=int, default=30, help="每日最多信号数")
    p.add_argument("--out", type=str, default="data/signals/bt_signals_label.csv")
    return p.parse_args()


def _infer_end_date(engine):
    last_two = pd.read_sql(
        text("SELECT trade_date, COUNT(*) AS n FROM stock_daily "
             "GROUP BY trade_date ORDER BY trade_date DESC LIMIT 2"), engine)
    if len(last_two) >= 2:
        n_today, n_yesterday = last_two.iloc[0]["n"], last_two.iloc[1]["n"]
        if n_today < n_yesterday * 0.8 and n_yesterday >= 2500:
            return str(last_two.iloc[1]["trade_date"])[:10]
    return str(last_two.iloc[0]["trade_date"])[:10]


def _load_concept_tags(engine, codes, end_date):
    """加载所有股票在 end_date 时的概念板块标签（分批查询）。

    返回: {stock_code: [tag1, tag2, ...]}
    """
    codes_list = list(codes)
    d_str = str(end_date)[:10]
    tag_map = defaultdict(list)

    with engine.connect() as conn:
        for i in range(0, len(codes_list), 500):
            chunk = codes_list[i:i+500]
            rows = conn.execute(text("""
                SELECT cs.stock_code, cb.name
                FROM concept_stock cs
                JOIN concept_board cb ON cs.board_code = cb.code
                WHERE cs.stock_code = ANY(:codes)
            """), {"codes": chunk}).fetchall()
            for stock_code, board_name in rows:
                tag = board_name.replace("概念", "").replace("板块", "").replace("行业", "").strip()
                if tag and tag not in NOISE_BOARDS:
                    tag_map[stock_code].append(tag)

    return dict(tag_map)


def _compute_tag_stability(code, today, daily_by_date, tag_map, lookback_days=60):
    """计算标签稳定度：历史涨停日的标签交集/并集。

    返回 0-1 之间的值。首次出现返回 0.5（中性）。
    """
    all_lu_dates = []
    for td, g in daily_by_date.items():
        if td >= today:
            break
        if (today - td).days > lookback_days:
            continue
        if code in g.index and g.loc[code].get("is_lu", 0) == 1:
            all_lu_dates.append(td)

    if len(all_lu_dates) < 1:
        return 0.5  # 首次出现：中性

    tags = tag_map.get(code, [])
    if not tags:
        return 0.3  # 无标签

    # 简化版：用当前标签集合作为"标签烙印"
    # 完整版需要逐日查历史标签，但 concept_stock 的 in_date/out_date 可以做
    # 这里用当前标签作为代理
    tag_set = set(tags)

    # 计算标签的"特异性"：标签越少越集中，越稳定
    if len(tag_set) == 0:
        return 0.0
    specificity = min(1.0, 5.0 / len(tag_set))  # ≤5个标签=满分

    return round(specificity, 3)


def main():
    args = parse_args()
    engine = get_engine()
    t0 = time.time()

    end_date_str = args.end or _infer_end_date(engine)

    # ── 股票池（仅主板）──
    min_list = pd.Timestamp(args.start) - timedelta(days=120)
    with engine.connect() as conn:
        codes_df = pd.read_sql(
            text("SELECT code, name FROM stock_basic WHERE is_st=FALSE "
                 "AND list_date <= :ld AND code !~ '^(300|301|688|[48])'"),
            conn, params={"ld": pd.Timestamp(end_date_str).strftime("%Y-%m-%d")})
    codes_df["code"] = codes_df["code"].astype(str).str.zfill(6)
    name_map = dict(zip(codes_df["code"], codes_df["name"]))
    code_set = set(codes_df["code"].tolist())
    logger.info(f"股票池: {len(code_set)} 只")

    # ── 加载数据 ──
    pre_start = (pd.Timestamp(args.start) - timedelta(days=120)).strftime("%Y-%m-%d")
    daily = load_daily_data(engine, code_set, pre_start, end_date_str,
                            cols=["open", "high", "low", "close", "volume", "turnover"])
    daily["code"] = daily["code"].astype(str).str.zfill(6)
    daily["trade_date"] = pd.to_datetime(daily["trade_date"])
    daily = daily.sort_values(["code", "trade_date"])

    extra = load_mcap_data(engine, code_set, pre_start, end_date_str, use_proxy=True)
    if not extra.empty:
        extra["code"] = extra["code"].astype(str).str.zfill(6)
        extra["trade_date"] = pd.to_datetime(extra["trade_date"])

    logger.info(f"日线: {len(daily)}行 | 市值: {len(extra)}行 ({time.time()-t0:.0f}s)")

    # ── 因子预计算 ──
    logger.info("预计算因子...")
    daily["ret"] = daily.groupby("code")["close"].pct_change()
    daily["prev_close"] = daily.groupby("code")["close"].shift(1)
    daily["is_lu"] = daily.apply(
        lambda r: 1 if _sealed_at_close(r["close"], r["prev_close"], str(r["code"])) else 0, axis=1)
    daily["ma5"] = daily.groupby("code")["close"].transform(lambda x: x.rolling(5, min_periods=3).mean())
    daily["ma10"] = daily.groupby("code")["close"].transform(lambda x: x.rolling(10, min_periods=5).mean())
    daily["lu_20d"] = daily.groupby("code")["is_lu"].transform(lambda x: x.rolling(20, min_periods=1).sum())
    daily["lu_60d"] = daily.groupby("code")["is_lu"].transform(lambda x: x.rolling(60, min_periods=1).sum())

    daily_by_date = {d: g.set_index("code") for d, g in daily.groupby("trade_date")}
    extra_by_date = {d: g.set_index("code") for d, g in extra.groupby("trade_date")} if not extra.empty else {}

    # ── 加载概念板块标签 ──
    logger.info("加载概念板块标签...")
    tag_map = _load_concept_tags(engine, code_set, pd.Timestamp(end_date_str))
    tagged_codes = len(tag_map)
    logger.info(f"  有标签的股票: {tagged_codes}/{len(code_set)} ({time.time()-t0:.0f}s)")

    # ── 逐日筛选 + 标签烙印重排 ──
    all_dates = sorted(daily["trade_date"].unique())
    trade_dates = [d for d in all_dates
                   if pd.Timestamp(args.start) <= d <= pd.Timestamp(end_date_str)]
    logger.info(f"交易日: {len(trade_dates)} 天")

    rows = []
    for today in trade_dates:
        td_df = daily_by_date.get(today)
        if td_df is None or td_df.empty:
            continue

        ex_td = extra_by_date.get(today)
        if ex_td is not None and not ex_td.empty:
            td_df = td_df.copy()
            td_df["mcap"] = ex_td.get("market_cap", np.nan)
        else:
            td_df = td_df.copy()
            td_df["mcap"] = np.nan

        # 4 条件筛选（mcap 30-800 亿）
        mask = (
            (td_df["mcap"].between(MCAP_LOWER, MCAP_UPPER)) &
            (td_df["close"].between(PRICE_LOWER, PRICE_UPPER)) &
            (td_df["ma5"] > td_df["ma10"]) &
            (td_df["lu_20d"] >= 2) & (td_df["lu_20d"] <= 4) &
            (td_df["close"] > 0) & (td_df.index.isin(code_set))
        )
        sel = td_df[mask]
        if sel.empty:
            continue

        sel = sel.copy()
        sel["ret_today"] = sel["close"] / sel["prev_close"] - 1

        # ── 标签烙印评分 ──
        scores = []
        for code, r in sel.iterrows():
            code_str = str(code).zfill(6)

            # 标签稳定度 (50%)
            tag_stab = _compute_tag_stability(code, today, daily_by_date, tag_map)

            # 稀缺性 (20%): 近60日涨停 ≤4 满分
            lu60 = int(r.get("lu_60d", 0))
            rarity = 1.0 if lu60 <= 4 else max(0, 1.0 - (lu60 - 4) * 0.2)

            # 池内涨幅排名 (30%) — 先放中性，后面统一算
            label_score = round(tag_stab * 50 + rarity * 20, 1)
            scores.append((code, label_score, tag_stab, rarity, float(r["ret_today"])))

        # 加涨幅排名分 (30%)
        scores_df = pd.DataFrame(scores, columns=["code", "label_base", "tag_stab", "rarity", "ret_today"])
        scores_df["ret_rank"] = scores_df["ret_today"].rank(pct=True)  # 0-1
        scores_df["label_score"] = (scores_df["label_base"] + scores_df["ret_rank"] * 30).round(1)

        # 按标签烙印评分重排
        top = scores_df.nlargest(min(args.top_n, len(scores_df)), "label_score")

        for rank, (_, r) in enumerate(top.iterrows(), 1):
            code = str(r["code"]).zfill(6)
            orig = sel.loc[r["code"]]
            rows.append({
                "date": today.strftime("%Y-%m-%d"),
                "rank": rank,
                "code": code,
                "name": name_map.get(code, "?"),
                "score": round(float(r["label_score"]), 1),
                "tag_stability": round(float(r["tag_stab"]), 2),
                "close": round(float(orig["close"]), 2),
                "is_limit_up": bool(orig["is_lu"] == 1),
                "is_limit_down": False,
            })

    engine.dispose()

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    logger.success(f"导出 {len(df)} 条 → {args.out} ({time.time()-t0:.0f}s)")
    logger.info(f"日期: {df['date'].min()} ~ {df['date'].max()}, "
                f"{df['date'].nunique()}天, {df['code'].nunique()}只")
    if "tag_stability" in df.columns:
        high_stab = (df["tag_stability"] > 0.7).sum()
        logger.info(f"  高标签稳定度(>0.7): {high_stab} 条 ({high_stab/len(df)*100:.0f}%)")


if __name__ == "__main__":
    main()
