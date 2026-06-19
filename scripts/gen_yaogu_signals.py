#!/usr/bin/env python
"""妖股池信号生成器 —— 6规则评分，对齐管线 CSV 格式。

在涨停日评估: 一字板(+3) + 低振幅<8%(+2) + 缩量板(+1) + 非量能极值(+1) + 连板≥2(+1) + 缩量整理(+1)
评分 ≥ min_score 入选。

用法:
    python scripts/gen_yaogu_signals.py --start 2020-01-01 --min-score 3
"""

from __future__ import annotations

import argparse, os, sys, time
import numpy as np
import pandas as pd
from datetime import timedelta
from loguru import logger
from sqlalchemy import text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.db import get_engine
from data.loader import load_daily_data

# 涨停阈值
_DEFAULT_MULT = 1.9899

def _is_at_limit_up(close, prev_close, code):
    if pd.isna(close) or pd.isna(prev_close) or prev_close <= 0:
        return False
    return close >= round(prev_close * 1.9899, 4)


def parse_args():
    p = argparse.ArgumentParser(description="妖股池信号生成")
    p.add_argument("--start", type=str, default="2020-01-01")
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--min-score", type=int, default=3, help="最低妖股评分")
    p.add_argument("--top-n", type=int, default=30, help="每日最多信号数")
    p.add_argument("--out", type=str, default="data/signals/bt_signals_yaogu.csv")
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


def main():
    args = parse_args()
    engine = get_engine()
    t0 = time.time()

    end_date_str = args.end or _infer_end_date(engine)

    # ── 股票池（仅主板）──
    min_list = pd.Timestamp(args.start) - timedelta(days=120)
    with engine.connect() as conn:
        codes_df = pd.read_sql(
            text("SELECT code, name FROM stock_basic WHERE is_st=FALSE AND list_date <= :ld AND code !~ '^(300|301|688|[48])'"),
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
    logger.info(f"日线: {len(daily)}行 ({time.time()-t0:.0f}s)")

    # ── 因子预计算 ──
    logger.info("预计算因子...")
    daily["ret"] = daily.groupby("code")["close"].pct_change()
    daily["prev_close"] = daily.groupby("code")["close"].shift(1)
    daily["is_lu"] = daily.apply(
        lambda r: 1 if _is_at_limit_up(r["close"], r["prev_close"], str(r["code"])) else 0,
        axis=1)
    daily["vol_ma20"] = daily.groupby("code")["volume"].transform(lambda x: x.rolling(20, min_periods=5).mean())
    daily["vol_std20"] = daily.groupby("code")["volume"].transform(lambda x: x.rolling(20, min_periods=5).std())
    daily["hl_range"] = daily["high"] - daily["low"]
    daily["amplitude"] = np.where(
        daily["is_lu"] == 1,
        daily["hl_range"] / daily["prev_close"].replace(0, np.nan), np.nan)

    # 连板
    def _calc_streak(s):
        cnt, res = 0, []
        for v in s:
            cnt = cnt + 1 if v else 0
            res.append(cnt)
        return pd.Series(res, index=s.index)
    daily["lu_streak"] = daily.groupby("code")["is_lu"].transform(_calc_streak)

    # 预分组
    daily_by_date = {d: g.set_index("code") for d, g in daily.groupby("trade_date")}
    logger.info(f"因子完成 ({time.time()-t0:.0f}s)")

    # ── 逐日筛选 ──
    all_dates = sorted(daily["trade_date"].unique())
    trade_dates = [d for d in all_dates
                   if pd.Timestamp(args.start) <= d <= pd.Timestamp(end_date_str)]
    logger.info(f"交易日: {len(trade_dates)} 天")

    rows = []
    for today in trade_dates:
        td_df = daily_by_date.get(today)
        if td_df is None or td_df.empty:
            continue

        # 只评涨停股
        lu_today = td_df[td_df["is_lu"] == 1]
        if lu_today.empty:
            continue

        # 6规则评分
        for code, r in lu_today.iterrows():
            yiziban = 1 if abs(r["hl_range"]) < r["close"] * 0.001 else 0
            amp_val = float(r["amplitude"]) if pd.notna(r.get("amplitude")) else 1.0
            vol_avg = float(r["vol_ma20"]) if pd.notna(r.get("vol_ma20")) and r["vol_ma20"] > 0 else 1.0
            vol_std = float(r["vol_std20"]) if pd.notna(r.get("vol_std20")) else 0
            vol_intensity = r["volume"] / vol_avg if vol_avg > 0 else 1
            vol_climax = r["volume"] / (vol_avg + 2 * vol_std) if (vol_avg + 2 * vol_std) > 0 else 1
            streak_val = int(r.get("lu_streak", 0))

            score = 0
            if yiziban: score += 3
            if pd.notna(amp_val) and amp_val < 0.08: score += 2
            if pd.notna(vol_intensity) and vol_intensity < 1.5: score += 1
            if pd.notna(vol_climax) and vol_climax < 0.8: score += 1
            if streak_val >= 2: score += 1

            if score >= args.min_score:
                ret_today = float(r["ret"]) if pd.notna(r.get("ret")) else 0
                rows.append({
                    "date": today.strftime("%Y-%m-%d"),
                    "code": str(code).zfill(6),
                    "score": score,
                    "ret_today": round(ret_today * 100, 1),
                })

    engine.dispose()

    if not rows:
        logger.warning("无信号")
        return

    df_raw = pd.DataFrame(rows)
    # 每日按 score 降序，取 top_n
    df_raw["rank"] = df_raw.groupby("date")["score"].rank(method="dense", ascending=False).astype(int)
    df_raw = df_raw[df_raw["rank"] <= args.top_n].copy()
    df_raw = df_raw.sort_values(["date", "score"], ascending=[True, False])

    # 组装标准输出
    out_rows = []
    for _, r in df_raw.iterrows():
        out_rows.append({
            "date": r["date"],
            "rank": int(r["rank"]),
            "code": r["code"],
            "name": name_map.get(r["code"], "?"),
            "score": float(r["score"]),
            "close": 0.0,  # 信号日是涨停日，实际买入在第一个非涨停日
            "is_limit_up": True,
            "is_limit_down": False,
        })

    df_out = pd.DataFrame(out_rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df_out.to_csv(args.out, index=False, encoding="utf-8-sig")
    logger.success(f"导出 {len(df_out)} 条 → {args.out} ({time.time()-t0:.0f}s)")
    logger.info(f"日期: {df_out['date'].min()} ~ {df_out['date'].max()}, "
                f"{df_out['date'].nunique()}天, {df_out['code'].nunique()}只")


if __name__ == "__main__":
    main()
