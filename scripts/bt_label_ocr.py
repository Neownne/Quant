#!/usr/bin/env python
"""标签烙印型再启动回测 — 基于 OCR 关键词 + 涨停池条件。

标签稳定度 = 历史涨停日的 OCR 关键词交集 / 并集
策略: 涨停池(mcap 30-500亿) → 标签烙印重排 → T+2 入场 → 妖股出场规则

用法:
    python scripts/bt_label_ocr.py --top-n 5
"""
import sys, os, argparse, time
from collections import defaultdict
import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy import text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.db import get_engine
from data.loader import load_daily_data
from config.settings import TradingConfig

# ── 参数 ──
MASTER_CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'data', 'arsenal', 'hotpoint', 'master.csv')
REBALANCE_DAYS = 5
MAX_WAIT = 2
MCAP_LOWER, MCAP_UPPER = 30, 500
DD_THRESHOLD = 0.25


def parse_args():
    p = argparse.ArgumentParser(description="标签烙印 OCR 回测")
    p.add_argument("--top-n", type=int, default=5)
    p.add_argument("--cash", type=float, default=1_000_000)
    p.add_argument("--label", default="label_ocr")
    return p.parse_args()


def _compute_tag_stability(code, today, keyword_history):
    """标签稳定度: 历史关键词交集/并集。"""
    if code not in keyword_history:
        return 0.5  # 首次出现
    tags_list = keyword_history[code]
    if len(tags_list) < 2:
        return 0.5
    # 最新一次 vs 之前所有次的交集/并集
    latest = tags_list[-1]
    earlier = set()
    for tags in tags_list[:-1]:
        earlier.update(tags)
    if not earlier or not latest:
        return 0.5
    intersection = latest & earlier
    union = latest | earlier
    return round(len(intersection) / len(union), 3) if union else 0.5


def main():
    args = parse_args()
    engine = get_engine()
    t0 = time.time()

    # ── 1. 加载 OCR 关键词数据 ──
    logger.info("加载 OCR 数据...")
    ocr_df = pd.read_csv(MASTER_CSV, dtype={'code': str, 'date': str})
    ocr_df['date_dt'] = pd.to_datetime(ocr_df['date'])
    ocr_df['keyword_set'] = ocr_df['keywords'].apply(
        lambda x: set(str(x).split('|')) if pd.notna(x) and str(x) != 'nan' else set())

    # 构建每只股票的关键词历史（按日期排序）
    keyword_history = defaultdict(list)
    for code, grp in ocr_df.groupby('code'):
        sorted_grp = grp.sort_values('date_dt')
        for _, row in sorted_grp.iterrows():
            keyword_history[code].append(row['keyword_set'])

    ocr_dates = sorted(ocr_df['date'].unique())
    ocr_codes = set(ocr_df['code'].unique())
    logger.info(f"OCR: {len(ocr_df)} 条, {len(ocr_dates)} 天, {len(ocr_codes)} 只")
    logger.info(f"日期: {ocr_dates[0]} ~ {ocr_dates[-1]}")

    # ── 2. 加载日线 + 市值 ──
    logger.info("加载日线数据...")
    all_codes = sorted(ocr_codes)
    start_d = ocr_dates[0]
    end_d = ocr_dates[-1]

    # Pre-load with 120-day lookback
    pre_start = (pd.Timestamp(start_d) - pd.Timedelta(days=120)).strftime('%Y-%m-%d')
    daily = load_daily_data(engine, all_codes, pre_start, end_d,
                            cols=["open", "high", "low", "close", "volume", "turnover"])
    daily["code"] = daily["code"].astype(str).str.zfill(6)
    daily["trade_date"] = pd.to_datetime(daily["trade_date"])
    daily = daily.sort_values(["code", "trade_date"])

    # Load market cap
    with engine.connect() as conn:
        extra = pd.read_sql(text("""
            SELECT code, trade_date, market_cap FROM stock_daily_extra
            WHERE code = ANY(:codes) AND trade_date BETWEEN :start AND :end
        """), conn, params={"codes": all_codes, "start": pre_start, "end": end_d})
    extra["code"] = extra["code"].astype(str).str.zfill(6)
    extra["trade_date"] = pd.to_datetime(extra["trade_date"])

    logger.info(f"日线: {len(daily)}行, 市值: {len(extra)}行 ({time.time()-t0:.0f}s)")

    # ── 3. 因子预计算 ──
    logger.info("预计算因子...")
    daily["prev_close"] = daily.groupby("code")["close"].shift(1)
    daily["ret"] = daily.groupby("code")["close"].pct_change()

    # Limit-up detection (sealed at close)
    def _mult(c):
        c = str(c).zfill(6)
        for pfx, m in [("688",1.19899),("300",1.19899),("301",1.19899),("4",1.29899),("8",1.29899)]:
            if c.startswith(pfx): return m
        return 1.09899

    daily["limit_price"] = daily.apply(
        lambda r: round(r["prev_close"] * _mult(r["code"]), 4)
        if pd.notna(r["prev_close"]) and r["prev_close"] > 0 else 0, axis=1)
    daily["is_lu"] = (daily["close"] >= daily["limit_price"]).astype(int)
    daily["ma5"] = daily.groupby("code")["close"].transform(lambda x: x.rolling(5, min_periods=3).mean())
    daily["ma10"] = daily.groupby("code")["close"].transform(lambda x: x.rolling(10, min_periods=5).mean())
    daily["ma20"] = daily.groupby("code")["close"].transform(lambda x: x.rolling(20, min_periods=5).mean())
    daily["lu_20d"] = daily.groupby("code")["is_lu"].transform(lambda x: x.rolling(20, min_periods=1).sum())

    daily_by_date = {d: g.set_index("code") for d, g in daily.groupby("trade_date")}
    extra_by_date = {d: g.set_index("code") for d, g in extra.groupby("trade_date")} if not extra.empty else {}

    # ── 4. 逐日生成信号 ──
    logger.info("生成标签烙印信号...")
    all_trade_dates = sorted(daily["trade_date"].unique())
    start_dt, end_dt = pd.Timestamp(start_d), pd.Timestamp(end_d)
    trade_dates = [d for d in all_trade_dates if start_dt <= d <= end_dt]

    signal_rows = []
    for today in trade_dates:
        td_str = str(today.date())
        td_df = daily_by_date.get(today)
        if td_df is None or td_df.empty:
            continue

        ex_td = extra_by_date.get(today)
        if ex_td is not None and not ex_td.empty:
            td_df = td_df.copy()
            td_df["mcap"] = ex_td.get("market_cap", np.nan)
        else:
            continue

        # 涨停池 4 条件
        mask = (
            td_df["mcap"].between(MCAP_LOWER, MCAP_UPPER) &
            td_df["close"].between(5, 100) &
            (td_df["ma5"] > td_df["ma10"]) &
            (td_df["lu_20d"] >= 2) & (td_df["lu_20d"] <= 4) &
            (td_df["close"] > 0)
        )
        sel = td_df[mask]
        if sel.empty:
            continue

        # 只用在 OCR 数据中的股票
        sel = sel[sel.index.isin(ocr_codes)]
        if sel.empty:
            continue

        sel = sel.copy()
        sel["ret_today"] = sel["close"] / sel["prev_close"] - 1

        # 标签烙印评分
        scores = []
        for code, r in sel.iterrows():
            code_str = str(code).zfill(6)
            tag_stab = _compute_tag_stability(code_str, td_str, keyword_history)
            tag_score = tag_stab * 50  # 标签稳定度 50%

            # 稀缺性 20%
            lu20 = int(r.get("lu_20d", 0))
            rarity = 1.0 if lu20 <= 4 else max(0, 1.0 - (lu20 - 4) * 0.2)
            rarity_score = rarity * 20

            scores.append((code, tag_score + rarity_score, tag_stab, rarity, float(r["ret_today"])))

        scores_df = pd.DataFrame(scores, columns=["code", "label_base", "tag_stab", "rarity", "ret_today"])
        scores_df["ret_rank"] = scores_df["ret_today"].rank(pct=True)
        scores_df["label_score"] = (scores_df["label_base"] + scores_df["ret_rank"] * 30).round(1)

        top = scores_df.nlargest(min(args.top_n, len(scores_df)), "label_score")

        for rank, (_, s) in enumerate(top.iterrows(), 1):
            code = str(s["code"]).zfill(6)
            r = sel.loc[s["code"]]
            signal_rows.append({
                "date": td_str,
                "rank": rank,
                "code": code,
                "name": "",
                "score": round(float(s["label_score"]), 1),
                "tag_stability": round(float(s["tag_stab"]), 2),
                "close": round(float(r["close"]), 2),
                "is_limit_up": bool(r["is_lu"] == 1),
                "is_limit_down": False,
            })

    sig_df = pd.DataFrame(signal_rows)
    if sig_df.empty:
        logger.error("无信号生成")
        return

    sig_df["date"] = pd.to_datetime(sig_df["date"])
    sig_df["code"] = sig_df["code"].astype(str).str.zfill(6)

    # Add names
    with engine.connect() as conn:
        names = pd.read_sql(text("SELECT code, name FROM stock_basic WHERE code = ANY(:codes)"),
                            conn, params={"codes": sig_df["code"].unique().tolist()})
    name_map = dict(zip(names["code"].astype(str).str.zfill(6), names["name"]))
    sig_df["name"] = sig_df["code"].map(name_map).fillna("?")

    logger.info(f"信号: {len(sig_df)} 条, {sig_df['date'].nunique()} 天, {sig_df['code'].nunique()} 只")

    # ── 5. 回测（复用 bt_label 逻辑）──
    logger.info("=" * 60)
    logger.info("回测...")

    all_dates = sorted(daily_by_date.keys())
    date_idx = {d: i for i, d in enumerate(all_dates)}

    # 信号 → 买入窗口
    sig_by_date = {}
    for d, g in sig_df.groupby("date"):
        sig_by_date[d] = g.sort_values("score", ascending=False)

    buy_signals = []
    for sig_date, sigs in sig_by_date.items():
        idx = date_idx.get(sig_date)
        if idx is None:
            continue
        for _, s in sigs.iterrows():
            code = s["code"]
            bought = False
            for offset in range(1, MAX_WAIT + 1):
                nxt = idx + offset
                if nxt >= len(all_dates):
                    break
                nd = all_dates[nxt]
                ndf = daily_by_date.get(nd)
                if ndf is None or code not in ndf.index:
                    continue
                r = ndf.loc[code]
                px, prev_c = r["close"], r.get("prev_close")
                if pd.notna(prev_c) and prev_c > 0:
                    if TradingConfig.is_at_limit_up(px, prev_c, code):
                        continue
                    if TradingConfig.is_at_limit_down(px, prev_c, code):
                        continue
                buy_signals.append({
                    "date": nd, "code": code, "score": int(s["score"]),
                    "close": float(px), "signal_date": sig_date, "wait_days": offset,
                })
                bought = True
                break

    buy_df = pd.DataFrame(buy_signals)
    sig_by_date_buy = {}
    for d, g in buy_df.groupby("date"):
        sig_by_date_buy[d] = g.sort_values("score", ascending=False)

    logger.info(f"买入信号: {len(sig_df)}条 → 可买入{len(buy_df)}条")

    # ── 回测循环 ──
    cash = args.cash
    positions = {}
    equity, trade_log = [], []
    trade_count = 0

    NET_SELL = 1.0 - TradingConfig.SLIPPAGE - TradingConfig.COMMISSION - TradingConfig.STAMP_DUTY
    BUY_COST = 1.0 + TradingConfig.COMMISSION + TradingConfig.SLIPPAGE
    peak_value, frozen, frozen_days = args.cash, False, 0

    # 名字映射
    name_map_full = dict(zip(names["code"].astype(str).str.zfill(6), names["name"]))

    for i, td in enumerate(all_dates):
        td_df = daily_by_date.get(td)
        if td_df is None:
            continue
        px_map = td_df["close"].to_dict()
        prev_map = {c: r["prev_close"] for c, r in td_df.iterrows() if pd.notna(r.get("prev_close"))}
        ma20_map = td_df["ma20"].to_dict()

        # 更新持仓
        for code, pos in list(positions.items()):
            cur_px = px_map.get(code, pos["entry_price"])
            pos["current_price"], pos["hold_days"] = cur_px, pos["hold_days"] + 1
            if cur_px > pos.get("peak_price", 0):
                pos["peak_price"] = cur_px

        pos_val = sum(p["shares"] * p.get("current_price", p["entry_price"]) for p in positions.values())
        total = cash + pos_val
        equity.append({"date": td.strftime("%Y-%m-%d"), "value": round(total, 2), "cash": round(cash, 2)})

        # 熔断
        if total > peak_value:
            peak_value = total
        dd = (peak_value - total) / peak_value if peak_value > 0 else 0
        if dd > DD_THRESHOLD and not frozen:
            frozen, frozen_days = True, 0
            logger.info(f"  [{td.strftime('%Y-%m-%d')}] DD {dd:.1%} 熔断")
        if frozen:
            frozen_days += 1
        if frozen and frozen_days > 60:
            frozen, peak_value = False, total

        # 退出检查
        for code, pos in list(positions.items()):
            cur_px, sell_reason = pos["current_price"], None

            ma20 = ma20_map.get(code)
            if ma20 and cur_px < ma20 and pos["hold_days"] > 5:
                sell_reason = "破MA20"
            elif pos["hold_days"] >= 7 and pos.get("peak_price", 0) > pos["entry_price"] * 1.05:
                if cur_px < pos["peak_price"] * 0.88:
                    sell_reason = "移动止盈"

            code_ret = daily[(daily["code"] == code) & (daily["trade_date"] <= td)]
            stock_vol = code_ret.tail(20)["ret"].std() if len(code_ret) >= 10 else 0
            stop_pct = max(0.08, stock_vol * 2) if pd.notna(stock_vol) and stock_vol > 0 else 0.08
            if cur_px < pos["entry_price"] * (1 - stop_pct):
                sell_reason = f"止损({stop_pct:.0%})"

            if sell_reason:
                prev_c = prev_map.get(code)
                if prev_c and TradingConfig.is_at_limit_down(cur_px, prev_c, code):
                    continue
                proceeds = pos["shares"] * cur_px * NET_SELL
                cash += proceeds
                pnl = (cur_px / pos["entry_price"] - 1) * 100
                trade_log.append({
                    "日期": td.strftime("%Y-%m-%d"), "操作": f"卖出({sell_reason})",
                    "股票代码": code, "股票名称": name_map_full.get(code, ""),
                    "入场价": round(pos["entry_price"], 2), "当前价/出场价": round(cur_px, 2),
                    "盈亏%": round(pnl, 2), "股数": pos["shares"],
                    "入场日期": pos["entry_date"], "持有天数": pos["hold_days"],
                    "总资产": round(cash, 2), "当前现金": round(cash, 2),
                })
                trade_count += 1
                del positions[code]

        # 买入
        if not frozen and td in sig_by_date_buy:
            sigs = sig_by_date_buy[td]
            available = args.top_n - len(positions)
            for _, s in sigs.iterrows():
                if available <= 0:
                    break
                code = s["code"]
                if code in positions:
                    continue
                prev_c = prev_map.get(code)
                if prev_c and TradingConfig.is_at_limit_up(s["close"], prev_c, code):
                    continue
                px = s["close"]
                shares = int(cash / max(available, 1) / px / 100) * 100
                if shares <= 0:
                    continue
                cost = shares * px * BUY_COST
                if cost > cash:
                    continue
                cash -= cost
                positions[code] = {
                    "entry_price": px, "entry_date": td.strftime("%Y-%m-%d"),
                    "shares": shares, "peak_price": px, "hold_days": 0, "current_price": px,
                }
                trade_log.append({
                    "日期": td.strftime("%Y-%m-%d"), "操作": "买入",
                    "股票代码": code, "股票名称": name_map_full.get(code, ""),
                    "入场价": round(px, 2), "当前价/出场价": "",
                    "盈亏%": "", "股数": shares,
                    "入场日期": td.strftime("%Y-%m-%d"), "持有天数": 0,
                    "总资产": round(cash + shares * px, 2), "当前现金": round(cash, 2),
                })
                available -= 1

    # ── 输出 ──
    final_val = cash + sum(p["shares"] * p.get("current_price", p["entry_price"]) for p in positions.values())
    total_ret = (final_val / args.cash - 1) * 100

    # Metrics
    if equity:
        eq = pd.DataFrame(equity)
        eq["date"] = pd.to_datetime(eq["date"])
        eq["ret"] = eq["value"].pct_change()
        days = (eq["date"].max() - eq["date"].min()).days
        annual_ret = (final_val / args.cash) ** (365 / max(days, 1)) - 1
        sharpe = eq["ret"].mean() / eq["ret"].std() * np.sqrt(252) if eq["ret"].std() > 0 else 0
        peak = eq["value"].cummax()
        dd_series = (eq["value"] - peak) / peak
        max_dd = dd_series.min()
        win_trades = [t for t in trade_log if "卖出" in str(t.get("操作", ""))]
        wins = sum(1 for t in win_trades if float(t.get("盈亏%", 0)) > 0)

        print(f"\n{'='*60}")
        print(f"  标签烙印型再启动 — 回测结果")
        print(f"{'='*60}")
        print(f"  区间: {eq['date'].min().date()} ~ {eq['date'].max().date()} ({days}天)")
        print(f"  最终资产: {final_val:,.0f} | 总收益: {total_ret:+.1f}%")
        print(f"  年化收益: {annual_ret*100:+.1f}% | 夏普: {sharpe:.2f} | 最大回撤: {max_dd*100:.1f}%")
        print(f"  交易: {len(win_trades)} 笔 | 胜率: {wins}/{len(win_trades)} = {wins/max(len(win_trades),1)*100:.0f}%")

    # 保存交割单
    if trade_log:
        trades_df = pd.DataFrame(trade_log)
        out_path = f"data/backtest_trades/trades_{args.label}_{args.top_n}_ocr.csv"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        trades_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        logger.success(f"交割单: {out_path} ({len(trades_df)} 条)")

    elapsed = time.time() - t0
    logger.success(f"全部完成 ({elapsed:.0f}s)")

    engine.dispose()


if __name__ == "__main__":
    main()
