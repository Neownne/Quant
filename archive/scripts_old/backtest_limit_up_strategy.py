#!/usr/bin/env python
"""涨停策略简单回测 —— 路径A：逐日筛选 + 等权持有，滚动模拟。

策略条件（5选4）：
  1. 市值 50–300 亿
  2. 股价 5–50 元
  3. MA5 > MA10
  4. 近一月涨停 > 1 次（日收益 ≥ 10%）
  5. 近 10 日无跌停（日收益 > −10%）

用法:
    python scripts/backtest_limit_up_strategy.py
    python scripts/backtest_limit_up_strategy.py --start 2024-01-01 --top-n 15
    python scripts/backtest_limit_up_strategy.py --min-conditions 3 --no-mcap
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import date, timedelta
from loguru import logger
from sqlalchemy import text

from data.db import get_engine
from config.settings import TradingConfig


def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="涨停策略简单回测")
    p.add_argument("--start", type=str, default="2025-06-01",
                   help="回测起始日期（默认 2025-06-01）")
    p.add_argument("--end", type=str, default=None,
                   help="回测结束日期（默认最新）")
    # 筛选参数
    p.add_argument("--mcap-min", type=float, default=50.0)
    p.add_argument("--mcap-max", type=float, default=300.0)
    p.add_argument("--price-min", type=float, default=5.0)
    p.add_argument("--price-max", type=float, default=50.0)
    p.add_argument("--limit-up-pct", type=float, default=0.099)
    p.add_argument("--limit-up-lookback", type=int, default=20)
    p.add_argument("--limit-up-count", type=int, default=1, help="涨停次数 > 该值")
    p.add_argument("--limit-down-pct", type=float, default=-0.099)
    p.add_argument("--limit-down-lookback", type=int, default=10)
    p.add_argument("--min-conditions", type=int, default=4)
    p.add_argument("--adaptive-ld", action="store_true",
                   help="自适应跌停: 高波动时含跌停(排雷)，低波动时去跌停(追动量)")
    p.add_argument("--ld-vol-lookback", type=int, default=20,
                   help="自适应跌停的波动率计算窗口(交易日)")
    p.add_argument("--ld-vol-threshold", type=float, default=0.18,
                   help="自适应跌停的波动率阈值(高于此值启用跌停过滤)")
    p.add_argument("--rebalance", type=str, default="daily",
                   choices=["daily", "3day", "weekly"],
                   help="调仓频率（daily=每日, 3day=每3交易日, weekly=每周）")
    p.add_argument("--no-mcap", action="store_true", help="不使用市值条件")
    p.add_argument("--mcap-proxy", action="store_true",
                   help="用隐含股本估算历史市值（无 stock_daily_extra 的日期用 close×隐含股本近似）")
    # 组合参数
    p.add_argument("--top-n", type=int, default=20, help="最多持有 N 只")
    p.add_argument("--benchmark", type=str, default="000300", help="基准指数")
    p.add_argument("--min-listed-days", type=int, default=120)
    # 风控参数
    p.add_argument("--exclude-yesterday-drop", type=float, default=0,
                   help="排除昨跌超过X%%的股票（如0.03=排除昨跌>3%%的票）")
    p.add_argument("--min-turnover", type=float, default=0,
                   help="最低换手率要求（如0.02=换手率<2%%的不买）")
    p.add_argument("--pause-after-losses", type=str, default="",
                   help="连续亏N天→空仓M天，格式N,M（如2,2）")
    p.add_argument("--dd-reduce", type=str, default="",
                   help="回撤超X%%→持仓降到N只，格式X,N（如0.15,3）")
    p.add_argument("--dd-stop", type=float, default=0,
                   help="回撤超X%%→清仓等回到均线（如0.20）")
    p.add_argument("--min-signals", type=int, default=0,
                   help="当日通过数少于此值→不交易")
    p.add_argument("--min-top-lu", type=int, default=0,
                   help="Top-1涨停次数少于此值→不交易")
    p.add_argument("--trend-filter", action="store_true",
                   help="CSI1000<MA60时清仓")
    p.add_argument("--entry-close", action="store_true",
                   help="T+1收盘买入(非开盘), 延迟1天入场")
    p.add_argument("--rank-mode", type=str, default="top",
                   choices=["top", "bottom", "median"],
                   help="排名方式: top=涨停多优先, bottom=涨停少优先, median=接近中位数")
    p.add_argument("--skip-top", type=int, default=0,
                   help="跳过前N名, 如1=买2-6名")
    # S组-评分优化
    p.add_argument("--lu-score", action="store_true",
                   help="涨停最优区间评分(2-6次加分,7+次惩罚)")
    p.add_argument("--lu-decay", action="store_true", help="涨停次数时间衰减加权")
    p.add_argument("--lu-quality", action="store_true", help="涨停质量加权(收盘/最高价)")
    p.add_argument("--lu-streak", action="store_true", help="连板加分")
    p.add_argument("--lu-turnover", action="store_true", help="换手率因子加权")
    p.add_argument("--lu-volume", action="store_true", help="放量涨停加权")
    # E组-出场信号
    p.add_argument("--skip-open-gap", type=float, default=0,
                   help="T+1开盘相对昨日收盘跌幅超此阈值则跳过买入(如0.03=3%%)")
    p.add_argument("--exit-ma5", action="store_true", help="跌破MA5出场")
    p.add_argument("--exit-trailing", type=float, default=0, help="高点回落X%%止盈")
    p.add_argument("--exit-stop", type=float, default=0, help="个股跌超X%%硬止损")
    p.add_argument("--exit-max-hold", type=int, default=0, help="最多持有N天")
    # P组-仓位
    p.add_argument("--pos-score-weight", action="store_true", help="评分加权仓位")
    p.add_argument("--pos-min-turnover", type=int, default=0, help="换手<此数不调仓")
    p.add_argument("--pos-min-candidates", type=int, default=0, help="候选<此数空仓")
    # M组-择时
    p.add_argument("--timing-lu-ma", action="store_true", help="涨停家数<MA20空仓")
    p.add_argument("--timing-dual", action="store_true", help="CSI1000+通过数双重过滤")
    return p.parse_args()


def load_universe(engine, date_str, min_listed_days):
    """获取某日的候选股票池（非ST，已上市足够天数）"""
    min_list_date = pd.Timestamp(date_str) - timedelta(days=min_listed_days)
    codes = pd.read_sql(
        f"SELECT code, name FROM stock_basic "
        f"WHERE is_st = FALSE AND list_date <= %s",
        engine, params=(min_list_date,),
    )
    return set(codes["code"].tolist()), dict(zip(codes["code"], codes["name"]))


def run_backtest(args):
    engine = get_engine()
    tc = TradingConfig()

    # ── 1. 确定回测区间 ──
    if args.end:
        end_date_str = args.end
    else:
        last_two = pd.read_sql(
            "SELECT trade_date, COUNT(*) as n FROM stock_daily "
            "GROUP BY trade_date ORDER BY trade_date DESC LIMIT 2", engine)
        if len(last_two) >= 2 and last_two.iloc[1]["n"] > last_two.iloc[0]["n"] * 0.8:
            end_date_str = str(last_two.iloc[1]["trade_date"])[:10]
            logger.info(f"最新日({str(last_two.iloc[0]['trade_date'])[:10]})数据可能不完整"
                       f"({last_two.iloc[0]['n']}只 vs {last_two.iloc[1]['n']}只), 回退到 {end_date_str}")
        else:
            end_date_str = str(last_two.iloc[0]["trade_date"])[:10]

    # 往前多取一段以保证均线/涨跌停可算
    pre_start = pd.Timestamp(args.start) - timedelta(days=max(args.limit_up_lookback, args.limit_down_lookback) + 30)
    pre_start_str = pre_start.strftime("%Y-%m-%d")

    logger.info(f"回测区间: {args.start} → {end_date_str}（预热从 {pre_start_str}）")

    # ── 2. 初始候选池（用回测最后一天圈定，避免 survivorship 偏差过大）──
    code_set, _ = load_universe(engine, end_date_str, args.min_listed_days)
    logger.info(f"候选池（基于最新日）: {len(code_set)} 只")

    # ── 3. 一次性拉取全部日线 ──
    code_tuple = tuple(code_set)
    daily = pd.read_sql(
        f"SELECT code, trade_date, open, close FROM stock_daily "
        f"WHERE code IN ({','.join(['%s']*len(code_tuple))}) "
        f"AND trade_date BETWEEN %s AND %s "
        f"ORDER BY code, trade_date",
        engine, params=(*code_tuple, pre_start_str, end_date_str),
    )
    daily = daily.sort_values(["code", "trade_date"]).reset_index(drop=True)
    daily["trade_date"] = pd.to_datetime(daily["trade_date"])
    daily["ret"] = daily.groupby("code")["close"].pct_change()  # close-to-close (涨停/跌停判定用)
    # 次日盘中收益（open→close，消除隔夜缺口偏差）
    daily["ret_oc"] = daily["close"] / daily["open"] - 1
    daily["ma5"] = daily.groupby("code")["close"].transform(lambda x: x.rolling(5, min_periods=5).mean())
    daily["ma10"] = daily.groupby("code")["close"].transform(lambda x: x.rolling(10, min_periods=10).mean())
    logger.info(f"日线: {len(daily)} 行, {daily['code'].nunique()} 只")

    # ── 4. 市值数据（一次性拉取）+ 隐含股本代理 ──
    implied_shares = {}  # code -> shares (亿股)
    if not args.no_mcap:
        extra = pd.read_sql(
            f"SELECT code, trade_date, market_cap FROM stock_daily_extra "
            f"WHERE code IN ({','.join(['%s']*len(code_tuple))}) "
            f"AND trade_date BETWEEN %s AND %s",
            engine, params=(*code_tuple, pre_start_str, end_date_str),
        )
        extra = extra.sort_values(["code", "trade_date"]).reset_index(drop=True)
        extra["trade_date"] = pd.to_datetime(extra["trade_date"])
        extra_by_date = {d: g.set_index("code") for d, g in extra.groupby("trade_date")}
        extra_dates = sorted(extra_by_date.keys())
        logger.info(f"市值数据: {len(extra)} 行, {extra['code'].nunique()} 只")

        if args.mcap_proxy:
            # 为每只股票算隐含股本 = 最新 market_cap / 当日 close
            daily_indexed = daily.set_index(["code", "trade_date"])
            latest_extra = extra.sort_values("trade_date").groupby("code").last()
            for code, row in latest_extra.iterrows():
                mcap_date = row["trade_date"]
                if (code, mcap_date) in daily_indexed.index:
                    close_p = daily_indexed.loc[(code, mcap_date), "close"]
                    if close_p > 0:
                        implied_shares[code] = row["market_cap"] / close_p
            logger.info(f"隐含股本: {len(implied_shares)} 只")
    else:
        extra = pd.DataFrame(columns=["code", "trade_date", "market_cap"])
        extra_by_date = {}
        extra_dates = []
        logger.info("市值条件已禁用")

    # ── 5. 基准指数 ──
    benchmark = pd.read_sql(
        f"SELECT trade_date, close FROM index_daily "
        f"WHERE code = %s AND trade_date BETWEEN %s AND %s "
        f"ORDER BY trade_date",
        engine, params=(args.benchmark, args.start, end_date_str),
    )
    benchmark = benchmark.set_index("trade_date")["close"]
    benchmark_ret = benchmark.pct_change().dropna()
    logger.info(f"基准 {args.benchmark}: {len(benchmark)} 天")

    # ── 5.4 自适应跌停: 预计算上证指数滚动波动率 ──
    adaptive_ld_vol = {}
    if args.adaptive_ld:
        idx_vol_data = pd.read_sql(
            f"SELECT trade_date, close FROM index_daily "
            f"WHERE code='000001' AND trade_date BETWEEN %s AND %s "
            f"ORDER BY trade_date",
            engine, params=(pre_start_str, end_date_str),
        )
        idx_vol_data["trade_date"] = pd.to_datetime(idx_vol_data["trade_date"])
        idx_vol_data["ret"] = idx_vol_data["close"].pct_change()
        idx_vol_data["vol"] = idx_vol_data["ret"].rolling(args.ld_vol_lookback).std() * np.sqrt(252)
        adaptive_ld_vol = dict(zip(idx_vol_data["trade_date"], idx_vol_data["vol"]))
        logger.info(f"自适应跌停: lookback={args.ld_vol_lookback}d threshold={args.ld_vol_threshold:.0%}")

    # ── 5.5 CSI1000 趋势（D1风控用）──
    csi1k_data = {}
    if args.trend_filter:
        csi1k = pd.read_sql(
            f"SELECT trade_date, close FROM index_daily "
            f"WHERE code='000852' AND trade_date BETWEEN %s AND %s",
            engine, params=(args.start, end_date_str),
        )
        csi1k["trade_date"] = pd.to_datetime(csi1k["trade_date"])
        csi1k["ma60"] = csi1k["close"].rolling(60, min_periods=30).mean()
        csi1k_data = dict(zip(csi1k["trade_date"], csi1k["close"] > csi1k["ma60"]))
        logger.info(f"CSI1000趋势数据: {len(csi1k_data)} 天")

    # ── 6. 获取全部交易日 ──
    all_dates = sorted(daily["trade_date"].unique())
    trade_dates = [d for d in all_dates if d >= pd.Timestamp(args.start) and d <= pd.Timestamp(end_date_str)]
    logger.info(f"回测交易日: {len(trade_dates)} 天")

    # ── 7. 逐日模拟 ──
    # 预处理：按日期建立 daily 索引加速
    daily_by_date = {d: g for d, g in daily.groupby("trade_date")}

    # 单边交易成本率（买入: commission+slippage, 卖出: commission+stamp_duty+slippage）
    COST_BUY = tc.COMMISSION + tc.SLIPPAGE
    COST_SELL = tc.COMMISSION + tc.STAMP_DUTY + tc.SLIPPAGE
    COST_ROUNDTRIP = COST_BUY + COST_SELL

    portfolio_rets_gross = []   # 毛收益（不含成本）
    portfolio_rets_net = []     # 净收益（含成本）
    daily_details = []
    prev_holdings = set()       # 当前持仓
    nav_h = [1.0]               # 净值序列（风控用）
    pause_counter = [0]         # 暂停计数器（可变对象跨迭代）
    reduce_n = args.top_n       # 降仓后的持仓数（默认不变）
    hold_tracker = {}           # 持仓追踪: code -> {entry_date, entry_price, high_water, days_held}

    for i, today in enumerate(trade_dates):
        if today not in daily_by_date:
            continue

        # ── 判断是否为调仓日 ──
        if args.rebalance == "weekly":
            is_rebalance = (i == 0) or (today.isocalendar()[1] != trade_dates[i-1].isocalendar()[1])
        elif args.rebalance == "3day":
            is_rebalance = (i == 0) or (i % 3 == 0)
        else:
            is_rebalance = True

        today_data = daily_by_date[today].set_index("code")

        # ── 涨跌停统计 ──
        lookback_start = today - timedelta(days=max(args.limit_up_lookback, args.limit_down_lookback) + 5)
        lb_data = daily[(daily["trade_date"] >= lookback_start) & (daily["trade_date"] <= today)]

        lu = lb_data[lb_data["ret"] >= args.limit_up_pct]
        lu_counts = lu.groupby("code").size()

        ld = lb_data[lb_data["ret"] <= args.limit_down_pct]
        ld_codes = set(ld["code"].unique()) if not ld.empty else set()

        # 市值
        mcap_series = None       # 来自 stock_daily_extra 的精确市值
        mcap_proxy_avail = False  # 是否可用代理
        if not args.no_mcap:
            avail_dates = sorted([d for d in extra_dates if d <= today], reverse=True)
            if avail_dates:
                mcap_series = extra_by_date[avail_dates[0]]["market_cap"]
            mcap_proxy_avail = args.mcap_proxy and bool(implied_shares)

        # ── 逐股判定 ──
        passed = []
        for code in today_data.index:
            if code not in code_set:
                continue
            row = today_data.loc[code]
            close_p = row["close"]
            ma5 = row.get("ma5")
            ma10 = row.get("ma10")
            if pd.isna(close_p) or close_p <= 0:
                continue

            # C1: 市值（精确 > 代理 > 跳过）
            if args.no_mcap:
                c1 = True
            elif mcap_series is not None and code in mcap_series.index and not pd.isna(mcap_series.loc[code]):
                c1 = args.mcap_min <= mcap_series.loc[code] <= args.mcap_max
            elif mcap_proxy_avail and code in implied_shares:
                proxy_mcap = implied_shares[code] * close_p
                c1 = args.mcap_min <= proxy_mcap <= args.mcap_max
            else:
                c1 = False  # 无市值数据，条件不通过
            c2 = args.price_min <= close_p <= args.price_max
            c3 = (not pd.isna(ma5)) and (not pd.isna(ma10)) and (ma5 > ma10)
            lu_n = int(lu_counts.get(code, 0))
            c4 = lu_n > args.limit_up_count
            c5 = code not in ld_codes

            # ── 自适应跌停: 高波动含跌停(排雷)，低波动去跌停(追动量) ──
            if args.adaptive_ld:
                vol_now = adaptive_ld_vol.get(today, np.nan)
                if not pd.isna(vol_now) and vol_now > args.ld_vol_threshold:
                    # 高波动 → 5条件严格模式 (含跌停排雷)
                    n_pass = sum([c1, c2, c3, c4, c5])
                    effective_min = 5
                else:
                    # 低波动 → 4条件宽松模式 (去跌停追动量)
                    n_pass = sum([c1, c2, c3, c4])
                    effective_min = 4
            else:
                n_pass = sum([c1, c2, c3, c4, c5])
                effective_min = args.min_conditions

            if n_pass >= effective_min:
                score = float(lu_n)  # 基础分=涨停次数
                # Lu-score: 最优区间加权
                if args.lu_score:
                    if lu_n <= 3: score = lu_n + 1.0       # 2-3次: +1分(早期动量)
                    elif lu_n <= 5: score = lu_n + 2.0     # 4-5次: +2分(强动量)
                    elif lu_n == 6: score = lu_n + 1.0     # 6次: +1分(高动量风险上升)
                    else: score = lu_n - 2.0               # 7+次: -2分(过度追高)
                # S1: 时间衰减
                if args.lu_decay:
                    code_lu_dates = lb_data[(lb_data["code"]==code) & (lb_data["ret"]>=0.099)]["trade_date"]
                    if len(code_lu_dates) > 0:
                        weights = [max(0.1, 1.0 - (today - d).days / 20) for d in code_lu_dates if today >= d]
                        score = sum(weights) if weights else lu_n
                # S2: 涨停质量
                if args.lu_quality:
                    code_lu = lb_data[(lb_data["code"]==code) & (lb_data["ret"]>=0.099)]
                    if len(code_lu) > 0:
                        q_scores = [min(1.5, max(0.3, (r["close"]/r["high"]) if r["high"]>0 else 1.0)) for _, r in code_lu.iterrows()]
                        score *= np.mean(q_scores) if q_scores else 1.0
                # S3: 连板加分
                if args.lu_streak:
                    code_rets = lb_data[lb_data["code"]==code].sort_values("trade_date")["ret"]
                    streak = 0; max_streak = 0
                    for r in code_rets:
                        if r >= 0.099: streak += 1; max_streak = max(max_streak, streak)
                        else: streak = 0
                    if max_streak >= 2: score += (max_streak - 1) * 1.0
                # S4/S5 skipped for now (need turnover/volume data)
                passed.append((code, score, close_p))

        # ── 风控：个股过滤（排除昨跌超阈值）──
        if args.exclude_yesterday_drop > 0:
            yesterday_ret_map = {}
            if "ret" in today_data.columns:
                yesterday_ret_map = today_data["ret"].dropna().to_dict()
            passed = [(c, lu, cp) for c, lu, cp in passed
                      if c not in yesterday_ret_map or yesterday_ret_map[c] >= -args.exclude_yesterday_drop]

        # ── 风控：信号质量 ──
        if args.min_signals > 0 and len(passed) < args.min_signals:
            passed = []  # 候选太少，今日不交易
        if args.min_top_lu > 0 and passed:
            top_lu = passed[0][1]
            if top_lu < args.min_top_lu:
                passed = []  # 领头羊不够强

        # ── 风控：连续亏损暂停 ──
        pause_active = False
        if args.pause_after_losses:
            parts = args.pause_after_losses.split(",")
            n_loss = int(parts[0]); m_pause = int(parts[1])
            # 检查最近N天是否连续亏损
            recent_rets = portfolio_rets_net[-n_loss:] if len(portfolio_rets_net) >= n_loss else []
            if len(recent_rets) == n_loss and all(r < 0 for r in recent_rets):
                pause_counter[0] = m_pause  # 触发暂停
            if pause_counter[0] > 0:
                pause_active = True
                pause_counter[0] -= 1

        # ── 风控：回撤降仓/清仓 ──
        peak_nav = max(nav_h[-200:]) if len(nav_h) >= 200 else max(nav_h) if nav_h else 1.0
        current_nav = nav_h[-1] if nav_h else 1.0
        dd_from_peak = current_nav / peak_nav - 1
        dd_reduce_active = False

        if args.dd_stop > 0 and dd_from_peak < -args.dd_stop:
            passed = []  # 清仓
            dd_reduce_active = True
        elif args.dd_reduce:
            parts = args.dd_reduce.split(",")
            dd_threshold = float(parts[0]); reduce_n = int(parts[1])
            if dd_from_peak < -dd_threshold:
                dd_reduce_active = True

        # ── 风控：趋势过滤 ──
        trend_skip = False
        if args.trend_filter:
            csi1k_today = csi1k_data.get(today)
            if csi1k_today is not None and not csi1k_today:
                trend_skip = True  # CSI1000 < MA60

        # ── 选股 ──
        if is_rebalance and not pause_active and not trend_skip:
            if args.rank_mode == "top":
                passed.sort(key=lambda x: x[1], reverse=True)
            elif args.rank_mode == "bottom":
                passed.sort(key=lambda x: x[1])  # 涨停少的排前面
            elif args.rank_mode == "median":
                lu_values = [x[1] for x in passed]
                median_lu = np.median(lu_values)
                passed.sort(key=lambda x: abs(x[1] - median_lu))
            if args.skip_top > 0 and len(passed) > args.skip_top:
                passed = passed[args.skip_top:]  # 跳过前N名
            n_select = min(reduce_n, args.top_n) if dd_reduce_active else args.top_n
            selected = passed[:n_select]
            selected_set = set(s[0] for s in selected)
        elif pause_active or trend_skip:
            selected = []
            selected_set = set()  # 空仓
        else:
            selected = []  # 非调仓日不选新股
            selected_set = prev_holdings

        # ── E组: 出场信号 ──
        forced_exits = set()
        if args.exit_ma5 or args.exit_trailing > 0 or args.exit_stop > 0 or args.exit_max_hold > 0:
            for code in list(selected_set):
                if code not in today_data.index: continue
                r = today_data.loc[code]
                cp = r["close"]
                # E1: 跌破MA5
                if args.exit_ma5:
                    ma5 = r.get("ma5")
                    if not pd.isna(ma5) and cp < ma5:
                        forced_exits.add(code); continue
                # E2: 移动止盈
                if args.exit_trailing > 0 and code in hold_tracker:
                    hw = hold_tracker[code]["high_water"]
                    if cp < hw * (1 - args.exit_trailing):
                        forced_exits.add(code); continue
                # E3: 硬止损
                if args.exit_stop > 0 and code in hold_tracker:
                    ep = hold_tracker[code]["entry_price"]
                    if cp < ep * (1 - args.exit_stop):
                        forced_exits.add(code); continue
                # E4: 持有天数上限
                if args.exit_max_hold > 0 and code in hold_tracker:
                    if hold_tracker[code]["days_held"] >= args.exit_max_hold:
                        forced_exits.add(code); continue
            selected_set -= forced_exits

        # ── P3: 最小换手 ──
        if args.pos_min_turnover > 0 and is_rebalance:
            new_codes = selected_set - prev_holdings
            old_codes = prev_holdings - selected_set
            if len(new_codes) + len(old_codes) < args.pos_min_turnover:
                selected_set = prev_holdings  # 不调仓

        # ── P4: 候选不足空仓 ──
        if args.pos_min_candidates > 0 and len(passed) < args.pos_min_candidates:
            selected_set = set()

        # ── M1: 涨停家数均线 ──
        if args.timing_lu_ma:
            total_lu_today = len(lb_data[lb_data["ret"] >= 0.099]["code"].unique())
            lu_ma20 = np.mean([len(daily[(daily["trade_date"]>=today-timedelta(days=i+20)) & (daily["trade_date"]<=today-timedelta(days=i)) & (daily["ret"]>=0.099)]["code"].unique()) for i in range(20)]) if len(all_dates) > 20 else total_lu_today
            if total_lu_today < lu_ma20:
                selected_set = set()

        # ── M2: 双重过滤 ──
        if args.timing_dual:
            csi1k_up = csi1k_data.get(today, True) if csi1k_data else True
            if not csi1k_up and len(passed) < 15:
                selected_set = set()

        # ── 次日收益 ──
        entry_delay = 2 if args.entry_close else 1
        next_date_idx = all_dates.index(today) + entry_delay
        if next_date_idx >= len(all_dates):
            break

        next_date = all_dates[next_date_idx]
        if next_date not in daily_by_date:
            portfolio_rets_gross.append(0.0)
            portfolio_rets_net.append(0.0)
            daily_details.append({"date": today, "n_stocks": len(selected), "ret_gross": 0.0, "ret_net": 0.0, "turnover": 0})
            continue

        next_data = daily_by_date[next_date].set_index("code")
        stock_rets = []
        gap_skipped = 0
        for code in selected_set:
            if code in next_data.index:
                r = next_data.loc[code].get("ret_oc")  # 用盘中收益(open→close)，消除隔夜偏差
                # ── 开盘跌幅过滤: 新买入的票如果T+1开盘暴跌则跳过 ──
                if args.skip_open_gap > 0 and code not in prev_holdings:
                    today_close = today_data.loc[code, "close"] if code in today_data.index else None
                    next_open = next_data.loc[code].get("open") if "open" in next_data.columns else None
                    if today_close and next_open and not pd.isna(today_close) and not pd.isna(next_open) and today_close > 0:
                        gap = (next_open - today_close) / today_close
                        if gap < -args.skip_open_gap:
                            stock_rets.append(0.0)  # 不买入，现金收益
                            gap_skipped += 1
                            continue
                stock_rets.append(r if not pd.isna(r) else 0.0)
            else:
                stock_rets.append(0.0)

        day_ret_gross = np.mean(stock_rets) if stock_rets else 0.0

        # ── 换手率 & 成本（仅调仓日产生交易成本）──
        if is_rebalance:
            if prev_holdings:
                n_new = len(selected_set - prev_holdings)
                n_old_sold = len(prev_holdings - selected_set)
                turnover_pct = max(n_new, n_old_sold) / max(len(selected_set), 1)
            else:
                turnover_pct = 1.0  # 首日建仓
            day_cost = turnover_pct * COST_ROUNDTRIP
        else:
            turnover_pct = 0.0
            day_cost = 0.0
        day_ret_net = day_ret_gross - day_cost

        portfolio_rets_gross.append(day_ret_gross)
        portfolio_rets_net.append(day_ret_net)
        nav_h.append(nav_h[-1] * (1 + day_ret_net))  # 风控用净值追踪
        daily_details.append({
            "date": today, "next_date": next_date,
            "n_stocks": len(selected_set), "turnover": turnover_pct,
            "ret_gross": day_ret_gross, "ret_net": day_ret_net,
            "top_score": round(float(selected[0][1]), 1) if selected else 0,
            "is_rebalance": is_rebalance,
        })
        # ── 更新持仓追踪 ──
        for code in selected_set:
            if code not in hold_tracker:
                hold_tracker[code] = {"entry_date": today, "entry_price": today_data.loc[code, "close"] if code in today_data.index else 0, "high_water": today_data.loc[code, "close"] if code in today_data.index else 0, "days_held": 0}
            else:
                hold_tracker[code]["days_held"] += 1
                if code in today_data.index:
                    cp = today_data.loc[code, "close"]
                    hold_tracker[code]["high_water"] = max(hold_tracker[code]["high_water"], cp)
        # 清理已卖出的
        for code in list(hold_tracker.keys()):
            if code not in selected_set:
                del hold_tracker[code]

        prev_holdings = selected_set

        if (i + 1) % 50 == 0:
            cum_gross = np.prod([1 + r for r in portfolio_rets_gross]) - 1
            cum_net = np.prod([1 + r for r in portfolio_rets_net]) - 1
            logger.info(f"  进度 {i+1}/{len(trade_dates)} | 毛收益 {cum_gross*100:.1f}% | "
                        f"净收益 {cum_net*100:.1f}% | 持仓 {len(selected_set)} 只 | "
                        f"换手 {turnover_pct:.0%} | {'调仓' if is_rebalance else '持有'}")

    # ── 8. 计算指标 ──
    idx = trade_dates[:len(portfolio_rets_net)]
    rets_net = pd.Series(portfolio_rets_net, index=idx)
    rets_gross = pd.Series(portfolio_rets_gross, index=idx)
    cum_net = (1 + rets_net).cumprod() - 1
    cum_gross = (1 + rets_gross).cumprod() - 1

    # 对齐基准
    bench_aligned = benchmark_ret.reindex(rets_net.index).fillna(0)
    bench_cum = (1 + bench_aligned).cumprod() - 1

    def metrics(r, cum):
        n = len(r)
        ann_r = (1 + cum.iloc[-1]) ** (252 / n) - 1 if n > 0 else 0
        ann_v = r.std() * np.sqrt(252) if n > 0 else 0
        sh = ann_r / ann_v if ann_v > 0 else 0
        dd = ((1 + cum) / (1 + cum.cummax()) - 1).min()
        wr = (r > 0).mean()
        return ann_r, ann_v, sh, dd, wr

    ann_net, vol_net, sh_net, dd_net, wr_net = metrics(rets_net, cum_net)
    ann_gross, vol_gross, sh_gross, dd_gross, wr_gross = metrics(rets_gross, cum_gross)
    bench_ann = (1 + bench_cum.iloc[-1]) ** (252 / len(rets_net)) - 1 if len(rets_net) > 0 else 0

    avg_turnover = np.mean([d["turnover"] for d in daily_details]) if daily_details else 0

    # ── 9. 输出 ──
    print(f"\n{'='*80}")
    print(f"  涨停策略回测结果")
    print(f"  区间: {args.start} → {end_date_str}（{len(rets_net)} 个交易日）")
    print(f"  条件: 市值{args.mcap_min}-{args.mcap_max}亿 | 股价{args.price_min}-{args.price_max}元")
    print(f"        MA5>MA10 | 涨停>{args.limit_up_count}次 | 无跌停 | {args.min_conditions}/5通过")
    print(f"  组合: Top-{args.top_n} 等权 | {args.rebalance}频调仓 | 成本: 买{COST_BUY*100:.3f}% 卖{COST_SELL*100:.3f}%")
    print(f"{'='*80}")
    print(f"  {'【毛收益（不含成本）】':<30}")
    print(f"  {'累计收益:':<20} {cum_gross.iloc[-1]*100:>8.1f}%")
    print(f"  {'年化收益:':<20} {ann_gross*100:>8.1f}%")
    print(f"  {'Sharpe Ratio:':<20} {sh_gross:>8.2f}")
    print(f"  {'最大回撤:':<20} {dd_gross*100:>8.1f}%")
    print(f"{'─'*80}")
    print(f"  {'【净收益（含成本）】':<30}")
    print(f"  {'累计收益:':<20} {cum_net.iloc[-1]*100:>8.1f}%")
    print(f"  {'年化收益:':<20} {ann_net*100:>8.1f}%")
    print(f"  {'年化波动:':<20} {vol_net*100:>8.1f}%")
    print(f"  {'Sharpe Ratio:':<20} {sh_net:>8.2f}")
    print(f"  {'最大回撤:':<20} {dd_net*100:>8.1f}%")
    print(f"  {'胜率:':<20} {wr_net*100:>8.1f}%")
    print(f"  {'日均换手率:':<20} {avg_turnover*100:>8.1f}%")
    print(f"  {'日均持仓:':<20} {np.mean([d['n_stocks'] for d in daily_details]):>8.1f} 只")
    print(f"{'─'*80}")
    print(f"  基准 {args.benchmark}:")
    print(f"  {'累计收益:':<20} {bench_cum.iloc[-1]*100:>8.1f}%")
    print(f"  {'年化收益:':<20} {bench_ann*100:>8.1f}%")
    print(f"  {'超额收益(年化,净):':<20} {(ann_net - bench_ann)*100:>+8.2f}%")
    print(f"  {'超额收益(年化,毛):':<20} {(ann_gross - bench_ann)*100:>+8.2f}%")
    print(f"{'='*80}")

    # 年度统计
    if len(rets_net) > 0:
        rets_net.index = pd.DatetimeIndex(rets_net.index)
        yearly = rets_net.groupby(rets_net.index.year).apply(lambda x: (1 + x).prod() - 1)
        print(f"\n  年度收益（净）:")
        for yr, r in yearly.items():
            print(f"    {yr}: {r*100:+.1f}%")
        print()

    return {
        "cum_net": cum_net,
        "cum_gross": cum_gross,
        "bench_cum": bench_cum,
        "rets_net": rets_net,
        "rets_gross": rets_gross,
        "sharpe_net": sh_net,
        "max_dd_net": dd_net,
        "daily_details": pd.DataFrame(daily_details),
    }


if __name__ == "__main__":
    args = parse_args()
    result = run_backtest(args)
    if result is not None and len(result["rets_net"]) > 0:
        rets = result["rets_net"]
        n = len(rets)
        ann = (1 + result["cum_net"].iloc[-1]) ** (252 / n) - 1
        sh = result["sharpe_net"]
        dd = result["max_dd_net"]
        wr = (rets > 0).mean()

        # 基准年化
        bench_ann = (1 + result["bench_cum"].iloc[-1]) ** (252 / n) - 1 if len(result["bench_cum"]) > 0 else 0

        import json
        from sqlalchemy import text
        engine = get_engine()
        try:
            with engine.begin() as conn:
                sc = conn.execute(text(
                    "SELECT id FROM strategy_configs WHERE name='涨停策略'"
                )).fetchone()
                sid = sc[0] if sc else conn.execute(text(
                    "INSERT INTO strategy_configs (name, type, description) "
                    "VALUES ('涨停策略', 'static', '5条件规则筛选(5选4): "
                    "市值50-300亿/股价5-50元/MA5>MA10/近月涨停>1次/近10日无跌停') RETURNING id"
                )).fetchone()[0]

                version_str = f"lu5"  # 涨停Top-5
                if args.exit_stop > 0: version_str += "s"    # stop
                # E4标识: 去跌停+宽市值
                if args.limit_down_pct < -0.5 and args.mcap_min <= 30 and args.mcap_max >= 500:
                    version_str += "E4"
                if args.exit_trailing > 0: version_str += "t"
                if args.exit_ma5: version_str += "m"
                if args.lu_decay: version_str += "d"
                if args.rank_mode != "top": version_str += args.rank_mode[0]
                if args.entry_close: version_str += "c"
                sv = conn.execute(text(
                    "SELECT id FROM strategy_versions WHERE strategy_id=:s AND version=:v"
                ), {"s": sid, "v": version_str}).fetchone()
                vid = sv[0] if sv else conn.execute(text(
                    "INSERT INTO strategy_versions (strategy_id, version, algorithm_type, feature_list_version) "
                    "VALUES (:s, :v, 'rule_filter', 'lu_v1') RETURNING id"
                ), {"s": sid, "v": version_str}).fetchone()[0]

                end_d = pd.read_sql("SELECT MAX(trade_date) FROM stock_daily", engine).iloc[0, 0].strftime("%Y-%m-%d")
                metrics = {
                    "start_date": args.start, "end_date": end_d,
                    "annual_return": round(float(ann), 4), "sharpe": round(float(sh), 3),
                    "max_drawdown": round(float(dd), 4), "win_rate": round(float(wr), 4),
                    "n_days": n, "top_n": args.top_n, "min_conditions": args.min_conditions,
                    "rebalance": args.rebalance, "benchmark_annual": round(float(bench_ann), 4),
                    "mcap_proxy": args.mcap_proxy,
                }

                nav_dates = rets.index.strftime("%Y-%m-%d").tolist()
                nav_vals = (1 + rets).cumprod().values
                eq_dict = {nav_dates[i]: round(float(nav_vals[i]), 6) for i in range(n)}
                ret_dict = {nav_dates[i]: round(float(rets.iloc[i]), 6) for i in range(n)}

                conn.execute(text(
                    "INSERT INTO backtest_results (version_id, start_date, end_date, "
                    "quality, metrics_json, equity_curve_json, daily_returns_json) "
                    "VALUES (:v, :s, :e, 'valid', :m, :eq, :dr)"
                ), {"v": vid, "s": args.start, "e": end_d,
                    "m": json.dumps(metrics, ensure_ascii=False),
                    "eq": json.dumps(eq_dict), "dr": json.dumps(ret_dict)})
                logger.info(f"已写入DB (涨停策略 {version_str}, version_id={vid})")
        except Exception as e:
            logger.warning(f"DB写入失败: {e}")
        finally:
            engine.dispose()
