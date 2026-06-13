#!/usr/bin/env python
"""大小票智能切换 v2.0 —— 大盘侧改用涨停策略。

CSI1000 > MA60 → 小票反转（跌多的1000-3000名里选15只，11因子ML，周频）
CSI1000 < MA60 → 大票切换为涨停策略（50-300亿中盘动量，5条件筛选，日频）

涨停策略条件（5选4）：
  1. 市值 50–300 亿
  2. 股价 5–50 元
  3. MA5 > MA10
  4. 近一月涨停 > 1 次
  5. 近 10 日无跌停

前视审计: 所有条件仅用当日收盘前已知数据。T+1 开盘执行。
"""
from __future__ import annotations
import sys, os, argparse, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from datetime import timedelta
from loguru import logger
from sqlalchemy import text
from data.db import get_engine
from models.trainer import walk_forward_train_by_regime
from models.regime import detect_regime, REGIME_PARAMS
from config.settings import TradingConfig
from portfolio.selector import select_topk_ndrop, filter_stocks
from small_cap.backtest import load_universe, load_financial, build_factors as build_sc_factors, _filter_sc_risks

# ── 涨停策略参数 ──
LU_MCAP_MIN = 50.0
LU_MCAP_MAX = 300.0
LU_PRICE_MIN = 5.0
LU_PRICE_MAX = 50.0
LU_PCT = 0.099
LU_LOOKBACK = 20
LU_COUNT = 1          # 涨停次数 > 此值
LD_PCT = -0.099
LD_LOOKBACK = 10
LU_MIN_CONDITIONS = 4  # 5选4
LU_TOP_N = 15          # 大票侧最多持15只


def compute_limit_up_data(ohlcv, extra_df):
    """为涨停策略预计算所有所需数据。

    返回: DataFrame with columns: trade_date, code, close, ma5, ma10, ret, market_cap
    """
    df = ohlcv.sort_values(["code", "trade_date"]).reset_index(drop=True)
    df["trade_date"] = pd.to_datetime(df["trade_date"])

    # 日收益
    df["ret"] = df.groupby("code")["close"].pct_change()

    # 均线
    df["ma5"] = df.groupby("code")["close"].transform(
        lambda x: x.rolling(5, min_periods=5).mean())
    df["ma10"] = df.groupby("code")["close"].transform(
        lambda x: x.rolling(10, min_periods=10).mean())

    # 市值（真实 + 隐含股本代理）
    if extra_df is not None and not extra_df.empty:
        extra = extra_df.copy()
        extra["trade_date"] = pd.to_datetime(extra["trade_date"])
        # 真实市值
        df = df.merge(
            extra[["code", "trade_date", "market_cap"]],
            on=["code", "trade_date"], how="left")

        # 隐含股本代理：对没有市值的日期，用最近已知市值 / 当时股价 * 当前股价
        latest_mcap = extra.sort_values("trade_date").groupby("code").last()
        implied = {}
        for code, row in latest_mcap.iterrows():
            code_data = df[df["code"] == code]
            mcap_date = row["trade_date"]
            mcap_val = row["market_cap"]
            on_mcap_date = code_data[code_data["trade_date"] == mcap_date]
            if not on_mcap_date.empty:
                ref_close = on_mcap_date["close"].iloc[0]
                if ref_close > 0:
                    implied[code] = mcap_val / ref_close

        if implied:
            # 为每只股票填充代理市值
            for code, shares in implied.items():
                mask = df["code"] == code
                df.loc[mask, "proxy_mcap"] = df.loc[mask, "close"] * shares
            df["market_cap"] = df["market_cap"].fillna(df["proxy_mcap"])
    else:
        df["market_cap"] = np.nan

    return df


def run_lu_screening(day_data, lb_data, mcap_threshold=True):
    """在给定日对候选池执行涨停策略筛选。

    day_data: 当日数据 (DataFrame indexed by code, with close/ma5/ma10/market_cap)
    lb_data: 回看期数据 (用于涨停统计)
    mcap_threshold: 是否要求市值条件

    返回: [(code, lu_count, close), ...] 按涨停次数降序
    """
    # 涨停统计
    lu_mask = lb_data["ret"] >= LU_PCT
    lu_counts = lb_data[lu_mask].groupby("code").size()

    # 跌停标记
    ld_mask = lb_data["ret"] <= LD_PCT
    ld_codes = set(lb_data[ld_mask]["code"].unique())

    passed = []
    for code, row in day_data.iterrows():
        close_p = row.get("close")
        if pd.isna(close_p) or close_p <= 0:
            continue
        ma5 = row.get("ma5")
        ma10 = row.get("ma10")

        # 条件判定
        c1 = True
        if mcap_threshold:
            mcap = row.get("market_cap")
            c1 = (not pd.isna(mcap)) and (LU_MCAP_MIN <= mcap <= LU_MCAP_MAX)

        c2 = LU_PRICE_MIN <= close_p <= LU_PRICE_MAX
        c3 = (not pd.isna(ma5)) and (not pd.isna(ma10)) and (ma5 > ma10)
        lu_n = int(lu_counts.get(code, 0))
        c4 = lu_n > LU_COUNT
        c5 = code not in ld_codes

        n_pass = sum([c1, c2, c3, c4, c5])
        if n_pass >= LU_MIN_CONDITIONS:
            passed.append((code, lu_n, close_p))

    passed.sort(key=lambda x: x[1], reverse=True)
    return passed


def main():
    t0 = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="20200101")
    parser.add_argument("--end", default="20260529")
    parser.add_argument("--train-years", type=int, default=3)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--lu-top-n", type=int, default=LU_TOP_N,
                        help="涨停侧最多持有数量")
    parser.add_argument("--lu-min-cond", type=int, default=LU_MIN_CONDITIONS,
                        help="涨停策略最少通过条件数")
    args = parser.parse_args()
    if args.quick:
        args.start = "20230101"; args.train_years = 1

    engine = get_engine()

    # ── 双候选池 ──
    # 小票：成交额排名 1000-3000（不变）
    sc_codes = load_universe(engine, args.start, args.end, rank_lo=1000, rank_hi=3000)
    # 大票侧：成交额 Top-1000（涨停策略需要中盘覆盖，放宽到1000）
    lc_sql = f"""
        SELECT code FROM (
            SELECT code, ROW_NUMBER() OVER (ORDER BY SUM(amount) DESC) AS rn
            FROM stock_daily WHERE trade_date BETWEEN '{args.start}' AND '{args.end}'
            GROUP BY code
        ) t WHERE rn <= 1000
    """
    lc_codes = pd.read_sql(lc_sql, engine)["code"].tolist()
    logger.info(f"小票:{len(sc_codes)}只  涨停侧候选:{len(lc_codes)}只")

    # ── 数据加载 ──
    def load_ohlcv(codes, label=""):
        cl = ",".join([f"'{c}'" for c in codes])
        df = pd.read_sql(
            f"SELECT code,trade_date,open,high,low,close,volume,amount,turnover "
            f"FROM stock_daily WHERE code IN ({cl}) "
            f"AND trade_date BETWEEN '{args.start}' AND '{args.end}' "
            f"ORDER BY code,trade_date", engine)
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        return df

    sc_ohlcv = load_ohlcv(sc_codes)
    lc_ohlcv = load_ohlcv(lc_codes)
    fin_df = load_financial(engine, sc_codes)

    # 市值数据
    lc_cl = ",".join([f"'{c}'" for c in lc_codes])
    extra_df = pd.read_sql(
        f"SELECT code, trade_date, market_cap FROM stock_daily_extra "
        f"WHERE code IN ({lc_cl}) "
        f"AND trade_date BETWEEN '{args.start}' AND '{args.end}'",
        engine)

    # ── 大票侧：涨停策略数据准备 ──
    lu_data = compute_limit_up_data(lc_ohlcv, extra_df if not extra_df.empty else None)
    lu_data["trade_date"] = pd.to_datetime(lu_data["trade_date"])
    all_lu_dates = sorted(lu_data["trade_date"].unique())
    lu_by_date = {d: g.set_index("code") for d, g in lu_data.groupby("trade_date")}
    logger.info(f"涨停数据: {len(lu_data)}行, {lu_data['code'].nunique()}只")

    # 判断市值阈值是否可用（有真实或代理数据时启用）
    has_mcap = lu_data["market_cap"].notna().mean() > 0.1
    logger.info(f"市值覆盖率: {lu_data['market_cap'].notna().mean()*100:.0f}%，"
                f"阈值过滤={'启用' if has_mcap else '禁用'}")

    # ── 小票因子（不变）──
    sc_dataset, sc_factors = build_sc_factors(sc_ohlcv, fin_df)
    sc_dataset = sc_dataset.dropna(subset=["ret_5d"])
    sc_dataset["label"] = (sc_dataset["ret_5d"] > 0).astype(int)
    sc_dataset = sc_dataset.dropna(subset=["label"])

    # ── 指数 + 趋势 ──
    index_df = pd.read_sql(
        f"SELECT trade_date,code,close FROM index_daily "
        f"WHERE code IN ('000001','000852') "
        f"AND trade_date BETWEEN '{args.start}' AND '{args.end}' "
        f"ORDER BY trade_date", engine)
    index_df["trade_date"] = pd.to_datetime(index_df["trade_date"])
    sh_idx = index_df[index_df["code"] == "000001"][["trade_date", "close"]]
    csi1k = index_df[index_df["code"] == "000852"].copy()
    csi1k["ma60"] = csi1k["close"].rolling(60, min_periods=30).mean()
    csi_trend = dict(zip(
        csi1k["trade_date"].dt.strftime("%Y-%m-%d"),
        csi1k["close"] > csi1k["ma60"]))

    # ── ML 训练（仅小票侧，大票侧不需要 ML）──
    regime_df = detect_regime(sh_idx)
    sc_ml = walk_forward_train_by_regime(
        sc_dataset, sc_factors, regime_df,
        train_years=args.train_years, val_years=1)
    logger.info(f"小票ML: {len(sc_ml)}窗")

    # ── 价格 lookup ──
    def mk_lookup(ohlcv, col="close"):
        r = {}
        pv = ohlcv.pivot_table(
            index="trade_date", columns="code", values=col, aggfunc="last")
        for idx in pv.index:
            r[str(idx)[:10]] = {
                c: float(pv.loc[idx, c])
                for c in pv.columns if pd.notna(pv.loc[idx, c])}
        return r

    sc_close = mk_lookup(sc_ohlcv); sc_open = mk_lookup(sc_ohlcv, "open")
    lc_close = mk_lookup(lc_ohlcv); lc_open = mk_lookup(lc_ohlcv, "open")

    # ── 仿真 ──
    # 使用小票验证窗口作为统一时间轴
    sc_all_dates = sorted(set(sc_dataset["trade_date"].unique()))
    val_ranges = [(pd.Timestamp(w["val_start"]), pd.Timestamp(w["val_end"]))
                  for w in sc_ml]
    all_dates = [d for d in sc_all_dates
                 if any(vs <= d <= ve for vs, ve in val_ranges)
                 and d in all_lu_dates]
    logger.info(f"验证期: {len(all_dates)}天")

    nav = 1.0
    holdings, cb = [], {}  # cb: cost_basis
    nav_h, rets, nav_dates = [1.0], [], [str(all_dates[0])[:10]]
    window_metrics = []
    dc = 0

    sc_model_idx = 0
    sc_ens = sc_ml[0].get("ensemble") or sc_ml[0].get("model") if sc_ml else None
    lu_holdings = set()  # 涨停侧独立持仓
    sc_holdings = set()  # 小票侧独立持仓
    position_history = []
    nav_window_start = nav
    current_window = 0

    for i, dt in enumerate(all_dates[:-1]):
        nd = all_dates[i + 1]
        ds, nds = str(dt)[:10], str(nd)[:10]
        sc_dd = sc_dataset[sc_dataset["trade_date"] == dt]

        # WF 窗口切换（小票侧）
        if (sc_model_idx + 1 < len(sc_ml) and
                dt >= pd.Timestamp(sc_ml[sc_model_idx + 1]["val_start"])):
            win_ret = nav / nav_window_start - 1
            window_metrics.append({
                "win": sc_model_idx + 1,
                "ret": round(win_ret * 100, 1)})
            sc_model_idx += 1
            sc_ens = (sc_ml[sc_model_idx].get("ensemble") or
                      sc_ml[sc_model_idx].get("model"))
            nav_window_start = nav

        # 市场状态
        reg = "sideways"
        tr = regime_df[regime_df["trade_date"] <= dt]
        if not tr.empty:
            reg = str(tr["regime"].iloc[-1])
        rp = REGIME_PARAMS.get(reg, REGIME_PARAMS["sideways"])

        # ── 趋势强弱 → 大小票权重（v4: 涨停权重上限70% + 回撤控制）──
        csi1k_price = csi1k[csi1k["trade_date"] == dt]["close"]
        csi1k_ma = csi1k[csi1k["trade_date"] == dt]["ma60"]
        if (len(csi1k_price) > 0 and len(csi1k_ma) > 0 and
                csi1k_ma.iloc[0] > 0):
            ma_dev = (csi1k_price.iloc[0] / csi1k_ma.iloc[0] - 1)
            # 涨停权重：上限压到70%，给防御侧留够底仓
            lu_weight = np.clip(0.3 + ma_dev * 10, 0.1, 0.7)
            sc_weight = 1.0 - lu_weight
            total_exposure = (1.0 if ma_dev > -0.02
                              else max(0.5, 1.0 + ma_dev * 5))
        else:
            lu_weight = 0.5
            sc_weight = 0.5
            total_exposure = 1.0

        # 组合回撤控制：峰值回撤 > 20% 时，涨停侧再砍半
        peak_nav = max(nav_h[-100:]) if len(nav_h) >= 100 else max(nav_h)
        dd_from_peak = nav / peak_nav - 1
        if dd_from_peak < -0.20:
            lu_weight *= 0.5
            total_exposure = min(total_exposure, 0.7)

        # ── 小票调仓（周频：每5交易日）──
        is_sc_rb = (dc % 5 == 0) or (not sc_holdings)
        sc_sel = sc_holdings  # 默认不变
        sc_tb, sc_ts = set(), set()

        if is_sc_rb:
            if sc_ens is not None and not sc_dd.empty:
                dfac = sc_dd[["code"] + sc_factors].fillna(0.5).replace(
                    [np.inf, -np.inf], 0.5)
                dfac["regime"] = reg
                preds = sc_ens.predict(dfac)
                if not preds.empty:
                    cand = sc_dd[sc_dd["code"].isin(preds["code"])]
                    cand = filter_stocks(
                        cand[["code"]].drop_duplicates().assign(
                            name=cand["code"]),
                        ref_date=dt, exclude_st=True, min_list_days=120)
                    risks = _filter_sc_risks(cand, sc_dd, fin_df, dt)
                    valid = set(cand["code"].unique()) - risks
                    preds = preds[preds["code"].isin(valid)]
                    if not preds.empty:
                        ss = pd.Series(
                            preds["score"].values,
                            index=preds["code"].values).sort_values(
                            ascending=False)
                        sc_n = max(3, int(rp["top_n"] * sc_weight * total_exposure))
                        sc_nh, sc_tb, sc_ts = select_topk_ndrop(
                            ss, set(sc_holdings), K=sc_n, N=2)
                        sc_sel = set(sc_nh)
                    else:
                        sc_sel = set()
                else:
                    sc_sel = set()
            else:
                sc_sel = set()

        # 小票交易执行
        for c in sc_tb:
            ep = sc_open.get(nds, {}).get(c, sc_close.get(ds, {}).get(c, 0))
            if ep > 0: cb[c] = ep
        for c in sc_ts:
            cb.pop(c, None)

        # ── 涨停侧调仓（日频：每天换）──
        lu_sel = lu_holdings  # 默认不变
        lu_tb, lu_ts = set(), set()

        if dt in lu_by_date:
            today_lu = lu_by_date[dt]
            lookback_start = dt - timedelta(days=max(LU_LOOKBACK, LD_LOOKBACK) + 5)
            lb = lu_data[
                (lu_data["trade_date"] >= lookback_start) &
                (lu_data["trade_date"] <= dt)]

            lu_passed = run_lu_screening(today_lu, lb, mcap_threshold=has_mcap)
            lu_n_stocks = max(2, int(args.lu_top_n * lu_weight * total_exposure))
            lu_top = lu_passed[:lu_n_stocks]
            lu_sel = set(s[0] for s in lu_top)

            # 计算换手
            lu_tb = lu_sel - lu_holdings
            lu_ts = lu_holdings - lu_sel

            # 执行交易
            for c in lu_tb:
                ep = lc_open.get(nds, {}).get(c, lc_close.get(ds, {}).get(c, 0))
                if ep > 0: cb[c] = ep
            for c in lu_ts:
                cb.pop(c, None)

        # ── 合并持仓 ──
        sc_holdings = sc_sel
        lu_holdings = lu_sel
        all_holdings = list(sc_holdings | lu_holdings)

        dc += 1
        if not all_holdings:
            nav_h.append(nav)
            rets.append(0.0)
            nav_dates.append(nds)
            position_history.append({
                "date": nds, "codes": [],
                "daily_ret": 0.0,
                "mode": "cash",
                "sc_weight": round(sc_weight, 2)})
            continue

        # ── 日收益（混合持仓，等权）──
        dr = 0.0
        total_w = len(all_holdings)
        all_tb = lu_tb | sc_tb
        all_ts = lu_ts | sc_ts

        for c in all_holdings:
            ncp = (sc_close.get(nds, {}).get(c, 0) or
                   lc_close.get(nds, {}).get(c, 0))
            if ncp <= 0:
                continue
            if c in all_tb:
                e = cb.get(c, 0)
            else:
                e = (sc_close.get(ds, {}).get(c, 0) or
                     lc_close.get(ds, {}).get(c, 0))
            if e > 0 and ncp > 0:
                r = ncp / e - 1
                if abs(r) < 0.20:
                    dr += r / total_w

        # 成本（仅对新买入和卖出）
        wp = 1.0 / max(total_w, 1)
        cost = (len(all_tb) * wp * 0.0007 +   # 买入: 万7
                len(all_ts) * wp * 0.0012)     # 卖出: 万12
        nr = dr - cost
        nav *= (1 + nr)
        nav_h.append(nav)
        rets.append(nr)
        nav_dates.append(nds)
        mode_str = f"lu{lu_weight:.0%} sc{sc_weight:.0%}"
        position_history.append({
            "date": nds, "codes": all_holdings,
            "daily_ret": round(nr, 6),
            "mode": mode_str,
            "sc_n": len(sc_holdings),
            "lu_n": len(lu_holdings),
            "sc_weight": round(sc_weight, 2)})

        if (dc) % 100 == 0:
            logger.info(f"  进度 {dc}/{len(all_dates)-1} | NAV: {nav:.4f} | "
                        f"小票{len(sc_holdings)}只 涨停{len(lu_holdings)}只 "
                        f"lu={lu_weight:.0%} sc={sc_weight:.0%}")

    # 最后一个窗口
    win_ret = nav / nav_window_start - 1
    window_metrics.append({
        "win": len(window_metrics) + 1,
        "ret": round(win_ret * 100, 1)})

    # ── 指标 ──
    na = np.array(nav_h)
    td = len(na) - 1
    if td < 10:
        logger.error("天数不足")
        return

    tr = (na[-1] / na[0] - 1) * 100
    yr = max(td / 252, 0.2)
    cagr = ((na[-1] / na[0]) ** (1 / yr) - 1) * 100
    dr_arr = np.array(rets)
    sh = (float(np.sqrt(252) * np.mean(dr_arr) / np.std(dr_arr))
          if np.std(dr_arr) > 0 else 0)
    pk = np.maximum.accumulate(na)
    mdd = float(np.max((pk - na) / pk) * 100)
    wr = float(np.mean(dr_arr > 0)) * 100
    elapsed = time.time() - t0

    # 年度统计
    dates_dt = pd.DatetimeIndex([pd.Timestamp(d) for d in nav_dates])
    rets_s = pd.Series(rets, index=dates_dt[1:])
    yearly = rets_s.groupby(rets_s.index.year).apply(
        lambda x: (1 + x).prod() - 1)

    # 平均持仓
    avg_sc = np.mean([p.get("sc_n", 0) for p in position_history])
    avg_lu = np.mean([p.get("lu_n", 0) for p in position_history])

    print(f"\n{'='*70}")
    print(f"  大小票平滑分配 v4.0 —— 涨停替代大盘 + 权重上限70% + 回撤控制")
    print(f"  小票:{len(sc_codes)}只(ML 11因子)  涨停侧:{len(lc_codes)}只(规则筛选)")
    print(f"  涨停条件: 市值{LU_MCAP_MIN}-{LU_MCAP_MAX}亿 | 股价{LU_PRICE_MIN}-{LU_PRICE_MAX}元")
    print(f"          MA5>MA10 | 涨停>{LU_COUNT}次 | 无跌停 | {args.lu_min_cond}/5通过")
    print(f"  分配: 强势→涨停加码(上限70%) | 回撤>20%→涨停减半 | 弱势→小票防御")
    print(f"  耗时: {elapsed:.0f}s")
    print(f"{'='*70}")
    print(f"  累计收益:      {tr:>10.1f}%")
    print(f"  年化收益:      {cagr:>10.1f}%")
    print(f"  Sharpe Ratio:  {sh:>10.2f}")
    print(f"  最大回撤:      {mdd:>10.1f}%")
    print(f"  胜率:          {wr:>10.1f}%")
    print(f"  天数:          {td:>10}")
    print(f"  日均小票持仓:   {avg_sc:>10.1f}只")
    print(f"  日均涨停持仓:   {avg_lu:>10.1f}只")
    print(f"{'─'*70}")
    if window_metrics:
        wins = ["W{}:{:+.1f}%".format(w["win"], w["ret"])
                for w in window_metrics]
        print(f"  WF窗口: {wins}")
    print(f"\n  年度收益:")
    for yr_v, r in yearly.items():
        print(f"    {yr_v}: {r*100:+.1f}%")

    # ── 写入 DB ──
    try:
        with engine.begin() as conn:
            sc = conn.execute(text(
                "SELECT id FROM strategy_configs WHERE name='大小票平滑分配'"
            )).fetchone()
            sid = sc[0] if sc else conn.execute(text(
                "INSERT INTO strategy_configs (name, type, description) "
                "VALUES ('大小票平滑分配', 'ml', 'CSI1000相对MA60偏离度→涨停/小票比例平滑调整') RETURNING id"
            )).fetchone()[0]
            sv = conn.execute(text(
                "SELECT id FROM strategy_versions WHERE strategy_id=:s AND version='v4.0'"
            ), {"s": sid}).fetchone()
            vid = sv[0] if sv else conn.execute(text(
                "INSERT INTO strategy_versions (strategy_id, version, algorithm_type, feature_list_version) "
                "VALUES (:s, 'v4.0', 'dual_switch_lu', 'switch_v4') RETURNING id"
            ), {"s": sid}).fetchone()[0]

            m = {
                "start_date": args.start, "end_date": args.end,
                "total_return": round(tr / 100, 4),
                "annual_return": round(cagr / 100, 4),
                "sharpe": round(sh, 3),
                "max_drawdown": round(mdd / 100, 4),
                "win_rate": round(wr / 100, 4),
                "n_days": td,
                "wf_windows": len(window_metrics),
                "lu_weight_cap": 0.7,
                "dd_control": 0.20,
                "position_history": position_history[-40:],
            }
            eq_dict = {nav_dates[i]: round(float(v), 6)
                       for i, v in enumerate(nav_h) if i < len(nav_dates)}
            ret_dict = {nav_dates[i]: round(float(v), 6)
                        for i, v in enumerate(rets) if i < len(nav_dates)}
            conn.execute(text(
                "INSERT INTO backtest_results (version_id, start_date, end_date, "
                "quality, metrics_json, equity_curve_json, daily_returns_json) "
                "VALUES (:v, :s, :e, 'valid', :m, :eq, :dr)"
            ), {
                "v": vid, "s": args.start, "e": args.end,
                "m": json.dumps(m, ensure_ascii=False),
                "eq": json.dumps(eq_dict),
                "dr": json.dumps(ret_dict),
            })
            logger.info(f"已写入DB (大小票平滑分配 v4.0, version_id={vid})")
    except Exception as e:
        logger.warning(f"DB写入失败: {e}")
    engine.dispose()

    return {
        "nav": nav_h, "rets": rets, "dates": nav_dates,
        "total_return": tr, "cagr": cagr, "sharpe": sh, "max_dd": mdd,
        "win_rate": wr, "yearly": yearly,
        "positions": position_history,
    }


if __name__ == "__main__":
    main()
