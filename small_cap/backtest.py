#!/usr/bin/env python
"""小市值 alpha 策略回测 v2.0 — 快速向量化版本。

候选池：成交额排名 1000-3000（中低流动性小票）
因子：价值+反转+低波+质量（向量化计算）
调仓：周度 (forward_days=5, rebalance_freq=5)
"""
from __future__ import annotations
import sys, os, argparse, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from loguru import logger
from sqlalchemy import text
from data.db import get_engine
from models.trainer import walk_forward_train_by_regime
from models.regime import detect_regime, REGIME_PARAMS
from config.settings import TradingConfig
from portfolio.selector import select_topk_ndrop, filter_stocks


def load_universe(engine, start, end, min_amount=500_000, rank_lo=1000, rank_hi=3000):
    """加载小市值候选池：按成交额升序排名，取中间段。"""
    sql = f"""
        SELECT code FROM (
            SELECT code, ROW_NUMBER() OVER (ORDER BY SUM(amount) ASC) AS rn
            FROM stock_daily WHERE trade_date BETWEEN '{start}' AND '{end}'
            GROUP BY code HAVING SUM(amount)/COUNT(*) > {min_amount}
        ) t WHERE rn BETWEEN {rank_lo} AND {rank_hi}
    """
    return pd.read_sql(sql, engine)["code"].tolist()


def load_ohlcv(engine, codes, start, end):
    cl = ",".join([f"'{c}'" for c in codes])
    df = pd.read_sql(f"""
        SELECT code, trade_date, open, high, low, close, volume, amount, turnover
        FROM stock_daily WHERE code IN ({cl})
        AND trade_date BETWEEN '{start}' AND '{end}'
        ORDER BY code, trade_date
    """, engine)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


def load_financial(engine, codes):
    cl = ",".join([f"'{c}'" for c in codes])
    try:
        return pd.read_sql(f"""
            SELECT code, report_date, roe, gross_margin, total_assets, total_liability,
                   goodwill, holder_equity, operating_cash_flow, net_profit, bps, eps, revenue
            FROM stock_financial WHERE code IN ({cl}) ORDER BY code, report_date
        """, engine)
    except Exception:
        return None


def build_factors(ohlcv, fin_df):
    """向量化因子计算。所有操作在单个 DataFrame 上用 groupby().transform()。"""
    df = ohlcv.sort_values(["code", "trade_date"]).copy()
    g = df.groupby("code")
    c = df["close"]
    t = df.get("turnover", pd.Series(0, index=df.index))

    # === 反转 ===
    df["rev_20"] = -c.groupby(df["code"]).transform(lambda x: x.pct_change(20))
    df["rev_60"] = -c.groupby(df["code"]).transform(lambda x: x.pct_change(60))
    # RSI 简化
    delta = c.groupby(df["code"]).transform(lambda x: x.diff())
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.groupby(df["code"]).transform(lambda x: x.rolling(14, min_periods=5).mean())
    avg_loss = loss.groupby(df["code"]).transform(lambda x: x.rolling(14, min_periods=5).mean())
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))
    df["rsi_rev"] = -df["rsi_14"]

    # === 波动率 ===
    ret_1d = c.groupby(df["code"]).transform(lambda x: x.pct_change())
    df["vol_20"] = -ret_1d.groupby(df["code"]).transform(lambda x: x.rolling(20, min_periods=10).std())
    # 60日回撤恢复
    hh = c.groupby(df["code"]).transform(lambda x: x.rolling(60, min_periods=20).max())
    df["dd_rev"] = -(c - hh) / hh.replace(0, np.nan)

    # === 换手 ===
    if t.notna().sum() > 0:
        t_ma5 = t.groupby(df["code"]).transform(lambda x: x.rolling(5, min_periods=3).mean())
        df["tov_shock"] = -(t / t_ma5.shift(1).replace(0, np.nan) - 1).abs()

    # === 价值 + 质量 (从 financial 做 merge_asof, 45天公告滞后) ===
    df["pb_raw"] = np.nan
    df["pe_raw"] = np.nan
    df["roe_raw"] = np.nan
    df["debt_raw"] = np.nan
    df["gw_raw"] = np.nan
    if fin_df is not None and not fin_df.empty:
        fin = fin_df.copy()
        fin["report_date"] = pd.to_datetime(fin["report_date"])
        # 45 天公告滞后：数据在实际公告后才可用
        fin["avail_date"] = fin["report_date"] + pd.DateOffset(days=45)
        fin["_roe"] = fin["roe"].replace([np.inf, -np.inf], np.nan)
        fin["_debt"] = np.where(fin["total_assets"].fillna(0) > 0,
                                 fin["total_liability"].fillna(0) / fin["total_assets"], np.nan)
        fin["_gw"] = np.where(fin["holder_equity"].fillna(0) > 0,
                               fin["goodwill"].fillna(0) / fin["holder_equity"], np.nan)
        fin["_bps"] = fin["bps"].replace(0, np.nan)
        fin["_eps"] = fin["eps"].replace(0, np.nan)
        for dt in sorted(df["trade_date"].unique()):
            dt_ts = pd.Timestamp(dt)
            avail = fin[fin["avail_date"] <= dt_ts]
            if avail.empty:
                continue
            latest = avail.sort_values("report_date").groupby("code").tail(1).set_index("code")
            mask = df["trade_date"] == dt_ts
            for fin_col, out_col in [
                ("_roe", "roe_raw"), ("_debt", "debt_raw"), ("_gw", "gw_raw"),
                ("_bps", "bps_val"), ("_eps", "eps_val")
            ]:
                if fin_col in latest.columns:
                    df.loc[mask, out_col] = df.loc[mask, "code"].map(latest[fin_col])
            # PB = close / bps (bps 来自最新财报)
            if "_bps" in latest.columns:
                bps_map = latest["_bps"]
                for idx in df[mask].index:
                    cd = df.loc[idx, "code"]
                    bps_v = bps_map.get(cd, np.nan)
                    close_v = df.loc[idx, "close"]
                    if pd.notna(bps_v) and bps_v > 0 and pd.notna(close_v):
                        df.loc[idx, "pb_raw"] = close_v / bps_v
            # PE = close / eps → E/P 收益率（越高越便宜）
            if "_eps" in latest.columns:
                eps_map = latest["_eps"]
                for idx in df[mask].index:
                    cd = df.loc[idx, "code"]
                    eps_v = eps_map.get(cd, np.nan)
                    close_v = df.loc[idx, "close"]
                    if pd.notna(eps_v) and eps_v > 0 and pd.notna(close_v):
                        df.loc[idx, "pe_raw"] = eps_v / close_v  # E/P = earnings yield

    # === ret_5d label + 截面排名 ===
    df["ret_5d"] = c.groupby(df["code"]).transform(lambda x: x.pct_change(5).shift(-5))

    # 全量因子 (IC 分析证实小票中低 PB/PE = 价值陷阱，高 PB/PE 反而好)
    factor_raws = ["rev_20", "rev_60", "rsi_rev", "vol_20", "dd_rev",
                   "pb_raw", "pe_raw", "roe_raw", "debt_raw", "gw_raw"]
    if "tov_shock" in df.columns:
        factor_raws.append("tov_shock")
    rank_ascending = {
        "pb_raw": False,   # 高 PB (成长型) 优于低 PB (价值陷阱)
        "pe_raw": False,   # 低 E/P (高 PE 成长) 优于高 E/P (价值陷阱)
        "debt_raw": True,  # 低负债
        "gw_raw": True,    # 低商誉
        "vol_20": True,    # 低波
    }
    for col in factor_raws:
        if col not in df.columns:
            continue
        ascending = rank_ascending.get(col, True)
        df[f"{col}_rank"] = df.groupby("trade_date")[col].rank(pct=True, ascending=ascending)

    factor_cols = [c for c in df.columns if c.endswith("_rank")]
    logger.info(f"因子: {len(factor_cols)} 个 — {factor_cols}")
    return df, factor_cols


def _filter_sc_risks(candidates, day_data, fin_df, trade_date):
    """小票专项排雷：退市风险、商誉暴雷、庄股。返回要排除的 code 集合。"""
    exclude = set()
    dt_ts = pd.Timestamp(trade_date)
    day_codes = set(candidates["code"].unique())

    # 1. 退市风险：名字含 *ST 或 退
    if "name" in candidates.columns:
        for _, row in candidates.iterrows():
            name = str(row.get("name", ""))
            if "*ST" in name or "退市" in name or "ST" in name:
                exclude.add(row["code"])

    # 2. 庄股检测：近20日日均换手率 < 0.05% 且 近5日价格几乎不变
    dd_slice = day_data[day_data["code"].isin(day_codes)]
    if "turnover" in dd_slice.columns:
        for code in day_codes:
            code_data = dd_slice[dd_slice["code"] == code]
            if code_data.empty:
                continue
            tov = code_data["turnover"]
            close = code_data["close"]
            # 用当天数据做快照（实际应看历史20日，这里简化）
            if len(tov) > 0 and tov.mean() < 0.0005:  # 换手率 < 0.05%
                if len(close) > 0 and close.std() / close.mean() < 0.01 if close.mean() > 0 else True:
                    exclude.add(code)

    # 3. 商誉暴雷：商誉/净资产 > 50%（从财务数据，带45天滞后）
    if fin_df is not None and not fin_df.empty:
        fin = fin_df.copy()
        fin["report_date"] = pd.to_datetime(fin["report_date"])
        cutoff = dt_ts - pd.DateOffset(days=45)
        fin_cut = fin[fin["report_date"] <= cutoff]
        if not fin_cut.empty:
            latest = fin_cut.sort_values("report_date").groupby("code").tail(1)
            for _, row in latest.iterrows():
                code = row["code"]
                if code not in day_codes:
                    continue
                equity = row.get("holder_equity", 0) or 1
                goodwill = row.get("goodwill", 0) or 0
                if goodwill / (abs(equity) + 1e-8) > 0.5:
                    exclude.add(code)

    return exclude


def main():
    t0 = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="20200101")
    parser.add_argument("--end", default="20260529")
    parser.add_argument("--train-years", type=int, default=3)
    parser.add_argument("--top-n", type=int, default=15)
    parser.add_argument("--ndrop-n", type=int, default=2)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--ncodes", type=int, default=2000, help="候选池大小")
    args = parser.parse_args()

    if args.quick:
        args.start = "20230101"
        args.train_years = 1
        args.top_n = 10

    engine = get_engine()

    # 1. 候选池
    codes = load_universe(engine, args.start, args.end, rank_lo=1000, rank_hi=1000 + args.ncodes)
    logger.info(f"候选池: {len(codes)} 只")

    # 2. 数据加载
    ohlcv = load_ohlcv(engine, codes, args.start, args.end)
    fin_df = load_financial(engine, codes)
    index_df = pd.read_sql(f"SELECT trade_date,close FROM index_daily WHERE code='000001' AND trade_date BETWEEN '{args.start}' AND '{args.end}' ORDER BY trade_date", engine)
    index_df["trade_date"] = pd.to_datetime(index_df["trade_date"])
    # 小票趋势过滤：CSI1000 60日均线
    csi1k = pd.read_sql(f"SELECT trade_date,close FROM index_daily WHERE code='000852' AND trade_date BETWEEN '{args.start}' AND '{args.end}' ORDER BY trade_date", engine)
    csi1k["trade_date"] = pd.to_datetime(csi1k["trade_date"])
    csi1k["ma60"] = csi1k["close"].rolling(60, min_periods=30).mean()
    csi_trend = dict(zip(csi1k["trade_date"].dt.strftime("%Y-%m-%d"),
                         csi1k["close"] > csi1k["ma60"]))

    # 3. 因子
    dataset, factor_cols = build_factors(ohlcv, fin_df)
    dataset = dataset.dropna(subset=["ret_5d"])

    # 4. Regime
    regime_df = detect_regime(index_df)

    # 5. Walk-Forward 训练
    # 创建 label: ret_5d > 0 → 1 (5日后上涨)
    dataset["label"] = (dataset["ret_5d"] > 0).astype(int)
    dataset = dataset.dropna(subset=["label"])
    logger.info(f"Walk-Forward: {len(factor_cols)} 因子, train={args.train_years}y")
    ml_results = walk_forward_train_by_regime(
        dataset, factor_cols, regime_df,
        train_years=args.train_years, val_years=1,
    )
    logger.info(f"窗口: {len(ml_results)} 个")

    # 6. 仿真
    all_dates = sorted(dataset["trade_date"].unique())
    # 价格查找表
    pl = {}
    pv = ohlcv.pivot_table(index="trade_date", columns="code", values="close", aggfunc="last")
    for idx in pv.index:
        pl[str(idx)[:10]] = {c: float(pv.loc[idx, c]) for c in pv.columns if pd.notna(pv.loc[idx, c])}
    ol = {}
    pv2 = ohlcv.pivot_table(index="trade_date", columns="code", values="open", aggfunc="last")
    for idx in pv2.index:
        ol[str(idx)[:10]] = {c: float(pv2.loc[idx, c]) for c in pv2.columns if pd.notna(pv2.loc[idx, c])}

    nav, holdings, cb, dc = 1.0, [], {}, 0
    nav_h, rets = [nav], []
    nav_dates = [str(all_dates[0])[:10]] if all_dates else ["start"]
    window_metrics = []
    position_history = []  # 记录每个调仓日的持仓（Web 展示用）

    for wi, wr in enumerate(ml_results):
        ens = wr.get("ensemble") or wr.get("model")
        if ens is None:
            continue
        nav_start = nav  # 记录窗口起始 NAV（用于过拟合检测）
        vs = pd.Timestamp(wr["val_start"])
        ve = pd.Timestamp(wr["val_end"])
        vd = dataset[(dataset["trade_date"] >= vs) & (dataset["trade_date"] <= ve)]
        if vd.empty:
            continue
        vdates = sorted(vd["trade_date"].unique())
        if len(vdates) < 5:
            continue

        for i, dt in enumerate(vdates[:-1]):
            dd = vd[vd["trade_date"] == dt]
            nd = vdates[i + 1]
            if dd.empty:
                continue
            ds = str(dt)[:10]
            reg = "sideways"
            if regime_df is not None:
                tr = regime_df[regime_df["trade_date"] <= dt]
                reg = str(tr["regime"].iloc[-1]) if not tr.empty else "sideways"
            rp = REGIME_PARAMS.get(reg, REGIME_PARAMS["sideways"])
            rb = rp.get("rebalance_freq", 5)
            is_rb = (dc % rb == 0) or (not holdings)

            tb = set()
            ts = set()

            if is_rb:
                dfac = dd[["code"] + factor_cols].fillna(0.5).replace([np.inf, -np.inf], 0.5)
                dfac["regime"] = reg
                preds = ens.predict(dfac)
                if preds.empty:
                    continue
                # 过滤：ST/次新 + 小票专项排雷
                cand = dd[dd["code"].isin(preds["code"])]
                cand = filter_stocks(cand[["code"]].drop_duplicates().assign(name=cand["code"]),
                                     ref_date=dt, exclude_st=True, min_list_days=120)
                # 小票专项排雷
                sc_risks = _filter_sc_risks(cand, dd, fin_df, dt)
                cand = cand[~cand["code"].isin(sc_risks)]
                valid = set(cand["code"].unique())
                preds = preds[preds["code"].isin(valid)]
                if preds.empty:
                    continue

                ss = pd.Series(preds["score"].values, index=preds["code"].values).sort_values(ascending=False)
                # 小票趋势过滤：CSI1000 < MA60 → 半仓
                sc_trend_up = csi_trend.get(ds, True)
                eff_top_n = max(3, rp["top_n"] // 2) if not sc_trend_up else rp["top_n"]
                nh, tb, ts = select_topk_ndrop(ss, set(holdings), K=eff_top_n, N=args.ndrop_n)

                # 成本基础
                no = ol.get(str(nd)[:10], {})
                tc = pl.get(ds, {})
                for c in tb:
                    ep = no.get(c, tc.get(c, 0))
                    if ep > 0:
                        cb[c] = ep
                for c in ts:
                    cb.pop(c, None)
                holdings = list(nh)

            dc += 1
            tcodes = list(holdings)
            if not tcodes:
                nav_h.append(nav)
                rets.append(0.0)
                nav_dates.append(str(nd)[:10])
                continue

            nc = pl.get(str(nd)[:10], {})
            tcmap = pl.get(ds, {})
            dret_vals = []
            for c in tcodes:
                ncp = nc.get(c, 0)
                if c in tb:
                    e = cb.get(c, 0)
                    if e > 0 and ncp > 0:
                        dret_vals.append(ncp / e - 1)
                else:
                    pc = tcmap.get(c, 0)
                    if pc > 0 and ncp > 0:
                        dret_vals.append(ncp / pc - 1)
            dr = float(np.mean(dret_vals)) if dret_vals else 0.0

            wp = 1.0 / max(len(tcodes), 1)
            SC_COMM = TradingConfig.COMMISSION
            SC_STAMP = TradingConfig.STAMP_DUTY
            SC_SLIP = 0.003  # 小票滑点 0.3%
            cost = (len(tb) * wp * (SC_COMM + SC_SLIP) +
                    len(ts) * wp * (SC_COMM + SC_STAMP + SC_SLIP))
            nr = dr - cost
            nav *= (1 + nr)
            nav_h.append(nav)
            rets.append(nr)
            nds_date = str(nd)[:10]
            nav_dates.append(nds_date)
            # 每天记录持仓及日收益（Web 展示用）
            pos_entry = {"date": nds_date, "codes": list(tcodes), "daily_ret": round(nr, 6)}
            if is_rb:
                pos_entry["n_buy"] = len(tb)
                pos_entry["n_sell"] = len(ts)
            position_history.append(pos_entry)

        # 逐窗口追踪（过拟合检测：各窗口收益应同向，不应一个暴涨一个暴跌）
        win_ret = nav / nav_start - 1
        window_metrics.append({"win": wi+1, "start": str(wr['val_start'])[:10],
                                "end": str(wr['val_end'])[:10],
                                "ret": round(win_ret * 100, 1)})

    # 7. 指标
    na = np.array(nav_h)
    td = len(na) - 1
    if td < 10:
        logger.error("仿真天数不足")
        return
    tr_pct = (na[-1] / na[0] - 1) * 100
    yr = max(td / 252, 0.2)
    cagr_pct = ((na[-1] / na[0]) ** (1 / yr) - 1) * 100
    dr_arr = np.array(rets)
    sh = float(np.sqrt(252) * np.mean(dr_arr) / np.std(dr_arr)) if np.std(dr_arr) > 0 else 0
    pk = np.maximum.accumulate(na)
    mdd_pct = float(np.max((pk - na) / pk) * 100)
    wr_pct = float(np.mean(dr_arr > 0)) * 100

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"小市值 alpha v2.0")
    annual_turnover = sum(len(p["codes"]) for p in position_history) / max(len(position_history), 1)
    print(f"候选池: {len(codes)}只 | 因子: {len(factor_cols)}个 | 滑点:0.3% | 耗时: {elapsed:.0f}s")
    print(f"总收益: {tr_pct:.1f}%  年化: {cagr_pct:.1f}%  Sharpe: {sh:.2f}  最大回撤: {mdd_pct:.1f}%")
    print(f"胜率: {wr_pct:.1f}%  天数: {td}  周均持仓: {annual_turnover:.0f}只  {args.start}-{args.end}")
    # 过拟合检测
    win_rets = [w["ret"] for w in window_metrics]
    if len(win_rets) >= 2:
        n_pos = sum(1 for r in win_rets if r > 0)
        win_strs = ["W{}:{:+.1f}%".format(w["win"], w["ret"]) for w in window_metrics]
        print(f"WF窗口: {win_strs}")
        # 过拟合检测：标准差/均值比（变异系数），越小越稳定
        win_rets_arr = np.array(win_rets)
        cv = float(np.std(win_rets_arr) / (abs(np.mean(win_rets_arr)) + 1))  # +1防除零
        print(f"  窗口收益波动(CV)={cv:.2f} {'✅稳定' if cv < 2 else '⚠️窗口间波动大'}")
        if n_pos == len(win_rets) or n_pos == 0:
            sign_label = "全正" if n_pos > 0 else "全负"
            print(f"  ⚠️ 过拟合风险：所有{len(win_rets)}个窗口{sign_label}，可能过拟合到特定市场")
        else:
            print(f"  ✅ 窗口一致性：{n_pos}/{len(win_rets)} 正收益，策略较稳健")

    # 写入 DB
    try:
        with engine.begin() as conn:
            sc = conn.execute(text("SELECT id FROM strategy_configs WHERE name = '小市值alpha'")).fetchone()
            sid = sc[0] if sc else conn.execute(text(
                "INSERT INTO strategy_configs (name, type, description) VALUES ('小市值alpha', 'ml', '小市值价值+质量+反转 v2.0') RETURNING id"
            )).fetchone()[0]
            sv = conn.execute(text("SELECT id FROM strategy_versions WHERE strategy_id=:s AND version='v2.0'"), {"s": sid}).fetchone()
            vid = sv[0] if sv else conn.execute(text(
                "INSERT INTO strategy_versions (strategy_id, version, algorithm_type, feature_list_version) VALUES (:s,'v2.0','small_cap','sc_v2.0') RETURNING id"
            ), {"s": sid}).fetchone()[0]
            m = {"start_date": args.start, "end_date": args.end,
                 "total_return": round(tr_pct / 100, 4), "annual_return": round(cagr_pct / 100, 4),
                 "sharpe": round(sh, 3), "max_drawdown": round(mdd_pct / 100, 4),
                 "win_rate": round(wr_pct / 100, 4), "n_days": td, "n_factors": len(factor_cols),
                 "universe": "小市值(成交额1000-3000)", "rebalance_freq": 5,
                 "position_history": position_history[-40:]}
            eq_dict = {nav_dates[i]: round(v, 6) for i, v in enumerate(nav_h) if i < len(nav_dates)}
            ret_dict = {nav_dates[i]: round(v, 6) for i, v in enumerate(rets) if i < len(nav_dates)}
            conn.execute(text("""
                INSERT INTO backtest_results (version_id, start_date, end_date, quality, metrics_json, equity_curve_json, daily_returns_json)
                VALUES (:v, :s, :e, 'valid', :m, :eq, :dr)
            """), {"v": vid, "s": args.start, "e": args.end, "m": json.dumps(m, ensure_ascii=False),
                   "eq": json.dumps(eq_dict), "dr": json.dumps(ret_dict)})
            logger.info(f"已写入 DB (version_id={vid})")
    except Exception as e:
        logger.warning(f"DB写入失败: {e}")

    engine.dispose()


if __name__ == "__main__":
    main()
