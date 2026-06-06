#!/usr/bin/env python
"""大小票智能切换策略。

CSI1000 > MA60 → 小票反转（跌多的2000只小票里选15只）
CSI1000 < MA60 → 大票反转（跌多的200只大票里选10只）

前视审计: 切换信号(MA60) + 因子(mom_20) 都是纯历史数据, T+1开盘执行。
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
from small_cap.backtest import load_universe, load_financial, build_factors as build_sc_factors, _filter_sc_risks


def main():
    t0 = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="20200101")
    parser.add_argument("--end", default="20260529")
    parser.add_argument("--train-years", type=int, default=3)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.start = "20230101"; args.train_years = 1

    engine = get_engine()

    # ── 双候选池 ──
    sc_codes = load_universe(engine, args.start, args.end, rank_lo=1000, rank_hi=3000)
    lc_sql = f"SELECT code FROM (SELECT code, ROW_NUMBER() OVER (ORDER BY SUM(amount) DESC) AS rn FROM stock_daily WHERE trade_date BETWEEN '{args.start}' AND '{args.end}' GROUP BY code) t WHERE rn <= 200"
    lc_codes = pd.read_sql(lc_sql, engine)["code"].tolist()
    logger.info(f"小票:{len(sc_codes)}只  大票:{len(lc_codes)}只")

    # ── 数据加载 ──
    def load_ohlcv(codes, label=""):
        cl = ",".join([f"'{c}'" for c in codes])
        df = pd.read_sql(f"SELECT code,trade_date,open,high,low,close,volume,amount,turnover FROM stock_daily WHERE code IN ({cl}) AND trade_date BETWEEN '{args.start}' AND '{args.end}' ORDER BY code,trade_date", engine)
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        return df

    sc_ohlcv = load_ohlcv(sc_codes)
    lc_ohlcv = load_ohlcv(lc_codes)
    fin_df = load_financial(engine, sc_codes)

    # 大票因子（纯动量，不用质量因子）
    lc = lc_ohlcv.sort_values(["code", "trade_date"])
    c_lc = lc.groupby("code")["close"]
    lc["mom_20"] = c_lc.transform(lambda x: x.pct_change(20))
    lc["ret_5d"] = c_lc.transform(lambda x: x.pct_change(5).shift(-5))
    lc["mom_20_rank"] = lc.groupby("trade_date")["mom_20"].rank(pct=True, ascending=True) # 跌最多=rank高
    lc["label"] = (lc["ret_5d"] > 0).astype(int)
    lc_dataset = lc.dropna(subset=["ret_5d", "label", "mom_20_rank"])

    # 小票因子（完整11因子）
    sc_dataset, sc_factors = build_sc_factors(sc_ohlcv, fin_df)
    sc_dataset = sc_dataset.dropna(subset=["ret_5d"])
    sc_dataset["label"] = (sc_dataset["ret_5d"] > 0).astype(int)
    sc_dataset = sc_dataset.dropna(subset=["label"])

    # ── 指数 + 趋势 ──
    index_df = pd.read_sql(f"SELECT trade_date,code,close FROM index_daily WHERE code IN ('000001','000852') AND trade_date BETWEEN '{args.start}' AND '{args.end}' ORDER BY trade_date", engine)
    index_df["trade_date"] = pd.to_datetime(index_df["trade_date"])
    sh_idx = index_df[index_df["code"]=="000001"][["trade_date","close"]]
    csi1k = index_df[index_df["code"]=="000852"].copy()
    csi1k["ma60"] = csi1k["close"].rolling(60, min_periods=30).mean()
    csi1k["ret_5d"] = csi1k["close"].pct_change(5)  # 5日动量
    csi_trend = dict(zip(csi1k["trade_date"].dt.strftime("%Y-%m-%d"), csi1k["close"] > csi1k["ma60"]))
    csi_mom = dict(zip(csi1k["trade_date"].dt.strftime("%Y-%m-%d"), csi1k["ret_5d"]))  # 快刹车信号

    # ── ML 训练 ──
    regime_df = detect_regime(sh_idx)
    sc_ml = walk_forward_train_by_regime(sc_dataset, sc_factors, regime_df, train_years=args.train_years, val_years=1)
    lc_ml = walk_forward_train_by_regime(lc_dataset, ["mom_20_rank"], regime_df, train_years=args.train_years, val_years=1)
    logger.info(f"ML: 小票{len(sc_ml)}窗  大票{len(lc_ml)}窗")

    # ── 价格表 ──
    def mk_lookup(ohlcv, col="close"):
        r = {}
        pv = ohlcv.pivot_table(index="trade_date", columns="code", values=col, aggfunc="last")
        for idx in pv.index:
            r[str(idx)[:10]] = {c: float(pv.loc[idx, c]) for c in pv.columns if pd.notna(pv.loc[idx, c])}
        return r
    sc_close = mk_lookup(sc_ohlcv); sc_open = mk_lookup(sc_ohlcv, "open")
    lc_close = mk_lookup(lc_ohlcv); lc_open = mk_lookup(lc_ohlcv, "open")

    # ── 仿真 ──
    all_dates = sorted(set(sc_dataset["trade_date"].unique()) & set(lc_dataset["trade_date"].unique()))
    # 仅验证期
    val_ranges = [(pd.Timestamp(w["val_start"]), pd.Timestamp(w["val_end"])) for w in sc_ml]
    all_dates = [d for d in all_dates if any(vs<=d<=ve for vs,ve in val_ranges)]
    logger.info(f"验证期: {len(all_dates)}天")

    nav = 1.0
    holdings, cb = [], {}
    nav_h, rets, nav_dates = [1.0], [], [str(all_dates[0])[:10]]
    window_metrics = []
    dc = 0

    sc_model_idx = lc_model_idx = 0
    sc_ens = sc_ml[0].get("ensemble") or sc_ml[0].get("model") if sc_ml else None
    lc_ens = lc_ml[0].get("ensemble") or lc_ml[0].get("model") if lc_ml else None
    position_history = []
    nav_window_start = nav; current_window = 0

    for i, dt in enumerate(all_dates[:-1]):
        nd = all_dates[i+1]; ds, nds = str(dt)[:10], str(nd)[:10]
        sc_dd = sc_dataset[sc_dataset["trade_date"]==dt]
        lc_dd = lc_dataset[lc_dataset["trade_date"]==dt]

        # WF窗口切换
        if sc_model_idx+1 < len(sc_ml) and dt >= pd.Timestamp(sc_ml[sc_model_idx+1]["val_start"]):
            win_ret = nav / nav_window_start - 1
            window_metrics.append({"win": sc_model_idx+1, "ret": round(win_ret*100,1)})
            sc_model_idx += 1; sc_ens = sc_ml[sc_model_idx].get("ensemble") or sc_ml[sc_model_idx].get("model")
            nav_window_start = nav
        if lc_model_idx+1 < len(lc_ml) and dt >= pd.Timestamp(lc_ml[lc_model_idx+1]["val_start"]):
            lc_model_idx += 1; lc_ens = lc_ml[lc_model_idx].get("ensemble") or lc_ml[lc_model_idx].get("model")

        reg = "sideways"
        tr = regime_df[regime_df["trade_date"]<=dt]
        if not tr.empty: reg = str(tr["regime"].iloc[-1])
        rp = REGIME_PARAMS.get(reg, REGIME_PARAMS["sideways"])
        is_rb = (dc % 5 == 0) or (not holdings)

        # ── 趋势强弱 → 小票/大票比例 + 总仓位 ──
        csi1k_price = csi1k[csi1k["trade_date"]==dt]["close"]
        csi1k_ma = csi1k[csi1k["trade_date"]==dt]["ma60"]
        if len(csi1k_price)>0 and len(csi1k_ma)>0 and csi1k_ma.iloc[0]>0:
            ma_dev = (csi1k_price.iloc[0] / csi1k_ma.iloc[0] - 1)
            # 大小票分配 (线性，70/30为基准)
            sc_weight = np.clip(0.3 + ma_dev * 10, 0.1, 0.9)
            total_exposure = 1.0 if ma_dev > -0.02 else max(0.5, 1.0 + ma_dev * 5)
        else:
            sc_weight = 0.5; total_exposure = 1.0

        tb, ts = set(), set()

        if is_rb:
            sc_n = 0; lc_n = 0
            # === 小票选股 ===
            if sc_ens is not None and not sc_dd.empty:
                dfac = sc_dd[["code"]+sc_factors].fillna(0.5).replace([np.inf,-np.inf],0.5)
                dfac["regime"] = reg
                preds = sc_ens.predict(dfac)
                if not preds.empty:
                    cand = sc_dd[sc_dd["code"].isin(preds["code"])]
                    cand = filter_stocks(cand[["code"]].drop_duplicates().assign(name=cand["code"]), ref_date=dt, exclude_st=True, min_list_days=120)
                    risks = _filter_sc_risks(cand, sc_dd, fin_df, dt)
                    valid = set(cand["code"].unique()) - risks
                    preds = preds[preds["code"].isin(valid)]
                    if not preds.empty:
                        ss = pd.Series(preds["score"].values, index=preds["code"].values).sort_values(ascending=False)
                        sc_n = max(3, int(rp["top_n"] * sc_weight * total_exposure))
                        sc_nh, sc_tb, sc_ts = select_topk_ndrop(ss, set(holdings), K=sc_n, N=2)
                        sc_sel = set(sc_nh)
                        tb.update(sc_tb); ts.update(sc_ts)
                        no = sc_open.get(nds,{}); tc = sc_close.get(ds,{})
                        for c in sc_tb:
                            ep = no.get(c, tc.get(c,0))
                            if ep>0: cb[c]=ep
                        for c in sc_ts: cb.pop(c,None)
                    else:
                        sc_sel = set()
                else:
                    sc_sel = set()
            else:
                sc_sel = set()

            # === 大票选股 ===
            lc_dd2 = lc_dd.copy()
            lc_dd2 = lc_dd2[lc_dd2["code"].isin(filter_stocks(lc_dd2[["code"]].drop_duplicates().assign(name=lc_dd2["code"]), ref_date=dt, exclude_st=True, min_list_days=120)["code"])]
            if len(lc_dd2) >= 10:
                lc_n = max(2, int(10 * (1-sc_weight) * total_exposure))
                top_lc = lc_dd2.nsmallest(lc_n, "mom_20")
                lc_sel = set(top_lc["code"].tolist())
                lc_tb = lc_sel - set(holdings); lc_ts = set()
                tb.update(lc_tb)
                no = lc_open.get(nds,{}); tc = lc_close.get(ds,{})
                for c in lc_tb:
                    ep = no.get(c, tc.get(c,0))
                    if ep>0: cb[c]=ep
            else:
                lc_sel = set()

            # 合并 + 清理旧持仓
            new_all = sc_sel | lc_sel
            old = set(holdings)
            tb = new_all - old
            ts = old - new_all
            holdings = list(new_all)
            for c in ts: cb.pop(c,None)

        dc += 1
        tcodes = list(holdings)
        if not tcodes:
            nav_h.append(nav); rets.append(0.0); nav_dates.append(nds)
            position_history.append({"date":nds,"codes":[],"daily_ret":0.0,"mode":"cash"})
            continue

        # 日收益（混合持仓，按权重加权）
        dr = 0.0
        if tcodes:
            total_w = len(tcodes)
            for c in tcodes:
                # 查价格：先小票表，再大票表
                ncp = sc_close.get(nds,{}).get(c,0) or lc_close.get(nds,{}).get(c,0)
                if ncp<=0: continue
                if c in tb: e = cb.get(c,0)
                else: e = sc_close.get(ds,{}).get(c,0) or lc_close.get(ds,{}).get(c,0)
                if e>0 and ncp>0:
                    r = ncp/e-1
                    if abs(r)<0.20: dr += r/total_w

        wp = 1.0/max(len(tcodes),1)
        cost = len(tb)*wp*0.0007 + len(ts)*wp*0.0012  # 平均成本
        nr = dr - cost
        nav *= (1+nr)
        nav_h.append(nav); rets.append(nr); nav_dates.append(nds)
        mode_str = f"sc{sc_weight:.0%}"
        position_history.append({"date":nds,"codes":list(tcodes),"daily_ret":round(nr,6),"mode":mode_str})

    # 最后一个窗口
    win_ret = nav / nav_window_start - 1
    window_metrics.append({"win": len(window_metrics)+1, "ret": round(win_ret*100,1)})

    # ── 指标 ──
    na = np.array(nav_h); td = len(na)-1
    if td<10: return logger.error("天数不足")
    tr = (na[-1]/na[0]-1)*100
    yr = max(td/252,0.2); cagr = ((na[-1]/na[0])**(1/yr)-1)*100
    dr_arr = np.array(rets); sh = float(np.sqrt(252)*np.mean(dr_arr)/np.std(dr_arr)) if np.std(dr_arr)>0 else 0
    pk = np.maximum.accumulate(na); mdd = float(np.max((pk-na)/pk)*100)
    wr = float(np.mean(dr_arr>0))*100
    elapsed = time.time()-t0

    print(f"\n{'='*60}")
    print(f"大小票平滑分配")
    print(f"候选池: 小票{len(sc_codes)}只/大票{len(lc_codes)}只 | 耗时:{elapsed:.0f}s")
    print(f"总收益: {tr:.1f}%  年化: {cagr:.1f}%  Sharpe: {sh:.2f}  最大回撤: {mdd:.1f}%")
    print(f"胜率: {wr:.1f}%  天数: {td}")
    if window_metrics:
        win_strs = ["W{}:{:+.1f}%".format(w["win"], w["ret"]) for w in window_metrics]
        print(f"WF窗口: {win_strs}")
    print(f"{args.start}-{args.end}")

    # DB
    try:
        with engine.begin() as conn:
            sc = conn.execute(text("SELECT id FROM strategy_configs WHERE name='大小票平滑分配'")).fetchone()
            sid = sc[0] if sc else conn.execute(text(
                "INSERT INTO strategy_configs (name, type, description) VALUES ('大小票平滑分配', 'ml', 'CSI1000相对MA60偏离度→大小票比例平滑调整+极端弱势提现') RETURNING id"
            )).fetchone()[0]
            sv = conn.execute(text("SELECT id FROM strategy_versions WHERE strategy_id=:s AND version='v1.0'"), {"s":sid}).fetchone()
            vid = sv[0] if sv else conn.execute(text(
                "INSERT INTO strategy_versions (strategy_id, version, algorithm_type, feature_list_version) VALUES (:s,'v1.0','dual_switch','switch_v1') RETURNING id"
            ), {"s":sid}).fetchone()[0]
            m = {"start_date":args.start,"end_date":args.end,
                 "total_return":round(tr/100,4),"annual_return":round(cagr/100,4),
                 "sharpe":round(sh,3),"max_drawdown":round(mdd/100,4),
                 "win_rate":round(wr/100,4),"n_days":td,"wf_windows":len(window_metrics),
                 "position_history": position_history[-40:]}
            eq_dict = {nav_dates[i]:round(v,6) for i,v in enumerate(nav_h) if i<len(nav_dates)}
            ret_dict = {nav_dates[i]:round(v,6) for i,v in enumerate(rets) if i<len(nav_dates)}
            conn.execute(text("INSERT INTO backtest_results (version_id,start_date,end_date,quality,metrics_json,equity_curve_json,daily_returns_json) VALUES (:v,:s,:e,'valid',:m,:eq,:dr)"), {"v":vid,"s":args.start,"e":args.end,"m":json.dumps(m,ensure_ascii=False),"eq":json.dumps(eq_dict),"dr":json.dumps(ret_dict)})
            logger.info(f"已写入DB (version_id={vid})")
    except Exception as e: logger.warning(f"DB:{e}")
    engine.dispose()

if __name__=="__main__":
    main()
