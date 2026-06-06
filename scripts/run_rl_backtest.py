#!/usr/bin/env python
"""RL 概念板块轮动回测：概念板块热度 → RL 动态板块权重 → ML 选股 + 板块偏移。

架构:
  ML Pipeline (XGBoost+LGBM) → stock scores
  RL Meta-Controller → 5 concept board group boosts
  Final score = ML_score * (1 + board_boost[stock])
  Top-K + NDrop → NAV simulation

用法:
    python scripts/run_rl_backtest.py --start 20160101 --end 20250601
    python scripts/run_rl_backtest.py --rl-off  # 纯 ML baseline
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy import text
from data.db import get_engine
from factors import ALL_FACTORS
from config.settings import TradingConfig
from models.dataset import build_factor_dataset
from models.trainer import walk_forward_train_by_regime, RegimeAwareEnsemble
from models.regime import detect_regime
from rl_dynamic.state_builder import StateBuilder
from rl_dynamic.factor_pool import FactorPool
from portfolio.selector import select_topk_ndrop


# ── 数据加载 ──────────────────────────────────────────────

def load_data(engine, start_date, end_date, universe_size=500):
    """加载 OHLCV + 指数 + extra_data。"""
    first_year = f"{int(start_date[:4])+1}{start_date[4:]}"
    codes = pd.read_sql(
        f"SELECT code FROM stock_daily "
        f"WHERE trade_date >= '{start_date}' AND trade_date <= '{first_year}' "
        f"GROUP BY code ORDER BY SUM(amount) DESC LIMIT {universe_size}",
        engine)["code"].tolist()
    code_list = ",".join([f"'{c}'" for c in codes])

    ohlcv = pd.read_sql(f"""
        SELECT code, trade_date, open, high, low, close, volume, amount, turnover
        FROM stock_daily WHERE code IN ({code_list})
        AND trade_date BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY code, trade_date
    """, engine)
    ohlcv["trade_date"] = pd.to_datetime(ohlcv["trade_date"])

    index_df = pd.read_sql(f"""
        SELECT trade_date, close FROM index_daily
        WHERE code='000001' AND trade_date BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY trade_date
    """, engine)
    index_df["trade_date"] = pd.to_datetime(index_df["trade_date"])

    extra = {}
    try:
        edf = pd.read_sql(f"SELECT code,trade_date,market_cap,pb FROM stock_daily_extra "
                          f"WHERE code IN ({code_list}) AND trade_date BETWEEN '{start_date}' AND '{end_date}'", engine)
        if not edf.empty:
            edf["log_mcap"] = np.log(edf["market_cap"].replace(0, np.nan))
            extra["log_mcap"] = edf[["code", "trade_date", "log_mcap"]]
            extra["pb"] = edf[["code", "trade_date", "pb"]]
    except Exception: pass
    try:
        fcols = ["net_profit","roe","bps","net_margin","revenue","eps","cash_flow",
                 "operating_cash_flow","total_assets","total_liability","goodwill",
                 "holder_equity","adjusted_profit"]
        fdf = pd.read_sql(f"SELECT code,report_date,{','.join(fcols)} FROM stock_financial "
                          f"WHERE code IN ({code_list}) AND report_date>='2018-01-01' ORDER BY code,report_date", engine)
        if not fdf.empty:
            for c in fcols:
                if c in fdf.columns: extra[c] = fdf[["code","report_date",c]].copy()
    except Exception: pass

    logger.info(f"数据: {len(ohlcv)}行, {ohlcv['code'].nunique()}只")
    return ohlcv, index_df, extra


# ── 板块映射 ──────────────────────────────────────────────

def load_board_map(engine, ohlcv_codes: set) -> dict:
    """加载股票→概念板块映射 (优先 TOP N 个板块的股票)。"""
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT cs.stock_code, cs.board_code
            FROM concept_stock cs
            WHERE cs.stock_code IN (
                SELECT DISTINCT code FROM stock_daily LIMIT 3000
            )
        """)).fetchall()
    mapping = {}
    for stock_code, board_code in rows:
        mapping.setdefault(stock_code, []).append(board_code)
    logger.info(f"板块映射: {len(mapping)} 只股票")
    return mapping


# ── 主流程 ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RL 概念板块轮动回测")
    parser.add_argument("--start", default="20200101")
    parser.add_argument("--end", default="20260501")
    parser.add_argument("--universe-size", type=int, default=500)
    parser.add_argument("--train-years", type=int, default=3)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--ndrop-n", type=int, default=2)
    parser.add_argument("--max-factors", type=int, default=15)
    parser.add_argument("--no-screen", action="store_true")
    parser.add_argument("--no-concept", action="store_true")
    parser.add_argument("--boost", type=float, default=0.5, help="概念板块热度固定加成系数 (0=off)")
    parser.add_argument("--rl-boost", action="store_true", help="GRPO 动态学习 boost 参数 (覆盖 --boost)")
    parser.add_argument("--rl-off", action="store_true", help="纯 ML baseline")
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    engine = get_engine()
    ohlcv, index_df, extra = load_data(engine, args.start, args.end, args.universe_size)

    # ── 因子筛选 ──
    fn = [f for f in ALL_FACTORS.keys() if f in ALL_FACTORS]  # 全量因子，匹配 web
    pool = FactorPool(fn)
    dataset = pool.compute_factors(ohlcv, extra)
    dataset["trade_date"] = pd.to_datetime(dataset["trade_date"])

    if not args.no_screen and len(fn) > args.max_factors:
        from rl_dynamic.trainer import _screen_factors
        selected = _screen_factors(dataset, fn, max_factors=args.max_factors)
    else:
        selected = fn[:args.max_factors]

    # ── 概念板块特征 ──
    concept_features = None
    cb_feature_cols = []
    if not args.no_concept:
        try:
            from rl_dynamic.concept_features import ConceptBoardFeatures
            concept_features = ConceptBoardFeatures(ohlcv, engine)
            logger.info(f"概念板块特征: {len(concept_features._cache)} 日期")

            # 把板块特征作为 ML 特征列加入 dataset
            logger.info("构建股票级概念板块特征...")
            cb_parts = []
            for dt in sorted(dataset["trade_date"].unique()):
                day_codes = dataset[dataset["trade_date"] == dt]["code"].tolist()
                if len(day_codes) < 10:
                    continue
                sf = concept_features.get_stock_features(dt, day_codes)
                sf["trade_date"] = pd.Timestamp(dt)
                cb_parts.append(sf)
            if cb_parts:
                cb_all = pd.concat(cb_parts, ignore_index=True)
                cb_all["trade_date"] = pd.to_datetime(cb_all["trade_date"])
                dataset = dataset.merge(cb_all, on=["code", "trade_date"], how="left")
                cb_feature_cols = ["cb_s_mom5", "cb_s_mom20", "cb_s_hot", "cb_s_nboards"]
                for col in cb_feature_cols:
                    if col in dataset.columns:
                        dataset[col] = dataset[col].fillna(0)
                # 把这些列加到 ML 因子列表里
                selected = selected + [c for c in cb_feature_cols if c in dataset.columns]
                logger.info(f"板块特征已加入数据集: {cb_feature_cols}")
        except Exception as e:
            logger.warning(f"概念板块加载失败: {e}")

    # ── ML 训练 (Walk-Forward, 三状态 Regime-Aware) ──
    regime_df = detect_regime(index_df)
    logger.info(f"ML 训练: {len(selected)} 因子, regime-aware, walk-forward {args.train_years}y/1y")
    ml_results = walk_forward_train_by_regime(
        dataset, selected, regime_df,
        train_years=args.train_years, val_years=1,
    )
    logger.info(f"ML 窗口: {len(ml_results)}个")

    all_dates = sorted(dataset["trade_date"].unique())
    ret_map = dataset.pivot_table(index="trade_date", columns="code",
                                   values="ret_1d", aggfunc="last")

    # 公共 StateBuilder + IC map + 预计算状态缓存
    n_sel = len(selected)
    sb = StateBuilder(n_factors=n_sel, concept_features=concept_features)
    ic_map_by_date = {}
    factor_cols_sel = [c for c in selected if c in dataset.columns]
    rolling_ic = {f: [] for f in factor_cols_sel}
    for dt in all_dates:
        day = dataset[dataset["trade_date"] == dt]
        if len(day) < 10:
            ic_map_by_date[dt] = {}
            continue
        ret_col = "ret_1d" if "ret_1d" in day.columns else None
        if ret_col:
            for f in factor_cols_sel:
                if f not in day.columns: continue
                sub = day[[f, ret_col]].replace([np.inf, -np.inf], np.nan).dropna()
                if len(sub) >= 10 and sub[f].nunique() > 1:
                    ic = sub[f].corr(sub[ret_col], method="spearman")
                    rolling_ic[f].append(float(ic) if not np.isnan(ic) else 0.0)
                else:
                    rolling_ic[f].append(0.0)
        ic_map_by_date[dt] = {
            i: float(np.mean(rolling_ic[f][-20:])) if rolling_ic[f] else 0.0
            for i, f in enumerate(factor_cols_sel)
        }

    # 预计算所有日期的状态 + 5日收益 (向量化, 只需 ~5s)
    logger.info("预计算每日状态缓存...")
    state_cache, rets_5d_cache = sb.build_all(ohlcv, index_df, ic_map_by_date)

    # ── GRPO 动态 Boost 训练（用因子 IC 做 reward，避免 ML 数据泄漏）──
    rl_controllers = {}  # window_idx → ConceptBoostController
    if args.rl_boost and concept_features is not None:
        from rl_dynamic.meta_controller import ConceptBoostController, train_boost_grpo
        logger.info("训练 GRPO 动态 Boost 控制器（因子 IC reward）...")

        for wi, wr in enumerate(ml_results):
            train_end = wr["train_end"]
            train_dates = [d for d in all_dates if d <= train_end]
            if len(train_dates) < 200:
                continue

            daily_states = {}
            factor_scores_map = {}
            fwd_rets = {}
            stock_5d_rets = {}
            for dt in train_dates:
                day = dataset[dataset["trade_date"] == dt]
                if len(day) < 10:
                    continue
                # 从缓存读状态 (无需重新 build)
                state = state_cache.get(dt)
                if state is None:
                    continue
                daily_states[str(dt.date())] = state

                # 股票列表 + 前向收益
                codes = day["code"].values
                rets = np.zeros(len(codes))
                if dt in ret_map.index:
                    for j, code in enumerate(codes):
                        if code in ret_map.columns:
                            r = ret_map.loc[dt, code]
                            if not np.isnan(r): rets[j] = r
                fwd_rets[str(dt.date())] = rets

                # 因子等权得分 (对齐 codes)
                fcols = [c for c in selected if c in day.columns]
                fmat = day[fcols].fillna(0).replace([np.inf, -np.inf], 0).values.astype(float)
                factor_scores_map[str(dt.date())] = fmat.mean(axis=1)

                # 近 5 日收益 (对齐 codes — 用 ohlcv pivot 计算确保长度一致)
                s5d = np.zeros(len(codes))
                for j, code in enumerate(codes):
                    hist = ohlcv[(ohlcv["code"] == code) &
                                 (ohlcv["trade_date"] <= dt)].tail(6)
                    if len(hist) >= 6:
                        c0, c5 = hist["close"].iloc[-1], hist["close"].iloc[-6]
                        if c5 > 0: s5d[j] = (c0 - c5) / c5
                stock_5d_rets[str(dt.date())] = s5d

            logger.info(f"  窗口{wi+1}: {len(daily_states)} 训练日")

            ctrl = ConceptBoostController(state_dim=sb.state_dim)
            ctrl = train_boost_grpo(
                ctrl, daily_states, factor_scores_map, fwd_rets, stock_5d_rets,
                epochs=args.epochs, M=8, lr=5e-3, device="cpu",
            )
            rl_controllers[wi] = ctrl

    # ── 仿真准备 ──
    INIT = TradingConfig.INITIAL_CASH
    COMM = TradingConfig.COMMISSION
    SLIP = TradingConfig.SLIPPAGE
    STAMP = TradingConfig.STAMP_DUTY

    nav = INIT; holdings = set()
    nav_history = [nav]; daily_rets = []

    for wr in ml_results:
        val_dates = [d for d in all_dates
                     if wr["train_end"] < d <= wr["val_end"]]
        if len(val_dates) < 20:
            continue

        # 用该窗口的模型仿真实证期
        ensemble = wr["ensemble"]  # RegimeAwareEnsemble
        # 确定当前验证期的大致 regime（用训练期末尾的数据）
        today_regime = "sideways"
        if regime_df is not None:
            tr = regime_df[regime_df["trade_date"] <= wr["train_end"]]
            if not tr.empty:
                today_regime = str(tr["regime"].iloc[-1])

        for dt in val_dates:
            day = dataset[dataset["trade_date"] == dt]
            if len(day) < 10:
                continue

            # Regime 路由：用该日期的市场状态
            day_regime = "sideways"
            if regime_df is not None:
                tr = regime_df[regime_df["trade_date"] <= dt]
                if not tr.empty:
                    day_regime = str(tr["regime"].iloc[-1])

            day_factor = day[["code"] + selected].fillna(0).replace([np.inf, -np.inf], 0)
            day_factor["regime"] = day_regime
            preds = ensemble.predict(day_factor)
            scores = pd.Series(preds["score"].values, index=preds["code"].values)

            # ── 概念板块热度差异化加成 ──
            use_rl = args.rl_boost and rl_controllers
            use_fixed = concept_features is not None and args.boost > 0 and not use_rl

            if use_rl or use_fixed:
                state = state_cache.get(dt)
                if state is None:
                    icm = ic_map_by_date.get(dt, {})
                    state = sb.build(ohlcv, index_df, icm, dt)

                if use_rl:
                    # GRPO 动态 boost
                    ctrl = rl_controllers.get(0)  # 使用第一个窗口的控制器（全局）
                    if ctrl is not None:
                        bp = ctrl.predict(state)
                        boost_level = float(bp[0])
                        rot_sens = float(bp[1])
                    else:
                        boost_level, rot_sens = 0.5, 5.0
                else:
                    # 固定启发式 boost
                    boost_level = args.boost
                    rot_sens = 5.0

                hot_mom = np.array(state[16:21], dtype=float)
                cb_rotation = float(state[23])
                hot_strength = float(np.mean(np.maximum(hot_mom, 0)))
                rotation_penalty = max(0, 1.0 - cb_rotation * rot_sens)
                global_boost = hot_strength * rotation_penalty * boost_level

                if global_boost > 0.001:
                    # 差异化加成：用股票自身近期动量作为"板块热度"的代理
                    codes_in_day = day["code"].values
                    stock_ret_5d = np.zeros(len(codes_in_day))
                    for i, code in enumerate(codes_in_day):
                        # 回溯 5 日的 close 数据
                        hist = ohlcv[(ohlcv["code"] == code) &
                                     (ohlcv["trade_date"] < dt)].tail(6)
                        if len(hist) >= 6:
                            c0 = hist["close"].iloc[-1]   # dt-1 close
                            c5 = hist["close"].iloc[-6]   # dt-6 close
                            if c5 > 0:
                                stock_ret_5d[i] = (c0 - c5) / c5

                    # Z-score 标准化 → 热点股票获得更大加成
                    mu, std = stock_ret_5d.mean(), stock_ret_5d.std()
                    if std > 1e-8:
                        stock_z = (stock_ret_5d - mu) / std
                        stock_boost = global_boost * np.clip(stock_z, -2, 3)
                        scores = scores * (1.0 + pd.Series(stock_boost, index=codes_in_day))
                    else:
                        scores = scores * (1.0 + global_boost)

            scores = scores.sort_values(ascending=False)

            new_holdings, to_buy, to_sell = select_topk_ndrop(
                scores, holdings, K=args.top_n, N=args.ndrop_n)
            holdings = set(new_holdings)

            K = max(len(new_holdings), 1)
            day_ret = 0.0; valid = 0
            if dt in ret_map.index:
                for code in new_holdings:
                    if code in ret_map.columns:
                        r = ret_map.loc[dt, code]
                        if not np.isnan(r):
                            day_ret += r; valid += 1
            day_ret = day_ret / max(valid, 1) if valid > 0 else 0.0

            wp = 1.0 / K
            cost_pct = (len(to_buy) * wp * (COMM + SLIP) +
                        len(to_sell) * wp * (STAMP + COMM + SLIP))
            net_ret = day_ret - cost_pct
            nav *= (1 + net_ret)
            nav_history.append(nav)
            daily_rets.append(net_ret)

    # ── 指标 ──
    nav_arr = np.array(nav_history)
    td = len(nav_arr) - 1
    if td < 10:
        logger.error("仿真天数不足"); return

    total_ret = (nav_arr[-1] / nav_arr[0] - 1) * 100
    years = max(td / 252, 0.2)
    cagr = ((nav_arr[-1] / nav_arr[0]) ** (1 / years) - 1) * 100
    dr = np.array(daily_rets)
    sh = float(np.sqrt(252) * np.mean(dr) / np.std(dr)) if np.std(dr) > 0 else 0
    pk = np.maximum.accumulate(nav_arr)
    mdd = float(np.max((pk - nav_arr) / pk) * 100)
    wr_pct = float(np.mean(np.array(daily_rets) > 0)) * 100

    print(f"\n{'='*50}")
    if args.rl_boost:
        label = "ML+GRPO动态boost"
    elif args.boost > 0 and not args.rl_off:
        label = f"ML+概念boost={args.boost}"
    else:
        label = "ML baseline"
    print(f"RL 板块轮动回测 ({label}, ML={args.train_years}y)")
    print(f"{'='*50}")
    print(f"总收益: {total_ret:.1f}%  年化: {cagr:.1f}%  Sharpe: {sh:.2f}")
    print(f"最大回撤: {mdd:.1f}%  胜率: {wr_pct:.1f}%  天数: {td}")
    print(f"因子: {len(selected)}个  |  {args.start}-{args.end}")

    engine.dispose()


if __name__ == "__main__":
    main()
