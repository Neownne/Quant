#!/usr/bin/env python
"""日频模拟盘每日驱动：同步 → 训练 → T+1执行 → 预测 → 写DB。

策略参数与回测完全一致（config/settings.py + models/regime.py）。
调仓频率由 REGIME_PARAMS.rebalance_freq 决定（默认周度，强牛日度）。

用法:
    python scripts/run_daily_paper.py                        # 今日收盘后跑
    python scripts/run_daily_paper.py --dry-run              # 试运行
    python scripts/run_daily_paper.py --strategies v1.5      # 只跑v1.5
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import date, timedelta
from loguru import logger
from sqlalchemy import text

from data.db import get_engine
from models.dataset import build_factor_dataset
from models.trainer import walk_forward_train_by_regime, RegimeAwareEnsemble
from models.regime import detect_regime, REGIME_PARAMS
from factors import ALL_FACTORS
from factors.screening import filter_factors_by_ic, select_orthogonal_factors
from portfolio.paper_engine import PaperEngine
from config.settings import TradingConfig
from config.paper_strategies import PAPER_STRATEGIES

# ── v1.4 因子：标准技术因子 + 基本面（与回测一致）──
V14_FACTOR_NAMES = [
    # 动量/趋势 (10)
    "mom_20", "mom_60", "ema_ratio_5_20", "macd_dif", "macd_signal", "macd_hist",
    "vwap_ratio", "vpt", "price_mom_5", "price_mom_10",
    # 反转/均值回归 (10)
    "rev_5", "rev_10", "rev_20", "rsi_7", "rsi_14", "bb_position", "bb_width",
    "overnight_ret", "intra_day_rev", "gap_ratio",
    # 波动/风险 (6)
    "vol_20", "atr_14", "vol_ratio_5_20", "down_vol_ratio", "tail_risk", "intra_vol",
    # 流动性/资金流 (10)
    "turnover_5", "turnover_ma_dev", "turnover_mom", "dollar_volume", "illiquidity",
    "force_index", "money_flow", "obv_roc", "volume_climax", "cwt",
    # 形态 (4)
    "upper_shadow", "lower_shadow", "body_ratio", "ret_asymmetry",
    # 基本面 (14)
    "log_mcap", "pb_pct", "sh_change",
    "fin_cashflow_gap", "fin_roe_quality", "fin_profit_cv",
    "fin_net_margin", "fin_bps_growth", "fin_revenue_stability",
    "fin_eps_growth", "fin_debt_ratio", "fin_goodwill_ratio",
    "fin_pledge_risk", "fin_audit_score",
]

# ── v1.5 因子：v1.4 + 日内分钟 + 市场宽度（与回测一致）──
V15_FACTOR_NAMES = V14_FACTOR_NAMES + [
    # 日内分钟 (7)
    "close_auction_strength", "pm_ret", "intra_vol_skew", "am_ret",
    "volume_concentration", "vwap_gap", "am_pm_divergence",
    # 市场宽度 (8)
    "mkt_adv_dec_ratio", "mkt_limit_up_n", "mkt_limit_down_n",
    "mkt_up_vol_ratio", "mkt_ret_mean", "mkt_ret_std",
    "mkt_turnover_mean", "mkt_active_pct",
]


def load_data(engine, start_date: str, end_date: str, universe_size: int = 500):
    """加载 OHLCV + 指数 + extra_data（所有策略共享）。"""
    codes = pd.read_sql(
        f"SELECT d.code FROM stock_daily d "
        f"JOIN stock_basic b ON d.code = b.code AND b.is_st = FALSE "
        f"WHERE d.trade_date BETWEEN '{start_date}' AND '{end_date}' "
        f"GROUP BY d.code ORDER BY SUM(d.amount) DESC LIMIT {universe_size}",
        engine,
    )["code"].tolist()
    code_list = ",".join([f"'{c}'" for c in codes])

    ohlcv = pd.read_sql(f"""
        SELECT code, trade_date, open, high, low, close, volume, amount, turnover
        FROM stock_daily WHERE code IN ({code_list})
        AND trade_date BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY code, trade_date
    """, engine)
    ohlcv["trade_date"] = pd.to_datetime(ohlcv["trade_date"])
    logger.info(f"OHLCV: {len(ohlcv)} 行, {ohlcv['code'].nunique()} 只")

    index_df = pd.read_sql(f"""
        SELECT trade_date, close FROM index_daily
        WHERE code = '000001' AND trade_date BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY trade_date
    """, engine)
    index_df["trade_date"] = pd.to_datetime(index_df["trade_date"])
    regime_df = detect_regime(index_df) if not index_df.empty else None

    extra_data = {}
    # 行业
    try:
        ind_df = pd.read_sql(f"SELECT code, industry_sw1 FROM stock_industry WHERE code IN ({code_list})", engine)
        if not ind_df.empty:
            all_dates = sorted(ohlcv["trade_date"].unique())
            extra_data["industry_sw1"] = ind_df.merge(
                pd.DataFrame({"trade_date": all_dates}), how="cross")[["code", "trade_date", "industry_sw1"]]
    except Exception as e: logger.warning(f"行业数据: {e}")

    # 估值
    try:
        extra_df = pd.read_sql(f"SELECT code, trade_date, market_cap, pb FROM stock_daily_extra "
                               f"WHERE code IN ({code_list}) AND trade_date BETWEEN '{start_date}' AND '{end_date}'", engine)
        if not extra_df.empty:
            extra_df["log_mcap"] = np.log(extra_df["market_cap"].replace(0, np.nan))
            extra_data["log_mcap"] = extra_df[["code", "trade_date", "log_mcap"]]
            extra_data["pb"] = extra_df[["code", "trade_date", "pb"]]
    except Exception as e: logger.warning(f"估值数据: {e}")

    # 股东
    try:
        sh_df = pd.read_sql(f"SELECT code, end_date AS trade_date, shareholder_count FROM stock_shareholder "
                            f"WHERE code IN ({code_list}) AND end_date BETWEEN '{start_date}' AND '{end_date}'", engine)
        if not sh_df.empty:
            extra_data["shareholder_count"] = sh_df[["code", "trade_date", "shareholder_count"]]
    except Exception as e: logger.warning(f"股东数据: {e}")

    # 质押
    try:
        pledge_df = pd.read_sql(f"SELECT code, trade_date, pledge_ratio FROM stock_pledge "
                                f"WHERE code IN ({code_list}) AND trade_date BETWEEN '{start_date}' AND '{end_date}'", engine)
        if not pledge_df.empty:
            extra_data["pledge_ratio"] = pledge_df[["code", "trade_date", "pledge_ratio"]]
    except Exception as e: logger.warning(f"质押数据: {e}")

    # 财务
    try:
        fin_cols = ["net_profit", "roe", "bps", "net_margin", "revenue", "eps",
                     "cash_flow", "operating_cash_flow", "total_assets", "total_liability",
                     "goodwill", "holder_equity", "adjusted_profit"]
        fin_df = pd.read_sql(f"SELECT code, report_date, {','.join(fin_cols)} FROM stock_financial "
                             f"WHERE code IN ({code_list}) AND report_date >= '2018-01-01' ORDER BY code, report_date", engine)
        if not fin_df.empty:
            for col in fin_cols:
                if col in fin_df.columns:
                    extra_data[col] = fin_df[["code", "report_date", col]].copy()
    except Exception as e: logger.warning(f"财务数据: {e}")

    # 日内因子（分钟数据）
    try:
        minute_sql = (f"SELECT code, trade_time, period, open, high, low, close, volume, amount "
                      f"FROM stock_minute WHERE code IN ({code_list}) "
                      f"AND trade_time >= '{start_date}'::timestamp "
                      f"AND trade_time < ('{end_date}'::date + interval '1 day')::timestamp "
                      f"ORDER BY code, trade_time")
        minute_df = pd.read_sql(minute_sql, engine)
        if not minute_df.empty:
            from factors.intraday_minute import build_intraday_daily_features
            extra_data.update(build_intraday_daily_features(minute_df))
    except Exception as e: logger.warning(f"分钟数据: {e}")

    # 市场宽度
    try:
        from factors.market_breadth import build_market_breadth_extra
        mkt = build_market_breadth_extra(ohlcv, codes)
        for col in [c for c in mkt.columns if c.startswith("mkt_")]:
            extra_data[col] = mkt[["code", "trade_date", col]]
    except Exception as e: logger.warning(f"市场宽度: {e}")

    return ohlcv, index_df, regime_df, extra_data


def run_strategy(strategy_cfg: dict, ohlcv: pd.DataFrame, index_df: pd.DataFrame,
                 regime_df, extra_data: dict, trade_date: pd.Timestamp, dry_run: bool = False):
    """执行单策略日频流程：训练 → T+1执行(调仓频率控制) → 预测 → 信号保存。"""
    name, ver = strategy_cfg["name"], strategy_cfg["version"]
    account_id, run_id = strategy_cfg["account_id"], strategy_cfg["run_id"]
    forward_days, train_years, top_n = strategy_cfg["forward_days"], strategy_cfg["train_years"], strategy_cfg["top_n"]

    # 因子选择
    mode = strategy_cfg.get("factor_mode", "standard")
    if mode == "all":
        factor_names = list(ALL_FACTORS.keys())
    elif mode == "full":
        factor_names = [f for f in V15_FACTOR_NAMES if f in ALL_FACTORS]
    else:
        factor_names = [f for f in V14_FACTOR_NAMES if f in ALL_FACTORS]
    logger.info(f"[{name} {ver}] 因子: {len(factor_names)}个, account={account_id}")

    # 数据集
    dataset = build_factor_dataset(ohlcv, factor_names, label_mode="binary",
                                   forward_days=forward_days, extra_data=extra_data,
                                   industry_neutralize=False)

    # 因子筛选
    factor_cols = filter_factors_by_ic(dataset, factor_names)
    if len(factor_cols) < 3:
        factor_cols = factor_names[:min(12, len(factor_names))]
    selected = select_orthogonal_factors(dataset, factor_cols, threshold=0.7)
    logger.info(f"[{name} {ver}] 因子: {len(factor_names)}→IC{len(factor_cols)}→正交{len(selected)}")

    if regime_df is None:
        logger.error(f"[{name} {ver}] 无市场状态数据")
        return None

    # 训练
    results = walk_forward_train_by_regime(dataset, selected, regime_df,
                                           train_years=train_years, val_years=1)
    if not results:
        logger.error(f"[{name} {ver}] 训练无结果")
        return None

    # 合并模型
    merged_ensembles, merged_factors = {}, []
    for r in results:
        ens = r.get("ensemble")
        if ens:
            for reg, m in ens.ensembles.items():
                if reg not in merged_ensembles:
                    merged_ensembles[reg] = m
            if not merged_factors:
                merged_factors = ens.factor_names
    if not merged_ensembles:
        return None
    any_m = next(iter(merged_ensembles.values()))
    for reg in ["bull", "bear", "sideways"]:
        if reg not in merged_ensembles:
            merged_ensembles[reg] = any_m
    ensemble = RegimeAwareEnsemble(merged_ensembles, merged_factors or selected)

    # 最新交易日 & 市场状态
    engine = get_engine()
    with engine.connect() as conn:
        r = conn.execute(text("SELECT MAX(trade_date) FROM stock_daily")).fetchone()
    latest_date = pd.Timestamp(r[0]) if r else trade_date
    pred_date = dataset["trade_date"].max()
    today_regime = "sideways"
    if regime_df is not None:
        tr = regime_df[regime_df["trade_date"] <= latest_date]
        if not tr.empty:
            today_regime = str(tr["regime"].iloc[-1])

    reg_params = REGIME_PARAMS.get(today_regime, REGIME_PARAMS["sideways"])
    rebalance_freq = reg_params["rebalance_freq"]

    today_factor = dataset[dataset["trade_date"] == pred_date].copy()
    if today_factor.empty:
        logger.error(f"[{name} {ver}] 无今日因子数据")
        return None
    today_factor["regime"] = today_regime

    # ── T+1 执行：只执行最近一个待执行日期，且受调仓频率控制 ──
    if not dry_run:
        with engine.connect() as conn:
            # 找最近一个待执行日期
            today_str = pred_date.strftime("%Y-%m-%d")
            prev = conn.execute(text("""
                SELECT DISTINCT ps.signal_date FROM paper_signals ps
                WHERE ps.run_id = :rid AND ps.signal_date < :today
                AND NOT EXISTS (
                    SELECT 1 FROM paper_positions pp
                    WHERE pp.run_id = ps.run_id AND pp.entry_date > ps.signal_date
                    AND pp.entry_date <= :today2
                ) ORDER BY ps.signal_date DESC LIMIT 1
            """), {"rid": run_id, "today": today_str, "today2": today_str}).fetchone()

            if prev:
                # 检查调仓频率
                last_trade = conn.execute(text("""
                    SELECT MAX(entry_date) FROM paper_positions WHERE run_id = :rid
                """), {"rid": run_id}).fetchone()[0]
                if last_trade:
                    # 计算距上次调仓的交易日数
                    trading_days = conn.execute(text("""
                        SELECT COUNT(*) FROM (
                            SELECT DISTINCT trade_date FROM stock_daily
                            WHERE trade_date > :last AND trade_date <= :today
                        ) t
                    """), {"last": str(last_trade), "today": today_str}).fetchone()[0]
                else:
                    trading_days = rebalance_freq  # 无历史持仓，默认触发

                if trading_days >= rebalance_freq or last_trade is None:
                    prev_date = pd.Timestamp(prev[0])
                    logger.info(f"[{name} {ver}] 调仓({prev_date.date()}信号→{pred_date.date()}执行, "
                                f"距上次{trading_days}日/频率{rebalance_freq}日)")
                    eng = PaperEngine(account_id=account_id, run_id=run_id, predictor=ensemble,
                                      top_n=top_n, rebalance_mode="ndrop")
                    result = eng.run_daily(trade_date=pred_date, factor_df=today_factor,
                                           ohlcv_data=ohlcv, index_ohlcv=index_df, regime=today_regime)
                    if result:
                        logger.info(f"[{name} {ver}] 总资产={result['total_value']:,.0f}, "
                                    f"买{result['n_buy_orders']}/卖{result['n_sell_orders']}")
                else:
                    logger.info(f"[{name} {ver}] 非调仓日(距上次{trading_days}日<频率{rebalance_freq}日)，跳过")
    else:
        engine.dispose()

    # ── 预测今日信号（始终生成，供Web展示"待执行"）──
    preds = ensemble.predict(today_factor)
    preds = preds.sort_values("score", ascending=False).reset_index(drop=True)

    if dry_run:
        logger.info(f"[{name} {ver}] DRY RUN")
        return {"preds": preds, "top_n": top_n, "regime": today_regime}

    with engine.begin() as conn:
        existing = conn.execute(text(
            "SELECT COUNT(*) FROM paper_signals WHERE run_id=:rid AND signal_date=:sd"
        ), {"rid": run_id, "sd": pred_date.date()}).fetchone()[0]
        if existing == 0:
            for rank, (_, row) in enumerate(preds.head(top_n).iterrows()):
                conn.execute(text("""
                    INSERT INTO paper_signals (run_id, signal_date, stock_code, predicted_score, rank)
                    VALUES (:rid, :sd, :sc, :ps, :rk)
                    ON CONFLICT (run_id, signal_date, stock_code) DO NOTHING
                """), {"rid": run_id, "sd": pred_date.date(), "sc": row["code"],
                       "ps": float(row["score"]), "rk": rank + 1})
            logger.info(f"[{name} {ver}] 信号: {min(top_n, len(preds))}只")

    # 每日估值（无论是否调仓、无论信号是否已存在，都记录）
    # 注意：用 trade_date（命令行参数）而非 pred_date（因子集最新日，会因标签前视而偏早）
    if not dry_run:
        try:
            pnl_date = trade_date.strftime("%Y-%m-%d")
            engine_pnl = get_engine()
            with engine_pnl.begin() as conn_pnl:
                cash_r = conn_pnl.execute(text("SELECT cash FROM paper_account WHERE id=:aid"), {"aid": account_id}).fetchone()
                cash = float(cash_r[0]) if cash_r else TradingConfig.INITIAL_CASH
                pos_val_f = conn_pnl.execute(text("""
                    SELECT COALESCE(SUM(pp.quantity * COALESCE(sd.close, sp.close)), 0)
                    FROM paper_positions pp
                    LEFT JOIN stock_daily sd ON pp.stock_code=sd.code AND sd.trade_date=:d
                    LEFT JOIN stock_daily sp ON pp.stock_code=sp.code AND sp.trade_date = (
                        SELECT MAX(trade_date) FROM stock_daily WHERE code=pp.stock_code AND trade_date<:d2)
                    WHERE pp.run_id=:rid AND pp.entry_date<=:d2 AND (pp.exit_date IS NULL OR pp.exit_date>:d2)
                """), {"rid": run_id, "d": pnl_date, "d2": pnl_date}).fetchone()[0] or 0
                total = cash + float(pos_val_f)
                prev_total = conn_pnl.execute(text("SELECT total_value FROM paper_daily_pnl WHERE account_id=:aid ORDER BY trade_date DESC LIMIT 1"), {"aid": account_id}).fetchone()
                prev_tv = float(prev_total[0]) if prev_total else TradingConfig.INITIAL_CASH
                dr = (total - prev_tv) / prev_tv if prev_tv > 0 else 0
                # 回撤 = (历史峰值 - 当前) / 历史峰值
                peak_val = conn_pnl.execute(text("SELECT COALESCE(MAX(total_value), :init) FROM paper_daily_pnl WHERE account_id=:aid"), {"aid": account_id, "init": TradingConfig.INITIAL_CASH}).fetchone()[0]
                peak_val = max(float(peak_val), total)
                dd = (peak_val - total) / peak_val if peak_val > 0 else 0
                conn_pnl.execute(text("""
                    INSERT INTO paper_daily_pnl (account_id, trade_date, cash, position_value, total_value, daily_return, drawdown)
                    VALUES (:aid, :d, :c, :pv, :tv, :dr, :dd)
                    ON CONFLICT (account_id, trade_date) DO UPDATE SET cash=:c2, position_value=:pv2, total_value=:tv2, daily_return=:dr2, drawdown=:dd2
                """), {"aid": account_id, "d": pnl_date, "c": cash, "pv": float(pos_val_f), "tv": total, "dr": dr, "dd": dd,
                       "c2": cash, "pv2": float(pos_val_f), "tv2": total, "dr2": dr, "dd2": dd})
            engine_pnl.dispose()
        except Exception as e:
            logger.warning(f"[{name} {ver}] 估值记录失败: {e}")

    engine.dispose()
    return {"preds": preds, "top_n": top_n, "regime": today_regime}


def main():
    parser = argparse.ArgumentParser(description="日频模拟盘每日驱动")
    parser.add_argument("--date", help="交易日（YYYY-MM-DD）")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-sync", action="store_true")
    parser.add_argument("--strategies", default="")
    args = parser.parse_args()

    trade_date = pd.Timestamp(args.date) if args.date else pd.Timestamp.now().normalize()
    if trade_date.date() >= date.today():
        trade_date = pd.Timestamp(date.today())
    logger.info(f"交易日: {trade_date.date()}")

    target_versions = set(v.strip() for v in args.strategies.split(",") if v.strip())
    strategies = [s for s in PAPER_STRATEGIES if not target_versions or s["version"] in target_versions]
    if not strategies:
        logger.warning("无匹配策略")
        return
    logger.info(f"策略: {[s['version'] for s in strategies]}")

    # 首次引导：确保 paper_account 存在
    engine = get_engine()
    with engine.begin() as c:
        for cfg in strategies:
            aid = cfg["account_id"]
            exists = c.execute(text("SELECT 1 FROM paper_account WHERE id=:aid"), {"aid": aid}).fetchone()
            if not exists:
                c.execute(text("INSERT INTO paper_account (id, name, initial_capital, cash) VALUES (:aid, :n, :cap, :cap)"),
                          {"aid": aid, "n": f"{cfg['name']}-{cfg['version']}", "cap": TradingConfig.INITIAL_CASH})
                logger.info(f"创建 paper_account id={aid}")
    engine.dispose()

    # 数据同步
    if not args.no_sync:
        try:
            engine = get_engine()
            with engine.connect() as c:
                latest = c.execute(text("SELECT MAX(trade_date) FROM stock_daily")).fetchone()
            today = date.today()
            if latest and latest[0] and (today - latest[0]).days >= 1:
                from data.sync import sync_stock_daily, sync_index_daily
                sync_start = (latest[0] - timedelta(days=3)).strftime("%Y%m%d")
                logger.info(f"数据同步: {sync_start} → 最新")
                sync_stock_daily(engine, start_date=sync_start, workers=8)
                sync_index_daily(engine, start_date=sync_start)
            else:
                logger.info(f"数据已最新 ({latest[0] if latest else '?'})")
            engine.dispose()
        except Exception as e:
            logger.warning(f"同步跳过: {e}")

    # 加载共享数据
    start_dt = trade_date - pd.DateOffset(years=5)
    ohlcv, index_df, regime_df, extra_data = load_data(
        get_engine(), start_dt.strftime("%Y%m%d"), trade_date.strftime("%Y%m%d"))

    for cfg in strategies:
        try:
            run_strategy(cfg, ohlcv, index_df, regime_df, extra_data, trade_date, dry_run=args.dry_run)
        except Exception as e:
            logger.error(f"[{cfg['name']} {cfg['version']}] 失败: {e}", exc_info=True)

    logger.info("Done.")


if __name__ == "__main__":
    main()
