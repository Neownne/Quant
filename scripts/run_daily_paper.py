#!/usr/bin/env python
"""日频模拟盘每日驱动：拉数据 → 训练 → 预测 → 执行 → 写DB。

支持多策略并行运行（由 config/paper_strategies.py 配置）。

用法:
    python scripts/run_daily_paper.py                        # 今日收盘后跑
    python scripts/run_daily_paper.py --date 2026-06-01      # 指定日期
    python scripts/run_daily_paper.py --dry-run              # 试运行（不写DB）
    python scripts/run_daily_paper.py --strategies v1.4      # 只跑指定版本

初始化（首次）:
    python scripts/init_paper_trading.py    # 创建 paper_account + paper_runs
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
from models.trainer import walk_forward_train_by_regime
from models.regime import detect_regime, REGIME_PARAMS
from factors import ALL_FACTORS
from factors.screening import filter_factors_by_ic, select_orthogonal_factors
from portfolio.paper_engine import PaperEngine
from portfolio.risk import check_drawdown_limit, check_index_crash
from config.settings import TradingConfig
from config.paper_strategies import PAPER_STRATEGIES

# ── 因子定义 ──
_FACTOR_PRESET_NAMES = [
    *["mom_20", "mom_60", "ema_ratio_5_20", "macd_dif", "macd_signal", "macd_hist",
      "vwap_ratio", "vpt", "money_flow", "force_index", "cwt", "vwap_momentum",
      "gap_ma_dev", "turnover_ma_dev", "turnover_ret_corr", "free_turnover_ratio", "turnover_mom"],
    *["rev_5", "rev_10", "rev_20", "rsi_7", "rsi_14", "bb_position", "bb_width",
      "overnight_ret", "overnight_ret_std", "open_auction_jump",
      "intra_day_rev", "upper_shadow", "lower_shadow", "body_ratio",
      "ret_asymmetry", "tail_risk", "gap_ratio", "intra_vol"],
    *["vol_20", "atr_14", "vol_ratio_5_20", "vol_of_vol", "down_vol_ratio",
      "beta_20", "vol_conv"],
    *["turnover_5", "turnover_skew", "turnover_cv", "turnover_breakout",
      "volume_climax", "obv_roc", "amihud_5", "dollar_volume",
      "bid_ask_proxy", "illiquidity"],
    *["price_mom_5", "price_mom_10", "price_accel"],
    # v1.5 新增日内分钟因子
    *["close_auction_strength", "pm_ret", "intra_vol_skew", "corr_c_v", "am_ret",
      "volume_concentration", "vwap_gap", "am_pm_divergence"],
    # v1.5 市场宽度因子
    *["mkt_adv_dec_ratio", "mkt_limit_up_n", "mkt_limit_down_n", "mkt_up_vol_ratio",
      "mkt_ret_mean", "mkt_ret_std", "mkt_turnover_mean", "mkt_active_pct"],
]
_FUNDAMENTAL_NAMES = [
    "log_mcap", "pb_pct", "sh_change",
    "fin_cashflow_gap", "fin_roe_quality", "fin_profit_cv",
    "fin_net_margin", "fin_bps_growth", "fin_revenue_stability",
    "fin_eps_growth", "fin_debt_ratio", "fin_goodwill_ratio",
    "fin_pledge_risk", "fin_audit_score",
]
FACTOR_NAMES_STANDARD = [f for f in _FACTOR_PRESET_NAMES[:47] + _FUNDAMENTAL_NAMES if f in ALL_FACTORS]
FACTOR_NAMES_FULL = [f for f in _FACTOR_PRESET_NAMES + _FUNDAMENTAL_NAMES if f in ALL_FACTORS]


def load_data(engine, start_date: str, end_date: str, universe_size: int = 500):
    """加载 OHLCV + 指数 + extra_data（所有策略共享一次加载）。"""
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
            extra_data["industry_sw1"] = ind_df.merge(pd.DataFrame({"trade_date": all_dates}), how="cross")[["code", "trade_date", "industry_sw1"]]
    except Exception: pass

    # 估值
    try:
        extra_df = pd.read_sql(f"SELECT code, trade_date, market_cap, pb FROM stock_daily_extra WHERE code IN ({code_list}) AND trade_date BETWEEN '{start_date}' AND '{end_date}'", engine)
        if not extra_df.empty:
            extra_df["log_mcap"] = np.log(extra_df["market_cap"].replace(0, np.nan))
            extra_data["log_mcap"] = extra_df[["code", "trade_date", "log_mcap"]]
            extra_data["pb"] = extra_df[["code", "trade_date", "pb"]]
    except Exception: pass

    # 股东
    try:
        sh_df = pd.read_sql(f"SELECT code, end_date AS trade_date, shareholder_count FROM stock_shareholder WHERE code IN ({code_list}) AND end_date BETWEEN '{start_date}' AND '{end_date}'", engine)
        if not sh_df.empty:
            extra_data["shareholder_count"] = sh_df[["code", "trade_date", "shareholder_count"]]
    except Exception: pass

    # 质押
    try:
        pledge_df = pd.read_sql(f"SELECT code, trade_date, pledge_ratio FROM stock_pledge WHERE code IN ({code_list}) AND trade_date BETWEEN '{start_date}' AND '{end_date}'", engine)
        if not pledge_df.empty:
            extra_data["pledge_ratio"] = pledge_df[["code", "trade_date", "pledge_ratio"]]
    except Exception: pass

    # 财务
    try:
        fin_cols = ["net_profit", "roe", "bps", "net_margin", "revenue", "eps",
                     "cash_flow", "operating_cash_flow", "total_assets", "total_liability",
                     "goodwill", "holder_equity", "adjusted_profit"]
        fin_df = pd.read_sql(f"SELECT code, report_date, {','.join(fin_cols)} FROM stock_financial WHERE code IN ({code_list}) AND report_date >= '2018-01-01' ORDER BY code, report_date", engine)
        if not fin_df.empty:
            for col in fin_cols:
                if col in fin_df.columns:
                    extra_data[col] = fin_df[["code", "report_date", col]].copy()
    except Exception: pass

    # 市场宽度
    try:
        from factors.market_breadth import build_market_breadth_extra
        mkt = build_market_breadth_extra(ohlcv, codes)
        for col in [c for c in mkt.columns if c.startswith("mkt_")]:
            extra_data[col] = mkt[["code", "trade_date", col]]
    except Exception: pass

    # 日内因子
    try:
        minute_sql = f"SELECT code, trade_time, period, open, high, low, close, volume, amount FROM stock_minute WHERE code IN ({code_list}) AND trade_time >= '{start_date}'::timestamp AND trade_time < ('{end_date}'::date + interval '1 day')::timestamp ORDER BY code, trade_time"
        minute_df = pd.read_sql(minute_sql, engine)
        if not minute_df.empty:
            from factors.intraday_minute import build_intraday_daily_features
            minute_extra = build_intraday_daily_features(minute_df)
            extra_data.update(minute_extra)
    except Exception: pass

    return ohlcv, index_df, regime_df, extra_data


def run_strategy(strategy_cfg: dict, ohlcv: pd.DataFrame, index_df: pd.DataFrame,
                 regime_df: pd.DataFrame, extra_data: dict, trade_date: pd.Timestamp,
                 dry_run: bool = False):
    """执行单个策略的日频流程：训练 → 预测 → 信号保存 → T+1执行。"""
    name = strategy_cfg["name"]
    ver = strategy_cfg["version"]
    account_id = strategy_cfg["account_id"]
    run_id = strategy_cfg["run_id"]
    forward_days = strategy_cfg["forward_days"]
    train_years = strategy_cfg["train_years"]
    top_n = strategy_cfg["top_n"]

    factor_names = FACTOR_NAMES_FULL if strategy_cfg.get("factor_mode") == "full" else FACTOR_NAMES_STANDARD
    logger.info(f"[{name} {ver}] 因子数: {len(factor_names)}, account={account_id}, run={run_id}")

    # 构建数据集
    dataset = build_factor_dataset(
        ohlcv, factor_names, label_mode="binary", forward_days=forward_days,
        extra_data=extra_data if extra_data else None, industry_neutralize=False,
    )

    # IC + 正交筛选
    factor_cols = filter_factors_by_ic(dataset, factor_names)
    if len(factor_cols) < 3:
        factor_cols = factor_names[:min(12, len(factor_names))]
    selected = select_orthogonal_factors(dataset, factor_cols, threshold=0.7)
    logger.info(f"[{name} {ver}] 因子: {len(factor_names)} → IC{len(factor_cols)} → 正交{len(selected)}")

    # 训练
    if regime_df is None:
        logger.error(f"[{name} {ver}] 无法检测市场状态，跳过")
        return None

    results = walk_forward_train_by_regime(dataset, selected, regime_df, train_years=train_years, val_years=1)
    if not results:
        logger.error(f"[{name} {ver}] Regime训练无结果，跳过")
        return None

    # 合并多窗口模型
    from models.trainer import RegimeAwareEnsemble
    merged_ensembles, merged_factors = {}, []
    for r in results:
        ens = r.get("ensemble")
        if ens:
            for reg, model in ens.ensembles.items():
                if reg not in merged_ensembles:
                    merged_ensembles[reg] = model
            if not merged_factors:
                merged_factors = ens.factor_names
    if not merged_ensembles:
        logger.error(f"[{name} {ver}] 无有效模型，跳过")
        return None
    any_model = next(iter(merged_ensembles.values()))
    for reg in ["bull", "bear", "sideways"]:
        if reg not in merged_ensembles:
            merged_ensembles[reg] = any_model
    ensemble = RegimeAwareEnsemble(merged_ensembles, merged_factors or selected)

    # 今日市场状态
    engine = get_engine()
    with engine.connect() as conn:
        r = conn.execute(text("SELECT MAX(trade_date) FROM stock_daily")).fetchone()
    latest_date = pd.Timestamp(r[0]) if r else trade_date
    today_regime = "sideways"
    if regime_df is not None:
        tr = regime_df[regime_df["trade_date"] <= latest_date]
        if not tr.empty:
            today_regime = str(tr["regime"].iloc[-1])

    # 今日因子截面
    pred_date = dataset["trade_date"].max()
    today_factor = dataset[dataset["trade_date"] == pred_date].copy()
    if today_factor.empty:
        logger.error(f"[{name} {ver}] 无今日因子数据，跳过")
        return None
    today_factor["regime"] = today_regime

    # T+1 执行：处理待执行信号
    if not dry_run:
        with engine.connect() as conn:
            prev = conn.execute(text("""
                SELECT DISTINCT ps.signal_date FROM paper_signals ps
                WHERE ps.run_id = :rid AND NOT EXISTS (
                    SELECT 1 FROM paper_positions pp
                    WHERE pp.run_id = ps.run_id AND pp.entry_date = ps.signal_date
                ) ORDER BY ps.signal_date
            """), {"rid": run_id}).fetchall()
        for prow in prev:
            prev_date = pd.Timestamp(prow[0])
            prev_factor = dataset[dataset["trade_date"] == prev_date].copy()
            if prev_factor.empty:
                continue
            prev_regime = str(regime_df[regime_df["trade_date"] <= prev_date]["regime"].iloc[-1])
            prev_factor["regime"] = prev_regime
            eng = PaperEngine(account_id=account_id, run_id=run_id, predictor=ensemble,
                              top_n=top_n, rebalance_mode="ndrop")
            result = eng.run_daily(trade_date=prev_date, factor_df=prev_factor,
                                   ohlcv_data=ohlcv, index_ohlcv=index_df, regime=prev_regime)
            if result:
                logger.info(f"[{name} {ver}] T+1执行 {prev_date.date()}: 总资产={result['total_value']:,.0f}, "
                            f"买{result['n_buy_orders']}/卖{result['n_sell_orders']}")

    # 预测今日信号
    preds = ensemble.predict(today_factor)
    preds = preds.sort_values("score", ascending=False).reset_index(drop=True)

    # 保存信号
    if dry_run:
        logger.info(f"[{name} {ver}] DRY RUN — 信号不保存")
        return {"preds": preds, "top_n": top_n, "regime": today_regime}
    else:
        with engine.begin() as conn:
            existing = conn.execute(text(
                "SELECT COUNT(*) FROM paper_signals WHERE run_id = :rid AND signal_date = :sd"
            ), {"rid": run_id, "sd": pred_date.date()}).fetchone()[0]
            if existing == 0:
                for rank, (_, row) in enumerate(preds.head(top_n).iterrows()):
                    conn.execute(text("""
                        INSERT INTO paper_signals (run_id, signal_date, stock_code, predicted_score, rank)
                        VALUES (:rid, :sd, :sc, :ps, :rk)
                    """), {"rid": run_id, "sd": pred_date.date(), "sc": row["code"],
                           "ps": float(row["score"]), "rk": rank + 1})
                logger.info(f"[{name} {ver}] 信号已保存: {min(top_n, len(preds))} 只")
            else:
                logger.info(f"[{name} {ver}] 信号已存在 ({existing}条)")

            # 首次建仓：如无任何持仓，用最新有信号的日期立即建仓
            pos_count = conn.execute(text(
                "SELECT COUNT(*) FROM paper_positions WHERE run_id = :rid"
            ), {"rid": run_id}).fetchone()[0]
            if pos_count == 0:
                # 用最早的有信号日期建仓
                first_sig = conn.execute(text(
                    "SELECT MIN(signal_date) FROM paper_signals WHERE run_id = :rid"
                ), {"rid": run_id}).fetchone()[0]
                if first_sig:
                    logger.info(f"[{name} {ver}] 首次建仓 (信号日={first_sig}) ...")
                    first_factor = dataset[dataset["trade_date"] == pd.Timestamp(first_sig)].copy()
                    if not first_factor.empty:
                        first_regime = str(regime_df[regime_df["trade_date"] <= pd.Timestamp(first_sig)]["regime"].iloc[-1])
                        first_factor["regime"] = first_regime
                        eng = PaperEngine(account_id=account_id, run_id=run_id, predictor=ensemble,
                                          top_n=top_n, rebalance_mode="ndrop")
                        result = eng.run_daily(trade_date=pd.Timestamp(first_sig), factor_df=first_factor,
                                               ohlcv_data=ohlcv, index_ohlcv=index_df, regime=first_regime)
                        if result:
                            logger.info(f"[{name} {ver}] 建仓完成: 总资产={result['total_value']:,.0f}, "
                                        f"买{result['n_buy_orders']}/卖{result['n_sell_orders']}")

    return {"ensemble": ensemble, "preds": preds, "top_n": top_n, "regime": today_regime}


def main():
    parser = argparse.ArgumentParser(description="日频模拟盘每日驱动 — 多策略")
    parser.add_argument("--date", help="交易日（YYYY-MM-DD），默认今天")
    parser.add_argument("--dry-run", action="store_true", help="试运行不写DB")
    parser.add_argument("--no-sync", action="store_true", help="跳过数据同步")
    parser.add_argument("--strategies", default="", help="只跑指定版本（逗号分隔，如 v1.4,v1.5）")
    args = parser.parse_args()

    trade_date = pd.Timestamp(args.date) if args.date else pd.Timestamp.now().normalize()
    if trade_date.date() >= date.today():
        trade_date = pd.Timestamp(date.today())
    logger.info(f"交易日: {trade_date.date()}")

    # 确定要跑的策略
    target_versions = set(v.strip() for v in args.strategies.split(",") if v.strip())
    strategies = [s for s in PAPER_STRATEGIES if not target_versions or s["version"] in target_versions]
    if not strategies:
        logger.warning("没有匹配的策略，跳过")
        return

    logger.info(f"策略: {len(strategies)} 个 ({[s['version'] for s in strategies]})")

    # 数据同步（每次运行都检查并同步到最新）
    if not args.no_sync:
        try:
            engine = get_engine()
            with engine.connect() as c:
                latest = c.execute(text("SELECT MAX(trade_date) FROM stock_daily")).fetchone()
            from data.sync import sync_stock_daily
            sync_start = (latest[0] - timedelta(days=3)).strftime("%Y%m%d") if latest and latest[0] else (date.today() - timedelta(days=10)).strftime("%Y%m%d")
            logger.info(f"数据同步: {sync_start} → 最新")
            sync_stock_daily(engine, start_date=sync_start, workers=1)
            engine.dispose()
        except Exception as e:
            logger.warning(f"同步跳过: {e}")

    # 加载数据（所有策略共享）
    start_dt = trade_date - pd.DateOffset(years=5)  # 多留余量
    start_str = start_dt.strftime("%Y%m%d")
    end_str = trade_date.strftime("%Y%m%d")
    ohlcv, index_df, regime_df, extra_data = load_data(get_engine(), start_str, end_str)

    # 逐策略执行
    for cfg in strategies:
        try:
            run_strategy(cfg, ohlcv, index_df, regime_df, extra_data, trade_date, dry_run=args.dry_run)
        except Exception as e:
            logger.error(f"[{cfg['name']} {cfg['version']}] 执行失败: {e}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
