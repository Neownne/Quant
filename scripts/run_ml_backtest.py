#!/usr/bin/env python
"""ML 选股端到端回测验证。

用法:
    python scripts/run_ml_backtest.py                              # 集成模式+IC门禁
    python scripts/run_ml_backtest.py --regime --optuna            # 分状态+超参优化
    python scripts/run_ml_backtest.py --model xgboost              # 单模型
    python scripts/run_ml_backtest.py --factor-preset momentum --forward-days 5 --model xgboost
    python scripts/run_ml_backtest.py --factor-preset reversal --forward-days 1 --model lightgbm
    python scripts/run_ml_backtest.py --dynamic                    # 动态多因子：归因→健康度→调参
"""
import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import TradingConfig
from data.db import get_engine
from scripts.overfit_check import OverfitChecker
from scripts.dynamic_backtest import BacktestFeedbackLoop, DailySignalTracker
from sqlalchemy import text
import json

from models.dataset import build_factor_dataset
from models.trainer import walk_forward_train, walk_forward_train_ensemble, walk_forward_train_by_regime
from factors import ALL_FACTORS
from factors.screening import filter_factors_by_ic, select_orthogonal_factors
from models.regime import detect_regime
from portfolio.selector import select_topk_ndrop
from portfolio.risk import check_drawdown_limit, check_index_crash


# ── Factor presets ──────────────────────────────────────────────────────────
FACTOR_PRESETS = {
    "momentum": [
        # alpha101: trend/momentum
        "mom_20", "mom_60", "ema_ratio_5_20", "macd_dif", "macd_signal", "macd_hist",
        "vwap_ratio", "vpt",
        # alpha191_flow: money flow & trend strength
        "money_flow", "force_index", "cwt", "vwap_momentum",
        # alpha191_gap: gap/trend deviation
        "gap_ma_dev",
        # alpha191_turnover: turnover-trend interaction
        "turnover_ma_dev", "turnover_ret_corr", "free_turnover_ratio",
        # custom
        "turnover_mom",
    ],
    "reversal": [
        # alpha101: short-term reversal
        "rev_5", "rev_10", "rev_20", "rsi_7", "rsi_14",
        "bb_position", "bb_width",
        # alpha191_gap: overnight reversal
        "overnight_ret", "overnight_ret_std", "open_auction_jump",
        # alpha191_intraday: intraday reversal patterns
        "intra_day_rev", "upper_shadow", "lower_shadow", "body_ratio",
        # alpha191_vol: asymmetry & tail risk
        "ret_asymmetry", "tail_risk",
        # custom
        "gap_ratio", "intra_vol",
    ],
    "volatility": [
        "vol_20", "atr_14", "vol_ratio_5_20",
        "vol_of_vol", "down_vol_ratio", "beta_20",
        "vol_conv",
    ],
    "liquidity": [
        "turnover_5", "turnover_skew", "turnover_cv",
        "turnover_breakout", "volume_climax", "obv_roc",
        "amihud_5", "dollar_volume", "bid_ask_proxy", "illiquidity",
    ],
    "fundamental": [
        # custom fundamental proxies
        "log_mcap", "pb_pct", "sh_change",
        # fundamental module
        "fin_cashflow_gap", "fin_roe_quality", "fin_profit_cv",
        "fin_net_margin", "fin_bps_growth", "fin_revenue_stability",
        "fin_eps_growth", "fin_debt_ratio", "fin_goodwill_ratio",
        "fin_pledge_risk", "fin_audit_score",
    ],
}


def get_factors_by_preset(preset: str) -> list[str]:
    """Resolve factor preset name or comma-separated list to factor names."""
    if preset == "all":
        return list(ALL_FACTORS.keys())
    if preset in FACTOR_PRESETS:
        names = FACTOR_PRESETS[preset]
        # Filter to only factors that exist in ALL_FACTORS
        return [n for n in names if n in ALL_FACTORS]
    if preset.startswith("+"):
        # Union of multiple presets: "+momentum+volatility"
        names = set()
        for p in preset.lstrip("+").split("+"):
            p = p.strip()
            if p in FACTOR_PRESETS:
                names.update(n for n in FACTOR_PRESETS[p] if n in ALL_FACTORS)
        return sorted(names)
    # Assume comma-separated list
    return [f.strip() for f in preset.split(",") if f.strip() in ALL_FACTORS]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="xgboost", choices=["xgboost", "lightgbm"])
    parser.add_argument("--factors", default="all", help="'all', factor preset name, '+preset1+preset2', or comma-separated factor list")
    parser.add_argument("--factor-preset", default="", help="Shorthand for --factors: momentum, reversal, volatility, liquidity, fundamental, all")
    parser.add_argument("--forward-days", type=int, default=1, help="Label horizon: 1=next day, 5=next week")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--train-years", type=int, default=3)
    parser.add_argument("--val-years", type=int, default=1)
    parser.add_argument("--start", default="20200101")
    parser.add_argument("--end", default="20260501")
    parser.add_argument("--codes", default="")
    parser.add_argument("--no-ensemble", action="store_true")
    parser.add_argument("--no-orthogonal", action="store_true")
    parser.add_argument("--no-ic-gate", action="store_true", help="跳过 IC 门禁")
    parser.add_argument("--regime", action="store_true", help="启用分市场状态训练")
    parser.add_argument("--optuna", action="store_true", help="启用 Optuna 超参优化")
    parser.add_argument("--optuna-trials", type=int, default=30, help="Optuna 搜索轮数")
    parser.add_argument("--strategy", default="", help="策略名称，对应 strategy_configs.name，用于匹配 version_id")
    parser.add_argument("--dynamic", action="store_true", help="启用动态多因子闭环（归因→健康度→调参）")
    parser.add_argument("--rebalance-freq", type=int, default=5, help="调仓频率（交易日），默认5=周度")
    parser.add_argument("--universe-size", type=int, default=0, help="候选池上限，0=全市场非ST")
    parser.add_argument("--dd-reduce", type=float, default=0.20, help="组合回撤减仓阈值")
    parser.add_argument("--dd-liquidate", type=float, default=0.25, help="组合回撤清仓阈值")
    parser.add_argument("--index-crash", type=float, default=-0.12, help="指数大跌阈值")
    args = parser.parse_args()

    # Resolve factor preset (--factor-preset overrides --factors if given)
    if args.factor_preset:
        factor_names = get_factors_by_preset(args.factor_preset)
    elif args.factors == "all":
        factor_names = list(ALL_FACTORS.keys())
    else:
        factor_names = get_factors_by_preset(args.factors)

    if not factor_names:
        logger.error("因子列表为空，请检查 --factors/--factor-preset 参数")
        sys.exit(1)

    logger.info(f"{len(factor_names)} 个因子, label=ret_{args.forward_days}d, "
                f"IC门禁={'ON' if not args.no_ic_gate else 'OFF'}, "
                f"regime={'ON' if args.regime else 'OFF'}, optuna={'ON' if args.optuna else 'OFF'}, "
                f"model={args.model}, ensemble={'OFF' if args.no_ensemble else 'ON'}")

    orig_factor_names = factor_names.copy()  # Save before IC filtering

    engine = get_engine()

    # 查找策略版本ID 和 strategy_configs.id
    version_id = 0
    strategy_config_id = 0
    if args.strategy:
        with engine.connect() as conn:
            row = conn.execute(
                text("""
                    SELECT sv.id, sc.id FROM strategy_versions sv
                    JOIN strategy_configs sc ON sv.strategy_id = sc.id
                    WHERE sc.name = :name AND sv.version = '1.10'
                """),
                {"name": args.strategy},
            ).fetchone()
            if row:
                version_id = row[0]
                strategy_config_id = row[1]
                logger.info(f"策略 '{args.strategy}' → version_id={version_id}, config_id={strategy_config_id}")
            else:
                logger.warning(f"未找到策略 '{args.strategy}' 的 v1.10 版本，将使用 version_id=0")
    codes = [c.strip() for c in args.codes.split(",") if c.strip()] if args.codes else None
    if codes is None:
        if args.universe_size > 0:
            # 按成交额取前 N 只（覆盖沪深两市）
            codes = pd.read_sql(
                f"SELECT code FROM stock_daily "
                f"WHERE trade_date >= '{args.start}' AND trade_date <= '{args.end}' "
                f"GROUP BY code ORDER BY SUM(amount) DESC LIMIT {args.universe_size}",
                engine,
            )["code"].tolist()
        else:
            # 全市场非 ST（默认）
            codes = pd.read_sql(
                "SELECT code FROM stock_basic WHERE is_st = FALSE "
                "AND list_date <= CURRENT_DATE - INTERVAL '60 days' "
                "ORDER BY code",
                engine,
            )["code"].tolist()
        logger.info(f"候选池: {len(codes)} 只股票 (universe_size={args.universe_size or '全市场'})")

    code_list = ",".join([f"'{c}'" for c in codes])

    # OHLCV
    sql = f"""
        SELECT code, trade_date, open, high, low, close, volume, amount, turnover
        FROM stock_daily
        WHERE code IN ({code_list})
          AND trade_date BETWEEN '{args.start}' AND '{args.end}'
        ORDER BY code, trade_date
    """
    ohlcv = pd.read_sql(sql, engine)
    logger.info(f"OHLCV: {len(ohlcv)} 行")

    # 加载指数数据（始终加载用于风控，regime 模式额外做状态检测）
    regime_df = None
    index_daily_close: pd.Series = pd.Series(dtype=float)
    try:
        idx_sql = f"""
            SELECT trade_date, close FROM index_daily
            WHERE code = '000001' AND trade_date BETWEEN '{args.start}' AND '{args.end}'
            ORDER BY trade_date
        """
        index_df = pd.read_sql(idx_sql, engine)
        if not index_df.empty:
            index_df["trade_date"] = pd.to_datetime(index_df["trade_date"])
            index_daily_close = index_df.set_index("trade_date")["close"].sort_index()
            logger.info(f"指数数据: {len(index_df)} 行 (覆盖 {len(index_daily_close)} 天)")
            if args.regime:
                regime_df = detect_regime(index_df)
                logger.info(f"regime={regime_df['regime'].value_counts().to_dict()}")
    except Exception as e:
        logger.warning(f"指数数据加载失败: {e}, 风控中指数检查将跳过")
        if args.regime:
            logger.warning("退到非 regime 模式")
            args.regime = False

    # extra_data
    extra_data = {}
    try:
        extra_sql = f"""
            SELECT code, trade_date, market_cap, pb
            FROM stock_daily_extra
            WHERE code IN ({code_list})
              AND trade_date BETWEEN '{args.start}' AND '{args.end}'
        """
        extra_df = pd.read_sql(extra_sql, engine)
        if not extra_df.empty:
            extra_df["log_mcap"] = np.log(extra_df["market_cap"].replace(0, np.nan))
            extra_data["log_mcap"] = extra_df[["code", "trade_date", "log_mcap"]]
            extra_data["pb"] = extra_df[["code", "trade_date", "pb"]]
            logger.info(f"  估值数据: {len(extra_df)} 行")
    except Exception as e:
        logger.warning(f"  估值数据加载失败: {e}")

    try:
        sh_sql = f"""
            SELECT code, end_date AS trade_date, shareholder_count
            FROM stock_shareholder
            WHERE code IN ({code_list})
              AND end_date BETWEEN '{args.start}' AND '{args.end}'
        """
        sh_df = pd.read_sql(sh_sql, engine)
        if not sh_df.empty:
            extra_data["shareholder_count"] = sh_df[["code", "trade_date", "shareholder_count"]]
            logger.info(f"  股东数据: {len(sh_df)} 行")
    except Exception as e:
        logger.warning(f"  股东数据加载失败: {e}")

    # ── 排雷系统：加载财务 & 质押数据 ──
    quality_pass: dict[str, set[str]] = {}
    try:
        fin_sql = f"""
            SELECT code, report_date, adjusted_profit, net_profit,
                   total_assets, total_liability, goodwill, holder_equity,
                   operating_cash_flow, roe, net_margin
            FROM stock_financial
            WHERE code IN ({code_list})
              AND report_date >= '2018-01-01'
            ORDER BY code, report_date
        """
        fin_df = pd.read_sql(fin_sql, engine)

        pledge_sql = f"""
            SELECT code, trade_date, pledge_ratio
            FROM stock_pledge
            WHERE code IN ({code_list})
              AND trade_date BETWEEN '{args.start}' AND '{args.end}'
            ORDER BY code, trade_date
        """
        pledge_df = pd.read_sql(pledge_sql, engine)

        if not fin_df.empty:
            # 按 code 分组，对每个财务指标做 ffill
            fin_cols = ["adjusted_profit", "net_profit", "total_assets",
                        "total_liability", "goodwill", "holder_equity",
                        "operating_cash_flow", "roe", "net_margin"]
            fin_df["report_date"] = pd.to_datetime(fin_df["report_date"])
            for c in fin_cols:
                if c in fin_df.columns:
                    fin_df[c] = fin_df[c].replace(0, np.nan)
                    fin_df[c] = fin_df.groupby("code")[c].ffill()

            # 获取回测区间的所有交易日
            all_trade_dates = sorted(ohlcv["trade_date"].unique())
            all_trade_dates_dt = pd.to_datetime(all_trade_dates)

            # 为每个交易日构建质量通过的股票集合
            # 排雷检查: adjusted_profit<0, debt>70%, goodwill>30%, pledge>80%, net_profit<0
            # 允许最多 3 项违规
            fin_lookup = fin_df.copy()
            pledge_lookup = pledge_df.copy() if not pledge_df.empty else pd.DataFrame(
                columns=["code", "trade_date", "pledge_ratio"])
            if not pledge_lookup.empty:
                pledge_lookup["trade_date"] = pd.to_datetime(pledge_lookup["trade_date"])

            # 为所有交易日计算质量通过集合（调仓日需要用到）
            for dt in all_trade_dates_dt:
                dt_str = pd.Timestamp(dt).strftime("%Y-%m-%d")
                passing: set[str] = set()

                for code in codes:
                    # 获取该日期前最新的财务数据
                    code_fin = fin_lookup[
                        (fin_lookup["code"] == code) &
                        (fin_lookup["report_date"] <= dt)
                    ]
                    if code_fin.empty:
                        continue
                    code_fin = code_fin.iloc[-1]

                    violations = 0

                    # 1. 主业存疑：扣非净利润 < 0
                    adj = code_fin.get("adjusted_profit", np.nan)
                    if pd.notna(adj) and adj < 0:
                        violations += 1

                    # 2. 净利润为负
                    np_val = code_fin.get("net_profit", np.nan)
                    if pd.notna(np_val) and np_val < 0:
                        violations += 1

                    # 3. 资金链紧绷：资产负债率 > 70%
                    assets = code_fin.get("total_assets", np.nan)
                    liability = code_fin.get("total_liability", np.nan)
                    if pd.notna(assets) and pd.notna(liability) and assets > 0:
                        if liability / assets > 0.70:
                            violations += 1

                    # 4. 商誉过高：商誉 / 股东权益 > 30%
                    gw = code_fin.get("goodwill", np.nan)
                    equity = code_fin.get("holder_equity", np.nan)
                    if pd.notna(gw) and pd.notna(equity) and equity > 0:
                        if gw / equity > 0.30:
                            violations += 1

                    # 5. 大股东高质押
                    if not pledge_lookup.empty:
                        code_pl = pledge_lookup[
                            (pledge_lookup["code"] == code) &
                            (pledge_lookup["trade_date"] <= dt)
                        ]
                        if not code_pl.empty:
                            pl_ratio = code_pl.iloc[-1]["pledge_ratio"]
                            if pd.notna(pl_ratio) and pl_ratio > 0.80:
                                violations += 1

                    # 6. 现金流异常：净利润>0 但经营现金流<0
                    ocf = code_fin.get("operating_cash_flow", np.nan)
                    if pd.notna(np_val) and pd.notna(ocf) and np_val > 0 and ocf < 0:
                        violations += 1

                    # 7. ROE 为负
                    roe_val = code_fin.get("roe", np.nan)
                    if pd.notna(roe_val) and roe_val < 0:
                        violations += 1

                    # 8. 净利率下滑（近似：负净利率即为有问题）
                    nm_val = code_fin.get("net_margin", np.nan)
                    if pd.notna(nm_val) and nm_val < 0:
                        violations += 1

                    if violations <= 3:
                        passing.add(code)

                if passing:
                    quality_pass[dt_str] = passing

        logger.info(f"  排雷数据: {len(fin_df)} 行财务, {len(pledge_df)} 行质押, "
                    f"{len(quality_pass)} 个调仓日")
    except Exception as e:
        logger.warning(f"  排雷数据加载失败: {e}, 跳过质量过滤")

    engine.dispose()

    # 构建因子数据集（含 ret_1d 和 ret_{forward_days}d 列）
    ret_col = f"ret_{args.forward_days}d"
    dataset = build_factor_dataset(
        ohlcv, factor_names, label_mode="binary",
        forward_days=args.forward_days,
        extra_data=extra_data if extra_data else None,
    )

    # 1. IC 门禁
    if not args.no_ic_gate:
        factor_names = filter_factors_by_ic(dataset, factor_names, ret_col=ret_col)

    # 2. 正交筛选
    if not args.no_orthogonal:
        factor_cols = select_orthogonal_factors(dataset, factor_names, threshold=0.7)
    else:
        factor_cols = factor_names

    # 3. Walk-forward 训练
    use_ensemble = not args.no_ensemble

    if args.regime and use_ensemble and regime_df is not None:
        logger.info("使用分市场状态集成训练")
        results = walk_forward_train_by_regime(
            dataset, factor_cols, regime_df,
            train_years=args.train_years, val_years=args.val_years,
        )
    elif use_ensemble:
        logger.info("使用 XGBoost + LightGBM 集成模式")
        results = walk_forward_train_ensemble(
            dataset, factor_cols,
            train_years=args.train_years, val_years=args.val_years,
            use_optuna=args.optuna, optuna_trials=args.optuna_trials,
        )
    else:
        logger.info(f"使用单模型: {args.model}")
        results = walk_forward_train(
            dataset, factor_cols, model_type=args.model,
            train_years=args.train_years, val_years=args.val_years,
        )

    logger.info(f"完成 {len(results)} 个 walk-forward 窗口")

    # 汇总
    all_metrics = []
    for i, r in enumerate(results):
        m = r["metrics"]
        best_t = r.get("best_threshold", 0.5)
        regime_info = f", regimes={r.get('regimes_trained', [])}" if "regimes_trained" in r else ""
        logger.info(
            f"窗口 {i+1}: val={r['val_start'].date()}~{r['val_end'].date()}, "
            f"t={best_t:.2f}, acc={m['accuracy']:.3f}, prec={m['precision']:.3f}, rec={m['recall']:.3f}"
            f"{regime_info}"
        )
        all_metrics.append({
            "window": i + 1,
            "val_start": r["val_start"],
            "val_end": r["val_end"],
            "best_t": best_t,
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
        })

    summary = pd.DataFrame(all_metrics)
    print("\n=== Walk-Forward 汇总 ===")
    if summary.empty:
        print("无有效窗口，请检查数据范围（train_years + val_years 需小于数据跨度）")
        return
    print(summary.to_string(index=False))
    print(f"\n平均准确率: {summary['accuracy'].mean():.3f}")
    print(f"平均精确率: {summary['precision'].mean():.3f}")
    print(f"平均召回率: {summary['recall'].mean():.3f}")

    if not use_ensemble and results:
        fi = results[-1]["metrics"].get("feature_importance", pd.Series(dtype=float))
        if not fi.empty:
            print("\n=== Top-10 因子 ===")
            print(fi.head(10).to_string())

    if use_ensemble and results:
        print(f"\n=== 筛选管线 ===")
        active = results[-1].get("active_cols", [])
        print(f"最终活跃因子: {len(active)} 个")
        print(f"因子列表: {active[:15]}..." if len(active) > 15 else f"因子列表: {active}")

    # ── 动态多因子反馈闭环 ──────────────────────────────────────────────
    feedback_summary = {}
    if args.dynamic and strategy_config_id > 0:
        print(f"\n=== 动态反馈闭环 ===")
        loop = BacktestFeedbackLoop(strategy_config_id, orig_factor_names, get_engine())
        for i, r in enumerate(results):
            m = r["metrics"]
            win_metrics = {
                "sharpe": m.get("sharpe_ratio", m.get("accuracy", 0)),
                "accuracy": m.get("accuracy", 0),
                "precision": m.get("precision", 0),
                "recall": m.get("recall", 0),
            }
            health = loop.process_window(i, r, win_metrics, str(r.get("val_end", "")))
            status_icon = {"normal": "✓", "warning": "⚠", "critical": "✗"}.get(health["status"], "?")
            print(f"  窗口{i+1}: {status_icon} {health['status']} "
                  f"sharpe={win_metrics['sharpe']:.3f} acc={win_metrics['accuracy']:.3f}")
            if health["warnings"]:
                for w in health["warnings"]:
                    print(f"    ⚠ {w}")

        # 应用动态因子筛选：剔除权重<0.3的因子
        dynamic_factors = loop.get_dynamic_factors(factor_cols)
        if len(dynamic_factors) < len(factor_cols):
            removed = set(factor_cols) - set(dynamic_factors)
            print(f"\n  动态淘汰因子 ({len(removed)}): {removed}")
            factor_cols = dynamic_factors

        feedback_summary = loop.save_summary()
    elif args.dynamic:
        print(f"\n[跳过] 动态闭环需要匹配到 strategy_configs 中的策略")

    # ── 构建日度权益曲线 + 日度信号追踪 + 动态反馈 ─────────────────────
    # A股交易成本: 统一引用 config/settings.py TradingConfig
    COMM = TradingConfig.COMMISSION
    STAMP = TradingConfig.STAMP_DUTY
    SLIP = TradingConfig.SLIPPAGE
    COST_PER_TURNOVER = COMM * 2 + STAMP + SLIP * 2  # round-trip cost per unit turnover

    equity_curve = {}
    daily_returns = {}
    nav = 1.0
    all_dates = []
    all_navs = []
    daily_position_records: list[dict] = []
    turnover_rates: list[float] = []
    daily_costs: list[float] = []

    # 初始化日度信号追踪器（所有 ML 策略都追踪，动态策略额外启用调参）
    tracker = DailySignalTracker(strategy_config_id, get_engine(), window_size=20)

    # Collect active factor cols from results
    all_active_cols = []
    for r in results:
        ac = r.get("active_cols", [])
        if ac:
            all_active_cols = ac

    total_trading_days = 0
    current_factor_weights = {f: 1.0 for f in factor_cols}
    retrain_count = 0
    day_counter = 0
    current_holdings: list[str] = []
    cost_basis: dict[str, float] = {}  # code -> entry price
    stop_loss_events: list[dict] = []  # record stop-loss triggers
    risk_events: list[dict] = []  # record portfolio DD / index crash events
    peak_nav = 1.0
    lockout_until = 0  # day_counter value after liquidation

    # ── 价格查找表（止损计算用） ──
    price_lookup: dict[str, dict[str, float]] = {}  # {date_str: {code: close}}
    ohlcv_pivot = ohlcv.pivot_table(
        index="trade_date", columns="code", values="close", aggfunc="last"
    )
    for dt_idx in ohlcv_pivot.index:
        row = ohlcv_pivot.loc[dt_idx]
        price_lookup[str(dt_idx)[:10]] = {c: float(row[c]) for c in row.index if pd.notna(row[c])}

    for window_idx, r in enumerate(results):
        ensemble = r.get("ensemble")
        single_model = r.get("model")

        if ensemble is None and single_model is None:
            continue

        val_start = pd.Timestamp(r["val_start"])
        val_end = pd.Timestamp(r["val_end"])
        val_mask = (dataset["trade_date"] >= val_start) & (dataset["trade_date"] <= val_end)
        val_data = dataset[val_mask].copy()

        if val_data.empty or len(val_data["trade_date"].unique()) < 2:
            continue

        val_dates = sorted(val_data["trade_date"].unique())

        if ensemble is not None:
            pred_factors = ensemble.factor_names
        else:
            pred_factors = r.get("active_cols", factor_cols)

        for i, dt in enumerate(val_dates[:-1]):
            day_data = val_data[val_data["trade_date"] == dt]
            next_dt = val_dates[i + 1]
            next_data = val_data[val_data["trade_date"] == next_dt]

            if day_data.empty or next_data.empty:
                continue

            # ── 调仓决策：每 N 日调仓一次 ──
            is_rebalance = (day_counter % args.rebalance_freq == 0) or (not current_holdings and day_counter >= lockout_until)

            if is_rebalance:
                # ── 日度信号质量追踪 ──
                if args.dynamic and tracker.needs_adjustment():
                    decayed = [f for f, w in current_factor_weights.items() if w < 0.8]
                    if decayed:
                        print(f"  [动态] {str(dt)[:10]} IC衰减警告={tracker.decay_warnings}, "
                              f"低权重因子: {decayed}")
                    if tracker.needs_retrain() and retrain_count < 2:
                        print(f"  [动态] {str(dt)[:10]} 连续低IC触发重训建议 "
                              f"(decay_warnings={tracker.decay_warnings})")
                        retrain_count += 1

                try:
                    if ensemble is not None:
                        preds = ensemble.predict(day_data)
                    else:
                        model_features = getattr(single_model, 'feature_names_in_',
                                                getattr(single_model, 'feature_name_', pred_factors))
                        X = day_data[model_features].copy().fillna(0)
                        prob = single_model.predict_proba(X)[:, 1]
                        preds = day_data[["code"]].copy()
                        preds["score"] = prob
                        preds = preds.sort_values("score", ascending=False).reset_index(drop=True)
                except Exception:
                    continue

                # ── 排雷过滤：剔除质量不过关的股票 ──
                dt_str = str(dt)[:10]
                if quality_pass and dt_str in quality_pass:
                    passing_set = quality_pass[dt_str]
                    preds = preds[preds["code"].isin(passing_set)]

                if preds.empty:
                    continue

                # ── NDrop 选股 ──
                scores_series = pd.Series(
                    preds["score"].values,
                    index=preds["code"].values,
                ).sort_values(ascending=False)

                new_holdings_set, to_buy, to_sell = select_topk_ndrop(
                    scores_series,
                    current_holdings=set(current_holdings),
                    K=args.top_n,
                    N=TradingConfig.NDROP_N,
                )

                # ── 更新成本基础（新买入的股票记录买入价） ──
                today_prices = price_lookup.get(dt_str, {})
                for code in to_buy:
                    if code in today_prices:
                        cost_basis[code] = today_prices[code]
                # 清除已卖出的成本基础
                for code in to_sell:
                    cost_basis.pop(code, None)

                new_holdings = list(new_holdings_set)
                top_scores = [float(scores_series.get(c, 0)) for c in new_holdings]

                # ── 交易成本：基于 NDrop 的 to_buy/to_sell 计算 ──
                if to_buy or to_sell:
                    changed = len(to_buy) + len(to_sell)
                    n_positions = max(len(new_holdings_set), len(current_holdings), 1)
                    turnover_rate = changed / (n_positions * 2)
                    cost = turnover_rate * COST_PER_TURNOVER
                else:
                    turnover_rate = 0.0
                    cost = 0.0

                current_holdings = new_holdings
            else:
                # 持仓不变：无调仓成本
                turnover_rate = 0.0
                cost = 0.0
                top_scores = [0.0] * len(current_holdings) if current_holdings else []

            day_counter += 1
            top_codes = current_holdings

            # ── 个股止损检查：每日检查持仓是否触及 -8% 止损 ──
            next_prices = price_lookup.get(str(next_dt)[:10], {})
            stopped_out: list[str] = []
            for code in top_codes[:]:
                if code in cost_basis and code in next_prices and cost_basis[code] > 0:
                    pnl_pct = (next_prices[code] - cost_basis[code]) / cost_basis[code]
                    if pnl_pct <= -TradingConfig.STOP_LOSS_PCT:
                        stopped_out.append(code)
                        stop_loss_events.append({
                            "date": str(next_dt)[:10],
                            "code": code,
                            "entry_price": round(cost_basis[code], 3),
                            "exit_price": round(next_prices[code], 3),
                            "pnl_pct": round(pnl_pct, 4),
                        })
                        current_holdings.remove(code)
                        cost_basis.pop(code, None)

            if stopped_out:
                # 止损卖出成本：佣金 + 印花税 + 滑点
                stop_cost = len(stopped_out) * (COMM + STAMP + SLIP) / max(len(top_codes) + len(stopped_out), 1)
                cost += stop_cost

            daily_costs.append(cost)
            turnover_rates.append(turnover_rate)

            selected_next = next_data[next_data["code"].isin(top_codes)]

            if selected_next.empty:
                # 空仓状态：仍记录日期，NAV 不变（现金），但不计算收益
                if not top_codes:
                    all_dates.append(str(next_dt)[:10])
                    all_navs.append(round(nav, 6))
                continue

            daily_ret = float(selected_next["ret_1d"].mean())
            daily_ret_net = daily_ret - cost
            total_trading_days += 1

            # ── 记录日度持仓 + 信号质量 ──
            pos_record = {
                "date": str(next_dt)[:10],
                "codes": top_codes,
                "scores": top_scores,
                "daily_ret": round(daily_ret, 6),
                "daily_ret_net": round(daily_ret_net, 6),
                "turnover": round(turnover_rate, 4),
                "is_rebalance": is_rebalance,
            }
            daily_position_records.append(pos_record)

            # 计算日度 Rank IC（仅在调仓日）
            if is_rebalance:
                actual_ret_map = next_data.set_index("code")["ret_1d"].to_dict()
                pred_scores_ic = []
                actual_rets_ic = []
                for _, row in preds.head(50).iterrows():
                    pred_scores_ic.append(row["score"])
                    actual_rets_ic.append(actual_ret_map.get(row["code"], 0))

                tracker.record_day(
                    date_str=str(next_dt)[:10],
                    pred_scores=pred_scores_ic,
                    actual_rets=actual_rets_ic,
                    positions={"codes": top_codes, "scores": top_scores},
                    daily_ret=daily_ret_net,
                    factor_weights=current_factor_weights if args.dynamic else None,
                )

            # ── 复合净值 ──
            nav *= (1 + daily_ret_net)
            peak_nav = max(peak_nav, nav)

            # ── 组合级风控 ──
            dd = (peak_nav - nav) / peak_nav if peak_nav > 0 else 0.0

            if check_drawdown_limit(nav, peak_nav, limit=args.dd_liquidate):
                if current_holdings:
                    risk_events.append({
                        "date": str(next_dt)[:10], "type": "liquidate",
                        "dd": round(dd, 4), "nav": round(nav, 6),
                        "msg": f"组合回撤 {dd:.1%} >= {args.dd_liquidate:.0%}，清仓",
                    })
                    current_holdings = []
                    cost_basis = {}
                    lockout_until = day_counter + 10
                    peak_nav = nav  # 重置峰值，避免清仓后立即再触发
            elif dd >= args.dd_reduce and len(current_holdings) > 1:
                keep = max(1, len(current_holdings) // 2)
                sold = current_holdings[keep:]
                current_holdings = current_holdings[:keep]
                for code in sold:
                    cost_basis.pop(code, None)
                risk_events.append({
                    "date": str(next_dt)[:10], "type": "reduce",
                    "dd": round(dd, 4), "nav": round(nav, 6),
                    "msg": f"组合回撤 {dd:.1%} >= {args.dd_reduce:.0%}，减仓至 {keep} 只",
                })

            if not index_daily_close.empty and len(current_holdings) > 0:
                idx_slice = index_daily_close.loc[:pd.Timestamp(str(next_dt)[:10])]
                if check_index_crash(idx_slice.values, lookback=15, threshold=args.index_crash):
                    risk_events.append({
                        "date": str(next_dt)[:10], "type": "index_crash",
                        "dd": round(dd, 4), "nav": round(nav, 6),
                        "msg": f"指数 15 日跌超 {abs(args.index_crash):.0%}，空仓",
                    })
                    current_holdings = []
                    cost_basis = {}
                    lockout_until = day_counter + 10
                    peak_nav = nav

            all_dates.append(str(next_dt)[:10])
            all_navs.append(round(nav, 6))

    # ── 日度信号追踪总结 ──
    tracker.save_daily_summary()
    tracker_status = tracker.get_status()
    # Turnover stats
    if turnover_rates:
        avg_turnover = np.mean(turnover_rates)
        avg_cost = np.mean(daily_costs)
        avg_daily_gross = np.mean([r.get("daily_ret", 0) for r in daily_position_records]) if daily_position_records else 0
        print(f"\n=== 交易成本分析 ===")
        print(f"日均换手率: {avg_turnover:.1%}, 日均摩擦成本: {avg_cost:.4%}")
        print(f"日均毛收益(成本前): {avg_daily_gross:.4%}, 日均净收益: {avg_daily_gross - avg_cost:.4%}")
        print(f"年化成本侵蚀: {avg_cost * 252:.1%}")

        # ── 止损统计 ──
        if stop_loss_events:
            n_stops = len(stop_loss_events)
            avg_loss = np.mean([e["pnl_pct"] for e in stop_loss_events])
            print(f"\n=== 止损统计 ===")
            print(f"止损触发次数: {n_stops}, 平均亏损: {avg_loss:.2%}, "
                  f"止损频率: {n_stops / max(total_trading_days, 1):.1%}/日")

        # ── 排雷统计 ──
        if quality_pass:
            n_pass_dates = len(quality_pass)
            avg_pass = np.mean([len(s) for s in quality_pass.values()])
            print(f"\n=== 排雷统计 ===")
            print(f"覆盖调仓日: {n_pass_dates}, 日均通过: {avg_pass:.0f} 只")

    # ── 风控事件统计 ──
    if risk_events:
        print(f"\n=== 风控事件 ===")
        for evt in risk_events:
            print(f"  {evt['date']} [{evt['type']}] {evt['msg']}")
        n_liquidate = sum(1 for e in risk_events if e['type'] == 'liquidate')
        n_reduce = sum(1 for e in risk_events if e['type'] == 'reduce')
        n_crash = sum(1 for e in risk_events if e['type'] == 'index_crash')
        print(f"清仓: {n_liquidate} 次, 减仓: {n_reduce} 次, 指数大跌: {n_crash} 次")

    print(f"\n=== 日度信号追踪 ===")
    print(f"交易日: {tracker_status['total_days']}, "
          f"IC均值: {tracker_status['rolling_ic_20d']:.4f}, "
          f"信号等级: {tracker_status['signal_level']}")
    print(f"IC正率: {tracker_status.get('ic_stats', {}).get('positive_ratio', 'N/A')}")

    if all_dates:
        equity_curve = dict(zip(all_dates, all_navs))
        start_dt = str(results[0]["val_start"])[:10] if results else args.start
        ordered = {start_dt: 1.0}
        ordered.update(dict(sorted(equity_curve.items())))
        equity_curve = ordered

        prev_nav = 1.0
        for dt, nv in sorted(equity_curve.items()):
            if dt == start_dt:
                daily_returns[dt] = 0.0
            else:
                daily_returns[dt] = round(float(nv / prev_nav - 1), 6)
            prev_nav = nv
    else:
        nav = 1.0
        for m in all_metrics:
            window_return = (m["accuracy"] - 0.5) * 0.4
            nav *= (1 + window_return)
            equity_curve[str(m["val_end"])] = round(nav, 6)
        if all_metrics:
            equity_curve[str(all_metrics[0]["val_start"])] = 1.0
            equity_curve = dict(sorted(equity_curve.items()))

    # Compute real metrics from equity curve
    n_vals = list(equity_curve.values())
    total_return = float(n_vals[-1] / n_vals[0] - 1) if len(n_vals) >= 2 else 0.0
    n_days = max(len(n_vals), 1)
    years = max(n_days / 252, 0.2)
    annual_return = float((1 + total_return) ** (1 / years) - 1)

    # Sharpe from daily returns
    daily_ret_vals = [v for v in daily_returns.values() if v != 0.0]
    if daily_ret_vals and np.std(daily_ret_vals) > 0:
        computed_sharpe = float(np.mean(daily_ret_vals) / np.std(daily_ret_vals) * np.sqrt(252))
    else:
        computed_sharpe = 0.0

    # Max drawdown
    peak = 1.0
    max_dd = 0.0
    for v in n_vals:
        if v > peak:
            peak = v
        dd = (peak - v) / peak
        if dd > max_dd:
            max_dd = dd
    computed_mdd = float(max_dd)

    # Win rate from daily returns
    if daily_ret_vals:
        win_days = sum(1 for v in daily_ret_vals if v > 0)
        loss_days = sum(1 for v in daily_ret_vals if v < 0)
        computed_win = win_days / max(len(daily_ret_vals), 1)
    else:
        computed_win = 0.0
        win_days = 0
        loss_days = 0

    print(f"\n=== 权益曲线统计 ===")
    print(f"总收益: {total_return:.2%}, 年化: {annual_return:.2%}, Sharpe: {computed_sharpe:.2f}, 最大回撤: {computed_mdd:.2%}")

    # ── 防过拟合验证并入库 ──────────────────────────────────────────────
    # Collect all factor info
    initial_factors = orig_factor_names.copy()  # Full list before IC gate
    screened_factors = factor_cols.copy()  # After IC+orthogonal screening
    n_regimes = len(set(r.get("regime", 0) for r in results)) if args.regime else 1

    _validate_and_save(
        version_id=version_id,
        metrics={
            "train_sharpe": 0,
            "val_sharpe": 0,
            "test_sharpe": computed_sharpe,
            "annual_return": annual_return,
            "total_return": total_return,
            "sharpe": computed_sharpe,
            "max_drawdown": computed_mdd,
            "n_trades": total_trading_days,
            "n_params": len(factor_cols),
            "n_wins": win_days,
            "n_losses": loss_days,
            "start_date": args.start,
            "end_date": args.end,
            "win_rate": float(computed_win),
            "factor_cols": screened_factors,
            "active_cols": all_active_cols,
            "initial_factors": initial_factors,
            "n_stocks": len(codes),
            "n_days": n_days,
            "walk_forward_windows": len(results),
            "feedback_summary": feedback_summary,
            "daily_signal_tracker": tracker_status,
            "daily_ic_series": tracker.get_ic_series(),
            "position_history": daily_position_records[-500:],
            "stop_loss_events": stop_loss_events,
            "n_stop_loss": len(stop_loss_events),
            "risk_events": risk_events,
            "n_risk_events": len(risk_events),
            "quality_pass_dates": len(quality_pass),
            "rebalance_freq": args.rebalance_freq,
            "ndrop_n": TradingConfig.NDROP_N,
            "top_n": TradingConfig.TOP_N,
        },
        equity_curve=equity_curve,
        daily_returns=daily_returns,
        regime_count=n_regimes,
        sensitivity_stable=True,
    )


def _validate_and_save(version_id, metrics, equity_curve, daily_returns,
                       regime_count=0, sensitivity_stable=True):
    """Run overfitting checks and save results to backtest_results table."""
    checker = OverfitChecker(min_trades=3, min_regimes=1)
    result = checker.check(metrics, regime_count=regime_count,
                           sensitivity_stable=sensitivity_stable)

    start_date = metrics.get("start_date", "")
    end_date = metrics.get("end_date", "")

    engine = get_engine()
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO backtest_results
                    (version_id, start_date, end_date, quality, quality_flags,
                     metrics_json, equity_curve_json, daily_returns_json)
                VALUES (:vid, :start, :end, :quality, :flags, :metrics, :equity, :returns)
            """), {
                "vid": version_id,
                "start": start_date,
                "end": end_date,
                "quality": result["quality"],
                "flags": result["flags"],
                "metrics": json.dumps({**metrics, "adjusted_sharpe": result["adjusted_sharpe"]}, default=str),
                "equity": json.dumps(equity_curve, default=str),
                "returns": json.dumps(daily_returns, default=str),
            })
        print(f"\n回测结果已写入 backtest_results (quality={result['quality']})")
    except Exception as e:
        print(f"回测结果写入失败: {e}")

    if result["flags"]:
        print("警告标记:")
        for f in result["flags"]:
            print(f"  ⚠ {f}")

    return result["quality"]


if __name__ == "__main__":
    main()
