#!/usr/bin/env python
"""日频模拟盘每日驱动：拉数据 → 训练 → 预测 → 执行 → 写DB。

用法:
    python scripts/run_daily_paper.py                        # 今日收盘后跑
    python scripts/run_daily_paper.py --date 2026-06-01      # 指定日期
    python scripts/run_daily_paper.py --date 2026-06-01 --backfill  # 回填历史
    python scripts/run_daily_paper.py --dry-run              # 试运行（不写DB）

初始化（首次）:
    python scripts/init_paper_trading.py    # 创建 paper_account + paper_runs
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import date, timedelta, datetime
from loguru import logger
from sqlalchemy import text

from data.db import get_engine
from models.dataset import build_factor_dataset
from models.trainer import walk_forward_train_by_regime
from models.regime import detect_regime
from factors import ALL_FACTORS
from factors.screening import filter_factors_by_ic, select_orthogonal_factors
from portfolio.paper_engine import PaperEngine
from config.settings import TradingConfig

# ── 因子预设：与回测保持一致的66因子集（通过IC+正交自动筛选） ──
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
    # v1.13 新增 - 纯价格动量（不依赖换手率）
    *["price_mom_5", "price_mom_10", "price_accel"],
]
FACTOR_NAMES = [f for f in _FACTOR_PRESET_NAMES if f in ALL_FACTORS]


def load_data(engine, start_date: str, end_date: str, universe_size: int = 500):
    """加载 OHLCV + 指数 + 行业 + 估值数据。"""
    # 候选池：按成交额排序取 top-N（排除 ST）
    codes = pd.read_sql(
        f"SELECT d.code FROM stock_daily d "
        f"JOIN stock_basic b ON d.code = b.code AND b.is_st = FALSE "
        f"WHERE d.trade_date BETWEEN '{start_date}' AND '{end_date}' "
        f"GROUP BY d.code ORDER BY SUM(d.amount) DESC LIMIT {universe_size}",
        engine,
    )["code"].tolist()
    code_list = ",".join([f"'{c}'" for c in codes])

    # OHLCV
    ohlcv = pd.read_sql(f"""
        SELECT code, trade_date, open, high, low, close, volume, amount, turnover
        FROM stock_daily
        WHERE code IN ({code_list})
          AND trade_date BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY code, trade_date
    """, engine)
    ohlcv["trade_date"] = pd.to_datetime(ohlcv["trade_date"])
    logger.info(f"OHLCV: {len(ohlcv)} 行, {ohlcv['code'].nunique()} 只")

    # 指数
    index_df = pd.read_sql(f"""
        SELECT trade_date, close FROM index_daily
        WHERE code = '000001' AND trade_date BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY trade_date
    """, engine)
    index_df["trade_date"] = pd.to_datetime(index_df["trade_date"])
    regime_df = detect_regime(index_df) if not index_df.empty else None
    if regime_df is not None:
        logger.info(f"市场状态: {regime_df['regime'].value_counts().to_dict()}")

    # extra_data（仅行业数据，日频因子不需要估值）
    extra_data = {}

    try:
        ind_df = pd.read_sql(f"""
            SELECT code, industry_sw1 FROM stock_industry WHERE code IN ({code_list})
        """, engine)
        if not ind_df.empty:
            all_dates = sorted(ohlcv["trade_date"].unique())
            ind_df["trade_date"] = pd.to_datetime(ind_df["trade_date"])
            extra_data["industry_sw1"] = ind_df.merge(
                pd.DataFrame({"trade_date": all_dates}), how="cross"
            )[["code", "trade_date", "industry_sw1"]]
    except Exception:
        pass

    return ohlcv, index_df, regime_df, extra_data, codes


def main():
    parser = argparse.ArgumentParser(description="日频模拟盘每日驱动")
    parser.add_argument("--date", help="交易日（YYYY-MM-DD），默认今天")
    parser.add_argument("--account-id", type=int, default=1, help="paper_account.id")
    parser.add_argument("--run-id", type=int, default=1, help="paper_runs.id")
    parser.add_argument("--train-years", type=int, default=3)
    parser.add_argument("--forward-days", type=int, default=5)
    parser.add_argument("--top-n", type=int, default=TradingConfig.TOP_N)
    parser.add_argument("--universe-size", type=int, default=500)
    parser.add_argument("--dry-run", action="store_true", help="试运行不写DB")
    parser.add_argument("--backfill", action="store_true", help="从 paper_runs.start_date 回填到指定日期")
    parser.add_argument("--no-sync", action="store_true", help="跳过数据同步")
    args = parser.parse_args()

    trade_date = pd.Timestamp(args.date) if args.date else pd.Timestamp.now().normalize()
    if trade_date.date() >= date.today():
        trade_date = pd.Timestamp(date.today())  # 防止未来日期
    logger.info(f"交易日: {trade_date.date()}")

    engine = get_engine()

    # ── 检查账户和 run ──
    with engine.connect() as conn:
        acc = conn.execute(text("SELECT id, cash FROM paper_account WHERE id = :aid"), {"aid": args.account_id}).fetchone()
        run = conn.execute(text("SELECT id, start_date, strategy_id, version_id FROM paper_runs WHERE id = :rid"), {"rid": args.run_id}).fetchone()
    if not acc or not run:
        logger.error("账户或 run 不存在，请先运行 init_paper_trading.py")
        sys.exit(1)
    logger.info(f"账户 id={acc[0]}, 现金={acc[1]:,.0f}, run={run[0]}")

    # 始终加载足够历史数据用于训练（至少 train_years + val_years）
    start_dt = trade_date - pd.DateOffset(years=args.train_years + 2)  # 多留 2 年余量
    start_str = start_dt.strftime("%Y%m%d")
    end_str = trade_date.strftime("%Y%m%d")

    # ── 1. 增量同步日线数据 ──
    try:
        from data.sync import main as sync_main
    except ImportError:
        sync_main = None
    if not args.no_sync:
        logger.info("检查数据新鲜度 ...")
    try:
        sync_engine = get_engine()
        with sync_engine.connect() as c:
            latest = c.execute(text("SELECT MAX(trade_date) FROM stock_daily")).fetchone()
        if latest and latest[0]:
            days_behind = (date.today() - latest[0]).days
            if days_behind > 3:
                logger.info(f"  数据落后 {days_behind} 天，执行增量同步 ...")
                from data.sync import sync_stock_daily as do_sync
                do_sync(sync_engine, start_date=(date.today() - timedelta(days=max(days_behind, 5))).strftime("%Y%m%d"), workers=1)
            else:
                logger.info(f"  数据已是最新 ({latest[0]})，跳过同步")
        sync_engine.dispose()
    except Exception as e:
        logger.warning(f"  数据检查跳过: {e}")

    # ── 2. 加载数据 ──
    ohlcv, index_df, regime_df, extra_data, codes = load_data(
        engine, start_str, end_str, args.universe_size)

    with engine.connect() as conn:
        trade_date_dt = conn.execute(
            text("SELECT MAX(trade_date) FROM stock_daily")
        ).fetchone()
    latest_date = pd.Timestamp(trade_date_dt[0])
    logger.info(f"最新数据日: {latest_date.date()}")

    # ── 3. 构建因子数据集 ──
    dataset = build_factor_dataset(
        ohlcv, FACTOR_NAMES,
        label_mode="binary", forward_days=args.forward_days,
        extra_data=extra_data if extra_data else None,
        industry_neutralize=False,
    )

    # ── 4. IC + 正交筛选 ──
    factor_cols = filter_factors_by_ic(dataset, FACTOR_NAMES)
    if len(factor_cols) < 3:
        factor_cols = FACTOR_NAMES[:min(12, len(FACTOR_NAMES))]
    selected = select_orthogonal_factors(dataset, factor_cols, threshold=0.7)
    logger.info(f"因子: {len(FACTOR_NAMES)} → IC{len(factor_cols)} → 正交{len(selected)}")

    # ── 5. 训练模型（Regime 模式，用 walk-forward 最后一个窗口的模型） ──
    if regime_df is not None:
        logger.info("Regime 训练 ...")
        results = walk_forward_train_by_regime(
            dataset, selected, regime_df,
            train_years=args.train_years, val_years=1,
        )
        if results:
            # Merge models from all windows: pick best ensemble per regime
            from models.trainer import RegimeAwareEnsemble
            merged_ensembles = {}
            merged_factors = []
            all_trained = set()
            for r in results:
                ens = r.get("ensemble")
                if ens:
                    for reg, model in ens.ensembles.items():
                        if reg not in merged_ensembles:
                            merged_ensembles[reg] = model
                            all_trained.add(reg)
                    if not merged_factors:
                        merged_factors = ens.factor_names
            # Fallback: use any available model for missing regimes
            if merged_ensembles:
                any_model = next(iter(merged_ensembles.values()))
                for reg in ["bull", "bear", "sideways"]:
                    if reg not in merged_ensembles:
                        merged_ensembles[reg] = any_model
            ensemble = RegimeAwareEnsemble(merged_ensembles, merged_factors or selected)
            trained_regimes = list(all_trained)
            logger.info(f"模型就绪: regimes={trained_regimes} (merged from {len(results)} windows)")
        else:
            logger.error("Regime 训练无结果")
            sys.exit(1)
    else:
        logger.error("无法检测市场状态")
        sys.exit(1)

    # ── 6. 获取今日市场状态 ──
    today_regime = "sideways"
    if regime_df is not None:
        today_rows = regime_df[regime_df["trade_date"] <= latest_date]
        if not today_rows.empty:
            today_regime = str(today_rows["regime"].iloc[-1])
    logger.info(f"今日市场状态: {today_regime}")

    # ── 7. 构建今日因子截面（用于预测） ──
    pred_date = dataset["trade_date"].max()
    today_factor = dataset[dataset["trade_date"] == pred_date].copy()
    if today_factor.empty:
        logger.error(f"无 {pred_date.date()} 的因子数据")
        sys.exit(1)

    # 注入 regime
    today_factor["regime"] = today_regime

    # ── 8. 预测 ──
    preds = ensemble.predict(today_factor)
    preds = preds.sort_values("score", ascending=False).reset_index(drop=True)
    logger.info(f"预测: {len(preds)} 只, Top-5: {preds['code'].head(5).tolist()}")

    # ── 9. PaperEngine 执行 ──
    if args.dry_run:
        logger.info("[DRY RUN] 跳过执行")
        print(f"\n=== 今日信号 ({pred_date.date()}) === 市场: {today_regime}")
        for i, row in preds.head(args.top_n).iterrows():
            print(f"  {i+1}. {row['code']}  score={row['score']:.4f}")
    else:
        engine_paper = PaperEngine(
            account_id=args.account_id,
            run_id=args.run_id,
            predictor=ensemble,
            top_n=args.top_n,
            rebalance_mode="ndrop",
        )
        result = engine_paper.run_daily(
            trade_date=pred_date,
            factor_df=today_factor,
            ohlcv_data=ohlcv,
            index_ohlcv=index_df,
            regime=today_regime,
        )
        if result:
            logger.info(
                f"执行完成: 总资产={result['total_value']:,.0f}, "
                f"买入={result['n_buy_orders']}, 卖出={result['n_sell_orders']}, "
                f"成本={result['cost']:,.0f}"
            )

    engine.dispose()
    logger.info("Done.")


if __name__ == "__main__":
    main()
