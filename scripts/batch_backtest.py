"""
批量回测：所有股票 × 策略参数网格 × 牛熊市周期对比。

用法：
    python scripts/batch_backtest.py                          # 全量运行（8 并发）
    python scripts/batch_backtest.py --workers 16             # 指定并发数
    python scripts/batch_backtest.py --limit 50               # 只测前 50 只
    python scripts/batch_backtest.py --resume                 # 从断点继续

输出：
    output/batch_results.csv     — 逐笔回测明细
    output/batch_summary.md      — 排名汇总报告
"""
import argparse
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from typing import Any

import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from data.db import init_db, get_engine

OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "output")
RESULTS_CSV = os.path.join(OUTPUT_DIR, "batch_results.csv")

# ---- 策略参数网格 ----
STRATEGY_GRID: dict[str, list[dict[str, Any]]] = {
    "双均线交叉": [
        {"fast": 5, "slow": 20},
        {"fast": 5, "slow": 60},
        {"fast": 10, "slow": 60},
        {"fast": 20, "slow": 120},
    ],
    "MACD金叉死叉": [
        {"fast": 12, "slow": 26, "signal": 9},
        {"fast": 8, "slow": 17, "signal": 6},
        {"fast": 26, "slow": 52, "signal": 12},
    ],
    "RSI超买超卖": [
        {"period": 14, "oversold": 30, "overbought": 70},
        {"period": 7, "oversold": 20, "overbought": 80},
        {"period": 21, "oversold": 25, "overbought": 75},
    ],
}

# ---- 牛熊市周期定义 ----
MARKET_PERIODS: dict[str, tuple[str, str]] = {
    "全周期":       ("20200101", "20260524"),
    "🐂 结构牛2020-21": ("20200101", "20210218"),
    "🐻 慢熊2021-24":  ("20210218", "20240205"),
    "📈 反弹2024-26":  ("20240205", "20260524"),
}

# ---- 资金 / 费用 ----
FIXED_CASH = 1_000_000
FIXED_COMMISSION = 0.00009
FIXED_STAMP_DUTY = 0.0005


def _load_ohlcv(code: str, start: str, end: str):
    """加载单只股票 OHLCV，带 worker 进程级缓存。"""
    from sqlalchemy import text
    from data.db import get_engine

    key = (code, start, end)
    cache = _load_ohlcv._cache
    if key in cache:
        return cache[key]

    engine = get_engine()
    sql = """
        SELECT trade_date, open, high, low, close, volume, amount
        FROM stock_daily
        WHERE code = :code AND trade_date BETWEEN :start AND :end
        ORDER BY trade_date
    """
    with engine.connect() as conn:
        df = pd.read_sql_query(
            text(sql), conn, params={"code": code, "start": start, "end": end}
        )
    engine.dispose()
    if not df.empty:
        df["trade_date"] = pd.to_datetime(df["trade_date"])
    cache[key] = df
    return df


_load_ohlcv._cache = {}


def _run_single_combo(args: tuple) -> dict[str, Any] | None:
    """
    执行单个 股票×策略×参数×周期 组合的回测。
    在 worker 进程中运行，使用进程级数据缓存。
    """
    code, name, strategy_name, params, period_name, p_start, p_end, index_df_data = args

    try:
        df = _load_ohlcv(code, p_start, p_end)
    except Exception:
        return None

    if df.empty or len(df) < 100:
        return None

    # 按周期过滤（处理 sub-period）
    mask = (df["trade_date"] >= pd.Timestamp(p_start)) & (
        df["trade_date"] <= pd.Timestamp(p_end)
    )
    period_df = df[mask]
    if len(period_df) < 100:
        return None

    # 重建 index_df（从 pickled data 恢复）
    index_df = None
    if index_df_data is not None:
        import io
        index_df = pd.read_parquet(io.BytesIO(index_df_data))

    # 动态加载策略
    from strategies import get_all_strategies
    all_strats = get_all_strategies()
    strategy_class = all_strats.get(strategy_name)
    if strategy_class is None:
        return None

    # [架构重构] app.utils.backtest_runner 已移除，此处需要重构
    from app.utils.backtest_runner import run_backtest  # noqa: F401 (removed in v2.0)

    try:
        result = run_backtest(
            strategy_class=strategy_class,
            df=period_df,
            strategy_params=params,
            initial_cash=FIXED_CASH,
            commission=FIXED_COMMISSION,
            stamp_duty=FIXED_STAMP_DUTY,
            index_df=index_df,
            batch_mode=True,
        )
    except Exception:
        return None

    m = result["metrics"]
    return {
        "code": code,
        "name": name,
        "strategy": strategy_name,
        "params": str(params),
        "period": period_name,
        "total_return_pct": round(m.get("total_return", 0) * 100, 2),
        "annual_return_pct": round(m.get("annual_return", 0) * 100, 2),
        "max_drawdown_pct": round(m.get("max_drawdown", 0), 2),
        "sharpe_ratio": round(m.get("sharpe_ratio", 0), 2),
        "win_rate_pct": round(m.get("win_rate", 0) * 100, 1),
        "total_trades": m.get("total_trades", 0),
        "final_value": round(m.get("final_value", 0), 0),
    }


def get_stock_universe(engine, limit: int | None = None) -> list[tuple[str, str]]:
    """获取日线数据充足的股票列表。"""
    sql = """
        SELECT code, name FROM stock_basic
        WHERE name NOT LIKE '%%ST%%'
          AND name NOT LIKE '%%退市%%'
        ORDER BY code
    """
    df = pd.read_sql(sql, engine)
    if limit:
        df = df.head(limit)
    return list(zip(df["code"], df["name"]))


def load_existing_results() -> set[tuple]:
    """加载已有结果，返回已完成的 (code, strategy, params, period) 集合。"""
    if not os.path.exists(RESULTS_CSV):
        return set()
    df = pd.read_csv(RESULTS_CSV)
    existing = set()
    for _, row in df.iterrows():
        existing.add((row["code"], row["strategy"], row["params"], row["period"]))
    return existing


def _build_combo_tasks(
    stocks: list[tuple[str, str]],
    index_df_bytes: bytes | None,
    existing: set[tuple],
) -> list[tuple]:
    """构建所有待执行的 combo 任务列表。"""
    tasks = []
    for code, name in stocks:
        for strategy_name, param_sets in STRATEGY_GRID.items():
            for params in param_sets:
                for period_name, (p_start, p_end) in MARKET_PERIODS.items():
                    key = (code, strategy_name, str(params), period_name)
                    if key in existing:
                        continue
                    tasks.append((
                        code, name, strategy_name, params,
                        period_name, p_start, p_end, index_df_bytes,
                    ))
    return tasks


def run_batch(workers: int = 8, limit: int | None = None, resume: bool = False):
    engine = get_engine()
    stocks = get_stock_universe(engine, limit=limit)
    engine.dispose()

    # 加载上证指数数据，序列化为 bytes（跨进程传递）
    # [架构重构] app.utils.backtest_runner 已移除，此处需要重构
    from app.utils.backtest_runner import load_index_data  # noqa: F401 (removed in v2.0)
    print("加载上证指数数据 ...")
    index_df = load_index_data("20150101", "20300101")
    print(f"  上证指数: {len(index_df)} 条")

    import io
    buf = io.BytesIO()
    index_df.to_parquet(buf, index=False)
    index_df_bytes = buf.getvalue()

    existing = load_existing_results() if resume else set()
    if existing:
        print(f"跳过已完成: {len(existing)} 组")

    # 构建所有任务
    tasks = _build_combo_tasks(stocks, index_df_bytes, existing)
    total_tasks = len(tasks)
    combos_per_stock = sum(len(v) for v in STRATEGY_GRID.values()) * len(MARKET_PERIODS)
    print(f"回测范围: {len(stocks)} 只股票 × {combos_per_stock} 组 = {total_tasks + len(existing)} 总组合")
    print(f"待执行: {total_tasks} 组（跳过 {len(existing)} 组已完成）")
    print(f"并发数: {workers}")
    print()

    if not tasks:
        print("所有任务已完成，无需运行。")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_rows: list[dict] = []
    if resume and os.path.exists(RESULTS_CSV):
        existing_df = pd.read_csv(RESULTS_CSV)
        all_rows = existing_df.to_dict("records")

    start_time = time.time()
    completed = 0
    last_save = 0
    save_interval = max(100, total_tasks // 200)  # 自动调整保存频率

    with ProcessPoolExecutor(max_workers=workers) as executor:
        # 分批提交，避免一次性创建过多 Future 对象
        batch_size = 5000
        for batch_start in range(0, total_tasks, batch_size):
            batch = tasks[batch_start:batch_start + batch_size]
            futures = {executor.submit(_run_single_combo, t): t for t in batch}

            for future in as_completed(futures):
                completed += 1
                try:
                    row = future.result(timeout=120)
                except Exception:
                    row = None

                if row is not None:
                    key = (row["code"], row["strategy"], row["params"], row["period"])
                    if key not in existing:
                        existing.add(key)
                        all_rows.append(row)

                # 定期保存
                if completed - last_save >= save_interval:
                    pd.DataFrame(all_rows).to_csv(RESULTS_CSV, index=False)
                    last_save = completed

            # 每批结束后保存
            pd.DataFrame(all_rows).to_csv(RESULTS_CSV, index=False)

            elapsed = time.time() - start_time
            rate = completed / elapsed if elapsed > 0 else 0
            remaining = total_tasks - completed
            eta_min = remaining / rate / 60 if rate > 0 else 0
            print(
                f"\r进度: {completed}/{total_tasks} ({100*completed/total_tasks:.1f}%)  "
                f"速率: {rate:.1f}组/秒  预计剩余: {eta_min:.0f}分钟  ",
                end="", flush=True,
            )

    print()
    # 最终保存
    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(RESULTS_CSV, index=False)
    print(f"\n结果已保存: {RESULTS_CSV} ({len(results_df)} 条)")

    # 生成汇总报告
    generate_summary(results_df)


def generate_summary(df: pd.DataFrame):
    """生成排名汇总 Markdown 报告。"""
    if df.empty:
        return

    summary_path = os.path.join(OUTPUT_DIR, "batch_summary.md")

    top_return = df.nlargest(20, "total_return_pct")
    top_winrate = df[df["total_trades"] >= 10].nlargest(20, "win_rate_pct")

    import numpy as np
    df_clean = df.copy()
    df_clean["total_return_pct"] = df_clean["total_return_pct"].replace([np.inf, -np.inf], np.nan)

    strategy_stats = df_clean.groupby("strategy").agg(
        平均收益率=("total_return_pct", "mean"),
        最高收益率=("total_return_pct", "max"),
        平均胜率=("win_rate_pct", "mean"),
        平均回撤=("max_drawdown_pct", "mean"),
        样本数=("total_return_pct", "count"),
    ).round(2)

    period_stats = df_clean.groupby("period").agg(
        平均收益率=("total_return_pct", "mean"),
        平均胜率=("win_rate_pct", "mean"),
        样本数=("total_return_pct", "count"),
    ).round(2)

    lines = [
        "# 批量回测汇总报告",
        "",
        f"回测参数：本金 {FIXED_CASH/1e4:.0f}万 | 佣金万{FIXED_COMMISSION*1e4:.1f} | 印花税万{FIXED_STAMP_DUTY*1e4:.1f}",
        "",
        "---",
        "",
        "## 按策略汇总",
        "",
        "| 策略 | 平均收益率% | 最高收益率% | 平均胜率% | 平均回撤% | 样本数 |",
        "|------|-----------|-----------|----------|----------|-------|",
    ]
    for s_name, row in strategy_stats.iterrows():
        lines.append(
            f"| {s_name} | {row['平均收益率']:.1f} | {row['最高收益率']:.1f} | "
            f"{row['平均胜率']:.1f} | {row['平均回撤']:.1f} | {int(row['样本数'])} |"
        )

    lines += [
        "",
        "## 按市场周期汇总",
        "",
        "| 周期 | 平均收益率% | 平均胜率% | 样本数 |",
        "|------|-----------|----------|-------|",
    ]
    for p_name, row in period_stats.iterrows():
        lines.append(
            f"| {p_name} | {row['平均收益率']:.1f} | "
            f"{row['平均胜率']:.1f} | {int(row['样本数'])} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 收益率 Top 20",
        "",
        "| 排名 | 代码 | 名称 | 策略 | 参数 | 周期 | 收益率% | 年化% | 回撤% | 夏普 | 胜率% |",
        "|------|------|------|------|------|------|--------|-------|-------|------|-------|",
    ]
    for rank, (_, row) in enumerate(top_return.iterrows(), 1):
        lines.append(
            f"| {rank} | {row['code']} | {row['name']} | {row['strategy']} | "
            f"{row['params']} | {row['period']} | {row['total_return_pct']:.1f} | "
            f"{row['annual_return_pct']:.1f} | {row['max_drawdown_pct']:.1f} | "
            f"{row['sharpe_ratio']:.1f} | {row['win_rate_pct']:.1f} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 胜率 Top 20（交易次数 ≥ 10）",
        "",
        "| 排名 | 代码 | 名称 | 策略 | 参数 | 周期 | 胜率% | 收益率% | 回撤% | 交易数 |",
        "|------|------|------|------|------|------|-------|--------|-------|--------|",
    ]
    for rank, (_, row) in enumerate(top_winrate.iterrows(), 1):
        lines.append(
            f"| {rank} | {row['code']} | {row['name']} | {row['strategy']} | "
            f"{row['params']} | {row['period']} | {row['win_rate_pct']:.1f} | "
            f"{row['total_return_pct']:.1f} | {row['max_drawdown_pct']:.1f} | "
            f"{int(row['total_trades'])} |"
        )

    with open(summary_path, "w") as f:
        f.write("\n".join(lines))

    print(f"汇总报告: {summary_path}")

    print("\n===== 关键发现 =====")
    best = df.loc[df["total_return_pct"].idxmax()]
    mask = df["total_trades"] >= 10
    best_wr = df[mask].loc[df[mask]["win_rate_pct"].idxmax()]
    print(f"最高收益: {best['code']} {best['name']} {best['strategy']} {best['params']} {best['period']} → {best['total_return_pct']:.1f}%")
    print(f"最高胜率: {best_wr['code']} {best_wr['name']} {best_wr['strategy']} {best_wr['params']} {best_wr['period']} → {best_wr['win_rate_pct']:.1f}% ({int(best_wr['total_trades'])}笔交易)")


def main():
    parser = argparse.ArgumentParser(description="批量策略回测")
    parser.add_argument("--workers", type=int, default=8, help="并发进程数")
    parser.add_argument("--limit", type=int, default=None, help="限制股票数量")
    parser.add_argument("--resume", action="store_true", help="从断点继续")
    args = parser.parse_args()

    init_db()
    run_batch(workers=args.workers, limit=args.limit, resume=args.resume)


if __name__ == "__main__":
    main()
