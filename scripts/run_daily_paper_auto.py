#!/usr/bin/env python
"""每日自动化任务 —— 数据同步 + 双模拟盘 + ETF监控。

纯 cron 方案，一行命令搞定所有：
  ① 增量同步 stock_daily + index_daily
  ② 涨停 Top-5 模拟盘调仓
  ③ 大小票 v4.0 模拟盘调仓
  ④ ETF 三因子信号更新

用法:
    python scripts/run_daily_paper_auto.py              # 立即执行
    python scripts/run_daily_paper_auto.py --dry-run    # 试运行
    python scripts/run_daily_paper_auto.py --strategy lu  # 只跑涨停

crontab (每30分钟, 17:00~次日8:30):
    0,30 17-23,0-8 * * * cd /Users/chenwan/Documents/quant && .venv/bin/python scripts/run_daily_paper_auto.py >> logs/cron.log 2>&1
"""
import sys, os, time, argparse, traceback
from datetime import datetime, date, timedelta
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from loguru import logger
from sqlalchemy import text

from data.db import get_engine
from config.settings import TradingConfig

# ── 数据源配置 ──
DATA_SOURCES = [
    {
        "name": "新浪(ak.stock_zh_a_daily)",
        "fetch_func": "fetch_sina",
        "priority": 1,
    },
    {
        "name": "腾讯(ak.stock_zh_a_hist_tx)",
        "fetch_func": "fetch_tx",
        "priority": 2,
    },
]

TARGET_COUNT = 4500   # 至少拉到这么多只股票才算成功
MAX_RETRIES = 5        # 最多重试次数
RETRY_INTERVAL = 1800  # 重试间隔(秒) = 30分钟


def fetch_sina(code, start, end):
    """新浪数据源"""
    import akshare as ak
    prefix = 'sz' if code.startswith(('0', '3')) else 'sh'
    try:
        df = ak.stock_zh_a_daily(symbol=f'{prefix}{code}', start_date=start,
                                  end_date=end, adjust='qfq')
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.rename(columns={'date': 'trade_date', 'vol': 'volume'})
        df['code'] = code
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        if 'volume' not in df.columns: df['volume'] = 0
        if 'amount' not in df.columns: df['amount'] = 0
        if 'turnover' not in df.columns: df['turnover'] = np.nan
        for c in ['open','high','low','close','volume','amount','turnover']:
            if c not in df.columns: df[c] = 0
        return df[['code','trade_date','open','high','low','close','volume','amount','turnover']]
    except Exception:
        return pd.DataFrame()


def fetch_tx(code, start, end):
    """腾讯数据源（备用）"""
    import akshare as ak
    prefix = 'sz' if code.startswith(('0', '3')) else 'sh'
    try:
        df = ak.stock_zh_a_hist_tx(symbol=f'{prefix}{code}', start_date=start,
                                    end_date=end, adjust='qfq')
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.rename(columns={'date': 'trade_date'})
        df['code'] = code
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df['volume'] = 0
        df['turnover'] = np.nan
        if 'amount' in df.columns:
            df['amount'] = df['amount'] * 10000
        else:
            df['amount'] = 0
        for c in ['open','high','low','close','volume','amount','turnover']:
            if c not in df.columns: df[c] = 0
        return df[['code','trade_date','open','high','low','close','volume','amount','turnover']]
    except Exception:
        return pd.DataFrame()


def sync_with_fallback(engine, trade_date_str):
    """增量同步（主源→腾讯补漏），返回同步到的股票数"""
    from data.sync import sync_stock_daily

    start_fmt = trade_date_str.replace('-', '')
    logger.info(f"增量同步 {trade_date_str} ...")

    # 1. 主力源：新浪（增量，只拉缺数据的股票）
    try:
        sync_stock_daily(engine, start_date=start_fmt, workers=8)
    except Exception as e:
        logger.warning(f"主力源失败: {e}")

    # 2. 检查覆盖率
    cnt = pd.read_sql(f"SELECT COUNT(*) FROM stock_daily WHERE trade_date='{trade_date_str}'", engine).iloc[0,0]
    logger.info(f"  主力源后: {cnt} 只")

    if cnt >= TARGET_COUNT:
        logger.success(f"同步完成: {cnt} 只")
        return cnt

    # 3. 腾讯补漏：对缺失的活跃股用腾讯API拉
    logger.info(f"  不足 {TARGET_COUNT}, 腾讯补漏...")
    missing = pd.read_sql(f"""
        SELECT code FROM stock_basic WHERE is_st=FALSE
        AND code NOT IN (SELECT DISTINCT code FROM stock_daily WHERE trade_date='{trade_date_str}')
    """, engine)['code'].tolist()

    # 只补TOP活跃股
    active = pd.read_sql("""
        SELECT code FROM (SELECT code, SUM(amount) as amt FROM stock_daily
        WHERE trade_date >= CURRENT_DATE - 30 GROUP BY code ORDER BY amt DESC LIMIT 1000) t
    """, engine)['code'].tolist()
    missing = [c for c in missing if c in set(active)]
    logger.info(f"  待补: {len(missing)} 只")

    if not missing:
        logger.warning(f"无可补股票, 当前 {cnt} 只")
        return cnt

    from data.db import upsert_df
    all_data = []
    for i, code in enumerate(missing):
        df = fetch_tx(code, start_fmt, start_fmt)
        if not df.empty: all_data.append(df)
        if (i+1) % 200 == 0:
            logger.info(f"  腾讯进度 {i+1}/{len(missing)}")

    if all_data:
        result = pd.concat(all_data, ignore_index=True)
        upsert_df(result, 'stock_daily', engine)
        cnt2 = pd.read_sql(f"SELECT COUNT(*) FROM stock_daily WHERE trade_date='{trade_date_str}'", engine).iloc[0,0]
        logger.success(f"补完后: {cnt2} 只")
        return cnt2

    return cnt


def sync_index(engine):
    """同步指数日线"""
    import akshare as ak
    from data.db import upsert_df
    idx_codes = ['000001','000300','000852','399001','399006','000688','000905']
    latest = pd.read_sql("SELECT MAX(trade_date) FROM index_daily", engine).iloc[0, 0]
    all_data = []
    for code in idx_codes:
        try:
            prefix = 'sz' if code.startswith('3') else 'sh'
            df = ak.stock_zh_index_daily_tx(symbol=f'{prefix}{code}',
                                             start_date=latest.strftime('%Y%m%d'),
                                             end_date=date.today().strftime('%Y%m%d'))
            if df is not None and not df.empty:
                df = df.rename(columns={'date': 'trade_date', 'vol': 'volume'})
                df['code'] = code
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                if 'amount' not in df.columns: df['amount'] = 0
                cols = ['code','trade_date','open','high','low','close','volume','amount']
                all_data.append(df[[c for c in cols if c in df.columns]])
        except Exception as ex:
            logger.warning(f"指数 {code} 失败: {ex}")
    if all_data:
        result = pd.concat(all_data, ignore_index=True)
        upsert_df(result, 'index_daily', engine)
        logger.info(f"指数同步: {len(result)} 条")


def get_latest_trade_date(engine):
    """获取最近的交易日"""
    row = pd.read_sql("SELECT MAX(trade_date) FROM stock_daily", engine).iloc[0, 0]
    return str(row) if row else None


def run_paper_trading(trade_date, strategy='all', dry_run=False):
    """跑模拟盘。strategy: 'lu'/'switch'/'all'"""
    import subprocess
    scripts = []
    if strategy in ('lu', 'all'):
        scripts.append(('涨停Top-5', 'scripts/run_daily_paper_lu.py'))
    if strategy in ('switch', 'all'):
        scripts.append(('大小票v4.0', 'scripts/run_daily_paper_switch.py'))

    for name, script in scripts:
        cmd = [sys.executable, script, '--date', trade_date, '--no-sync']
        if dry_run: cmd.append('--dry-run')
        logger.info(f"--- {name} ---")
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        for line in (r.stdout + r.stderr).split('\n'):
            line = line.strip()
            if any(kw in line for kw in ['交易日', '筛选结果', '执行完成', '估值更新',
                                           '清仓', '弱势', '强势', 'CSI1000',
                                           '涨停侧', '小票侧', '执行:', '估值:']):
                logger.info(f"  {line}")
        if r.returncode != 0:
            logger.error(f"{name} 执行失败: {r.stderr[-300:]}")


def sync_and_trade(engine, strategy='all', dry_run=False):
    """完整流程：同步 → 模拟盘"""
    print(f"\n{'='*60}")
    print(f"  自动模拟盘 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    # 1. 同步指数
    logger.info("[1/3] 同步指数...")
    sync_index(engine)

    # 2. 同步个股
    logger.info("[2/3] 同步个股日线...")
    trade_date = get_latest_trade_date(engine)
    if not trade_date:
        logger.error("无法确定最新交易日")
        return False

    today_str = date.today().strftime('%Y-%m-%d')
    if trade_date < today_str:
        # 今天的数据还没到，拉最新可用的
        logger.info(f"最新数据: {trade_date}, 今天 {today_str} 尚无数据")

    n_stocks = sync_with_fallback(engine, today_str)

    if n_stocks < TARGET_COUNT:
        # 数据不全，尝试拉最近可用的
        logger.warning(f"今天数据不足({n_stocks}/{TARGET_COUNT})，尝试补拉 {trade_date}")
        n2 = sync_with_fallback(engine, trade_date)
        if n2 >= TARGET_COUNT:
            n_stocks = n2
            today_str = trade_date

    if n_stocks < TARGET_COUNT:
        logger.error(f"数据同步未达标 ({n_stocks}/{TARGET_COUNT})，跳过模拟盘")
        logger.info(f"将在 {RETRY_INTERVAL//60} 分钟后重试...")
        return False

    # 3. 跑模拟盘
    logger.info(f"[3/3] 模拟盘 (日期: {today_str})...")
    run_paper_trading(today_str, strategy=strategy, dry_run=dry_run)

    # 4. ETF监控
    logger.info("[4/4] ETF监控...")
    import subprocess
    subprocess.run([sys.executable, 'scripts/run_etf_monitor.py'],
                   capture_output=True, text=True, timeout=300)

    show_status(engine)
    return True


def show_status(engine):
    """显示模拟盘状态"""
    try:
        pnl = pd.read_sql(
            "SELECT trade_date, total_value, daily_return, drawdown "
            "FROM paper_daily_pnl WHERE account_id=1 ORDER BY trade_date DESC LIMIT 5", engine)
        pos = pd.read_sql(
            "SELECT stock_code, entry_date FROM paper_positions "
            "WHERE run_id=1 AND exit_date IS NULL", engine)
        cash = pd.read_sql("SELECT cash FROM paper_account WHERE id=1", engine).iloc[0,0]

        print(f"\n  {'─'*40}")
        print(f"  模拟盘状态")
        print(f"  现金: {cash:,.0f}  |  持仓: {len(pos)} 只")
        if not pnl.empty:
            latest = pnl.iloc[0]
            print(f"  最新净值: {latest['total_value']:,.0f}  "
                  f"(日收益 {latest['daily_return']*100:+.2f}%)")
        print(f"  {'─'*40}\n")
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--strategy", type=str, default="all",
                        choices=["lu", "switch", "all"], help="策略选择")
    args = parser.parse_args()

    engine = get_engine()

    # 单次执行 + 内置重试
    success = sync_and_trade(engine, strategy=args.strategy, dry_run=args.dry_run)
    if not success and not args.dry_run:
        for attempt in range(1, MAX_RETRIES + 1):
            logger.info(f"重试 {attempt}/{MAX_RETRIES}，等待 {RETRY_INTERVAL//60} 分钟...")
            time.sleep(RETRY_INTERVAL)
            success = sync_and_trade(engine, strategy=args.strategy, dry_run=False)
            if success:
                break

    engine.dispose()


if __name__ == "__main__":
    main()
