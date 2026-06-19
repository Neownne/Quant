#!/usr/bin/env python
"""每日自动化信号系统 —— 根据当前时间自动选午盘/日终模式。

自动模式:
    python scripts/run_daily_signals.py --send-email
    → < 11:30: 等到上午收盘 → 午盘扫描（腾讯实时行情 + 邮件）
    → 11:30–15:00: 午盘扫描（腾讯实时行情 + 邮件）
    → > 15:00: 日终扫描（数据同步 + 日线信号 + 邮件）

手动覆盖:
    python scripts/run_daily_signals.py --now            # 立即执行，跳过等待
    python scripts/run_daily_signals.py --intraday       # 强制午盘模式
    python scripts/run_daily_signals.py --date 2026-06-13 # 指定日期回测
    python scripts/run_daily_signals.py --dry-run         # 试运行，不写文件/不发邮件
    python scripts/run_daily_signals.py --no-sync         # 跳过数据同步
    python scripts/run_daily_signals.py --exclude-gem-star # 排除创业/科创板

6阶段流水线（日终模式）:
  Phase 0: 时间门控 + 模式判断
  Phase 1: 同步前数据质量检查
  Phase 2: 增量数据同步（指数 + 日线 + 市值 + 腾讯补漏）
  Phase 3: 同步后数据质量验证
  Phase 4: 一次性数据加载 + 公共因子预计算
  Phase 5: 三池信号扫描（涨停池 / 妖股池 / 牛股池）
  Phase 6: 输出 & 推送（报告 + JSON + 同花顺导入 + 邮件）
"""

from __future__ import annotations

import argparse, os, sys, json, csv, smtplib, time, urllib.request
from datetime import date, datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy import text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from data.db import get_engine
from data.loader import load_daily_data, load_mcap_data
from data.sync import (
    check_data_quality,
    sync_stock_daily,
    sync_index_daily,
    sync_daily_extra,
    _get_trading_calendar,
    _latest_trading_day,
)
from config.settings import TradingConfig

OUT_DIR = "data/arsenal"
os.makedirs(OUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# 常量
# ═══════════════════════════════════════════════════════════════

TARGET_STOCK_COUNT = 4000   # 同步后至少要有这么多只股票
MAX_SYNC_RETRIES = 3        # 同步失败重试次数
SYNC_RETRY_INTERVAL = 30    # 重试间隔(秒)

# 涨停阈值
_DEFAULT_MULT = 1.9899


def _is_at_limit_up(close: float, prev_close: float, code: str, tolerance: float = 1.0) -> bool:
    """A股涨停价判断。涨停价 = round(prev_close × 1.9899, 4)
    tolerance<1.0 时放宽到近涨停区。"""
    if pd.isna(close) or pd.isna(prev_close) or prev_close <= 0:
        return False
    limit_price = round(prev_close * 1.9899, 4)
    return close >= limit_price * tolerance


# ═══════════════════════════════════════════════════════════════
# 邮箱配置
# ═══════════════════════════════════════════════════════════════

EMAIL_CONFIG = {
    "smtp_host": os.getenv("SMTP_HOST", "smtp.qq.com"),
    "smtp_port": int(os.getenv("SMTP_PORT", "465")),
    "user": os.getenv("SMTP_USER", ""),
    "password": os.getenv("SMTP_PASS", ""),
    "from_addr": os.getenv("EMAIL_FROM", ""),
    "to": os.getenv("EMAIL_TO", ""),
}


# ═══════════════════════════════════════════════════════════════
# Phase 0: 时间门控
# ═══════════════════════════════════════════════════════════════

def _is_trading_day(d: date | None = None) -> bool:
    """判断是否为 A 股交易日。"""
    d = d or date.today()
    calendar = _get_trading_calendar()
    if calendar:
        return str(d) in calendar
    # fallback: 周末非交易日
    return d.weekday() < 5


def _is_after_market_close() -> bool:
    """判断当前时间是否已过 15:00 收盘。"""
    now = datetime.now()
    return now.hour >= 15


def _is_after_morning_close() -> bool:
    """判断当前时间是否已过 11:30 上午收盘。"""
    now = datetime.now()
    return now.hour > 11 or (now.hour == 11 and now.minute >= 30)


def _wait_until_morning_close():
    """等到当日 11:30 上午收盘。"""
    now = datetime.now()
    close_time = now.replace(hour=11, minute=30, second=0, microsecond=0)
    if now >= close_time:
        return
    wait_seconds = (close_time - now).total_seconds()
    logger.info(f"距上午收盘还有 {wait_seconds/60:.0f} 分钟，等待中...")


def _expected_latest_data_date() -> date:
    """预期 DB 中应该有的最新交易日。

    15:00 前 → 昨天（今天日线还没出）
    15:00 后 → 今天
    """
    today = date.today()
    calendar = _get_trading_calendar()
    if calendar and str(today) in calendar and _is_after_market_close():
        return today
    # 找今天之前的最近交易日
    d = today - timedelta(days=1)
    for _ in range(10):
        if calendar:
            if str(d) in calendar:
                return d
        elif d.weekday() < 5:
            return d
        d -= timedelta(days=1)
    return today


def _is_data_stale(engine) -> bool:
    """检查 DB 是否缺预期交易日数据（昨晚没跑日终 → 数据滞后）。"""
    with engine.connect() as conn:
        latest_db = conn.execute(
            text("SELECT MAX(trade_date) FROM stock_daily")
        ).scalar()
    if latest_db is None:
        return True
    latest_db = pd.Timestamp(latest_db).date()
    expected = _expected_latest_data_date()
    return latest_db < expected


def _is_eod_report_missing() -> bool:
    """检查预期交易日是否还没出过日终报告。"""
    expected = _expected_latest_data_date()
    date_tag = expected.strftime("%Y%m%d")
    report_path = f"{OUT_DIR}/daily_report_{date_tag}.txt"
    return not os.path.exists(report_path)
    time.sleep(wait_seconds + 5)


def _wait_until_close():
    """等到当日 15:00 收盘后。"""
    now = datetime.now()
    close_time = now.replace(hour=15, minute=0, second=0, microsecond=0)
    if now >= close_time:
        return
    wait_seconds = (close_time - now).total_seconds()
    logger.info(f"距收盘还有 {wait_seconds/60:.0f} 分钟，等待中...")
    time.sleep(wait_seconds + 10)  # +10s 确保数据已落地


# ═══════════════════════════════════════════════════════════════
# Phase 1 & 3: 数据质量检查
# ═══════════════════════════════════════════════════════════════

def _print_quality_report(qr: dict, phase: str = "同步前"):
    """打印数据质量报告摘要。"""
    tables = ["stock_daily", "stock_daily_extra", "index_daily", "stock_basic"]
    logger.info(f"[{phase}] 数据质量:")
    for tbl in tables:
        info = qr.get(tbl, {})
        status = info.get("status", "?")
        icon = {"ok": "✅", "stale": "⚠️", "low_coverage": "⚠️", "missing": "❌"}.get(status, "❓")
        logger.info(f"  {icon} {tbl}: {info.get('n_records',0):,}条, "
                    f"{info.get('n_codes',0)}只, 最新{info.get('latest_date','N/A')} "
                    f"(stale={info.get('stale_days',0)}d)")


def _check_today_coverage(engine, trade_date_str: str) -> int:
    """检查 T 日股票覆盖率，返回股票数。"""
    with engine.connect() as conn:
        cnt = conn.execute(
            text("SELECT COUNT(*) FROM stock_daily WHERE trade_date = :d"),
            {"d": trade_date_str},
        ).scalar()
    return int(cnt)


# ═══════════════════════════════════════════════════════════════
# Phase 2: 数据同步
# ═══════════════════════════════════════════════════════════════

def _fetch_tx(code: str, start_fmt: str) -> pd.DataFrame | None:
    """腾讯数据源补漏（单只股票）。"""
    try:
        import akshare as ak
        prefix = 'sz' if str(code).startswith(('0', '3')) else 'sh'
        df = ak.stock_zh_a_hist_tx(symbol=f'{prefix}{code}', start_date=start_fmt,
                                    end_date=start_fmt, adjust='qfq')
        if df is None or df.empty:
            return None
        df = df.rename(columns={'date': 'trade_date'})
        df['code'] = str(code)
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df['volume'] = 0
        df['turnover'] = np.nan
        if 'amount' in df.columns:
            df['amount'] = df['amount'] * 10000
        else:
            df['amount'] = 0
        for c in ['open', 'high', 'low', 'close', 'volume', 'amount', 'turnover']:
            if c not in df.columns:
                df[c] = 0
        return df[['code', 'trade_date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'turnover']]
    except Exception:
        return None


def _tx_fallback(engine, trade_date_str: str):
    """腾讯数据源补漏：对缺失的活跃股用腾讯 API 拉取。"""
    from data.db import upsert_df

    # 找出有缺的活跃股
    with engine.connect() as conn:
        missing = pd.read_sql(text("""
            SELECT code FROM stock_basic WHERE is_st = FALSE
            AND code NOT IN (SELECT DISTINCT code FROM stock_daily WHERE trade_date = :d)
        """), conn, params={"d": trade_date_str})['code'].tolist()

        active = pd.read_sql(text("""
            SELECT code FROM (
                SELECT code, SUM(amount) AS amt FROM stock_daily
                WHERE trade_date >= CURRENT_DATE - 30 GROUP BY code ORDER BY amt DESC LIMIT 1000
            ) t
        """), conn)['code'].tolist()

    missing = [c for c in missing if c in set(active)]
    if not missing:
        logger.info("  无需腾讯补漏")
        return

    start_fmt = trade_date_str.replace('-', '')
    logger.info(f"  腾讯补漏: {len(missing)} 只...")

    all_data = []
    for i, code in enumerate(missing):
        df = _fetch_tx(code, start_fmt)
        if df is not None:
            all_data.append(df)
        if (i + 1) % 200 == 0:
            logger.info(f"  腾讯进度 {i+1}/{len(missing)}")

    if all_data:
        result = pd.concat(all_data, ignore_index=True)
        upsert_df(result, 'stock_daily', engine)
        cnt = _check_today_coverage(engine, trade_date_str)
        logger.info(f"  腾讯补完后: {cnt} 只")


def sync_all(engine, trade_date_str: str) -> bool:
    """同步指数 + 日线 + 市值 + 腾讯补漏。返回是否达标。"""
    start_fmt = trade_date_str.replace('-', '')

    # 1. 同步指数
    logger.info("[sync] 同步指数日线...")
    try:
        sync_index_daily(engine, start_date=start_fmt)
    except Exception as e:
        logger.warning(f"指数同步失败: {e}")

    # 2. 同步个股日线
    logger.info("[sync] 同步个股日线...")
    try:
        sync_stock_daily(engine, start_date=start_fmt, workers=8)
    except Exception as e:
        logger.warning(f"个股日线同步失败: {e}")

    cnt = _check_today_coverage(engine, trade_date_str)
    logger.info(f"  主力源后: {cnt} 只")

    # 3. 腾讯补漏
    if cnt < TARGET_STOCK_COUNT:
        _tx_fallback(engine, trade_date_str)
        cnt = _check_today_coverage(engine, trade_date_str)

    # 4. 同步市值
    logger.info("[sync] 同步市值数据...")
    try:
        sync_daily_extra(engine, start_date=start_fmt, workers=8)
    except Exception as e:
        logger.warning(f"市值同步失败: {e}")

    return cnt >= TARGET_STOCK_COUNT


# ═══════════════════════════════════════════════════════════════
# Phase 4: 数据加载 + 公共因子预计算
# ═══════════════════════════════════════════════════════════════

def load_and_precompute(engine, target_date: pd.Timestamp, exclude_gem_star: bool = False):
    """加载全市场数据并预计算公共因子。

    返回: (daily, extra, name_map, ind_map, td, prev_td, csi_snapshot)
    """
    t0 = time.time()

    # ── 股票池 ──
    min_list = target_date - timedelta(days=252)
    with engine.connect() as conn:
        codes_df = pd.read_sql(
            text("SELECT code, name, industry FROM stock_basic "
                 "WHERE is_st=FALSE AND list_date <= :ld AND code !~ '^(300|301|688|[48])'"),
            conn, params={"ld": min_list.strftime("%Y-%m-%d")})
    codes_df['code'] = codes_df['code'].astype(str).str.zfill(6)
    name_map = dict(zip(codes_df['code'], codes_df['name']))
    ind_map = dict(zip(codes_df['code'], codes_df['industry'].fillna('其他')))

    codes = codes_df['code'].tolist()
    logger.info(f"  股票池: {len(codes)} 只")

    # ── 加载日线 + 市值 ──
    pre_start = (target_date - timedelta(days=120)).strftime("%Y-%m-%d")
    end_str = target_date.strftime("%Y-%m-%d")

    daily = load_daily_data(engine, codes, pre_start, end_str,
                            cols=['open', 'high', 'low', 'close', 'volume', 'turnover'])
    daily['code'] = daily['code'].astype(str).str.zfill(6)
    daily['trade_date'] = pd.to_datetime(daily['trade_date'])
    daily = daily.sort_values(['code', 'trade_date'])

    extra = load_mcap_data(engine, codes, pre_start, end_str, use_proxy=True)
    if not extra.empty:
        extra['code'] = extra['code'].astype(str).str.zfill(6)
        extra['trade_date'] = pd.to_datetime(extra['trade_date'])

    logger.info(f"  日线: {len(daily)} 行 | 市值: {len(extra)} 行")

    # ── 前收盘 & 收益率 ──
    daily['ret'] = daily.groupby('code')['close'].pct_change()
    daily['prev_close'] = daily.groupby('code')['close'].shift(1)

    # ── 确定 T 日和前一日 ──
    all_dates = sorted(daily['trade_date'].unique())
    if target_date not in all_dates:
        # 回退到最近交易日
        target_date = all_dates[-1]
    td_idx = all_dates.index(target_date)
    prev_td = all_dates[td_idx - 1] if td_idx > 0 else target_date

    # ── 公共因子预计算 ──
    # 板别感知涨停标记
    daily['is_lu'] = daily.apply(
        lambda r: 1 if _is_at_limit_up(r['close'], r['prev_close'], str(r['code']), tolerance=0.98) else 0,
        axis=1,
    )

    # 均线
    daily['ma5'] = daily.groupby('code')['close'].transform(lambda x: x.rolling(5, min_periods=3).mean())
    daily['ma10'] = daily.groupby('code')['close'].transform(lambda x: x.rolling(10, min_periods=5).mean())
    daily['ma20'] = daily.groupby('code')['close'].transform(lambda x: x.rolling(20, min_periods=5).mean())
    daily['ma40'] = daily.groupby('code')['close'].transform(lambda x: x.rolling(40, min_periods=20).mean())

    # 量能
    daily['vol_ma20'] = daily.groupby('code')['volume'].transform(lambda x: x.rolling(20, min_periods=5).mean())
    daily['vol_ma40'] = daily.groupby('code')['volume'].transform(lambda x: x.rolling(40, min_periods=20).mean())
    daily['vol_std20'] = daily.groupby('code')['volume'].transform(lambda x: x.rolling(20, min_periods=5).std())

    # 换手率均线
    if 'turnover' in daily.columns:
        daily['to_ma20'] = daily.groupby('code')['turnover'].transform(lambda x: x.rolling(20, min_periods=5).mean())

    # 涨停统计（板别感知）
    daily['lu_20d'] = daily.groupby('code')['is_lu'].transform(lambda x: x.rolling(20, min_periods=1).sum())
    daily['lu_60d'] = daily.groupby('code')['is_lu'].transform(lambda x: x.rolling(60, min_periods=30).sum())

    # 连板数
    def _calc_streak(s):
        cnt, res = 0, []
        for v in s:
            cnt = cnt + 1 if v else 0
            res.append(cnt)
        return pd.Series(res, index=s.index)
    daily['lu_streak'] = daily.groupby('code')['is_lu'].transform(_calc_streak)

    # 波动率
    daily['ret_vol_20'] = daily.groupby('code')['ret'].transform(lambda x: x.rolling(20, min_periods=10).std())

    # 振幅/封板质量
    daily['hl_range'] = daily['high'] - daily['low']
    daily['seal_quality'] = np.where(
        daily['is_lu'] == 1, daily['close'] / daily['high'].replace(0, np.nan), np.nan)
    daily['amplitude'] = np.where(
        daily['is_lu'] == 1,
        daily['hl_range'] / daily['prev_close'].replace(0, np.nan), np.nan)

    # ── CSI1000 快照 ──
    csi_snapshot = {}
    csi = pd.read_sql(
        text("SELECT trade_date, close FROM index_daily WHERE code='000852' "
             "AND trade_date BETWEEN :s AND :e ORDER BY trade_date"),
        engine, params={"s": pre_start, "e": end_str})
    if not csi.empty:
        csi['trade_date'] = pd.to_datetime(csi['trade_date'])
        csi['ma60'] = csi['close'].rolling(60, min_periods=30).mean()
        csi_td = csi[csi['trade_date'] == target_date]
        if not csi_td.empty:
            csi_snapshot = {
                'value': float(csi_td['close'].iloc[0]),
                'ma60': float(csi_td['ma60'].iloc[0]) if pd.notna(csi_td['ma60'].iloc[0]) else 0,
                'trend': '上升' if float(csi_td['close'].iloc[0]) > float(csi_td['ma60'].iloc[0]) else '下降',
            }

    elapsed = time.time() - t0
    logger.info(f"  数据加载+预计算完成 ({elapsed:.0f}s), T={target_date.date()}, "
                f"前日={prev_td.date()}")

    return daily, extra, name_map, ind_map, target_date, prev_td, csi_snapshot


# ═══════════════════════════════════════════════════════════════
# Phase 4b: 腾讯实时行情 + 日内数据加载（--intraday 模式）
# ═══════════════════════════════════════════════════════════════

def _fetch_tencent_quotes(codes: list[str], batch_size: int = 300) -> pd.DataFrame:
    """通过腾讯 API 拉取实时行情。

    返回: DataFrame[code, name, price, prev_close, open, high, low, volume]
    时间戳存储在 df.attrs['timestamp'] 中。
    """
    # 按交易所分组
    sh = [c for c in codes if c.startswith(('6', '9'))]
    sz = [c for c in codes if c.startswith(('0', '3', '4', '8'))]

    all_rows = []
    timestamp = ""

    for market, code_list in [('sh', sh), ('sz', sz)]:
        for i in range(0, len(code_list), batch_size):
            batch = code_list[i:i + batch_size]
            ids = ','.join(f'{market}{c}' for c in batch)
            url = f'http://qt.gtimg.cn/q={ids}'

            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                resp = urllib.request.urlopen(req, timeout=15)
                data = resp.read().decode('gbk')

                for line in data.strip().split('\n'):
                    if not line.strip() or '="' not in line:
                        continue
                    try:
                        content = line.split('="')[1].rstrip('";\n')
                        fields = content.split('~')
                        if len(fields) < 35:
                            continue
                        code = fields[2]
                        ts = fields[30] if len(fields) > 30 else ""
                        if ts and not timestamp:
                            timestamp = ts
                        all_rows.append({
                            'code': code,
                            'name': fields[1],
                            'price': float(fields[3]) if fields[3] and fields[3] != '0.00' else 0,
                            'prev_close': float(fields[4]) if fields[4] else 0,
                            'open': float(fields[5]) if fields[5] else 0,
                            'volume': int(fields[6]) if fields[6] else 0,
                            'high': float(fields[33]) if fields[33] else 0,
                            'low': float(fields[34]) if fields[34] else 0,
                        })
                    except (ValueError, IndexError):
                        continue

            except Exception as e:
                logger.warning(f"腾讯行情 {market} batch {i} 请求失败: {e}")

            time.sleep(0.3)

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df['code'] = df['code'].astype(str).str.zfill(6)
        df.attrs['timestamp'] = timestamp
    return df


def load_intraday_data(engine, target_date: pd.Timestamp, exclude_gem_star: bool = False):
    """日内模式：DB 历史数据（≤T-1）+ 腾讯实时行情（T日当前）。

    返回: (daily, extra, name_map, ind_map, target_date, prev_td, csi_snapshot, rt_ts)
    """
    t0 = time.time()

    # ── 股票池 ──
    min_list = target_date - timedelta(days=252)
    with engine.connect() as conn:
        codes_df = pd.read_sql(
            text("SELECT code, name, industry FROM stock_basic "
                 "WHERE is_st=FALSE AND list_date <= :ld AND code !~ '^(300|301|688|[48])'"),
            conn, params={"ld": min_list.strftime("%Y-%m-%d")})
    codes_df['code'] = codes_df['code'].astype(str).str.zfill(6)
    name_map = dict(zip(codes_df['code'], codes_df['name']))
    ind_map = dict(zip(codes_df['code'], codes_df['industry'].fillna('其他')))
    codes = codes_df['code'].tolist()
    logger.info(f"  股票池: {len(codes)} 只")

    # ── 加载历史日线（到 T-1）──
    pre_start = (target_date - timedelta(days=120)).strftime("%Y-%m-%d")
    # T-1 日期
    yesterday = (target_date - timedelta(days=1)).strftime("%Y-%m-%d")

    daily = load_daily_data(engine, codes, pre_start, yesterday,
                            cols=['open', 'high', 'low', 'close', 'volume', 'turnover'])
    daily['code'] = daily['code'].astype(str).str.zfill(6)
    daily['trade_date'] = pd.to_datetime(daily['trade_date'])
    daily = daily.sort_values(['code', 'trade_date'])

    # 市值
    extra = load_mcap_data(engine, codes, pre_start, yesterday, use_proxy=True)
    if not extra.empty:
        extra['code'] = extra['code'].astype(str).str.zfill(6)
        extra['trade_date'] = pd.to_datetime(extra['trade_date'])

    logger.info(f"  历史日线: {len(daily)} 行 | 市值: {len(extra)} 行")

    # ── 市值 T-1 前填到 T 日（实时行情无市值，用最近一日近似）──
    if not extra.empty:
        latest_extra = extra.sort_values(['code', 'trade_date']).groupby('code').tail(1).copy()
        latest_extra['trade_date'] = target_date
        extra = pd.concat([extra, latest_extra], ignore_index=True)

    # ── 拉取腾讯实时行情 ──
    logger.info("  拉取腾讯实时行情...")
    rt = _fetch_tencent_quotes(codes)
    rt_ts = rt.attrs.get('timestamp', '')
    if rt_ts:
        ts_str = f"{rt_ts[:4]}-{rt_ts[4:6]}-{rt_ts[6:8]} {rt_ts[8:10]}:{rt_ts[10:12]}:{rt_ts[12:14]}"
        logger.info(f"  实时行情时间戳: {ts_str}")
    logger.info(f"  实时行情: {len(rt)} 只")

    if rt.empty:
        logger.error("  腾讯实时行情无数据！")
        return None

    # ── 把实时行情构造为 T 日伪日线 ──
    rt_rows = []
    for _, r in rt.iterrows():
        if r['price'] <= 0 or r['prev_close'] <= 0:
            continue
        rt_rows.append({
            'code': str(r['code']).zfill(6),
            'trade_date': target_date,
            'open': r['open'] if r['open'] > 0 else r['price'],
            'high': r['high'] if r['high'] > 0 else r['price'],
            'low': r['low'] if r['low'] > 0 else r['price'],
            'close': r['price'],
            'volume': r['volume'],
            'turnover': np.nan,
        })
    rt_df = pd.DataFrame(rt_rows)

    # ── 拼到历史日线末尾 ──
    daily = pd.concat([daily, rt_df], ignore_index=True)
    daily = daily.sort_values(['code', 'trade_date'])

    # ── 确定 T 日前一个交易日 ──
    all_dates = sorted(daily['trade_date'].unique())
    td_idx = all_dates.index(target_date)
    prev_td = all_dates[td_idx - 1] if td_idx > 0 else target_date
    logger.info(f"  T={target_date.date()} (实时), 前日={prev_td.date()}")

    # ── 公共因子预计算（和 load_and_precompute 一致）──
    daily['ret'] = daily.groupby('code')['close'].pct_change()
    daily['prev_close'] = daily.groupby('code')['close'].shift(1)

    daily['is_lu'] = daily.apply(
        lambda r: 1 if _is_at_limit_up(r['close'], r['prev_close'], str(r['code']), tolerance=0.98) else 0,
        axis=1,
    )

    daily['ma5'] = daily.groupby('code')['close'].transform(lambda x: x.rolling(5, min_periods=3).mean())
    daily['ma10'] = daily.groupby('code')['close'].transform(lambda x: x.rolling(10, min_periods=5).mean())
    daily['ma20'] = daily.groupby('code')['close'].transform(lambda x: x.rolling(20, min_periods=5).mean())
    daily['ma40'] = daily.groupby('code')['close'].transform(lambda x: x.rolling(40, min_periods=20).mean())

    daily['vol_ma20'] = daily.groupby('code')['volume'].transform(lambda x: x.rolling(20, min_periods=5).mean())
    daily['vol_ma40'] = daily.groupby('code')['volume'].transform(lambda x: x.rolling(40, min_periods=20).mean())
    daily['vol_std20'] = daily.groupby('code')['volume'].transform(lambda x: x.rolling(20, min_periods=5).std())

    if 'turnover' in daily.columns:
        daily['to_ma20'] = daily.groupby('code')['turnover'].transform(lambda x: x.rolling(20, min_periods=5).mean())

    daily['lu_20d'] = daily.groupby('code')['is_lu'].transform(lambda x: x.rolling(20, min_periods=1).sum())
    daily['lu_60d'] = daily.groupby('code')['is_lu'].transform(lambda x: x.rolling(60, min_periods=30).sum())

    def _calc_streak(s):
        cnt, res = 0, []
        for v in s:
            cnt = cnt + 1 if v else 0
            res.append(cnt)
        return pd.Series(res, index=s.index)
    daily['lu_streak'] = daily.groupby('code')['is_lu'].transform(_calc_streak)

    daily['ret_vol_20'] = daily.groupby('code')['ret'].transform(lambda x: x.rolling(20, min_periods=10).std())

    daily['hl_range'] = daily['high'] - daily['low']
    daily['seal_quality'] = np.where(
        daily['is_lu'] == 1, daily['close'] / daily['high'].replace(0, np.nan), np.nan)
    daily['amplitude'] = np.where(
        daily['is_lu'] == 1,
        daily['hl_range'] / daily['prev_close'].replace(0, np.nan), np.nan)

    # CSI1000（用最近一个历史日）
    csi_snapshot = {}
    csi = pd.read_sql(
        text("SELECT trade_date, close FROM index_daily WHERE code='000852' "
             "AND trade_date BETWEEN :s AND :e ORDER BY trade_date"),
        engine, params={"s": pre_start, "e": yesterday})
    if not csi.empty:
        csi['trade_date'] = pd.to_datetime(csi['trade_date'])
        csi['ma60'] = csi['close'].rolling(60, min_periods=30).mean()
        latest_csi = csi.iloc[-1]
        csi_snapshot = {
            'value': float(latest_csi['close']),
            'ma60': float(latest_csi['ma60']) if pd.notna(latest_csi['ma60']) else 0,
            'trend': '上升' if float(latest_csi['close']) > float(latest_csi['ma60']) else '下降',
        }

    elapsed = time.time() - t0
    logger.info(f"  日内数据加载完成 ({elapsed:.0f}s)")

    return daily, extra, name_map, ind_map, target_date, prev_td, csi_snapshot, rt_ts


# ═══════════════════════════════════════════════════════════════
# Phase 5.1: 涨停池
# ═══════════════════════════════════════════════════════════════

def screen_limit_up(daily, extra, target_date, prev_td, name_map, ind_map):
    """涨停池：4条件筛选（市值/股价/均线/涨停次数）。

    使用预计算的因子列，不再跨股票查表。
    """
    td_mask = daily['trade_date'] == target_date
    today = daily[td_mask].set_index('code')
    prev = daily[daily['trade_date'] == prev_td].set_index('code')

    # 市值
    if not extra.empty:
        ex_td = extra[extra['trade_date'] == target_date].set_index('code')
        today['mcap'] = ex_td.get('market_cap', np.nan) if not ex_td.empty else np.nan
    else:
        today['mcap'] = np.nan

    # 当日涨幅（板别感知涨停判断已在 is_lu 中）
    today['prev_close'] = prev['close']
    today['ret_calc'] = today['close'] / today['prev_close'] - 1

    # 4条件筛选（不要求当日涨停，只要近20日涨停2-4次）
    mask = (
        (today['mcap'].between(30, 500)) &
        (today['close'].between(5, 100)) &
        (today['ma5'] > today['ma10']) &
        (today['lu_20d'] >= 2) & (today['lu_20d'] <= 4) &
        (today['close'] > 0)
    )
    lu = today[mask].copy()
    if lu.empty:
        return pd.DataFrame()

    lu['name'] = lu.index.map(name_map)
    lu['industry'] = lu.index.map(ind_map)
    lu['ret_pct'] = (lu['ret_calc'] * 100).round(1)
    lu['lu_20d_count'] = lu['lu_20d'].astype(int)

    result = lu.nlargest(100, 'ret_calc')[
        ['name', 'industry', 'close', 'mcap', 'ret_pct', 'lu_20d_count']
    ].copy()
    result.columns = ['名称', '行业', '收盘价', '市值(亿)', '今日涨幅', '近20日涨停']
    result['代码'] = result.index
    return result


# ═══════════════════════════════════════════════════════════════
# Phase 5.2: 妖股池
# ═══════════════════════════════════════════════════════════════

def screen_yaogu(daily, target_date, prev_td, name_map, ind_map, min_score: int = 3):
    """妖股池：6规则实时评分 ≥ min_score。

    使用预计算的 is_lu/lu_streak/vol_ma20/vol_std20/seal_quality/amplitude/hl_range。
    """
    td_mask = daily['trade_date'] == target_date
    today = daily[td_mask].set_index('code')
    prev = daily[daily['trade_date'] == prev_td].set_index('code')
    today['prev_close'] = prev['close']

    # 只看涨停股
    lu_today = today[today['is_lu'] == 1]
    if lu_today.empty:
        return pd.DataFrame()

    # 预计算 low_vol_streak（连续缩量天数）
    low_vol_streak_map = {}
    for code in lu_today.index:
        # 取该股近20日数据
        code_mask = (daily['code'] == code) & (daily['trade_date'] <= target_date)
        code_data = daily[code_mask].tail(20)
        streak = 0
        if len(code_data) >= 5:
            vol_mean = code_data['vol_ma20'].mean() if 'vol_ma20' in code_data.columns else code_data['volume'].mean()
            for _, row in code_data.iterrows():
                if row['volume'] < vol_mean * 0.7:
                    streak += 1
                else:
                    streak = 0
        low_vol_streak_map[code] = streak

    signals = []
    for code in lu_today.index:
        r = lu_today.loc[code]

        yiziban = 1 if abs(r['hl_range']) < r['close'] * 0.001 else 0
        amp_val = float(r['amplitude']) if pd.notna(r.get('amplitude')) else 1.0
        vol_avg = float(r['vol_ma20']) if pd.notna(r.get('vol_ma20')) and r['vol_ma20'] > 0 else 1.0
        vol_std = float(r['vol_std20']) if pd.notna(r.get('vol_std20')) else 0

        vol_intensity = r['volume'] / vol_avg if vol_avg > 0 else 1
        vol_climax = r['volume'] / (vol_avg + 2 * vol_std) if (vol_avg + 2 * vol_std) > 0 else 1
        streak_val = int(r.get('lu_streak', 0))

        # 6规则评分
        score = 0
        if yiziban: score += 3
        if pd.notna(amp_val) and amp_val < 0.08: score += 2
        if pd.notna(vol_intensity) and vol_intensity < 1.5: score += 1
        if pd.notna(vol_climax) and vol_climax < 0.8: score += 1
        if streak_val >= 2: score += 1
        if low_vol_streak_map.get(code, 0) >= 1: score += 1

        if score >= min_score:
            ret_today = (r['close'] / r['prev_close'] - 1) if pd.notna(r.get('prev_close')) and r['prev_close'] > 0 else 0
            signals.append({
                'code': code, 'name': name_map.get(code, '?'),
                'industry': ind_map.get(code, '?'),
                'close': r['close'], 'yaogu_score': score,
                'yiziban': yiziban, 'streak': streak_val,
                'ret_today': round(ret_today * 100, 1),
                'seal': round(float(r.get('seal_quality', 0)), 3) if pd.notna(r.get('seal_quality')) else 0,
                'vol_ratio': round(float(vol_intensity), 2) if pd.notna(vol_intensity) else 0,
            })

    return pd.DataFrame(signals).sort_values('yaogu_score', ascending=False) if signals else pd.DataFrame()


# ═══════════════════════════════════════════════════════════════
# Phase 5.3: 牛股池（委托 screen_bull.py）
# ═══════════════════════════════════════════════════════════════

def screen_bull(daily, extra, target_date, name_map, ind_map, exclude_gem_star=False):
    """牛股池：委托 screen_bull.screen()，传入预加载数据。"""
    from scripts.screen_bull import screen as _bull_screen
    return _bull_screen(
        date_str=target_date.strftime("%Y-%m-%d"),
        exclude_gem_star=exclude_gem_star,
        daily_df=daily,
        extra_df=extra,
        name_map=name_map,
        ind_map=ind_map,
    )


# ═══════════════════════════════════════════════════════════════
# Phase 5.4: 市场快照
# ═══════════════════════════════════════════════════════════════

def build_market_snapshot(daily, target_date, csi_snapshot):
    """构建市场概况。"""
    today_data = daily[daily['trade_date'] == target_date]
    if today_data.empty:
        return {'date': str(target_date.date()), 'total_stocks': 0, 'avg_ret': 0,
                'up_ratio': 0, 'lu_count': 0}
    rets = today_data['ret'].dropna()
    return {
        'date': str(target_date.date()),
        'total_stocks': len(today_data),
        'avg_ret': float(rets.mean()) if len(rets) > 0 else 0,
        'up_ratio': float((rets > 0).mean()) if len(rets) > 0 else 0,
        'lu_count': int(today_data['is_lu'].sum()),
        'csi1000': csi_snapshot.get('value', 0),
        'csi1000_ma60': csi_snapshot.get('ma60', 0),
        'csi_trend': csi_snapshot.get('trend', 'N/A'),
    }


# ═══════════════════════════════════════════════════════════════
# Phase 5.5: 三池交集
# ═══════════════════════════════════════════════════════════════

def find_intersections(limit_up_df, yaogu_df, bull_df):
    """找三池交集。

    返回每个交集的信息 dict，含:
      - codes: 交集代码集合
      - lu_details: 涨停池侧详情 (code → {name, ret, ...})
      - yg_details: 妖股池侧详情 (code → {name, score, ret, ...})
      - bull_details: 牛股池侧详情 (code → {name, score, ...})
    """
    intersections = {}

    def _codes(df):
        if df is None or df.empty:
            return set()
        if '代码' in df.columns:
            return set(df['代码'].tolist())
        if 'code' in df.columns:
            return set(df['code'].tolist())
        return set(df.index.tolist())

    def _lu_detail(df, codes):
        """从涨停池 DataFrame 提取指定代码的详情。"""
        detail = {}
        if df is None or df.empty or not codes:
            return detail
        for _, r in df.iterrows():
            c = r.get('代码', r.get('code', ''))
            if c in codes:
                detail[c] = {
                    'name': r.get('名称', r.get('name', '')),
                    'ret': f"{r.get('今日涨幅', 0):+.1f}%",
                    'industry': r.get('行业', r.get('industry', '')),
                }
        return detail

    def _yg_detail(df, codes):
        """从妖股池 DataFrame 提取指定代码的详情。"""
        detail = {}
        if df is None or df.empty or not codes:
            return detail
        for _, r in df.iterrows():
            c = r.get('code', r.get('代码', ''))
            if c in codes:
                detail[c] = {
                    'name': r.get('name', ''),
                    'score': int(r.get('yaogu_score', 0)),
                    'ret': f"{r.get('ret_today', 0):+.1f}%",
                    'streak': int(r.get('streak', 0)),
                    'yiziban': bool(r.get('yiziban', 0)),
                }
        return detail

    def _bull_detail(df, codes):
        """从牛股池 DataFrame 提取指定代码的详情。"""
        detail = {}
        if df is None or df.empty or not codes:
            return detail
        for _, r in df.iterrows():
            c = r.get('代码', '')
            if c in codes:
                detail[c] = {
                    'name': r.get('名称', ''),
                    'score': float(r.get('牛股评分', 0)),
                    'vs_ma40': f"{float(r.get('vsMA40(%)', 0)):+.0f}%",
                    'vol_ratio': float(r.get('量比', 0)),
                }
        return detail

    pairs = [
        ('涨停∩妖股', limit_up_df, yaogu_df, [_lu_detail, _yg_detail]),
        ('涨停∩牛股', limit_up_df, bull_df, [_lu_detail, _bull_detail]),
        ('妖股∩牛股', yaogu_df, bull_df, [_yg_detail, _bull_detail]),
    ]

    for label, df_a, df_b, detail_fns in pairs:
        codes_a, codes_b = _codes(df_a), _codes(df_b)
        common = codes_a & codes_b
        intersections[label] = {
            'codes': common,
            'detail_a': detail_fns[0](df_a, common),
            'detail_b': detail_fns[1](df_b, common),
        }

    # 三池交集
    c1, c2, c3 = _codes(limit_up_df), _codes(yaogu_df), _codes(bull_df)
    triple = c1 & c2 & c3
    intersections['涨停∩妖股∩牛股'] = {
        'codes': triple,
        'lu_detail': _lu_detail(limit_up_df, triple),
        'yg_detail': _yg_detail(yaogu_df, triple),
        'bull_detail': _bull_detail(bull_df, triple),
    }

    return intersections


# ═══════════════════════════════════════════════════════════════
# Phase 6: 输出 & 推送
# ═══════════════════════════════════════════════════════════════

def build_report(snapshot, limit_up, yaogu, bull, intersections=None, tag="", rt_ts=""):
    """生成文本报告。tag='intraday' 时为午盘报告。"""
    is_intraday = tag == 'intraday'
    title = "午盘信号速报" if is_intraday else "每日信号报告"
    lines = []
    lines.append("=" * 70)
    lines.append(f"  策略武器库 · {title}")
    lines.append(f"  日期: {snapshot.get('date', 'N/A')}")
    if rt_ts:
        ts_str = f"{rt_ts[:4]}-{rt_ts[4:6]}-{rt_ts[6:8]} {rt_ts[8:10]}:{rt_ts[10:12]}:{rt_ts[12:14]}"
        lines.append(f"  行情时间: {ts_str} (盘中实时)")
    lines.append("=" * 70)
    lines.append("")

    # 市场概况
    lines.append("【市场概况】")
    lines.append(f"  全市场: {snapshot.get('total_stocks', 0)}只 | "
                 f"均涨幅 {snapshot.get('avg_ret', 0):+.2%} | "
                 f"上涨比 {snapshot.get('up_ratio', 0):.0%}")
    lines.append(f"  CSI1000: {snapshot.get('csi1000', 0):.0f} | "
                 f"趋势: {snapshot.get('csi_trend', 'N/A')} | "
                 f"涨停: {snapshot.get('lu_count', 0)}只")
    lines.append("")

    # 涨停池（全量）
    lines.append(f"【涨停池】{len(limit_up)} 只 — 4条件(市值30-500亿/股价5-100/MA5>MA10/20日涨停2-4次)")
    if not limit_up.empty:
        for _, r in limit_up.iterrows():
            lu_tag = f" [涨停]" if r.get('今日涨幅', 0) >= 9.5 else ""
            lines.append(f"  {r['代码']} {r['名称']:<8s} "
                         f"涨幅{r.get('今日涨幅', 0):+.1f}% | "
                         f"20日涨停{int(r.get('近20日涨停', 0))}次{lu_tag} | {r.get('行业', '')}")
    else:
        lines.append("  (今日无符合条件的涨停股)")
    lines.append("")

    # 妖股池（全量）
    lines.append(f"【妖股池】{len(yaogu)} 只 — 6规则评分 ≥ 3")
    if not yaogu.empty:
        for _, r in yaogu.iterrows():
            tags = []
            if r.get('yiziban'): tags.append('一字板')
            if r.get('streak', 0) >= 2: tags.append(f"连板{int(r['streak'])}")
            lines.append(f"  {r['code']} {r['name']:<8s} "
                         f"评分{int(r['yaogu_score'])} | {r['ret_today']:+.1f}% | "
                         f"{' '.join(tags)}")
    else:
        lines.append("  (今日无妖股信号)")
    lines.append("")

    # 牛股池
    lines.append(f"【牛股池】{len(bull)} 只 — 缩量筑底，评分 Top-15")
    if not bull.empty:
        for _, r in bull.head(15).iterrows():
            lines.append(f"  {r['代码']} {r['名称']:<8s} "
                         f"评分{float(r['牛股评分']):.0f} | "
                         f"vsMA40={float(r['vsMA40(%)']):+.0f}% | "
                         f"量比{float(r['量比']):.2f}")
    else:
        lines.append("  (今日无符合条件的牛股)")
    lines.append("")

    # 交集
    if intersections:
        lines.append("【三池交集】")
        # 涨停∩妖股 — 重点展示详情
        lu_yg = intersections.get('涨停∩妖股', {})
        lu_yg_codes = lu_yg.get('codes', set())
        if lu_yg_codes:
            yg_d = lu_yg.get('detail_b', {})
            lu_d = lu_yg.get('detail_a', {})
            lines.append(f"  ★ 涨停∩妖股: {len(lu_yg_codes)}只")
            for c in sorted(lu_yg_codes):
                yg = yg_d.get(c, {})
                lu = lu_d.get(c, {})
                yiziban_tag = ' [一字板]' if yg.get('yiziban') else ''
                lines.append(f"    {c} {yg.get('name','?'):<8s} "
                           f"妖股评分{yg.get('score','?'):} | "
                           f"涨停{lu.get('ret','?')} | "
                           f"连板{yg.get('streak',0)}{yiziban_tag}")
            lines.append("")

        # 其他两两交集
        for key in ['涨停∩牛股', '妖股∩牛股']:
            inter = intersections.get(key, {})
            inter_codes = inter.get('codes', set())
            if inter_codes:
                lines.append(f"  {key}: {len(inter_codes)}只 → {', '.join(sorted(inter_codes))}")
        lines.append("")

        # 三池交集
        triple = intersections.get('涨停∩妖股∩牛股', {})
        triple_codes = triple.get('codes', set())
        if triple_codes:
            yg_d = triple.get('yg_detail', {})
            lines.append(f"  ★★★ 涨停∩妖股∩牛股: {len(triple_codes)}只")
            for c in sorted(triple_codes):
                yg = yg_d.get(c, {})
                lines.append(f"    {c} {yg.get('name','?')} 妖股评分{yg.get('score','?')}")
            lines.append("")
        lines.append("")

    # 规则
    lines.append("【各池规则】")
    lines.append("  涨停池: 4条件(市值30-500亿+股价5-100+MA5>MA10+20日涨停2-4次) → 按今日涨幅排序")
    lines.append("  妖股池: 一字板(+3) + 低振幅<8%(+2) + 缩量板(+1) + 非量能极值(+1)")
    lines.append("          + 连板≥2(+1) + 缩量整理≥1天(+1) → 评分≥3入选")
    lines.append("  牛股池: 市值5-50亿 + 收盘<MA40 + 缩量 + 20日波动<3% + 60日无涨停")
    lines.append("          → 综合评分(市值+偏离度+量比+波动+涨停次)")
    lines.append("")

    lines.append("=" * 70)
    lines.append("  报告结束。以上信号仅供参考，不构成投资建议。")
    lines.append("=" * 70)

    return "\n".join(lines)


def send_email(subject, body):
    """发送邮件。"""
    if not EMAIL_CONFIG['user'] or not EMAIL_CONFIG['to']:
        logger.warning("邮箱未配置，跳过发送")
        return False

    msg = MIMEMultipart()
    msg['From'] = EMAIL_CONFIG.get('from_addr', EMAIL_CONFIG['user'])
    recipients = [r.strip() for r in EMAIL_CONFIG['to'].split(',') if r.strip()]
    msg['To'] = ', '.join(recipients)
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain', 'utf-8'))

    try:
        if EMAIL_CONFIG['smtp_port'] == 465:
            server = smtplib.SMTP_SSL(EMAIL_CONFIG['smtp_host'], EMAIL_CONFIG['smtp_port'], timeout=15)
        else:
            server = smtplib.SMTP(EMAIL_CONFIG['smtp_host'], EMAIL_CONFIG['smtp_port'], timeout=15)
            server.starttls()
        server.login(EMAIL_CONFIG['user'], EMAIL_CONFIG['password'])
        server.sendmail(EMAIL_CONFIG['user'], recipients, msg.as_string())
        server.quit()
        logger.success(f"邮件已发送到 {EMAIL_CONFIG['to']}")
        return True
    except Exception as e:
        logger.error(f"邮件发送失败: {e}")
        return False


def save_outputs(snapshot, limit_up, yaogu, bull, tag=""):
    """保存报告、JSON、同花顺导入文件。tag='intraday' 时文件名加 _intraday 后缀。"""
    date_tag_val = snapshot['date'].replace('-', '')
    suffix = f"_{tag}" if tag else ""
    label = "午盘" if tag == 'intraday' else "日报"

    # 文本报告
    report = build_report(snapshot, limit_up, yaogu, bull, tag=tag)
    report_path = f"{OUT_DIR}/daily_report_{date_tag_val}{suffix}.txt"
    with open(report_path, 'w') as f:
        f.write(report)

    # 各池 JSON
    for name, df in [('limit_up', limit_up), ('yaogu', yaogu), ('bull', bull)]:
        if not df.empty:
            json_path = f"{OUT_DIR}/{name}_{date_tag_val}{suffix}.json"
            df_export = df.copy()
            if '代码' in df_export.columns:
                df_export['code'] = df_export['代码']
            df_export.to_json(json_path, orient='records', force_ascii=False, indent=2)

    # 同花顺导入
    all_codes = set()
    for df in [limit_up, yaogu, bull]:
        if df is not None and not df.empty:
            codes = df['代码'].tolist() if '代码' in df.columns else df.index.tolist()
            all_codes.update(str(c).zfill(6) for c in codes)
    ths_path = f"{OUT_DIR}/ths_import_{date_tag_val}{suffix}.txt"
    with open(ths_path, 'w') as f:
        for c in sorted(all_codes):
            f.write(f"{c}\n")

    print(f"\n  {label}报告: {report_path}")
    print(f"  同花顺导入: {ths_path} ({len(all_codes)}只)")
    return report


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="每日自动化信号系统")
    p.add_argument("--date", type=str, default=None, help="指定日期 YYYY-MM-DD（默认最新交易日）")
    p.add_argument("--now", action="store_true", help="立即执行，不等待收盘")
    p.add_argument("--dry-run", action="store_true", help="试运行：不写文件、不发邮件")
    p.add_argument("--no-sync", action="store_true", help="跳过数据同步（假设数据已最新）")
    p.add_argument("--send-email", action="store_true", help="强制发送邮件")
    p.add_argument("--exclude-gem-star", action="store_true", help="排除创业/科创板(300/301/688)")
    p.add_argument("--yaogu-min-score", type=int, default=3, help="妖股最低评分（默认3）")
    p.add_argument("--intraday", action="store_true", help="日内模式：用腾讯实时行情（上午盘后可用）")
    args = p.parse_args()

    t_start = time.time()
    print("=" * 60)
    print(f"  每日自动化信号系统 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    engine = get_engine()

    try:
        # ── Phase 0: 时间门控 + 模式判断 ──
        logger.info("═══ Phase 0: 时间门控 ═══")

        is_intraday = False  # 是否日内模式

        if args.date:
            # 指定日期 → 日终模式
            trade_date = pd.Timestamp(args.date)
            logger.info(f"  指定日期: {trade_date.date()} (日终模式)")
        elif args.intraday:
            # 强制日内模式
            if not _is_trading_day():
                logger.warning("今天非交易日，退出")
                return
            is_intraday = True
            if not args.now and not _is_after_morning_close():
                _wait_until_morning_close()
            trade_date = pd.Timestamp(_latest_trading_day())
            logger.info(f"  交易日: {trade_date.date()} | 强制日内模式")
        else:
            # ── 自动模式：根据当前时间 + 数据新鲜度判断 ──
            if not _is_trading_day():
                logger.warning("今天非交易日，退出")
                return

            now = datetime.now()
            data_stale = _is_data_stale(engine)
            report_missing = _is_eod_report_missing()  # 最新交易日还没出过报告

            if data_stale or report_missing:
                # 昨晚没跑 / 跑了但挂了 → 优先日终模式
                if data_stale:
                    logger.info("  检测到数据滞后（昨晚未跑日终），优先日终同步")
                else:
                    logger.info("  数据已同步但日终报告缺失，跳过同步直接出报告")
                    args.no_sync = True  # 数据已有，跳过同步加速
                is_intraday = False
            elif not args.now:
                if now.hour < 11 or (now.hour == 11 and now.minute < 30):
                    logger.info("  当前上午盘交易中，等待 11:30 上午收盘...")
                    _wait_until_morning_close()
                    is_intraday = True
                elif now.hour < 15:
                    is_intraday = True
                else:
                    is_intraday = False
            else:
                is_intraday = (now.hour < 15)

            trade_date = pd.Timestamp(_latest_trading_day())
            mode_label = "午盘扫描（腾讯实时行情）" if is_intraday else "日终扫描（同步+日线）"
            stale_tag = " (补昨日数据)" if data_stale else (" (补报告)" if report_missing else "")
            logger.info(f"  交易日: {trade_date.date()} | 自动 → {mode_label}{stale_tag}")

        trade_date_str = trade_date.strftime("%Y-%m-%d")

        rt_ts = None  # 日内时间戳（仅 intraday 模式）

        if is_intraday:
            # ── 日内模式：DB 历史 + 腾讯实时行情 ──
            logger.info("═══ 日内模式（腾讯实时行情 + DB 历史上下文）═══")
            result = load_intraday_data(engine, trade_date, exclude_gem_star=args.exclude_gem_star)
            if result is None:
                logger.error("日内数据加载失败，退出")
                return
            daily, extra, name_map, ind_map, target_date, prev_td, csi_snapshot, rt_ts = result

        elif not args.no_sync:
            # ── Phase 1: 同步前质量检查 ──
            logger.info("═══ Phase 1: 同步前质量检查 ═══")
            pre_qr = check_data_quality(engine)
            _print_quality_report(pre_qr, "同步前")

            # ── Phase 2: 数据同步 ──
            logger.info("═══ Phase 2: 数据同步 ═══")
            synced = sync_all(engine, trade_date_str)
            for attempt in range(1, MAX_SYNC_RETRIES + 1):
                if synced:
                    break
                logger.warning(f"数据同步未达标，{SYNC_RETRY_INTERVAL}s 后重试 ({attempt}/{MAX_SYNC_RETRIES})...")
                time.sleep(SYNC_RETRY_INTERVAL)
                synced = sync_all(engine, trade_date_str)

            if not synced:
                logger.error(f"数据同步失败（{MAX_SYNC_RETRIES}次重试后仍不达标），退出")
                return

            # ── Phase 3: 同步后质量验证 ──
            logger.info("═══ Phase 3: 同步后质量验证 ═══")
            post_qr = check_data_quality(engine)
            _print_quality_report(post_qr, "同步后")
            cnt = _check_today_coverage(engine, trade_date_str)
            logger.info(f"  T日覆盖: {cnt} 只 (目标 {TARGET_STOCK_COUNT})")

            # ── Phase 4: 数据加载 + 预计算 ──
            logger.info("═══ Phase 4: 数据加载 & 公共因子预计算 ═══")
            daily, extra, name_map, ind_map, target_date, prev_td, csi_snapshot = \
                load_and_precompute(engine, trade_date, exclude_gem_star=args.exclude_gem_star)
        else:
            # ── Phase 4: 数据加载 + 预计算（跳过同步）──
            logger.info("═══ Phase 4: 数据加载 & 公共因子预计算 ═══")
            daily, extra, name_map, ind_map, target_date, prev_td, csi_snapshot = \
                load_and_precompute(engine, trade_date, exclude_gem_star=args.exclude_gem_star)

        # ── Phase 5: 三池扫描 ──
        logger.info("═══ Phase 5: 三池信号扫描 ═══")
        snapshot = build_market_snapshot(daily, target_date, csi_snapshot)

        print("  扫描涨停池...")
        t0 = time.time()
        limit_up = screen_limit_up(daily, extra, target_date, prev_td, name_map, ind_map)
        logger.info(f"  涨停池: {len(limit_up)} 只 ({time.time()-t0:.0f}s)")

        print("  扫描妖股池...")
        t0 = time.time()
        yaogu = screen_yaogu(daily, target_date, prev_td, name_map, ind_map,
                             min_score=args.yaogu_min_score)
        logger.info(f"  妖股池: {len(yaogu)} 只 ({time.time()-t0:.0f}s)")

        print("  扫描牛股池...")
        t0 = time.time()
        bull = screen_bull(daily, extra, target_date, name_map, ind_map,
                          exclude_gem_star=args.exclude_gem_star)
        logger.info(f"  牛股池: {len(bull)} 只 ({time.time()-t0:.0f}s)")

        # 排除创业/科创（对涨停池和妖股池）
        if args.exclude_gem_star:
            if not limit_up.empty:
                limit_up = limit_up[~limit_up['代码'].astype(str).str.startswith(('300', '301', '688'))]
            if not yaogu.empty:
                code_col = 'code' if 'code' in yaogu.columns else '代码'
                yaogu = yaogu[~yaogu[code_col].astype(str).str.startswith(('300', '301', '688'))]
            logger.info(f"  排除创业/科创后: 涨停{len(limit_up)} 妖股{len(yaogu)}")

        # 交集
        intersections = find_intersections(limit_up, yaogu, bull)

        # ── Phase 6: 输出 & 推送 ──
        logger.info("═══ Phase 6: 输出 & 推送 ═══")

        phase6_tag = "intraday" if is_intraday else ""

        if args.dry_run:
            report = build_report(snapshot, limit_up, yaogu, bull, intersections,
                                  tag=phase6_tag, rt_ts=rt_ts or "")
            print("\n" + report)
            logger.info("  --dry-run: 已跳过文件写入和邮件发送")
        else:
            report = save_outputs(snapshot, limit_up, yaogu, bull, tag=phase6_tag)
            print("\n" + report)

            # 邮件
            should_email = args.send_email or bool(EMAIL_CONFIG['user'])
            if should_email:
                prefix = "午盘速报" if is_intraday else "日报"
                subject = (f"量化信号{prefix} {snapshot['date']} | "
                          f"涨停{len(limit_up)} 妖股{len(yaogu)} 牛股{len(bull)}")
                full_report = build_report(snapshot, limit_up, yaogu, bull, intersections,
                                          tag=phase6_tag, rt_ts=rt_ts or "")
                send_email(subject, full_report)

        elapsed = time.time() - t_start
        logger.success(f"全部完成 ({elapsed:.0f}s)")

    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
