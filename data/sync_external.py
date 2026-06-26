#!/usr/bin/env python
"""外部市场数据同步：北向资金、港股指数、美股指数、人民币汇率、中美利差。

首次运行全量同步，后续增量 upsert。

用法:
    python data/sync_external.py                  # 全量同步所有
    python data/sync_external.py --days 30        # 仅最近 30 天
"""

from __future__ import annotations

import argparse, os, sys, time
import pandas as pd
import numpy as np
from datetime import date, timedelta
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.db import get_engine, upsert_df, DDL_NORTH_FLOW, DDL_BOND_YIELD, DDL_FX_RATE
from sqlalchemy import text


# ═══════════════════════════════════════════════════════════════
# 北向资金
# ═══════════════════════════════════════════════════════════════

def sync_north_flow(engine=None, days: int | None = None):
    """同步北向资金历史净流入数据。"""
    import akshare as ak
    engine = engine or get_engine()
    logger.info("=" * 50)
    logger.info("同步北向资金 ...")

    try:
        df = ak.stock_hsgt_hist_em(symbol="北向资金")
    except Exception as e:
        logger.error(f"  北向资金获取失败: {e}")
        return 0

    df = df.rename(columns={
        "日期": "trade_date",
        "当日成交净买额": "net_flow",
        "买入成交额": "buy_amount",
        "卖出成交额": "sell_amount",
        "历史累计净买额": "accum_net",
        "当日资金流入": "quota_used",
        "当日余额": "quota_balance",
        "持股市值": "holdings_value",
        "领涨股-涨跌幅": "lead_stock_pct",
        "沪深300": "hs300",
        "沪深300-涨跌幅": "hs300_pct",
        "领涨股-代码": "lead_stock",
        "领涨股": "lead_stock_name",
    })
    df["trade_date"] = pd.to_datetime(df["trade_date"])

    # 保留需要的列
    cols = [c for c in DDL_NORTH_FLOW.split("trade_date") if c.strip()] if False else [
        "trade_date", "net_flow", "buy_amount", "sell_amount", "accum_net",
        "quota_balance", "holdings_value", "lead_stock", "lead_stock_name",
        "lead_stock_pct", "hs300", "hs300_pct",
    ]
    available = [c for c in cols if c in df.columns]
    df = df[available].copy()

    if days:
        cutoff = pd.Timestamp(date.today()) - timedelta(days=days)
        df = df[df["trade_date"] >= cutoff]

    n = upsert_df(df, "market_north_flow", engine)
    logger.success(f"  北向资金: {n} 条")
    return n


# ═══════════════════════════════════════════════════════════════
# 港股指数 → index_daily
# ═══════════════════════════════════════════════════════════════

HK_INDEX_MAP = {
    "HSI": "恒生指数",
    "HSCEI": "国企指数",
    "HSTECH": "恒生科技",
}


def sync_hk_index(engine=None, days: int | None = None):
    """同步港股指数日线到 index_daily。"""
    import akshare as ak
    engine = engine or get_engine()
    logger.info("=" * 50)
    logger.info("同步港股指数 ...")

    total = 0
    for code, name in HK_INDEX_MAP.items():
        try:
            df = ak.stock_hk_index_daily_sina(symbol=code)
        except Exception as e:
            logger.warning(f"  {name}({code}) 失败: {e}")
            continue

        df = df.rename(columns={
            "date": "trade_date", "open": "open", "high": "high",
            "low": "low", "close": "close", "volume": "volume",
            "amount": "amount",
        })
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df["code"] = f"HK_{code}"

        if days:
            cutoff = pd.Timestamp(date.today()) - timedelta(days=days)
            df = df[df["trade_date"] >= cutoff]

        keep = ["code", "trade_date", "open", "high", "low", "close", "volume", "amount"]
        df = df[[c for c in keep if c in df.columns]]

        n = upsert_df(df, "index_daily", engine)
        total += n
        logger.info(f"  {name}: {n} 条")

    logger.success(f"  港股指数: {total} 条")
    return total


# ═══════════════════════════════════════════════════════════════
# 美股指数 → index_daily
# ═══════════════════════════════════════════════════════════════

US_INDEX_MAP = {
    ".INX": "标普500",
    ".IXIC": "纳斯达克",
    ".DJI": "道琼斯",
}


def sync_us_index(engine=None, days: int | None = None):
    """同步美股指数日线到 index_daily。"""
    import akshare as ak
    engine = engine or get_engine()
    logger.info("=" * 50)
    logger.info("同步美股指数 ...")

    total = 0
    for code, name in US_INDEX_MAP.items():
        try:
            df = ak.index_us_stock_sina(symbol=code)
        except Exception as e:
            logger.warning(f"  {name}({code}) 失败: {e}")
            continue

        df = df.rename(columns={
            "date": "trade_date", "open": "open", "high": "high",
            "low": "low", "close": "close", "volume": "volume",
        })
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df["code"] = f"US_{code.replace('.','')}"
        df["amount"] = 0.0  # sina 不返回 amount

        if days:
            cutoff = pd.Timestamp(date.today()) - timedelta(days=days)
            df = df[df["trade_date"] >= cutoff]

        keep = ["code", "trade_date", "open", "high", "low", "close", "volume", "amount"]
        df = df[[c for c in keep if c in df.columns]]

        n = upsert_df(df, "index_daily", engine)
        total += n
        logger.info(f"  {name}: {n} 条")

    logger.success(f"  美股指数: {total} 条")
    return total


# ═══════════════════════════════════════════════════════════════
# 中美利差
# ═══════════════════════════════════════════════════════════════

def sync_bond_yield(engine=None, days: int | None = None):
    """同步中美利差数据。"""
    import akshare as ak
    engine = engine or get_engine()
    logger.info("=" * 50)
    logger.info("同步中美利差 ...")

    try:
        df = ak.bond_zh_us_rate()
    except Exception as e:
        logger.error(f"  中美利差获取失败: {e}")
        return 0

    df = df.rename(columns={
        "日期": "trade_date",
        "中国国债收益率2年": "cn_2y",
        "中国国债收益率5年": "cn_5y",
        "中国国债收益率10年": "cn_10y",
        "中国国债收益率30年": "cn_30y",
        "中国国债收益率10年-2年": "cn_10y_2y_spread",
        "美国国债收益率2年": "us_2y",
        "美国国债收益率5年": "us_5y",
        "美国国债收益率10年": "us_10y",
        "美国国债收益率30年": "us_30y",
        "美国国债收益率10年-2年": "us_10y_2y_spread",
    })
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["spread_cn_us_10y"] = df["cn_10y"] - df["us_10y"]

    cols = ["trade_date", "cn_2y", "cn_5y", "cn_10y", "cn_30y",
            "cn_10y_2y_spread", "us_2y", "us_5y", "us_10y", "us_30y",
            "us_10y_2y_spread", "spread_cn_us_10y"]
    df = df[[c for c in cols if c in df.columns]]

    if days:
        cutoff = pd.Timestamp(date.today()) - timedelta(days=days)
        df = df[df["trade_date"] >= cutoff]

    n = upsert_df(df, "market_bond_yield", engine)
    logger.success(f"  中美利差: {n} 条")
    return n


# ═══════════════════════════════════════════════════════════════
# 人民币汇率
# ═══════════════════════════════════════════════════════════════

def sync_fx_rate(engine=None, days: int | None = None):
    """同步人民币汇率。用 fx_spot_quote 获取当日实时报价，逐日积累历史。"""
    import akshare as ak
    engine = engine or get_engine()
    logger.info("=" * 50)
    logger.info("同步人民币汇率 ...")

    today = date.today()

    # 先尝试 forex_hist_em 拿历史（首次全量），失败则用 fx_spot_quote 记当日快照
    pairs_hist = {
        "USDCNH": "usd_cny",
        "EURCNH": "eur_cny",
        "JPYCNH": "jpy_cny",
        "HKDCNH": "hkd_cny",
        "GBPCNH": "gbp_cny",
    }

    all_data = []
    for fx_code, col_name in pairs_hist.items():
        try:
            df = ak.forex_hist_em(symbol=fx_code)
            df = df.rename(columns={"日期": "trade_date", "最新价": col_name})
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            all_data.append(df[["trade_date", col_name]].copy())
        except Exception:
            continue

    if all_data:
        merged = all_data[0]
        for df in all_data[1:]:
            merged = merged.merge(df, on="trade_date", how="outer")
        merged = merged.sort_values("trade_date")

        if days:
            cutoff = pd.Timestamp(today) - timedelta(days=days)
            merged = merged[merged["trade_date"] >= cutoff]
    else:
        # Fallback: 当日实时快照
        logger.info("  历史汇率不可用，使用 fx_spot_quote 当日快照")
        try:
            spot = ak.fx_spot_quote()
            pair_map = {
                "USD/CNY": "usd_cny", "EUR/CNY": "eur_cny",
                "100JPY/CNY": "jpy_cny", "HKD/CNY": "hkd_cny",
                "GBP/CNY": "gbp_cny",
            }
            row = {"trade_date": pd.Timestamp(today)}
            for _, r in spot.iterrows():
                col = pair_map.get(r["货币对"])
                if col:
                    row[col] = float(r["卖报价"])  # 卖报价 = 银行卖出价
            if len(row) > 1:
                merged = pd.DataFrame([row])
            else:
                logger.warning("  fx_spot_quote 无有效数据")
                return 0
        except Exception as e:
            logger.warning(f"  汇率快照也失败: {e}")
            return 0

    n = upsert_df(merged, "market_fx_rate", engine)
    logger.success(f"  人民币汇率: {n} 条")
    return n


# ═══════════════════════════════════════════════════════════════
# 表结构初始化
# ═══════════════════════════════════════════════════════════════

def ensure_tables(engine=None):
    """创建外部市场数据表（幂等）。"""
    engine = engine or get_engine()
    with engine.connect() as conn:
        conn.execute(text(DDL_NORTH_FLOW))
        conn.execute(text(DDL_BOND_YIELD))
        conn.execute(text(DDL_FX_RATE))
        conn.commit()
    logger.info("外部数据表已就绪")


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════

def sync_all(days: int | None = None):
    """全量同步所有外部市场数据。"""
    engine = get_engine()
    ensure_tables(engine)

    t0 = time.time()
    results = {}

    results["north_flow"] = sync_north_flow(engine, days=days)
    results["hk_index"] = sync_hk_index(engine, days=days)
    results["us_index"] = sync_us_index(engine, days=days)
    results["bond_yield"] = sync_bond_yield(engine, days=days)
    results["fx_rate"] = sync_fx_rate(engine, days=days)

    engine.dispose()
    elapsed = time.time() - t0
    total = sum(v for v in results.values() if v)
    logger.success(f"外部数据同步完成: {total} 条 ({elapsed:.0f}s)")
    return results


def main():
    p = argparse.ArgumentParser(description="外部市场数据同步")
    p.add_argument("--days", type=int, default=None,
                   help="仅同步最近 N 天（默认全量）")
    args = p.parse_args()
    sync_all(days=args.days)


if __name__ == "__main__":
    main()
