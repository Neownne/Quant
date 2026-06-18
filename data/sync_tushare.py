#!/usr/bin/env python
"""Tushare 数据同步模块 —— 补齐行业/板块/地区数据。

用法:
    python data/sync_tushare.py                     # 全量同步所有
    python data/sync_tushare.py --basic-only         # 仅股票基础信息
    python data/sync_tushare.py --industry-only      # 仅申万行业分类
    python data/sync_tushare.py --sector-only       # 仅同花顺板块
"""

from __future__ import annotations

import argparse, os, sys, time
import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy import text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from data.db import get_engine, upsert_df

TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN", "")

# tushare 120积分：stock_basic 1次/分钟，daily 50次/分钟
# 保守取 61s 确保所有接口不超限
_TUSHARE_COOLDOWN = 61
_LAST_API_CALL = 0.0      # 模块级，跨函数跟踪


def _init_pro():
    """初始化 tushare pro 接口（自动等限流冷却）。"""
    global _LAST_API_CALL
    _rate_limit_wait()
    if not TUSHARE_TOKEN:
        raise RuntimeError("TUSHARE_TOKEN 未配置，请在 .env 中设置")
    import tushare as ts
    ts.set_token(TUSHARE_TOKEN)
    return ts.pro_api()


def _rate_limit_wait():
    """如果距上次 API 调用不足 1 小时，等待剩余时间。"""
    global _LAST_API_CALL
    elapsed = time.time() - _LAST_API_CALL
    if _LAST_API_CALL > 0 and elapsed < _TUSHARE_COOLDOWN:
        wait = _TUSHARE_COOLDOWN - elapsed
        logger.info(f"  ⏳ tushare 限流冷却，等待 {wait/60:.0f} 分钟...")
        time.sleep(wait)
    _LAST_API_CALL = time.time()


def _ts_code_to_6digit(ts_code: str) -> str:
    """000001.SZ → 000001"""
    return str(ts_code).split(".")[0].zfill(6)


# ═══════════════════════════════════════════════════════════════
# 1. 股票基础信息
# ═══════════════════════════════════════════════════════════════

def sync_stock_basic_tushare(engine) -> int:
    """用 tushare stock_basic 补齐 stock_basic 表的 industry/area/list_date。

    不覆盖已有的 name（保留 akshare 来源）。
    返回更新的股票数。
    """
    logger.info("[tushare] 拉取 stock_basic ...")
    pro = _init_pro()

    df = pro.stock_basic(
        exchange='', list_status='L',
        fields='ts_code,symbol,name,area,industry,list_date,exchange,curr_type,is_hs',
    )
    if df is None or df.empty:
        logger.error("  tushare stock_basic 返回空")
        return 0

    df['code'] = df['ts_code'].apply(_ts_code_to_6digit)
    df['market'] = df['exchange'].map({'SSE': 'SH', 'SZSE': 'SZ', 'BSE': 'BJ'}).fillna('')
    df['list_date_ts'] = pd.to_datetime(df['list_date'], format='%Y%m%d', errors='coerce')

    logger.info(f"  tushare 返回: {len(df)} 只 (SH {(df['market']=='SH').sum()} "
                f"SZ {(df['market']=='SZ').sum()} BJ {(df['market']=='BJ').sum()})")

    # 更新已有股票的非空字段
    with engine.connect() as conn:
        for _, r in df.iterrows():
            updates = []
            params = {}

            if pd.notna(r.get('industry')) and str(r['industry']).strip():
                updates.append("industry = :ind")
                params['ind'] = str(r['industry']).strip()
            if pd.notna(r.get('area')) and str(r['area']).strip():
                updates.append("area = :area")
                params['area'] = str(r['area']).strip()
            if pd.notna(r.get('market')) and str(r['market']).strip():
                updates.append("market = :mkt")
                params['mkt'] = str(r['market']).strip()
            if pd.notna(r.get('list_date_ts')):
                updates.append("list_date = :ld")
                params['ld'] = r['list_date_ts'].strftime('%Y-%m-%d')

            if not updates:
                continue

            params['code'] = r['code']
            set_clause = ", ".join(updates)
            conn.execute(
                text(f"UPDATE stock_basic SET {set_clause} WHERE code = :code"),
                params,
            )

        # 统计
        for col in ['industry', 'area']:
            cnt = conn.execute(
                text(f"SELECT COUNT(*) FROM stock_basic WHERE {col} IS NOT NULL AND {col} != ''")
            ).scalar()
            logger.info(f"  stock_basic.{col}: {cnt} 有值")

        conn.commit()

    return len(df)


# ═══════════════════════════════════════════════════════════════
# 2. 申万行业分类（SW2021）
# ═══════════════════════════════════════════════════════════════

def sync_sw_industry(engine) -> int:
    """从 tushare 拉取申万 2021 行业分类，写入 stock_industry 表。

    填入 industry_sw1 (L1)、industry_sw2 (L2)、market。
    """
    logger.info("[tushare] 拉取申万 SW2021 行业分类 ...")
    pro = _init_pro()

    df = pro.industry(L='L', src='sw2021')
    if df is None or df.empty:
        logger.error("  tushare industry 返回空")
        return 0

    df['code'] = df['code'].apply(_ts_code_to_6digit)

    # 拆分 L1/L2
    l1 = df[df['level'] == 'L1'][['code', 'industry_name']].rename(
        columns={'industry_name': 'industry_sw1'})
    l2 = df[df['level'] == 'L2'][['code', 'industry_name']].rename(
        columns={'industry_name': 'industry_sw2'})

    merged = l1.merge(l2, on='code', how='outer')

    # 去重（同一股票可能属于多个申万行业，保留第一个）
    merged = merged.drop_duplicates(subset=['code'], keep='first')

    # 推断 market
    def _infer_market(c):
        c = str(c)
        if c.startswith('688'): return '科创板'
        if c.startswith(('300', '301')): return '创业板'
        if c.startswith(('8', '4')): return '北交所'
        return '主板'
    merged['market'] = merged['code'].apply(_infer_market)

    logger.info(f"  SW L1: {merged['industry_sw1'].nunique()} 个, "
                f"L2: {merged['industry_sw2'].notna().sum()} 只, "
                f"总股票: {len(merged)}")

    upsert_df(merged[['code', 'industry_sw1', 'industry_sw2', 'market']],
              'stock_industry', engine)

    # 验证
    with engine.connect() as conn:
        for col in ['industry_sw1', 'industry_sw2']:
            cnt = conn.execute(
                text(f"SELECT COUNT(*) FROM stock_industry WHERE {col} IS NOT NULL AND {col} != ''")
            ).scalar()
            logger.info(f"  stock_industry.{col}: {cnt} 有值")

    return len(merged)


# ═══════════════════════════════════════════════════════════════
# 3. 同花顺板块 + 成分股
# ═══════════════════════════════════════════════════════════════

def sync_ths_sectors(engine, include_members: bool = False) -> dict:
    """拉取同花顺行业/概念板块列表。

    include_members=True 时还会拉取每个板块的成分股（免费版不可用，需 400+ 次调用）。
    返回 {'boards': int, 'members': int}
    """
    logger.info("[tushare] 拉取同花顺板块列表 ...")
    pro = _init_pro()

    # ── 板块列表（1 次 API 调用）──
    df_idx = pro.ths_index()
    if df_idx is None or df_idx.empty:
        logger.error("  tushare ths_index 返回空")
        return {'boards': 0, 'members': 0}

    logger.info(f"  总板块: {len(df_idx)} (行业: {(df_idx['type']=='industry').sum()}, "
                f"概念: {(df_idx['type']=='concept').sum()})")

    # 写入 concept_board
    boards = df_idx[['ts_code', 'name', 'type']].copy()
    boards.columns = ['code', 'name', 'type']
    boards['stock_count'] = 0
    boards['updated_date'] = pd.Timestamp.now().strftime('%Y-%m-%d')
    upsert_df(boards, 'concept_board', engine)
    logger.info(f"  concept_board: {len(boards)} 条")

    # ── 成分股（需要逐板块调用，免费版不可行）──
    if not include_members:
        logger.info("  ⚠️ 跳过成分股同步（免费版需 400+ 次调用）。"
                    "升级 tushare 后用 --sector-only --with-members 拉取。")
        return {'boards': len(boards), 'members': 0}

    _sync_ths_members(engine, pro, df_idx)
    return {'boards': len(boards), 'members': 0}


def _sync_ths_members(engine, pro, df_idx):
    """同步板块成分股（需大量 API 调用）。"""
    logger.info(f"[tushare] 拉取 {len(df_idx)} 个板块成分股（免费版不可用）...")
    all_members = []
    board_count = 0
    skip_count = 0
    t0 = time.time()

    for _, row in df_idx.iterrows():
        ts_code = row['ts_code']
        try:
            _rate_limit_wait()
            mem = pro.ths_member(ts_code=ts_code, fields='ts_code,name,weight,in_date,out_date,is_new')
            if mem is not None and not mem.empty:
                mem['board_code'] = ts_code
                for _, m in mem.iterrows():
                    all_members.append({
                        'board_code': ts_code,
                        'stock_code': _ts_code_to_6digit(m['ts_code']),
                        'weight': m.get('weight'),
                        'in_date': pd.to_datetime(m['in_date'], format='%Y%m%d', errors='coerce')
                                   if pd.notna(m.get('in_date')) else None,
                        'out_date': pd.to_datetime(m['out_date'], format='%Y%m%d', errors='coerce')
                                    if pd.notna(m.get('out_date')) else None,
                    })
                board_count += 1
            else:
                skip_count += 1
        except Exception as e:
            logger.warning(f"  {ts_code} {row['name']} 成分股拉取失败: {e}")
            skip_count += 1

        if (board_count + skip_count) % 10 == 0:
            logger.info(f"  进度: {board_count + skip_count}/{len(df_idx)} "
                        f"(成功{board_count}, {time.time()-t0:.0f}s)")

    logger.info(f"  板块成分完成: {board_count} 板块, {len(all_members)} 条映射 "
                f"({time.time()-t0:.0f}s)")

    if all_members:
        mem_df = pd.DataFrame(all_members).drop_duplicates(
            subset=['board_code', 'stock_code'], keep='first'
        )
        upsert_df(mem_df, 'concept_stock', engine)
        logger.info(f"  concept_stock: {len(mem_df)} 条")

        # 更新 stock_count
        with engine.connect() as conn:
            conn.execute(text("""
                UPDATE concept_board b SET stock_count = (
                    SELECT COUNT(*) FROM concept_stock s WHERE s.board_code = b.code
                )
            """))
            conn.commit()


# ═══════════════════════════════════════════════════════════════
# 统一入口
# ═══════════════════════════════════════════════════════════════

def sync_all_tushare(engine=None) -> dict:
    """全量同步 tushare 行业/板块数据。

    免费版限流 1次/小时，三步间自动等待。预计总耗时 ~3 小时。
    """
    eng = engine or get_engine()
    own_engine = engine is None

    logger.info("══════════════════════════════════════════════════")
    logger.info("  Tushare 全量行业/板块同步 · 120积分 (50次/分)")
    logger.info("══════════════════════════════════════════════════")

    results = {}

    # Step 1: stock_basic — 1次/分钟
    _rate_limit_wait()
    try:
        results['basic'] = sync_stock_basic_tushare(eng)
    except Exception as e:
        logger.error(f"stock_basic 同步失败: {e}")
        results['basic'] = 0

    # Step 2: SW 行业分类 — 120积分无权限，跳过
    logger.info("[tushare] 申万行业分类: 120积分无 index_classify 权限，跳过")
    results['sw_industry'] = 0

    # Step 3: 同花顺板块 — 120积分无权限，跳过
    logger.info("[tushare] 同花顺板块: 120积分无 ths_index 权限，跳过")
    results['sectors'] = {'boards': 0, 'members': 0}

    logger.success(f"同步完成: stock_basic={results['basic']} 只")

    if own_engine:
        eng.dispose()

    return results


# ═══════════════════════════════════════════════════════════════
# 独立运行
# ═══════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Tushare 行业/板块数据同步")
    p.add_argument("--basic-only", action="store_true", help="仅股票基础信息")
    p.add_argument("--industry-only", action="store_true", help="仅申万行业分类")
    p.add_argument("--sector-only", action="store_true", help="仅同花顺板块")
    p.add_argument("--with-members", action="store_true", help="拉取板块成分股（需付费 token）")
    args = p.parse_args()

    engine = get_engine()

    if args.basic_only:
        sync_stock_basic_tushare(engine)
    elif args.industry_only:
        sync_sw_industry(engine)
    elif args.sector_only:
        sync_ths_sectors(engine, include_members=args.with_members)
    else:
        sync_all_tushare(engine)

    engine.dispose()


if __name__ == "__main__":
    main()
