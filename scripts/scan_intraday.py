#!/usr/bin/env python
"""午盘扫描：用腾讯实时行情扫描涨停/跌停/板块热度。

用法:
    python scripts/scan_intraday.py                  # 当前盘面
    python scripts/scan_intraday.py --no-sector      # 跳过板块热度
    python scripts/scan_intraday.py --top 20          # 展示Top-N
"""

from __future__ import annotations

import argparse, os, sys, time, urllib.request
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.db import get_engine
from sqlalchemy import text

# ═══════════════════════════════════════════════════════════════
# 涨停阈值（板别感知 —— 与 factors/limit_up.py 一致）
# ═══════════════════════════════════════════════════════════════

_DEFAULT_MULT = 1.9899


def _get_limit(code: str) -> float:
    """涨停阈值。"""
    return _DEFAULT_MULT


# ═══════════════════════════════════════════════════════════════
# 腾讯实时行情
# ═══════════════════════════════════════════════════════════════

def fetch_tencent_quotes(codes: list[str], batch_size: int = 300) -> pd.DataFrame:
    """通过腾讯 API 获取实时行情。

    返回 DataFrame: code, name, price, prev_close, open, high, low, volume
    时间戳格式: 20260617120500 → 上午盘 11:30 收盘快照
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
                print(f"  [WARN] {market} batch {i} 请求失败: {e}")

            time.sleep(0.3)  # 控制频率

    df = pd.DataFrame(all_rows)
    if df.empty:
        return df

    df['code'] = df['code'].astype(str).str.zfill(6)
    df['ret'] = ((df['price'] - df['prev_close']) / df['prev_close']).where(
        df['prev_close'] > 0, 0
    ) * 100

    # 板别感知涨停标记
    df['is_lu'] = df.apply(
        lambda r: 1 if r['ret'] >= _get_limit(r['code']) * 98 else 0, axis=1
    )

    # 附加属性
    df.attrs['timestamp'] = timestamp
    df.attrs['market'] = (
        '上午盘' if '11' in str(timestamp)[8:12] or '1130' in str(timestamp)[8:12] else
        '盘中' if int(str(timestamp)[8:12] or 0) < 1500 else '收盘'
    )

    return df


# ═══════════════════════════════════════════════════════════════
# 扫描
# ═══════════════════════════════════════════════════════════════

def scan():
    """拉取全市场行情并扫描。"""
    engine = get_engine()
    with engine.connect() as conn:
        codes_df = pd.read_sql(
            text('SELECT code, name, industry FROM stock_basic WHERE is_st=FALSE'),
            conn,
        )
    engine.dispose()

    codes_df['code'] = codes_df['code'].astype(str).str.zfill(6)
    name_map = dict(zip(codes_df['code'], codes_df['name']))
    ind_map = dict(zip(codes_df['code'], codes_df['industry'].fillna('其他')))

    print(f"拉取全市场行情... 股票池: {len(codes_df)} 只")
    t0 = time.time()
    df = fetch_tencent_quotes(codes_df['code'].tolist())
    elapsed = time.time() - t0
    ts = df.attrs.get('timestamp', '')
    market_tag = df.attrs.get('market', '')
    print(f"获取: {len(df)} 只 ({elapsed:.0f}s) | 时间戳: {ts} | {market_tag}")

    return df, name_map, ind_map


def print_summary(df: pd.DataFrame, name_map: dict, ind_map: dict, top_n: int = 10):
    """打印扫描摘要。"""
    if df.empty:
        print("无数据")
        return

    ts = df.attrs.get('timestamp', '')
    market_tag = df.attrs.get('market', '')

    print(f"\n{'='*60}")
    print(f"  午盘扫描 — {time.strftime('%Y-%m-%d %H:%M')} ({market_tag})")
    if ts:
        ts_str = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]} {ts[8:10]}:{ts[10:12]}:{ts[12:14]}"
        print(f"  行情时间戳: {ts_str}")
    print(f"{'='*60}")

    up_n = (df['ret'] > 0).sum()
    down_n = (df['ret'] < 0).sum()
    flat_n = (df['ret'] == 0).sum()
    print(f"  总: {len(df)} | 涨: {up_n} | 跌: {down_n} | 平: {flat_n}")
    print(f"  均涨幅: {df['ret'].mean():.2f}% | 涨停: {df['is_lu'].sum()} 只 | "
          f"跌停/准跌停: {(df['ret'] <= -9.5).sum()} 只")
    print()

    # ── 涨停股 ──
    lu = df[df['is_lu'] == 1].sort_values('ret', ascending=False)
    print(f"【涨停股】{len(lu)} 只")
    for _, r in lu.iterrows():
        seal_pct = r['price'] / r['high'] * 100 if r['high'] > 0 else 0
        if r['price'] == r['high'] == r['open']:
            seal_tag = '一字'
        elif seal_pct > 98:
            seal_tag = '封板'
        else:
            seal_tag = f'开板{seal_pct:.0f}%'
        ind = ind_map.get(r['code'], '')
        print(f"  {r['code']} {r['name']:<8s} {r['ret']:+.2f}% | {seal_tag} | {ind}")
    print()

    # ── 涨幅 Top-N（非涨停）──
    up = df[(df['is_lu'] == 0) & (df['ret'] > 0)].nlargest(top_n, 'ret')
    print(f"【涨幅 Top-{top_n}（非涨停）】")
    for _, r in up.iterrows():
        print(f"  {r['code']} {r['name']:<8s} {r['ret']:+.2f}% | {ind_map.get(r['code'], '')}")
    print()

    # ── 跌停 ──
    down = df[df['ret'] <= -9.5].sort_values('ret')
    print(f"【跌停/准跌停】{len(down)} 只")
    for _, r in down.iterrows():
        print(f"  {r['code']} {r['name']:<8s} {r['ret']:+.2f}% | {ind_map.get(r['code'], '')}")

    # ── 板块热度 ──
    df['industry'] = df['code'].map(ind_map)
    sector = df.groupby('industry').agg(
        n=('code', 'count'),
        avg_ret=('ret', 'mean'),
        lu_n=('is_lu', 'sum'),
        up_ratio=('ret', lambda x: (x > 0).mean()),
    ).reset_index()
    sector = sector[sector['n'] >= 5].sort_values('lu_n', ascending=False)

    print(f"\n【涨停最多的行业 Top-10】")
    for _, r in sector.head(10).iterrows():
        if r['lu_n'] > 0:
            bar = '█' * int(r['lu_n'])
            print(f"  {r['industry']:<12s} {int(r['lu_n']):>3}只涨停 | "
                  f"上涨比{r['up_ratio']:.0%} | 均涨幅{r['avg_ret']:+.2f}% {bar}")


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="午盘扫描 — 腾讯实时行情")
    p.add_argument("--top", type=int, default=10, help="涨幅榜 Top-N（默认 10）")
    p.add_argument("--no-sector", action="store_true", help="跳过板块热度")
    args = p.parse_args()

    df, name_map, ind_map = scan()
    print_summary(df, name_map, ind_map, top_n=args.top)


if __name__ == "__main__":
    main()
