#!/usr/bin/env python
"""每日信号聚合器 —— 三池并行扫描 + 交集分析 + 邮件推送。

涨停池: 涨停信号 (gen_signals 逻辑)
妖股池: 6规则评分 ≥ 3
牛股池: 缩量筑底筛选

用法:
    python scripts/run_daily_signals.py               # 扫描+打印
    python scripts/run_daily_signals.py --send-email   # 扫描+邮件
"""

from __future__ import annotations
import argparse, os, sys, json, smtplib
from datetime import date, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.db import get_engine
from data.loader import load_daily_data, load_mcap_data
from config.settings import TradingConfig

OUT_DIR = "data/arsenal"
os.makedirs(OUT_DIR, exist_ok=True)

# ── 邮箱配置 ──
EMAIL_CONFIG = {
    "smtp_host": os.environ.get("SMTP_HOST", "smtp.qq.com"),
    "smtp_port": int(os.environ.get("SMTP_PORT", "465")),
    "user": os.environ.get("SMTP_USER", ""),
    "password": os.environ.get("SMTP_PASS", ""),
    "to": os.environ.get("EMAIL_TO", ""),
}


def today_market_snapshot():
    """今日市场概况。"""
    engine = get_engine()
    with engine.connect() as conn:
        r = conn.execute(__import__('sqlalchemy').text(
            "SELECT MAX(trade_date) FROM stock_daily"
        )).fetchone()
    if not r or not r[0]:
        return {}
    td = pd.Timestamp(r[0])
    pre_start = (td - timedelta(days=90)).strftime("%Y-%m-%d")
    end_str = td.strftime("%Y-%m-%d")

    codes_df = pd.read_sql(
        __import__('sqlalchemy').text(
            "SELECT code FROM stock_basic WHERE is_st=FALSE"), engine
    )
    codes = [str(c).zfill(6) for c in codes_df['code'].tolist()]

    daily = load_daily_data(engine, codes, pre_start, end_str, cols=['close','volume','turnover'])
    daily['code'] = daily['code'].astype(str).str.zfill(6)
    daily['trade_date'] = pd.to_datetime(daily['trade_date'])
    daily = daily.sort_values(['code','trade_date'])
    daily['ret'] = daily.groupby('code')['close'].pct_change()

    # CSI1000
    csi = pd.read_sql(__import__('sqlalchemy').text(
        "SELECT trade_date, close FROM index_daily WHERE code='000852' "
        "AND trade_date BETWEEN :s AND :e ORDER BY trade_date"
    ), engine, params={"s": pre_start, "e": end_str})
    csi['trade_date'] = pd.to_datetime(csi['trade_date'])
    csi['ma60'] = csi['close'].rolling(60,min_periods=30).mean()
    engine.dispose()

    today_data = daily[daily['trade_date'] == td]
    prev_data = daily[daily['trade_date'] == td - timedelta(days=1)]

    snapshot = {
        'date': td.strftime("%Y-%m-%d"),
        'total_stocks': len(today_data),
        'avg_ret': float(today_data['ret'].mean()) if len(today_data) > 0 else 0,
        'up_ratio': float((today_data['ret'] > 0).mean()) if len(today_data) > 0 else 0,
        'lu_count': int((today_data['ret'] >= 0.09).sum()),
        'csi1000': float(csi[csi['trade_date']==td]['close'].iloc[0]) if len(csi[csi['trade_date']==td]) > 0 else 0,
        'csi1000_ma60': float(csi[csi['trade_date']==td]['ma60'].iloc[0]) if len(csi[csi['trade_date']==td]) > 0 else 0,
    }
    snapshot['csi_trend'] = '上升' if snapshot['csi1000'] > snapshot['csi1000_ma60'] else '下降'
    return snapshot


def screen_limit_up():
    """涨停池：今日涨停且满足基础筛选的股票。"""
    engine = get_engine()
    with engine.connect() as conn:
        r = conn.execute(__import__('sqlalchemy').text(
            "SELECT MAX(trade_date) FROM stock_daily"
        )).fetchone()
    td = pd.Timestamp(r[0])
    end_str = td.strftime("%Y-%m-%d")
    pre_start = (td - timedelta(days=90)).strftime("%Y-%m-%d")

    codes = pd.read_sql(__import__('sqlalchemy').text(
        "SELECT code, name, industry FROM stock_basic WHERE is_st=FALSE"), engine
    )
    codes['code'] = codes['code'].astype(str).str.zfill(6)
    name_map = dict(zip(codes['code'], codes['name']))
    ind_map = dict(zip(codes['code'], codes['industry'].fillna('其他')))

    daily = load_daily_data(engine, codes['code'].tolist(), pre_start, end_str,
                            cols=['open','high','low','close','volume','turnover'])
    daily['code'] = daily['code'].astype(str).str.zfill(6)
    daily['trade_date'] = pd.to_datetime(daily['trade_date'])
    daily = daily.sort_values(['code','trade_date'])
    daily['ret'] = daily.groupby('code')['close'].pct_change()
    daily['prev_close'] = daily.groupby('code')['close'].shift(1)

    extra = load_mcap_data(engine, codes['code'].tolist(), pre_start, end_str, use_proxy=True)
    extra['code'] = extra['code'].astype(str).str.zfill(6)
    extra['trade_date'] = pd.to_datetime(extra['trade_date'])
    engine.dispose()

    # 今日涨停
    all_dates = sorted(daily['trade_date'].unique())
    td_idx = all_dates.index(td) if td in all_dates else -1
    prev_td = all_dates[td_idx - 1] if td_idx > 0 else td

    today = daily[daily['trade_date'] == td].set_index('code')
    prev = daily[daily['trade_date'] == prev_td].set_index('code')
    ex_td = extra[extra['trade_date'] == td].set_index('code')

    today['prev_close'] = prev['close']
    today['ret_calc'] = today['close'] / today['prev_close'] - 1
    today['mcap'] = ex_td.get('market_cap', np.nan) if not ex_td.empty else np.nan

    # 涨停筛选
    lu = today[today['ret_calc'] >= 0.09].copy()
    if lu.empty:
        return pd.DataFrame()

    # 基础过滤
    mask = (
        (lu['mcap'].between(10, 500) if 'mcap' in lu.columns else True) &
        (lu['close'].between(3, 100)) &
        (lu['close'] > 0)
    )
    lu = lu[mask]
    lu['name'] = lu.index.map(name_map)
    lu['industry'] = lu.index.map(ind_map)
    lu['limit_up_score'] = (lu['ret_calc'] * 100).round(1)  # 涨停强度

    result = lu.nlargest(30, 'ret_calc')[
        ['name','industry','close','mcap','limit_up_score']
    ].copy()
    result.columns = ['名称','行业','收盘价','市值(亿)','涨停强度']
    result['代码'] = result.index
    return result


def screen_yaogu():
    """妖股池：6规则实时评分 ≥ 3。"""
    engine = get_engine()
    with engine.connect() as conn:
        r = conn.execute(__import__('sqlalchemy').text(
            "SELECT MAX(trade_date) FROM stock_daily"
        )).fetchone()
    td = pd.Timestamp(r[0])
    end_str = td.strftime("%Y-%m-%d")
    pre_start = (td - timedelta(days=90)).strftime("%Y-%m-%d")

    codes = pd.read_sql(__import__('sqlalchemy').text(
        "SELECT code, name, industry FROM stock_basic WHERE is_st=FALSE"), engine
    )
    codes['code'] = codes['code'].astype(str).str.zfill(6)
    name_map = dict(zip(codes['code'], codes['name']))
    ind_map = dict(zip(codes['code'], codes['industry'].fillna('其他')))

    daily = load_daily_data(engine, codes['code'].tolist(), pre_start, end_str,
                            cols=['open','high','low','close','volume','turnover'])
    daily['code'] = daily['code'].astype(str).str.zfill(6)
    daily['trade_date'] = pd.to_datetime(daily['trade_date'])
    daily = daily.sort_values(['code','trade_date'])
    daily['ret'] = daily.groupby('code')['close'].pct_change()
    daily['prev_close'] = daily.groupby('code')['close'].shift(1)
    engine.dispose()

    all_dates = sorted(daily['trade_date'].unique())
    td_idx = all_dates.index(td) if td in all_dates else -1
    prev_td = all_dates[td_idx - 1] if td_idx > 0 else td

    today = daily[daily['trade_date'] == td].set_index('code')
    prev_close = daily[daily['trade_date'] == prev_td].set_index('code')['close']
    today['prev_close'] = prev_close

    signals = []
    for code in today.index:
        cdata = daily[daily['code'] == code]
        if len(cdata) < 20:
            continue
        r_today = today.loc[code]
        ret_val = r_today['close'] / r_today['prev_close'] - 1 if pd.notna(r_today.get('prev_close')) and r_today['prev_close'] > 0 else 0

        limit = 0.20 if str(code).startswith(('688','300','301')) else 0.10
        is_lu = ret_val >= limit * 0.98

        if not is_lu:
            continue

        # 因子
        is_lu_s = (cdata['ret'] >= limit * 0.98)
        streak = 0
        for v in is_lu_s.iloc[::-1]:
            if v: streak += 1
            else: break

        hl = r_today['high'] - r_today['low']
        seal = r_today['close'] / r_today['high'] if r_today['high'] > 0 else 1
        vol_ratio = r_today['volume'] / cdata['volume'].tail(20).mean() if len(cdata) >= 20 else 1
        amp = hl / r_today['prev_close'] if pd.notna(r_today.get('prev_close')) and r_today['prev_close'] > 0 else 1
        yiziban = 1 if abs(hl) < r_today['close'] * 0.001 else 0
        vol_climax = r_today['volume'] / (cdata['volume'].tail(20).mean() + 2*cdata['volume'].tail(20).std()) if len(cdata) >= 20 else 1

        low_vol_streak = 0
        for _, row in cdata.tail(20).iterrows():
            v_avg = cdata['volume'].tail(20).mean()
            if row['volume'] < v_avg * 0.7: low_vol_streak += 1
            else: low_vol_streak = 0

        # 评分
        score = 0
        if yiziban: score += 3
        if pd.notna(amp) and amp < 0.08: score += 2
        if pd.notna(vol_ratio) and vol_ratio < 1.5: score += 1
        if vol_climax < 0.8 if pd.notna(vol_climax) else False: score += 1
        if streak >= 2: score += 1
        if low_vol_streak >= 1: score += 1

        if score >= 3:
            signals.append({
                'code': code, 'name': name_map.get(code, '?'),
                'industry': ind_map.get(code, '?'),
                'close': r_today['close'], 'yaogu_score': score,
                'yiziban': yiziban, 'streak': streak,
                'ret_today': round(ret_val*100, 1),
                'seal': round(seal, 3), 'vol_ratio': round(vol_ratio, 2),
            })

    return pd.DataFrame(signals).sort_values('yaogu_score', ascending=False) if signals else pd.DataFrame()


def screen_bull():
    """牛股池：复用 screen_bull.py。"""
    from scripts.screen_bull import screen
    return screen()


def find_intersections(limit_up_df, yaogu_df, bull_df):
    """找三池交集。"""
    intersections = {}
    pools = {'涨停': limit_up_df, '妖股': yaogu_df, '牛股': bull_df}

    for name1, name2 in [('涨停','妖股'), ('涨停','牛股'), ('妖股','牛股')]:
        if pools[name1].empty or pools[name2].empty:
            intersections[f'{name1}∩{name2}'] = pd.DataFrame()
            continue
        codes1 = set(pools[name1]['代码'].tolist() if '代码' in pools[name1].columns else pools[name1].index.tolist())
        codes2 = set(pools[name2]['代码'].tolist() if '代码' in pools[name2].columns else pools[name2].index.tolist())
        common = codes1 & codes2
        if common and not bull_df.empty:
            overlap = bull_df[bull_df['代码'].isin(common)] if '代码' in bull_df.columns else bull_df[bull_df.index.isin(common)]
        else:
            overlap = pd.DataFrame()
        intersections[f'{name1}∩{name2}'] = overlap

    # 三池交集
    if not limit_up_df.empty and not yaogu_df.empty and not bull_df.empty:
        c1 = set(limit_up_df['代码'].tolist() if '代码' in limit_up_df.columns else limit_up_df.index.tolist())
        c2 = set(yaogu_df['代码'].tolist() if '代码' in yaogu_df.columns else yaogu_df.index.tolist())
        c3 = set(bull_df['代码'].tolist() if '代码' in bull_df.columns else bull_df.index.tolist())
        triple = c1 & c2 & c3
        if triple and not bull_df.empty:
            triple_df = bull_df[bull_df['代码'].isin(triple)] if '代码' in bull_df.columns else bull_df[bull_df.index.isin(triple)]
        else:
            triple_df = pd.DataFrame()
        intersections['涨停∩妖股∩牛股'] = triple_df

    return intersections


def build_report(snapshot, limit_up, yaogu, bull, intersections):
    """生成文本报告。"""
    lines = []
    lines.append("=" * 70)
    lines.append(f"  策略武器库 · 每日信号报告")
    lines.append(f"  日期: {snapshot.get('date', 'N/A')}")
    lines.append("=" * 70)
    lines.append("")

    # 市场概况
    lines.append("【市场概况】")
    lines.append(f"  全市场: {snapshot.get('total_stocks',0)}只 | "
                 f"均涨幅 {snapshot.get('avg_ret',0):+.2%} | "
                 f"上涨比 {snapshot.get('up_ratio',0):.0%}")
    lines.append(f"  CSI1000: {snapshot.get('csi1000',0):.0f} | "
                 f"趋势: {snapshot.get('csi_trend','N/A')} | "
                 f"涨停: {snapshot.get('lu_count',0)}只")
    lines.append("")

    # 涨停池
    lines.append(f"【涨停池】{len(limit_up)} 只 — 今日涨停+基础过滤")
    if not limit_up.empty:
        for _, r in limit_up.head(10).iterrows():
            lines.append(f"  {r['代码']} {r['名称']:<8s} "
                         f"涨停{r['涨停强度']:+.1f}% | {r.get('行业','')}")
    else:
        lines.append("  (今日无符合条件的涨停股)")
    lines.append("")

    # 妖股池
    lines.append(f"【妖股池】{len(yaogu)} 只 — 6规则评分 ≥ 3")
    if not yaogu.empty:
        for _, r in yaogu.head(10).iterrows():
            tags = []
            if r.get('yiziban'): tags.append('一字板')
            if r.get('streak',0) >= 2: tags.append(f"连板{r['streak']}")
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
    lines.append("【池子交集】— 同时被多策略选中的高置信候选")
    for name, df in intersections.items():
        if df is not None and not df.empty:
            lines.append(f"  {name}: {len(df)}只")
            for _, r in df.head(5).iterrows():
                code = r.get('代码', r.name) if '代码' in df.columns else r.name
                name_str = r.get('名称', '')
                score = r.get('牛股评分', r.get('yaogu_score', ''))
                lines.append(f"    {code} {name_str:<8s} 评分{score}")
        else:
            lines.append(f"  {name}: 无")
    lines.append("")

    lines.append("=" * 70)
    lines.append("  报告结束。以上信号仅供参考，不构成投资建议。")
    lines.append("=" * 70)

    return "\n".join(lines)


def send_email(subject, body):
    """发送邮件。"""
    if not EMAIL_CONFIG['user'] or not EMAIL_CONFIG['to']:
        logger.warning("邮箱未配置，跳过发送。设置环境变量: QUANT_EMAIL_USER, QUANT_EMAIL_PASS, QUANT_EMAIL_TO")
        return False

    msg = MIMEMultipart()
    msg['From'] = EMAIL_CONFIG['user']
    msg['To'] = EMAIL_CONFIG['to']
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain', 'utf-8'))

    try:
        if EMAIL_CONFIG['smtp_port'] == 465:
            server = smtplib.SMTP_SSL(EMAIL_CONFIG['smtp_host'], EMAIL_CONFIG['smtp_port'], timeout=10)
        else:
            server = smtplib.SMTP(EMAIL_CONFIG['smtp_host'], EMAIL_CONFIG['smtp_port'], timeout=10)
            server.starttls()
        server.login(EMAIL_CONFIG['user'], EMAIL_CONFIG['password'])
        server.sendmail(EMAIL_CONFIG['user'], EMAIL_CONFIG['to'], msg.as_string())
        server.quit()
        logger.success(f"邮件已发送到 {EMAIL_CONFIG['to']}")
        return True
    except Exception as e:
        logger.error(f"邮件发送失败: {e}")
        return False


def main():
    p = argparse.ArgumentParser(description="每日信号聚合器")
    p.add_argument("--send-email", action="store_true", help="发送邮件")
    p.add_argument("--no-email", action="store_true", help="不发送邮件(默认)")
    args = p.parse_args()

    print("=" * 60)
    print("  策略武器库 · 扫描中...")
    print("=" * 60)

    # 市场概况
    snapshot = today_market_snapshot()
    print(f"  日期: {snapshot.get('date')} | CSI1000: {snapshot.get('csi_trend')} | 涨停: {snapshot.get('lu_count')}只")

    # 三池扫描
    print("  扫描涨停池...")
    limit_up = screen_limit_up()
    print(f"    涨停池: {len(limit_up)}只")

    print("  扫描妖股池...")
    yaogu = screen_yaogu()
    print(f"    妖股池: {len(yaogu)}只")

    print("  扫描牛股池...")
    bull = screen_bull()
    print(f"    牛股池: {len(bull)}只")

    # 交集
    intersections = find_intersections(limit_up, yaogu, bull)

    # 报告
    report = build_report(snapshot, limit_up, yaogu, bull, intersections)
    print("\n" + report)

    # 保存
    date_tag = snapshot['date'].replace('-', '')
    report_path = f"{OUT_DIR}/daily_report_{date_tag}.txt"
    with open(report_path, 'w') as f:
        f.write(report)

    # 保存各池数据
    for name, df in [('limit_up', limit_up), ('yaogu', yaogu), ('bull', bull)]:
        if not df.empty:
            json_path = f"{OUT_DIR}/{name}_{date_tag}.json"
            df_copy = df.copy()
            df_copy['code'] = df_copy.index if '代码' not in df_copy.columns else df_copy['代码']
            df_copy.to_json(json_path, orient='records', force_ascii=False, indent=2)

    # 邮件
    if args.send_email:
        subject = f"量化信号日报 {snapshot['date']} | 涨停{len(limit_up)} 妖股{len(yaogu)} 牛股{len(bull)}"
        send_email(subject, report)
    elif not args.no_email and EMAIL_CONFIG['user']:
        # 如果配置了邮箱，默认发送
        subject = f"量化信号日报 {snapshot['date']} | 涨停{len(limit_up)} 妖股{len(yaogu)} 牛股{len(bull)}"
        send_email(subject, report)

    print(f"\n报告已保存: {report_path}")


if __name__ == "__main__":
    main()
