#!/usr/bin/env python
"""ETF 三因子监测 — 国家队资金信号扫描。

用法:
    python scripts/run_etf_monitor.py                    # 最近交易日分析
    python scripts/run_etf_monitor.py --send             # 分析 + 发邮件
    python scripts/run_etf_monitor.py --date 2026-05-30  # 指定日期
    python scripts/run_etf_monitor.py --stats            # DB统计
"""
import sys, os, argparse, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import date, timedelta
from loguru import logger
from sqlalchemy import text

from data.db import get_engine
from etf_monitor.config import load_etfs, SIGNAL_HIGH, SIGNAL_MID
from etf_monitor.engine import analyze_all


def _send_etf_email(html_body: str, subject: str) -> bool:
    """复用项目 SMTP 配置发送邮件。"""
    import smtplib, os
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

    user = os.getenv("SMTP_USER", "")
    to = os.getenv("SMTP_TO", "")
    if not user or not to:
        print("邮箱未配置，跳过发送")
        return False

    msg = MIMEMultipart()
    msg["From"] = os.getenv("SMTP_FROM", user)
    msg["To"] = to
    msg["Subject"] = subject
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    try:
        host = os.getenv("SMTP_HOST", "smtp.qq.com")
        port = int(os.getenv("SMTP_PORT", "465"))
        if port == 465:
            server = smtplib.SMTP_SSL(host, port, timeout=15)
        else:
            server = smtplib.SMTP(host, port, timeout=15)
            server.starttls()
        server.login(user, os.getenv("SMTP_PASSWORD", ""))
        server.sendmail(user, [r.strip() for r in to.split(",") if r.strip()], msg.as_string())
        server.quit()
        print(f"邮件已发送到 {to}")
        return True
    except Exception as e:
        print(f"邮件发送失败: {e}")
        return False


def fetch_kline(engine, code, limit=60):
    """从 etf_daily 表获取ETF日线。"""
    try:
        df = pd.read_sql(text(f"""
            SELECT trade_date as date, open, high, low, close, volume
            FROM etf_daily WHERE code = '{code}'
            ORDER BY trade_date DESC LIMIT {limit}
        """), engine)
        return df.sort_values("date") if not df.empty else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def fetch_idx_kline(engine, limit=60):
    """获取沪深300指数日线。"""
    try:
        df = pd.read_sql(text(f"""
            SELECT trade_date as date, close FROM index_daily
            WHERE code = '000300' ORDER BY trade_date DESC LIMIT {limit}
        """), engine)
        return df.sort_values("date") if not df.empty else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def generate_html_report(results, target_date):
    """生成 HTML 报告。"""
    high_count = sum(1 for r in results if r.get("signal_level") == "high")
    mid_count = sum(1 for r in results if r.get("signal_level") == "mid")
    normal_count = sum(1 for r in results if r.get("signal_level") == "normal")

    # Overall signal
    if high_count >= 3:
        banner = ("🔴 高确信信号", "#c62828")
    elif high_count + mid_count >= 3:
        banner = ("🟡 中等信号", "#e65100")
    else:
        banner = ("⚪ 正常", "#2e7d32")

    rows_html = ""
    for r in results:
        if r.get("error"):
            rows_html += f"<tr><td>{r['code']}</td><td>{r['name']}</td><td colspan='10' style='color:#999'>{r['error']}</td></tr>"
            continue
        sig_color = {"high": "#c62828", "mid": "#e65100", "normal": "#2e7d32"}.get(r.get("signal_level", ""), "#666")
        sp = r.get("share_prob") or "—"
        rows_html += f"""<tr>
            <td>{r['code']}</td><td>{r['name']}</td>
            <td>{r.get('chg_pct',0):+.2f}%</td>
            <td>{r.get('vol_ratio',0):.2f}x</td>
            <td>{r.get('vol_prob',0):.0f}</td>
            <td>{r.get('dir_prob',0):.0f}</td>
            <td>{sp}</td>
            <td style='color:{sig_color};font-weight:bold;'>{r.get('composite_prob',0):.0f}%</td>
            <td>{r.get('signal_level','?')}</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>ETF 三因子监测 {target_date}</title>
<style>
    body {{ font-family: -apple-system, sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; }}
    .banner {{ padding: 16px; border-radius: 8px; color: white; font-size: 18px; margin-bottom: 16px; }}
    .cards {{ display: grid; grid-template-columns: repeat(4,1fr); gap: 12px; margin-bottom: 20px; }}
    .card {{ background: #f5f5f5; padding: 12px; border-radius: 6px; text-align: center; }}
    .card .num {{ font-size: 24px; font-weight: bold; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th,td {{ padding: 6px 8px; text-align: center; border-bottom: 1px solid #eee; }}
    th {{ background: #f5f5f5; }}
</style></head><body>
<h2>ETF 三因子监测 — {target_date}</h2>
<div class="banner" style="background:{banner[1]};">{banner[0]} (高确信{high_count}/中等{mid_count}/正常{normal_count})</div>
<div class="cards">
    <div class="card"><div class="num" style="color:#c62828;">{high_count}</div>高确信</div>
    <div class="card"><div class="num" style="color:#e65100;">{mid_count}</div>中等</div>
    <div class="card"><div class="num" style="color:#2e7d32;">{normal_count}</div>正常</div>
    <div class="card"><div class="num">{len(results)}</div>监控ETF</div>
</div>
<table><thead><tr>
    <th>代码</th><th>名称</th><th>涨跌</th><th>倍量</th><th>量能P</th><th>方向P</th><th>份额P</th><th>综合P</th><th>信号</th>
</tr></thead><tbody>{rows_html}</tbody></table>
<p style="color:#999;font-size:11px;margin-top:20px;">Quant 项目自动生成 | 数据来源: AKShare/腾讯财经</p>
</body></html>"""


def main():
    parser = argparse.ArgumentParser(description="ETF 三因子监测")
    parser.add_argument("--date", help="分析日期 (YYYY-MM-DD), 默认最近交易日")
    parser.add_argument("--send", action="store_true", help="发送邮件报告")
    parser.add_argument("--stats", action="store_true", help="显示数据库统计")
    args = parser.parse_args()

    engine = get_engine()

    if args.stats:
        try:
            cnt = pd.read_sql(text("SELECT COUNT(*) FROM etf_monitor_daily"), engine).iloc[0,0]
            print(f"etf_monitor_daily: {cnt} 条记录")
        except Exception as e:
            print(f"表不存在或查询失败: {e}")
        engine.dispose()
        return

    # 确定分析日期
    if args.date:
        target_date = args.date
    else:
        latest = pd.read_sql(text("SELECT MAX(trade_date) FROM stock_daily"), engine).iloc[0,0]
        target_date = str(latest)

    logger.info(f"ETF 三因子分析: {target_date}")

    # ── 自动同步 ETF 日线 ──
    try:
        latest_etf = pd.read_sql(text("SELECT MAX(trade_date) FROM etf_daily"), engine).iloc[0, 0]
        if latest_etf is None or str(latest_etf) < target_date:
            logger.info(f"ETF 日线过期 ({latest_etf}), 同步中...")
            from data.sync import sync_etf_daily
            sync_etf_daily(engine, start_date=str(latest_etf or '20240101'), workers=4)
            logger.info("ETF 日线同步完成")
        else:
            logger.info(f"ETF 日线已最新 ({latest_etf})")
    except Exception as e:
        logger.warning(f"ETF 日线同步跳过: {e}")

    # 获取 K 线数据
    kline_map = {}
    etfs = load_etfs(engine)
    logger.info(f"监控 {len(etfs)} 只 ETF")
    for code in etfs:
        kl = fetch_kline(engine, code, 60)
        if not kl.empty:
            kline_map[code] = kl
    idx_kl = fetch_idx_kline(engine, 60)

    # 拉取 ETF 份额数据
    shares_map = {}
    try:
        import akshare as ak
        for code, info in etfs.items():
            d = float(info.get("shares_yi", 0))
            # 份额数据：缓存昨日值计算日变化率
            cache_file = f"output/etf_shares_{code}.txt"
            prev = 0.0
            try:
                with open(cache_file) as f:
                    prev = float(f.read().strip())
            except Exception:
                pass
            delta_pct = (d - prev) / abs(prev) * 100 if prev > 0 else 0
            shares_map[code] = delta_pct
            # 缓存今日值
            os.makedirs("output", exist_ok=True)
            with open(cache_file, "w") as f:
                f.write(str(d))
        logger.info(f"份额数据: {len(shares_map)} 只 (缓存模式，需先有昨日数据)")
    except Exception as e:
        logger.warning(f"份额数据获取失败: {e}")

    results = analyze_all(kline_map, idx_kl, shares_map, etfs)

    # 打印结果
    for r in results:
        if r.get("error"):
            logger.warning(f"  {r['code']} {r['name']}: {r['error']}")
        else:
            logger.info(
                f"  {r['code']} {r['name']}: "
                f"量能{r['vol_prob']:.0f} 方向{r['dir_prob']:.0f} "
                f"综合{r['composite_prob']:.0f}% [{r['signal_level']}]"
            )

    # 保存到 DB
    try:
        with engine.begin() as conn:
            for r in results:
                if r.get("error"):
                    continue
                # Convert numpy types to Python native
                def _py(v):
                    if v is None: return None
                    if hasattr(v, 'item'): return v.item()
                    return v
                conn.execute(text("""
                    INSERT INTO etf_monitor_daily (date, code, name, close, chg_pct,
                        volume_ma20, vol_ratio, vol_prob, dir_prob, share_prob,
                        shares_delta_pct, composite_prob, signal_level)
                    VALUES (:d, :code, :name, :close, :chg, :vma, :vr, :vp, :dp, :sp, :sdp, :cp, :sl)
                    ON CONFLICT (date, code) DO UPDATE SET
                        vol_ratio=:vr2, vol_prob=:vp2, dir_prob=:dp2, composite_prob=:cp2, signal_level=:sl2
                """), {
                    "d": target_date, "code": r["code"], "name": r["name"],
                    "close": _py(r.get("close")), "chg": _py(r.get("chg_pct")),
                    "vma": _py(r.get("volume_ma20")), "vr": _py(r.get("vol_ratio")),
                    "vp": _py(r.get("vol_prob")), "dp": _py(r.get("dir_prob")),
                    "sp": _py(r.get("share_prob")), "sdp": _py(r.get("shares_delta_pct")),
                    "cp": _py(r.get("composite_prob")), "sl": r.get("signal_level"),
                    "vr2": _py(r.get("vol_ratio")), "vp2": _py(r.get("vol_prob")),
                    "dp2": _py(r.get("dir_prob")), "cp2": _py(r.get("composite_prob")),
                    "sl2": r.get("signal_level"),
                })
        logger.info(f"已写入 etf_monitor_daily: {len(results)} 条")
    except Exception as e:
        logger.error(f"DB写入失败: {e}")

    # HTML 报告
    html = generate_html_report(results, target_date)
    report_path = f"output/etf_report_{target_date}.html"
    os.makedirs("output", exist_ok=True)
    with open(report_path, "w") as f:
        f.write(html)
    logger.info(f"报告已保存: {report_path}")

    # 发送邮件（仅高确信时发送）
    if args.send:
        hc = sum(1 for r in results if r.get("signal_level") == "high")
        if hc > 0:
            _send_etf_email(html, subject=f"ETF 三因子监测 {target_date} — {hc}只高确信")
        else:
            logger.info("无高确信信号，跳过邮件发送")

    engine.dispose()


if __name__ == "__main__":
    main()
