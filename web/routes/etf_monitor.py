from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from sqlalchemy import text
from data.db import get_engine
from etf_monitor.config import ETFS

router = APIRouter()

@router.get("/etf", response_class=HTMLResponse)
async def etf_monitor_page():
    engine = get_engine()
    results = []
    try:
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT code, name, close, chg_pct, vol_ratio, vol_prob, dir_prob,
                       share_prob, composite_prob, signal_level, date
                FROM etf_monitor_daily
                WHERE date = (SELECT MAX(date) FROM etf_monitor_daily)
                ORDER BY code
            """)).fetchall()
        for r in rows:
            results.append(dict(zip(
                ["code","name","close","chg_pct","vol_ratio","vol_prob","dir_prob",
                 "share_prob","composite_prob","signal_level","date"], r)))
    except Exception:
        pass
    engine.dispose()

    rows_html = ""
    for r in results:
        sig_color = {"high":"#c62828","mid":"#e65100","normal":"#2e7d32"}.get(r.get("signal_level",""),"#666")
        sp = f"{r['share_prob']:.0f}" if r.get('share_prob') else "—"
        rows_html += f"""<tr>
            <td>{r['code']}</td><td>{r['name']}</td>
            <td>{r.get('chg_pct',0):+.2f}%</td><td>{r.get('vol_ratio',0):.2f}x</td>
            <td>{r.get('vol_prob',0):.0f}</td><td>{r.get('dir_prob',0):.0f}</td>
            <td>{sp}</td>
            <td style="color:{sig_color};font-weight:bold;">{r.get('composite_prob',0):.0f}%</td>
            <td>{r.get('signal_level','?')}</td></tr>"""

    return HTMLResponse(f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8"><title>ETF监测</title></head>
<body style="font-family:-apple-system,sans-serif;max-width:1000px;margin:0 auto;padding:20px;">
<h2>ETF 三因子监测 — 国家队资金信号</h2>
<p style="color:#999;">7只宽基ETF × 量能/方向/份额 三维监测</p>
<table style="width:100%;border-collapse:collapse;font-size:13px;">
<thead><tr style="background:#f5f5f5;"><th>代码</th><th>名称</th><th>涨跌</th><th>倍量</th><th>量能P</th><th>方向P</th><th>份额P</th><th>综合P</th><th>信号</th></tr></thead>
<tbody>{rows_html if rows_html else '<tr><td colspan="9" style="color:#999;padding:20px;">暂无数据，运行 scripts/run_etf_monitor.py 生成</td></tr>'}</tbody>
</table></body></html>""")
