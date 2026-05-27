import json
import os

from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from sqlalchemy import text

from data.db import get_engine

router = APIRouter(prefix="/api", tags=["api"])

WATCHLIST = {
    "default": ["000001", "000002", "600519", "300750", "002415"],
    "tech": ["300750", "002415", "002475", "688981", "300124"],
}


@router.get("/ping")
async def ping():
    return {"status": "ok"}


@router.get("/quotes/{group}")
async def get_quotes(group: str = "default"):
    codes = WATCHLIST.get(group, WATCHLIST["default"])
    engine = get_engine()
    try:
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT code, close, volume, amount
                FROM stock_daily
                WHERE code = ANY(:codes)
                  AND trade_date = (SELECT MAX(trade_date) FROM stock_daily)
                ORDER BY code
            """), {"codes": codes}).fetchall()
    except Exception:
        rows = []

    quotes = [{"code": r[0], "close": r[1] or 0, "volume": r[2] or 0, "amount": r[3] or 0} for r in rows]

    rows_html = ""
    for q in quotes:
        rows_html += f"""<tr style="cursor:pointer" hx-get="/api/kline/{q['code']}" hx-target="#kline-panel" hx-swap="innerHTML">
            <td>{q['code']}</td>
            <td>{q['close']:.2f}</td>
            <td>{q['volume']:.0f}</td>
            <td>{q['amount']:.0f}</td>
        </tr>"""

    html = f"""<table>
    <thead><tr><th>代码</th><th>现价</th><th>成交量</th><th>成交额</th></tr></thead>
    <tbody>{rows_html}</tbody>
    </table>"""
    return HTMLResponse(html)


@router.get("/kline/{code}")
async def get_kline(code: str):
    engine = get_engine()
    try:
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT trade_date, open, high, low, close, volume
                FROM stock_daily WHERE code = :code
                ORDER BY trade_date DESC LIMIT 250
            """), {"code": code}).fetchall()
    except Exception:
        rows = []

    rows = list(reversed(rows))
    dates = [str(r[0]) for r in rows]
    ohlc = [[float(r[1] or 0), float(r[2] or 0), float(r[3] or 0), float(r[4] or 0)] for r in rows]
    volumes = [int(r[5] or 0) for r in rows]

    option = {
        "grid": [{"left": "8%", "right": "2%", "top": "5%", "height": "65%"},
                 {"left": "8%", "right": "2%", "top": "75%", "height": "20%"}],
        "xAxis": [{"data": dates, "axisLabel": {"show": False}, "axisLine": {"lineStyle": {"color": "#444"}}},
                  {"data": dates, "axisLabel": {"rotate": 30, "color": "#889"}, "axisLine": {"lineStyle": {"color": "#444"}}}],
        "yAxis": [{"scale": True, "axisLine": {"lineStyle": {"color": "#444"}}, "splitLine": {"lineStyle": {"color": "#1a2a3a"}}},
                  {"scale": True, "axisLine": {"lineStyle": {"color": "#444"}}, "splitLine": {"lineStyle": {"color": "#1a2a3a"}}}],
        "series": [
            {"type": "candlestick", "data": ohlc, "xAxisIndex": 0, "yAxisIndex": 0,
             "itemStyle": {"color": "#ef5350", "color0": "#26a69a", "borderColor": "#ef5350", "borderColor0": "#26a69a"}},
            {"type": "bar", "data": volumes, "xAxisIndex": 1, "yAxisIndex": 1,
             "itemStyle": {"color": "#4fc3f7"}},
        ],
        "tooltip": {"trigger": "axis"},
    }

    html = f"""<div id="kline-chart" style="width:100%;height:500px;" data-chart='{json.dumps(option)}'></div>"""
    return HTMLResponse(html)


@router.get("/backtest-list")
async def list_backtests(strategy: str = "", quality: str = ""):
    """Return HTML table of backtest results. When DB is unavailable, show empty state."""
    engine = get_engine()
    try:
        with engine.connect() as conn:
            where = ["1=1"]
            params = {}
            if strategy:
                where.append("sc.name = :strategy")
                params["strategy"] = strategy
            if quality:
                where.append("br.quality = :quality")
                params["quality"] = quality
            rows = conn.execute(text(f"""
                SELECT sc.name, sv.version, br.start_date, br.end_date, br.quality,
                       br.quality_flags, br.id
                FROM backtest_results br
                JOIN strategy_versions sv ON br.version_id = sv.id
                JOIN strategy_configs sc ON sv.strategy_id = sc.id
                WHERE {' AND '.join(where)}
                ORDER BY br.created_at DESC LIMIT 50
            """), params).fetchall()
    except Exception:
        rows = []

    if not rows:
        return HTMLResponse("""<div style="padding:40px;text-align:center;color:#889;">
            <p>暂无回测数据</p><p style="font-size:12px;">运行 scripts/run_ml_backtest.py 生成回测结果</p>
        </div>""")

    rows_html = ""
    for r in rows:
        name, version, start, end, quality, flags, bt_id = r
        badge_class = f"badge-{quality}"
        rows_html += f"""<tr>
            <td>{name}</td><td>{version}</td><td>{start} ~ {end}</td>
            <td><span class="badge {badge_class}">{quality}</span></td>
            <td><button class="mock-button" hx-get="/api/backtest-equity/{bt_id}" hx-target="#equity-panel" hx-swap="innerHTML">权益曲线</button></td>
        </tr>"""

    html = f"""<table>
    <thead><tr><th>策略</th><th>版本</th><th>区间</th><th>质量</th><th>操作</th></tr></thead>
    <tbody>{rows_html}</tbody></table>"""
    return HTMLResponse(html)


@router.get("/backtest-equity/{backtest_id}")
async def get_equity_curve(backtest_id: int):
    engine = get_engine()
    try:
        with engine.connect() as conn:
            row = conn.execute(text(
                "SELECT equity_curve_json FROM backtest_results WHERE id = :id"
            ), {"id": backtest_id}).fetchone()
    except Exception:
        row = None

    if not row or not row[0]:
        return HTMLResponse("<p style='color:#889;padding:20px;'>无权益曲线数据</p>")

    curve = row[0] if isinstance(row[0], dict) else json.loads(str(row[0]))
    # curve is {date_str: nav, ...}
    dates = list(curve.keys()) if isinstance(curve, dict) else []
    values = list(curve.values()) if isinstance(curve, dict) else []

    option = {
        "xAxis": {"data": dates, "axisLabel": {"color": "#889", "rotate": 30}},
        "yAxis": {"scale": True, "axisLabel": {"color": "#889"}},
        "series": [{"type": "line", "data": values, "smooth": True,
                     "lineStyle": {"color": "#4fc3f7"}, "areaStyle": {"color": "rgba(79,195,247,0.1)"}}],
        "tooltip": {"trigger": "axis"},
        "grid": {"left": "10%", "right": "5%", "top": "5%", "bottom": "15%"},
    }

    html = f"""<div id="equity-chart" style="width:100%;height:400px;" data-chart='{json.dumps(option)}'></div>"""
    return HTMLResponse(html)


@router.get("/paper-runs")
async def list_paper_runs():
    engine = get_engine()
    try:
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT pr.id, sc.name, sv.version, pr.start_date, pr.end_date,
                       pr.initial_capital, pr.status
                FROM paper_runs pr
                JOIN strategy_configs sc ON pr.strategy_id = sc.id
                JOIN strategy_versions sv ON pr.version_id = sv.id
                ORDER BY pr.created_at DESC LIMIT 20
            """)).fetchall()
    except Exception:
        rows = []

    if not rows:
        return HTMLResponse("""<div style="padding:40px;text-align:center;color:#889;">
            <p>暂无模拟盘运行记录</p><p style="font-size:12px;">通过后台脚本启动模拟盘</p>
        </div>""")

    html = "<table><thead><tr><th>策略</th><th>版本</th><th>起始日</th><th>结束日</th><th>初始资金</th><th>状态</th><th>操作</th></tr></thead><tbody>"
    for r in rows:
        rid, name, ver, start, end, cap, status = r
        status_badge = {"running": "badge-valid", "paused": "badge-suspect", "stopped": "badge-invalid"}.get(status, "")
        html += f"""<tr>
            <td>{name}</td><td>{ver}</td><td>{start}</td><td>{end or '进行中'}</td>
            <td>{cap:,.0f}</td>
            <td><span class="badge {status_badge}">{status}</span></td>
            <td><button class="mock-button" hx-get="/api/paper-run/{rid}" hx-target="#paper-detail" hx-swap="innerHTML">查看</button></td>
        </tr>"""
    html += "</tbody></table>"
    return HTMLResponse(html)


@router.get("/paper-run/{run_id}")
async def get_paper_run_detail(run_id: int):
    engine = get_engine()
    try:
        with engine.connect() as conn:
            positions = conn.execute(text("""
                SELECT stock_code, entry_date, entry_price, exit_date, exit_price, quantity, pnl, pnl_pct
                FROM paper_positions WHERE run_id = :rid ORDER BY entry_date DESC LIMIT 50
            """), {"rid": run_id}).fetchall()
            signals = conn.execute(text("""
                SELECT signal_date, stock_code, predicted_score, rank
                FROM paper_signals WHERE run_id = :rid ORDER BY signal_date DESC LIMIT 30
            """), {"rid": run_id}).fetchall()
    except Exception:
        positions = []
        signals = []

    pos_html = "<table><thead><tr><th>代码</th><th>入场日</th><th>入场价</th><th>出场日</th><th>出场价</th><th>数量</th><th>盈亏</th><th>盈亏%</th></tr></thead><tbody>"
    for p in positions:
        code, ed, ep, xd, xp, qty, pnl, pct = p
        pnl_class = "up" if (pnl or 0) > 0 else "down"
        pos_html += f"""<tr>
            <td>{code}</td><td>{ed}</td><td>{ep:.2f}</td><td>{xd or '-'}</td><td>{xp or '-'}</td>
            <td>{qty}</td>
            <td class="{pnl_class}">{pnl:,.0f}</td>
            <td class="{pnl_class}">{pct:+.2f}%</td>
        </tr>"""
    pos_html += "</tbody></table>"

    sig_html = "<table><thead><tr><th>日期</th><th>代码</th><th>评分</th><th>排名</th></tr></thead><tbody>"
    for s in signals:
        sd, sc, score, rank = s
        sig_html += f"<tr><td>{sd}</td><td>{sc}</td><td>{score:.4f}</td><td>{rank}</td></tr>"
    sig_html += "</tbody></table>"

    total_pnl = sum((p[6] or 0) for p in positions)
    wins = sum(1 for p in positions if (p[7] or 0) > 0)

    html = f"""<div class="card"><h3>持仓明细 ({len(positions)}笔)</h3>{pos_html}</div>
    <div class="card"><h3>信号记录</h3>{sig_html}</div>
    <div class="card"><h3>汇总</h3><p>总盈亏: <span class="{'up' if total_pnl > 0 else 'down'}">{total_pnl:+,.0f}</span></p><p>胜率: {wins}/{len([p for p in positions if p[6] is not None])}</p></div>"""
    return HTMLResponse(html)


@router.get("/data-status")
async def get_data_status():
    engine = get_engine()
    tables = ["stock_daily", "stock_basic", "index_daily", "stock_daily_extra",
              "stock_shareholder", "stock_financial"]

    html = "<table><thead><tr><th>表名</th><th>最新日期</th><th>行数</th></tr></thead><tbody>"
    for t in tables:
        try:
            with engine.connect() as conn:
                row = conn.execute(text(
                    f"SELECT COALESCE(MAX(trade_date)::text, '无数据'), COUNT(*) FROM {t}" if t != "stock_basic"
                    else f"SELECT 'N/A', COUNT(*) FROM {t}"
                )).fetchone()
            html += f"<tr><td>{t}</td><td>{row[0]}</td><td>{row[1]:,}</td></tr>"
        except Exception:
            html += f"<tr><td>{t}</td><td style='color:#ef5350;'>查询失败</td><td>-</td></tr>"
    html += "</tbody></table>"

    # Add recent quality checks
    try:
        with engine.connect() as conn:
            qrows = conn.execute(text(
                "SELECT trade_date, check_name, CASE WHEN passed THEN 'PASS' ELSE 'FAIL' END, detail FROM data_quality_log ORDER BY trade_date DESC LIMIT 10"
            )).fetchall()
    except Exception:
        qrows = []

    qhtml = "<h4 style='margin-top:20px;'>最近质量校验</h4><table><thead><tr><th>日期</th><th>检查项</th><th>结果</th><th>详情</th></tr></thead><tbody>"
    for qr in qrows:
        td, cn, st, detail = qr
        qhtml += f"<tr><td>{td}</td><td>{cn}</td><td>{st}</td><td style='font-size:11px;'>{detail}</td></tr>"
    qhtml += "</tbody></table>" if qrows else "<p style='color:#889;'>暂无质量校验记录</p>"

    return HTMLResponse(html + qhtml)


@router.post("/sync/trigger")
async def trigger_sync():
    import subprocess, sys
    subprocess.Popen([sys.executable, "-m", "data.sync"],
                     cwd=os.path.join(os.path.dirname(__file__), "..", ".."))
    return HTMLResponse("<p style='color:#4fc3f7;'>同步已触发，正在后台运行... 刷新页面查看结果</p>")
