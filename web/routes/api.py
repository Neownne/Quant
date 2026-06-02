import json
import os

import numpy as np
from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from sqlalchemy import text

from data.db import get_engine

router = APIRouter(prefix="/api", tags=["api"])

WATCHLIST_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "config", "watchlist.json")

def _load_watchlist() -> dict[str, list[str]]:
    try:
        with open(WATCHLIST_FILE) as f:
            return json.load(f)
    except Exception:
        return {
            "default": ["000001", "000002", "600519", "300750", "002415"],
            "tech": ["300750", "002415", "002475", "688981", "300124"],
        }

def _save_watchlist(data: dict[str, list[str]]):
    os.makedirs(os.path.dirname(WATCHLIST_FILE), exist_ok=True)
    with open(WATCHLIST_FILE, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


@router.get("/ping")
async def ping():
    return {"status": "ok"}


@router.get("/quotes/{group}")
async def get_quotes(group: str = "default"):
    watchlist = _load_watchlist()
    codes = watchlist.get(group, watchlist.get("default", []))
    engine = get_engine()
    try:
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT sd.code, sd.close, sd.volume, sd.amount,
                       COALESCE(sb.name, sd.code) AS stock_name
                FROM stock_daily sd
                LEFT JOIN stock_basic sb ON sd.code = sb.code
                WHERE sd.code = ANY(:codes)
                  AND sd.trade_date = (SELECT MAX(trade_date) FROM stock_daily)
                ORDER BY sd.code
            """), {"codes": codes}).fetchall()
    except Exception:
        rows = []

    rows_html = ""
    for r in rows:
        code, close, volume, amount, name = r[0], r[1] or 0, r[2] or 0, r[3] or 0, r[4]
        rows_html += f"""<tr style="cursor:pointer" onclick="document.getElementById('kline-panel').setAttribute('hx-get','/api/kline/{code}');htmx.process(document.getElementById('kline-panel'));document.getElementById('kline-panel').dispatchEvent(new Event('loadKline'));">
            <td>{code}</td>
            <td style="max-width:100px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{name}</td>
            <td>{close:.2f}</td>
            <td>{volume:.0f}</td>
            <td>{amount:.0f}</td>
        </tr>"""

    html = f"""<table>
    <thead><tr><th>代码</th><th>名称</th><th>现价</th><th>成交量</th><th>成交额</th></tr></thead>
    <tbody>{rows_html}</tbody>
    </table>"""
    return HTMLResponse(html)


@router.post("/add-watch/{code}")
async def add_to_watchlist(code: str, group: str = "default"):
    """Add a code to the watchlist group."""
    watchlist = _load_watchlist()
    if group not in watchlist:
        watchlist[group] = []
    code = code.strip().upper()
    if code in watchlist[group]:
        return HTMLResponse(f"<span style='color:#ffb74d;'>{code} 已在自选中</span>")
    watchlist[group].append(code)
    _save_watchlist(watchlist)
    return HTMLResponse(f"<span style='color:#66bb6a;'>{code} 已加入 {group} 自选</span>")


@router.get("/kline/{code}")
async def get_kline(code: str):
    engine = get_engine()
    try:
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT trade_date, open, high, low, close, volume
                FROM stock_daily WHERE code = :code
                ORDER BY trade_date DESC LIMIT 1000
            """), {"code": code}).fetchall()
    except Exception:
        rows = []

    if not rows:
        return HTMLResponse("<p style='color:#999;padding:20px;'>无K线数据</p>")

    rows = list(reversed(rows))
    dates = [str(r[0]) for r in rows]
    # ECharts candlestick format: [open, close, low, high]
    ohlc = [[float(r[1] or 0), float(r[4] or 0), float(r[3] or 0), float(r[2] or 0)] for r in rows]
    volumes = [int(r[5] or 0) for r in rows]
    # Pre-compute volume colors: red for up (close >= open), green for down
    vol_colors = ["#c62828" if o[1] >= o[0] else "#2e7d32" for o in ohlc]

    # Default dataZoom to show last ~250 bars (~1 year); user can scroll back
    total_bars = len(dates)
    zoom_end = 100
    zoom_start = max(0, 100 - (250 / max(total_bars, 1)) * 100)

    option = {
        "grid": [{"left": "8%", "right": "2%", "top": "5%", "height": "55%"},
                 {"left": "8%", "right": "2%", "top": "70%", "height": "20%"}],
        "xAxis": [{"data": dates, "axisLabel": {"show": False}, "axisLine": {"lineStyle": {"color": "#ddd"}},
                    "axisTick": {"show": False}},
                  {"data": dates, "axisLabel": {"rotate": 0, "color": "#999", "fontSize": 10},
                   "axisLine": {"lineStyle": {"color": "#ddd"}}}],
        "yAxis": [{"scale": True, "axisLine": {"lineStyle": {"color": "#ddd"}},
                    "splitLine": {"lineStyle": {"color": "#f0f0f0"}},
                    "axisLabel": {"color": "#666"}},
                  {"scale": True, "axisLine": {"lineStyle": {"color": "#ddd"}},
                    "splitLine": {"lineStyle": {"color": "#f0f0f0"}},
                    "axisLabel": {"color": "#666", "fontSize": 10}}],
        "series": [
            {"name": code, "type": "candlestick", "data": ohlc, "xAxisIndex": 0, "yAxisIndex": 0,
             "itemStyle": {"color": "#c62828", "color0": "#2e7d32",
                           "borderColor": "#c62828", "borderColor0": "#2e7d32",
                           "borderWidth": 1}},
            {"name": "成交量", "type": "bar", "data": volumes, "xAxisIndex": 1, "yAxisIndex": 1,
             "itemStyle": {"color": "#90caf9"}},
        ],
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "cross"}},
        "dataZoom": [
            {"type": "inside", "xAxisIndex": [0, 1], "start": zoom_start, "end": zoom_end},
            {"type": "slider", "xAxisIndex": [0, 1], "start": zoom_start, "end": zoom_end,
             "height": 20, "bottom": 5},
        ],
    }

    html = f"""<div id="kline-chart" style="width:100%;height:500px;"></div>
<script>
(function(){{
    var el=document.getElementById('kline-chart');
    if(el && typeof echarts!=='undefined'){{
        if(el._echart) el._echart.dispose();
        var c=echarts.init(el);
        c.setOption({json.dumps(option)});
        el._echart=c;
        window.addEventListener('resize',function(){{c.resize();}});
    }}
}})();
</script>"""
    return HTMLResponse(html)


@router.get("/backtest-list")
async def list_backtests(strategy: str = "", quality: str = ""):
    """Return HTML table of backtest results with key metrics."""
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
                       br.quality_flags, br.id, br.metrics_json
                FROM backtest_results br
                JOIN strategy_versions sv ON br.version_id = sv.id
                JOIN strategy_configs sc ON sv.strategy_id = sc.id
                WHERE {' AND '.join(where)}
                ORDER BY br.created_at DESC LIMIT 50
            """), params).fetchall()
    except Exception:
        rows = []

    if not rows:
        return HTMLResponse("""<div style="padding:40px;text-align:center;color:#999;">
            <p>暂无回测数据</p><p style="font-size:12px;">运行 scripts/run_ml_backtest.py 生成回测结果</p>
        </div>""")

    rows_html = ""
    for r in rows:
        name, version, start, end, quality, flags, bt_id, metrics_raw = r
        # Parse metrics for key display values
        metrics = {}
        if metrics_raw:
            try:
                metrics = json.loads(str(metrics_raw)) if isinstance(metrics_raw, str) else metrics_raw
            except Exception:
                pass
        ann_ret = metrics.get("annual_return", 0) or 0
        mdd = metrics.get("max_drawdown", 0) or 0
        sharpe = metrics.get("sharpe", 0) or 0
        badge_class = f"badge-{quality}"
        ret_color = "#c62828" if ann_ret > 0 else "#2e7d32"
        rows_html += f"""<tr>
            <td>{name}</td><td>{version}</td><td>{start} ~ {end}</td>
            <td style="color:{ret_color};font-weight:600;">{ann_ret*100:+.1f}%</td>
            <td style="color:#c62828;">{mdd*100:.1f}%</td>
            <td>{sharpe:.2f}</td>
            <td><span class="badge {badge_class}">{quality}</span></td>
            <td><button class="mock-button" hx-get="/api/backtest-detail/{bt_id}" hx-target="#detail-panel" hx-swap="innerHTML" onclick="document.getElementById('detail-panel').style.display='block';">查看详情</button></td>
        </tr>"""

    html = f"""<table>
    <thead><tr><th>策略</th><th>版本</th><th>区间</th><th>年化收益</th><th>最大回撤</th><th>Sharpe</th><th>质量</th><th>操作</th></tr></thead>
    <tbody>{rows_html}</tbody></table>"""
    return HTMLResponse(html)


@router.get("/backtest-equity/{backtest_id}")
async def get_equity_curve(backtest_id: int):
    engine = get_engine()
    try:
        with engine.connect() as conn:
            row = conn.execute(text(
                "SELECT equity_curve_json, metrics_json, quality, quality_flags FROM backtest_results WHERE id = :id"
            ), {"id": backtest_id}).fetchone()
    except Exception:
        row = None

    if not row:
        return HTMLResponse("<p style='color:#999;padding:20px;'>无数据</p>")

    curve = row[0] if isinstance(row[0], dict) else (json.loads(str(row[0])) if row[0] else {})
    metrics = row[1] if isinstance(row[1], dict) else (json.loads(str(row[1])) if row[1] else {})
    quality = row[2] or "unknown"
    flags = row[3] or []

    # Equity curve chart
    dates = list(curve.keys()) if isinstance(curve, dict) and curve else []
    values = list(curve.values()) if isinstance(curve, dict) and curve else []

    chart_html = ""
    if dates and values:
        # Clean dates (remove time component)
        clean_dates = [d.split(" ")[0] if " " in str(d) else str(d) for d in dates]
        option = {
            "xAxis": {"data": clean_dates, "axisLabel": {"color": "#888", "rotate": 30, "fontSize": 11}},
            "yAxis": {"scale": True, "axisLabel": {"color": "#888", "fontSize": 11}},
            "series": [{"type": "line", "data": values, "smooth": True,
                         "lineStyle": {"color": "#1976d2"}, "areaStyle": {"color": "rgba(25,118,210,0.08)"}}],
            "tooltip": {"trigger": "axis"},
            "grid": {"left": "10%", "right": "5%", "top": "5%", "bottom": "15%"},
        }
        chart_html = f"""<div id="equity-chart" style="width:100%;height:400px;" data-chart='{json.dumps(option)}'></div>"""
    else:
        chart_html = "<p style='color:#999;padding:20px;'>暂无权益曲线数据（旧版记录）</p>"

    # Metrics display
    qclass = f"badge-{quality}"
    flags_html = ""
    if flags:
        flags_html = "<div style='margin-top:8px;'>" + "".join(
            f"<span style='display:inline-block;background:#fff3e0;color:#e65100;padding:2px 8px;border-radius:4px;font-size:11px;margin:2px 4px;'>{f}</span>"
            for f in flags) + "</div>"

    # Format key metrics
    metrics_rows = []
    key_metrics = [
        ("win_rate", "胜率", ".2%"),
        ("n_trades", "验证窗口数", "d"),
        ("n_params", "因子数", "d"),
        ("adjusted_sharpe", "调整后夏普", ".4f"),
        ("start_date", "起始日", ""),
        ("end_date", "结束日", ""),
    ]
    for k, label, fmt in key_metrics:
        v = metrics.get(k, "-")
        if isinstance(v, float) and fmt:
            if fmt == ".2%":
                v = f"{v*100:.1f}%"
            elif fmt == ".4f":
                v = f"{v:.4f}"
            elif fmt == "d":
                v = int(v)
        metrics_rows.append(f"<tr><td style='color:#888;'>{label}</td><td>{v}</td></tr>")

    return HTMLResponse(f"""
    {chart_html}
    <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-top:16px;">
        <div>
            <span class="badge {qclass}" style="margin-right:8px;">{quality}</span>
            {flags_html}
        </div>
        <div>
            <table style="font-size:13px;"><tbody>{"".join(metrics_rows)}</tbody></table>
        </div>
    </div>
    """)


@router.get("/backtest-detail/{backtest_id}")
async def get_backtest_detail(backtest_id: int):
    """综合回测详情：双线权益曲线（策略+基准）+ 全量指标 + 因子构成。"""
    engine = get_engine()

    # Load backtest record with strategy info
    try:
        with engine.connect() as conn:
            row = conn.execute(text("""
                SELECT br.equity_curve_json, br.metrics_json, br.daily_returns_json,
                       br.quality, br.quality_flags, br.start_date, br.end_date,
                       sc.name AS strategy_name, sc.type AS strategy_type,
                       sv.version
                FROM backtest_results br
                JOIN strategy_versions sv ON br.version_id = sv.id
                JOIN strategy_configs sc ON sv.strategy_id = sc.id
                WHERE br.id = :id
            """), {"id": backtest_id}).fetchone()
    except Exception:
        return HTMLResponse("<p style='color:#ef5350;padding:20px;'>查询失败</p>")

    if not row:
        return HTMLResponse("<p style='color:#999;padding:20px;'>无数据</p>")

    equity_curve = row[0] if isinstance(row[0], dict) else (json.loads(str(row[0])) if row[0] else {})
    metrics = row[1] if isinstance(row[1], dict) else (json.loads(str(row[1])) if row[1] else {})
    daily_returns_json = row[2] if isinstance(row[2], dict) else (json.loads(str(row[2])) if row[2] else {})
    quality = row[3] or "unknown"
    flags = row[4] or []
    start_date = str(row[5] or "")
    end_date = str(row[6] or "")
    strategy_name = row[7] or "Unknown"
    strategy_type = row[8] or ""
    version = row[9] or ""

    # Load benchmark data (上证指数 000001)
    bench_curve = {}
    try:
        with engine.connect() as conn:
            bench_rows = conn.execute(text("""
                SELECT trade_date, close FROM index_daily
                WHERE code = '000001' AND trade_date BETWEEN :s AND :e
                ORDER BY trade_date
            """), {"s": start_date, "e": end_date}).fetchall()
        if bench_rows:
            base_close = float(bench_rows[0][1])
            if base_close > 0:
                bench_curve = {str(r[0]): round(float(r[1]) / base_close, 6) for r in bench_rows}
    except Exception:
        pass

    engine.dispose()

    # ── Compute metrics ──────────────────────────────────────────────
    eq_dates = list(equity_curve.keys()) if equity_curve else []
    eq_vals = list(equity_curve.values()) if equity_curve else []

    # Strategy total return
    strat_total_return = float(eq_vals[-1] / eq_vals[0] - 1) if len(eq_vals) >= 2 else 0.0
    n_days = max(len(eq_vals), 1)
    years = max(n_days / 252, 0.2)
    strat_annual = float((1 + strat_total_return) ** (1 / years) - 1)

    # Benchmark total return
    bench_vals = list(bench_curve.values())
    bench_total = float(bench_vals[-1] / bench_vals[0] - 1) if len(bench_vals) >= 2 else 0.0
    bench_annual = float((1 + bench_total) ** (1 / years) - 1)

    # Excess return (alpha)
    excess_return = strat_total_return - bench_total

    # Sharpe from daily_returns
    dr_vals = [v for v in daily_returns_json.values() if v != 0]
    if dr_vals:
        strat_sharpe = float(np.mean(dr_vals) / np.std(dr_vals) * np.sqrt(252)) if np.std(dr_vals) > 0 else 0.0
    else:
        strat_sharpe = float(metrics.get("sharpe", 0) or 0)

    # Benchmark Sharpe
    if bench_vals and len(bench_vals) >= 2:
        bench_rets = [float(bench_vals[i] / bench_vals[i-1] - 1) for i in range(1, len(bench_vals))]
        bench_sharpe = float(np.mean(bench_rets) / np.std(bench_rets) * np.sqrt(252)) if np.std(bench_rets) > 0 else 0.0
    else:
        bench_sharpe = 0.0

    # Max drawdown
    strat_mdd = float(metrics.get("max_drawdown", 0) or 0)
    if strat_mdd == 0 and eq_vals:
        peak = eq_vals[0]
        for v in eq_vals:
            if v > peak:
                peak = v
            dd = (peak - v) / peak
            if dd > strat_mdd:
                strat_mdd = dd

    # Benchmark max drawdown
    bench_mdd = 0.0
    if bench_vals:
        peak = bench_vals[0]
        for v in bench_vals:
            if v > peak:
                peak = v
            dd = (peak - v) / peak
            if dd > bench_mdd:
                bench_mdd = dd

    # Win rate
    win_rate = float(metrics.get("win_rate", 0) or 0)
    n_trades = int(metrics.get("n_trades", 0) or 0)
    n_wins = int(metrics.get("n_wins", 0) or 0)
    n_losses = int(metrics.get("n_losses", 0) or 0)
    n_windows = int(metrics.get("walk_forward_windows", 0) or 0)

    # Daily signal tracker info
    signal_tracker = metrics.get("daily_signal_tracker", {})
    ic_series = metrics.get("daily_ic_series", {})
    position_history = metrics.get("position_history", [])

    # ── Build ECharts chart ──────────────────────────────────────────
    chart_html = ""
    if eq_dates and eq_vals:
        clean_dates = [d.split(" ")[0] if " " in str(d) else str(d) for d in eq_dates]

        # Benchmark data aligned to strategy dates
        bench_on_dates = []
        for d in clean_dates:
            v = bench_curve.get(d)
            if v is None:
                # Use closest previous value
                prev_val = None
                for bd in sorted(bench_curve.keys()):
                    if str(bd) <= d:
                        prev_val = bench_curve[bd]
                bench_on_dates.append(prev_val if prev_val is not None else "-")
            else:
                bench_on_dates.append(v)

        # ── 标注事件（窗口切换、因子调整、重训、风控） ──
        EVENT_STYLE = {
            "window_transition": {"color": "#1565c0", "label": "窗口"},
            "factor_discover":    {"color": "#2e7d32", "label": "+因子"},
            "factor_eliminate":   {"color": "#c62828", "label": "-因子"},
            "model_retrain":      {"color": "#e65100", "label": "重训"},
            "risk_liquidate":     {"color": "#b71c1c", "label": "清仓"},
            "risk_reduce":        {"color": "#f9a825", "label": "减仓"},
            "risk_index_crash":   {"color": "#6a1b9a", "label": "空仓"},
        }
        ann_events = metrics.get("annotation_events", []) or []
        mark_lines = []
        from collections import Counter
        event_counts: dict[str, int] = Counter()
        for evt in ann_events:
            d = str(evt.get("date", ""))
            t = evt.get("type", "")
            if d not in clean_dates and d not in set(clean_dates):
                continue  # date not in chart range
            style = EVENT_STYLE.get(t, {"color": "#999", "label": t[:3]})
            event_counts[t] += 1
            n = event_counts[t]
            mark_lines.append({
                "xAxis": d,
                "lineStyle": {"color": style["color"], "type": "dashed", "width": 1},
                "label": {
                    "show": True,
                    "position": "insideStartTop" if n % 2 == 0 else "insideEndTop",
                    "formatter": f"{style['label']}",
                    "fontSize": 9,
                    "color": style["color"],
                    "backgroundColor": "rgba(255,255,255,0.85)",
                    "padding": [1, 4],
                },
            })

        option = {
            "title": {"text": f"{strategy_name} vs 上证指数", "left": "center",
                      "textStyle": {"fontSize": 14, "color": "#333"}},
            "legend": {"data": ["策略权益", "上证指数"], "bottom": 0,
                       "textStyle": {"color": "#666"}},
            "xAxis": {"data": clean_dates, "axisLabel": {"rotate": 30, "fontSize": 10, "color": "#999"}},
            "yAxis": {"type": "value", "min": "dataMin",
                      "axisLabel": {"formatter": "{value}", "color": "#666"},
                      "splitLine": {"lineStyle": {"color": "#f0f0f0"}}},
            "series": [
                {"name": "策略权益", "type": "line", "data": eq_vals,
                 "smooth": True, "lineStyle": {"color": "#1976d2", "width": 2},
                 "itemStyle": {"color": "#1976d2"},
                 "markLine": {
                     "silent": True,
                     "symbol": "none",
                     "data": mark_lines,
                 } if mark_lines else None},
                {"name": "上证指数", "type": "line", "data": bench_on_dates,
                 "smooth": True, "lineStyle": {"color": "#9e9e9e", "width": 1.5, "type": "dashed"},
                 "itemStyle": {"color": "#9e9e9e"}},
            ],
            "tooltip": {"trigger": "axis",
                        "axisPointer": {"type": "cross"}},
            "toolbox": {"feature": {"dataZoom": {"yAxisIndex": "none"},
                                    "restore": {}, "saveAsImage": {}}},
            "dataZoom": [{"type": "inside", "start": 0, "end": 100},
                         {"type": "slider", "start": 0, "end": 100, "height": 20, "bottom": 25}],
            "grid": {"left": "8%", "right": "4%", "top": "12%", "bottom": "18%"},
        }
        chart_html = f"""<div id="detail-chart" style="width:100%;height:450px;"></div>
<script>
(function(){{
    var el=document.getElementById('detail-chart');
    if(el && typeof echarts!=='undefined'){{
        if(el._echart) el._echart.dispose();
        var c=echarts.init(el);
        c.setOption({json.dumps(option)});
        el._echart=c;
        window.addEventListener('resize',function(){{c.resize();}});
    }}
}})();
</script>"""
    else:
        chart_html = "<p style='color:#999;padding:20px;text-align:center;'>暂无权益曲线数据</p>"

    # ── Build metrics table ──────────────────────────────────────────
    def fmt_pct(v):
        return f"{v*100:+.2f}%" if isinstance(v, (int, float)) else str(v)

    def fmt_num(v):
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    metrics_table = f"""
    <table style="font-size:13px;width:100%;">
    <thead><tr><th>指标</th><th>策略</th><th>基准(上证指数)</th></tr></thead>
    <tbody>
    <tr><td>累计收益</td><td style="font-weight:600;">{fmt_pct(strat_total_return)}</td><td>{fmt_pct(bench_total)}</td></tr>
    <tr><td>年化收益</td><td style="font-weight:600;">{fmt_pct(strat_annual)}</td><td>{fmt_pct(bench_annual)}</td></tr>
    <tr><td>超额收益(Alpha)</td><td style="font-weight:600;color:{'#c62828' if excess_return > 0 else '#2e7d32'};">{fmt_pct(excess_return)}</td><td>-</td></tr>
    <tr><td>Sharpe比率</td><td style="font-weight:600;">{fmt_num(strat_sharpe)}</td><td>{fmt_num(bench_sharpe)}</td></tr>
    <tr><td>最大回撤</td><td style="font-weight:600;color:#c62828;">{fmt_pct(strat_mdd)}</td><td>{fmt_pct(bench_mdd)}</td></tr>
    <tr><td>日度胜率</td><td>{win_rate*100:.1f}%</td><td>-</td></tr>
    <tr><td>盈利/亏损日</td><td>{n_wins} / {n_losses}</td><td>-</td></tr>
    <tr><td>日度调仓次数</td><td>{n_trades}</td><td>-</td></tr>
    <tr><td>Walk-Forward窗口</td><td>{n_windows}</td><td>-</td></tr>
    <tr><td>回测区间</td><td colspan="2">{start_date} ~ {end_date} ({n_days} 个交易日)</td></tr>
    </tbody></table>"""

    # ── Build factor/params section ──────────────────────────────────
    factor_html = ""
    factor_cols = metrics.get("factor_cols", [])
    active_cols = metrics.get("active_cols", [])
    initial_factors = metrics.get("initial_factors", [])
    strategy_params = metrics.get("strategy_params", {})

    if strategy_type == "ml" and factor_cols:
        factor_html = "<h4 style='margin:16px 0 8px;'>因子构成</h4>"
        if initial_factors:
            factor_html += f"<p style='font-size:12px;color:#999;margin-bottom:4px;'>初始因子池: {len(initial_factors)} 个 → IC筛选+正交后: {len(factor_cols)} 个</p>"
        factor_html += "<div style='display:flex;flex-wrap:wrap;gap:4px;'>"
        for f in factor_cols:
            factor_html += f"<span style='background:#e3f2fd;color:#1565c0;padding:2px 8px;border-radius:4px;font-size:11px;'>{f}</span>"
        factor_html += "</div>"
        if active_cols and len(active_cols) != len(factor_cols):
            factor_html += "<p style='font-size:12px;color:#999;margin-top:4px;'>最终活跃因子: " + ", ".join(active_cols) + "</p>"
    elif strategy_type == "static" and strategy_params:
        factor_html = "<h4 style='margin:16px 0 8px;'>策略参数</h4>"
        factor_html += "<div style='display:flex;flex-wrap:wrap;gap:4px;'>"
        for k, v in strategy_params.items():
            factor_html += f"<span style='background:#f3e5f5;color:#7b1fa2;padding:2px 8px;border-radius:4px;font-size:11px;'>{k}={v}</span>"
        factor_html += "</div>"

    # ── Quality badges ──────────────────────────────────────────────
    qclass = f"badge-{quality}"
    flags_html = ""
    if flags:
        flags_html = "<div style='margin-top:8px;'>" + "".join(
            f"<span style='display:inline-block;background:#fff3e0;color:#e65100;padding:2px 8px;border-radius:4px;font-size:11px;margin:2px 4px;'>{f}</span>"
            for f in flags) + "</div>"

    # ── Signal quality section (ML strategies only) ──────────────────
    signal_html = ""
    if signal_tracker and strategy_type == "ml":
        signal_html = "<h4 style='margin:16px 0 8px;'>信号质量追踪</h4>"
        signal_html += "<p style='font-size:12px;color:#999;margin-bottom:4px;'>"
        signal_html += f"IC均值: {signal_tracker.get('rolling_ic_20d', 0):.4f} | "
        signal_html += f"信号等级: <b>{signal_tracker.get('signal_level', 'N/A')}</b> | "
        signal_html += f"跟踪天数: {signal_tracker.get('total_days', 0)}"
        signal_html += "</p>"

        # Build IC chart if data available
        daily_ic = ic_series.get("daily_ic", [])
        rolling_ic = ic_series.get("rolling_ic", [])
        if daily_ic:
            import json as _json
            # Sample every 5th point for performance
            sampled_daily = daily_ic[::5]
            sampled_rolling = rolling_ic[::5]
            ic_option = {
                "title": {"text": "日度 Rank IC (每5日采样)", "textStyle": {"fontSize": 12, "color": "#555"}},
                "xAxis": {"data": list(range(len(sampled_daily))), "show": False},
                "yAxis": {"type": "value", "name": "IC", "axisLabel": {"fontSize": 10}},
                "series": [
                    {"name": "日度IC", "type": "bar", "data": sampled_daily,
                     "itemStyle": {"color": "#bbdefb"}, "barWidth": "80%"},
                    {"name": "滚动IC(20日)", "type": "line", "data": sampled_rolling,
                     "lineStyle": {"color": "#1976d2", "width": 1.5}, "symbol": "none"},
                ],
                "legend": {"bottom": 0, "textStyle": {"fontSize": 10}},
                "grid": {"left": "12%", "right": "4%", "top": "15%", "bottom": "15%"},
            }
            signal_html += f"""<div id="signal-ic-chart" style="width:100%;height:200px;"></div>
<script>
(function(){{
    var el=document.getElementById('signal-ic-chart');
    if(el && typeof echarts!=='undefined'){{
        var c=echarts.init(el);
        c.setOption({_json.dumps(ic_option)});
        window.addEventListener('resize',function(){{c.resize();}});
    }}
}})();
</script>"""

    # ── Position history section ──────────────────────────────────────
    pos_html = ""
    if position_history and strategy_type == "ml":
        # Build name lookup from stock_basic
        name_map = {}
        all_pos_codes = set()
        for p in position_history[-20:]:
            all_pos_codes.update(p.get("codes", []))
        if all_pos_codes:
            try:
                with engine.connect() as conn:
                    cl = ",".join([f"'{c}'" for c in all_pos_codes])
                    names = conn.execute(text(f"SELECT code, name FROM stock_basic WHERE code IN ({cl})")).fetchall()
                    name_map = {r[0]: r[1] for r in names}
            except Exception:
                pass

        pos_html = "<h4 style='margin:16px 0 8px;'>近期持仓明细 (最近20个调仓日)</h4>"
        recent = position_history[-20:]
        pos_html += """<table style="font-size:11px;width:100%;border-collapse:collapse;"><thead>
        <tr style="background:#f5f5f5;"><th style="padding:4px;text-align:left;">日期</th>
        <th style="padding:4px;text-align:left;">持仓 (代码+名称)</th>
        <th style="padding:4px;text-align:center;">只数</th>
        <th style="padding:4px;text-align:right;">日收益</th></tr></thead><tbody>"""
        for p in reversed(recent):
            codes = p.get("codes", [])
            codes_display = " ".join(f"{c}({name_map.get(c, '?')})" for c in codes)
            ret = p.get("daily_ret", p.get("ret", 0))
            ret_color = "#c62828" if ret > 0 else "#2e7d32" if ret < 0 else "#666"
            pos_html += f"""<tr><td style="padding:4px;">{p['date']}</td>
            <td style="padding:4px;font-size:10px;">{codes_display}</td>
            <td style="padding:4px;text-align:center;">{len(codes)}</td>
            <td style="padding:4px;text-align:right;color:{ret_color};">{ret*100:+.2f}%</td></tr>"""
        pos_html += "</tbody></table>"

    # ── Assemble final HTML ──────────────────────────────────────────
    return HTMLResponse(f"""
    <div style="display:grid; grid-template-columns: 1fr 380px; gap: 20px;">
        <div>
            <div style="display:flex; align-items:center; gap:12px; margin-bottom:12px;">
                <h3 style="margin:0;">{strategy_name}</h3>
                <span style="color:#999;font-size:12px;">v{version}</span>
                <span class="badge {qclass}">{quality}</span>
            </div>
            {flags_html}
            {chart_html}
            {signal_html}
            {pos_html}
            {factor_html}
        </div>
        <div style="background:#fafafa; border-radius:8px; padding:16px; border:1px solid #eee;">
            <h4 style="margin:0 0 12px 0;color:#555;">指标详情</h4>
            {metrics_table}
        </div>
    </div>
    """)


@router.get("/factor-overview")
async def get_factor_overview():
    """因子监控：可用性、最新数据日期"""
    engine = get_engine()
    try:
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT factor_name, trade_date, data_ready_at, data_source, latency_ms
                FROM factor_availability
                ORDER BY trade_date DESC, factor_name
            """)).fetchall()
    except Exception:
        rows = []

    if not rows:
        return HTMLResponse("<p style='color:#999;padding:20px;'>暂无因子数据。运行因子计算 (factors/engine.py) 生成因子数据。</p>")

    # Group by trade_date
    from collections import defaultdict
    by_date = defaultdict(list)
    for r in rows:
        fn, td, dr, ds, lat = r
        by_date[str(td)].append({"name": fn, "ready_at": str(dr), "source": ds, "latency": lat})

    html = "<h4 style='margin-bottom:12px;'>因子就绪状态</h4>"
    html += "<table><thead><tr><th>交易日</th><th>因子数</th><th>最早就绪时间</th><th>详情</th></tr></thead><tbody>"
    for td in sorted(by_date.keys(), reverse=True)[:10]:
        factors = by_date[td]
        earliest = min(f["ready_at"] for f in factors)
        names = ", ".join(f["name"][:30] for f in factors[:5])
        more = f" +{len(factors)-5}个" if len(factors) > 5 else ""
        html += f"<tr><td>{td}</td><td>{len(factors)}</td><td>{earliest[:19]}</td><td style='font-size:11px;'>{names}{more}</td></tr>"
    html += "</tbody></table>"

    # Factor lineage summary
    try:
        with engine.connect() as conn:
            lineages = conn.execute(text(
                "SELECT factor_name, source_fields, last_validated_at FROM factor_lineage ORDER BY factor_name"
            )).fetchall()

        if lineages:
            html += "<h4 style='margin:20px 0 12px;'>因子血缘</h4>"
            html += "<table><thead><tr><th>因子名</th><th>上游字段</th><th>上次校验</th></tr></thead><tbody>"
            for lr in lineages:
                fn, sf, lv = lr
                html += f"<tr><td>{fn}</td><td style='font-size:11px;'>{', '.join(sf)}</td><td>{str(lv)[:19]}</td></tr>"
            html += "</tbody></table>"
    except Exception:
        pass

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
        return HTMLResponse("""<div style="padding:40px;text-align:center;color:#999;">
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
            <td><button class="mock-button" hx-get="/api/paper-run/{rid}?account_id=15" hx-target="#paper-detail" hx-swap="innerHTML">查看</button></td>
        </tr>"""
    html += "</tbody></table>"
    return HTMLResponse(html)


@router.get("/paper-run/{run_id}")
async def get_paper_run_detail(run_id: int, account_id: int = 15):
    engine = get_engine()
    try:
        with engine.connect() as conn:
            positions = conn.execute(text("""
                SELECT stock_code, entry_date, entry_price, exit_date, exit_price, quantity, pnl, pnl_pct
                FROM paper_positions WHERE run_id = :rid ORDER BY entry_date DESC LIMIT 50
            """), {"rid": run_id}).fetchall()
            signals = conn.execute(text("""
                SELECT ps.signal_date, ps.stock_code, ps.predicted_score, ps.rank
                FROM paper_signals ps
                WHERE ps.run_id = :rid
                AND EXISTS (SELECT 1 FROM paper_positions pp WHERE pp.run_id = ps.run_id AND pp.entry_date = ps.signal_date)
                ORDER BY ps.signal_date DESC LIMIT 30
            """), {"rid": run_id}).fetchall()
            # 股票名称映射
            all_codes = list(set(p[0] for p in positions) | set(s[1] for s in signals))
            name_map = {}
            if all_codes:
                cl = ",".join([f"'{c}'" for c in all_codes])
                names = conn.execute(text(f"SELECT code, name FROM stock_basic WHERE code IN ({cl})")).fetchall()
                name_map = {r[0]: r[1] for r in names}
            # 最新收盘价（用于计算当日涨幅）
            price_map = {}
            if all_codes:
                prices = conn.execute(text(f"""
                    SELECT DISTINCT ON (code) code, close,
                           (close - LAG(close) OVER (PARTITION BY code ORDER BY trade_date))
                           / NULLIF(LAG(close) OVER (PARTITION BY code ORDER BY trade_date), 0) * 100 as chg_pct
                    FROM stock_daily WHERE code IN ({cl})
                    ORDER BY code, trade_date DESC
                """)).fetchall()
                for r in prices:
                    price_map[r[0]] = (float(r[1]), float(r[2]) if r[2] else 0)
    except Exception:
        positions = []
        signals = []
        name_map = {}
        price_map = {}

    # 已平仓交易表
    pos_html = "<table><thead><tr><th>代码</th><th>名称</th><th>入场日</th><th>入场价</th><th>出场日</th><th>出场价</th><th>数量</th><th>盈亏</th><th>盈亏%</th></tr></thead><tbody>"
    for p in positions:
        code, ed, ep, xd, xp, qty, pnl, pct = p
        pnl_class = "up" if (pnl or 0) > 0 else "down"
        pos_html += f"""<tr>
            <td>{code}</td><td>{name_map.get(code, '?')}</td><td>{ed}</td><td>{ep:.2f}</td><td>{xd or '-'}</td><td>{xp or '-'}</td>
            <td>{qty}</td>
            <td class="{pnl_class}">{pnl:,.0f}</td>
            <td class="{pnl_class}">{pct:+.2f}%</td>
        </tr>"""
    pos_html += "</tbody></table>"

    sig_html = "<table><thead><tr><th>日期</th><th>代码</th><th>名称</th><th>评分</th><th>排名</th></tr></thead><tbody>"
    for s in signals:
        sd, sc, score, rank = s
        sig_html += f"<tr><td>{sd}</td><td>{sc}</td><td>{name_map.get(sc, '?')}</td><td>{score:.4f}</td><td>{rank}</td></tr>"
    sig_html += "</tbody></table>"

    open_pos = [p for p in positions if p[3] is None]
    closed_pos = [p for p in positions if p[3] is not None]

    total_pnl = sum((p[6] or 0) for p in closed_pos)
    wins = sum(1 for p in closed_pos if (p[7] or 0) > 0)

    # 账户概览
    pnl_summary = ""
    reg_label = "—"
    try:
        with engine.connect() as conn:
            daily = conn.execute(text("""
                SELECT trade_date, cash, position_value, total_value, daily_return, drawdown
                FROM paper_daily_pnl WHERE account_id = :aid
                ORDER BY trade_date DESC LIMIT 1
            """), {"aid": account_id}).fetchone()
            if daily:
                d, cash, pv, tv, dr, dd = daily
                # 获取当前市场状态
                try:
                    idx_row = conn.execute(text(
                        "SELECT close, AVG(close) OVER (ORDER BY trade_date ROWS BETWEEN 249 PRECEDING AND CURRENT ROW) as ma250, "
                        "(close - LAG(close,20) OVER (ORDER BY trade_date)) / NULLIF(LAG(close,20) OVER (ORDER BY trade_DATE), 0) as ret_20 "
                        "FROM index_daily WHERE code='000001' ORDER BY trade_date DESC LIMIT 1"
                    )).fetchone()
                    if idx_row and idx_row[1] and idx_row[2] is not None:
                        c, ma, r20 = float(idx_row[0]), float(idx_row[1]), float(idx_row[2])
                        if c > ma and r20 > 0.03: reg_label = "强牛"
                        elif c > ma and r20 > 0: reg_label = "弱牛"
                        elif c < ma and r20 < -0.03: reg_label = "快熊"
                        elif c < ma and r20 < 0: reg_label = "慢熊"
                        else: reg_label = "震荡"
                except Exception:
                    pass
            pnl_summary = f"""<div class="card"><h3>账户概览 <span style="font-size:12px;color:#1976d2;">市场: {reg_label}</span></h3>
            <p>日期: {d} | 现金: {cash:,.0f} | 持仓市值: {pv:,.0f}</p>
            <p>总资产: <strong>{tv:,.0f}</strong> | 日收益: {dr:+.2%} | 回撤: {dd:.2%}</p>
            </div>"""
    except Exception:
        pass

    # 当前持仓（含名称+当日涨幅+份额+市值+浮动盈亏）
    open_html = ""
    if open_pos:
        open_html = "<table><thead><tr><th>代码</th><th>名称</th><th>入场日</th><th>入场价</th><th>现价</th><th>股数</th><th>市值</th><th>浮动盈亏</th></tr></thead><tbody>"
        for p in open_pos:
            code, ed, ep, xd, xp, qty, pnl, pct = p
            price_info = price_map.get(code, (ep, 0))
            cur_price, chg_pct = price_info
            market_value = cur_price * qty
            cost_value = ep * qty
            float_pnl = (cur_price - ep) * qty if ep > 0 else 0
            float_pnl_pct = (cur_price / ep - 1) * 100 if ep > 0 else 0
            pnl_class = "up" if float_pnl > 0 else "down"
            open_html += f"""<tr>
                <td><strong>{code}</strong></td><td>{name_map.get(code, '?')}</td><td>{ed}</td>
                <td>{ep:.2f}</td><td>{cur_price:.2f}</td>
                <td>{qty}股</td><td>{market_value:,.0f}</td>
                <td class="{pnl_class}">{float_pnl:+,.0f} ({float_pnl_pct:+.2f}%)</td>
            </tr>"""
        total_mv = sum(p[5] * price_map.get(p[0], (p[2], 0))[0] for p in open_pos)
        open_html += "</tbody></table>"
        open_html += f"<p style='margin-top:8px;color:#888;'>总市值: {total_mv:,.0f} | 等权分配，每只约 {100/max(len(open_pos),1):.0f}% 仓位</p>"

    # ── 待执行信号（T+1：已生成但未入场） ──
    pending_html = ""
    try:
        with engine.connect() as conn:
            pending = conn.execute(text("""
                SELECT ps.signal_date, ps.stock_code, ps.predicted_score, ps.rank, sb.name
                FROM paper_signals ps
                LEFT JOIN stock_basic sb ON ps.stock_code = sb.code
                WHERE ps.run_id = :rid
                AND NOT EXISTS (
                    SELECT 1 FROM paper_positions pp
                    WHERE pp.run_id = ps.run_id AND pp.entry_date = ps.signal_date
                )
                ORDER BY ps.signal_date DESC, ps.rank
                LIMIT 15
            """), {"rid": run_id}).fetchall()
        if pending:
            pending_html = f"<div class='card'><h3>待执行信号 (T+1) <span style='font-size:11px;color:#e65100;'>共{len(pending)}条</span></h3>"
            pending_html += "<table><thead><tr><th>日期</th><th>代码</th><th>名称</th><th>评分</th><th>排名</th></tr></thead><tbody>"
            for p in pending:
                sd, sc, score, rank, nm = p
                pending_html += f"<tr><td>{sd}</td><td><strong>{sc}</strong></td><td>{nm or '?'}</td><td>{score:.4f}</td><td>#{rank}</td></tr>"
            pending_html += "</tbody></table></div>"
    except Exception:
        pass

    # 权益曲线 + 基准叠加 + 每日盈亏
    chart_html = ""
    try:
        with engine.connect() as conn:
            eq_rows = conn.execute(text("""
                SELECT trade_date, total_value, daily_return FROM paper_daily_pnl
                WHERE account_id = :aid ORDER BY trade_date
            """), {"aid": account_id}).fetchall()
        if len(eq_rows) > 1:
            import json as _json
            dates = [str(r[0]) for r in eq_rows]
            values = [float(r[1]) for r in eq_rows]
            daily_rets = [float(r[2] or 0) for r in eq_rows]
            start_val = values[0]

            # 基准：上证指数归一化
            benchmark_values = []
            try:
                bench_rows = conn.execute(text(f"""
                    SELECT trade_date, close FROM index_daily WHERE code='000001'
                    AND trade_date BETWEEN :d1 AND :d2 ORDER BY trade_date
                """), {"d1": eq_rows[0][0], "d2": eq_rows[-1][0]}).fetchall()
                if bench_rows and len(bench_rows) > 1:
                    b0 = float(bench_rows[0][1])
                    b_dates = {str(r[0]): float(r[1]) / b0 * start_val for r in bench_rows}
                    benchmark_values = [b_dates.get(d, None) for d in dates]
            except Exception:
                pass

            # 权益曲线（双线）
            series = [{"name": "策略净值", "type": "line", "data": values,
                        "lineStyle": {"color": "#26a69a"}, "areaStyle": {"color": "rgba(38,166,154,0.1)"}}]
            if any(v is not None for v in benchmark_values):
                series.append({"name": "上证基准", "type": "line", "data": benchmark_values,
                               "lineStyle": {"color": "#ccc", "type": "dashed"}, "areaStyle": {"opacity": 0}})

            eq_chart = _json.dumps({
                "tooltip": {"trigger": "axis"}, "legend": {"data": ["策略净值", "上证基准"]},
                "grid": {"left": 60, "right": 20, "top": 30, "bottom": 30},
                "xAxis": {"type": "category", "data": dates, "axisLabel": {"rotate": 45, "fontSize": 9}},
                "yAxis": {"type": "value", "name": "净值"},
                "series": series,
            })

            # 每日盈亏柱状图
            bar_colors = [{"value": v, "itemStyle": {"color": "#c62828" if v < 0 else "#2e7d32"}} for v in daily_rets]
            pnl_chart = _json.dumps({
                "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
                "grid": {"left": 60, "right": 20, "top": 20, "bottom": 30},
                "xAxis": {"type": "category", "data": dates, "axisLabel": {"rotate": 45, "fontSize": 9}},
                "yAxis": {"type": "value", "name": "日收益"},
                "series": [{"name": "日收益", "type": "bar", "data": daily_rets,
                            "itemStyle": {"color": "function(p){return p.data>=0?'#2e7d32':'#c62828';}"}}],
            })

            chart_html = (
                f'<div class="card"><h3>权益曲线</h3><div class="chart-container" data-chart=\'{eq_chart}\' style="width:100%;height:350px;"></div></div>'
                f'<div class="card"><h3>每日盈亏</h3><div class="chart-container" data-chart=\'{pnl_chart}\' style="width:100%;height:250px;"></div></div>'
            )
    except Exception:
        pass

    # 持仓行业分布
    sector_html = ""
    try:
        with engine.connect() as conn:
            sec_rows = conn.execute(text("""
                SELECT COALESCE(si.industry_sw1, '未知') as sector, SUM(pp.quantity) as total_qty
                FROM paper_positions pp
                LEFT JOIN stock_industry si ON pp.stock_code = si.code
                WHERE pp.run_id = :rid AND pp.exit_date IS NULL
                GROUP BY COALESCE(si.industry_sw1, '未知') ORDER BY total_qty DESC
            """), {"rid": run_id}).fetchall()
        if sec_rows:
            import json as _json
            sec_data = [{"name": r[0], "value": int(r[1])} for r in sec_rows]
            sec_chart = _json.dumps({
                "tooltip": {"trigger": "item", "formatter": "{b}: {c} 股 ({d}%)"},
                "series": [{"type": "pie", "radius": ["30%", "70%"], "data": sec_data,
                            "label": {"fontSize": 10}, "emphasis": {"itemStyle": {"shadowBlur": 10}}}],
            })
            sector_html = f'<div class="card"><h3>持仓行业分布</h3><div class="chart-container" data-chart=\'{sec_chart}\' style="width:100%;height:250px;"></div></div>'
    except Exception:
        pass

    # ── 汇总 + 历史日收益 ──
    summary_card = ""
    try:
        with engine.connect() as conn:
            daily_rows = conn.execute(text("""
                SELECT trade_date, total_value, daily_return, drawdown
                FROM paper_daily_pnl WHERE account_id = :aid ORDER BY trade_date
            """), {"aid": account_id}).fetchall()
        if daily_rows:
            latest = daily_rows[-1]
            tv = float(latest[1]); dr_val = float(latest[2] or 0)
            first_tv = float(daily_rows[0][1])
            total_ret = (tv - first_tv) / first_tv if first_tv > 0 else 0
            max_dd = min((float(r[3]) for r in daily_rows if r[3] is not None), default=0)

            # 现金和持仓市值
            try:
                with engine.connect() as conn2:
                    cash_r = conn2.execute(text('SELECT cash FROM paper_account WHERE id=:aid'), {'aid': account_id}).fetchone()
                    cash_val = float(cash_r[0]) if cash_r else 0
            except Exception:
                cash_val = 0
            pos_mv = sum(p[5] * price_map.get(p[0], (p[2], 0))[0] for p in open_pos) if open_pos else 0

            summary_card = f"""<div class="card"><h3>汇总</h3>
            <table style="width:100%"><tr>
                <td>总资产: <strong>{tv:,.0f}</strong></td>
                <td>现金: {cash_val:,.0f}</td>
                <td>股票市值: {pos_mv:,.0f}</td>
                <td>日收益: <span class="{'up' if dr_val>0 else 'down'}">{dr_val:+.2%}</span></td>
                <td>最大回撤: <span class="down">{max_dd:.2%}</span></td>
            </tr></table>"""
            if total_pnl != 0 or len(closed_pos) > 0:
                summary_card += f"<p style='margin-top:8px;'>已平仓盈亏: <span class=\"{'up' if total_pnl > 0 else 'down'}\">{total_pnl:+,.0f}</span> | 胜率: {wins}/{len(closed_pos)} ({wins/max(len(closed_pos),1)*100:.0f}%)</p>"
            summary_card += "</div>"

            # 历史日收益表
            recent = daily_rows[-20:]
            history_html = '<div class="card"><h3>每日估值记录</h3><table><thead><tr><th>日期</th><th>总资产</th><th>日收益</th><th>回撤</th></tr></thead><tbody>'
            for d in reversed(recent):
                dr_c = "up" if (d[2] or 0) > 0 else "down"
                history_html += f'<tr><td>{d[0]}</td><td>{d[1]:,.0f}</td><td class="{dr_c}">{(d[2] or 0):+.2%}</td><td class="down">{(d[3] or 0):.2%}</td></tr>'
            history_html += '</tbody></table></div>'
        else:
            history_html = ""
    except Exception:
        summary_card = ""
        history_html = ""

    html = f"""{pnl_summary}
    {summary_card}
    {history_html}
    <div class="card"><h3>当前持仓 ({len(open_pos)}只)</h3>{open_html or '<p>无持仓</p>'}</div>
    {chart_html}
    {pending_html}
    {sector_html}
    <div class="card"><h3>已平仓交易 ({len(closed_pos)}笔)</h3>{pos_html.replace('<th>代码</th><th>入场日</th>', '<th>代码</th><th>入场日</th><th>出场日</th><th>出场价</th>')
    if not closed_pos else pos_html}</div>
    <div class="card"><h3>近期信号</h3>{sig_html}</div>
    <div class="card"><h3>汇总</h3><p>总盈亏: <span class="{'up' if total_pnl > 0 else 'down'}">{total_pnl:+,.0f}</span></p><p>胜率: {wins}/{len(closed_pos)} ({wins/max(len(closed_pos),1)*100:.0f}%)</p></div>"""
    return HTMLResponse(html)


@router.get("/data-status")
async def get_data_status():
    engine = get_engine()
    # (表名, 日期列)
    tables = [
        ("stock_basic", None),
        ("stock_daily", "trade_date"),
        ("index_daily", "trade_date"),
        ("stock_daily_extra", "trade_date"),
        ("stock_minute", "trade_time"),
        ("stock_shareholder", "end_date"),
        ("stock_financial", "report_date"),
        ("backtest_results", "created_at"),
        ("paper_runs", "created_at"),
        ("data_quality_log", "trade_date"),
    ]

    html = "<table><thead><tr><th>表名</th><th>最新日期</th><th>行数</th></tr></thead><tbody>"
    for t, date_col in tables:
        try:
            with engine.connect() as conn:
                if date_col:
                    row = conn.execute(text(
                        f'SELECT COALESCE(MAX({date_col})::text, \'无数据\'), COUNT(*) FROM {t}'
                    )).fetchone()
                else:
                    row = conn.execute(text(f"SELECT 'N/A', COUNT(*) FROM {t}")).fetchone()
            html += f"<tr><td>{t}</td><td>{row[0]}</td><td>{row[1]:,}</td></tr>"
        except Exception as e:
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
    qhtml += "</tbody></table>" if qrows else "<p style='color:#999;'>暂无质量校验记录</p>"

    return HTMLResponse(html + qhtml)


@router.post("/paper/create")
async def create_paper_run(strategy: str = "ML-默认集成", capital: float = 1_000_000,
                           start_date: str = "", end_date: str = ""):
    """创建模拟盘运行记录。需要 strategy_configs 中有对应策略。"""
    from datetime import date
    engine = get_engine()
    try:
        with engine.connect() as conn:
            # Find strategy version
            row = conn.execute(text("""
                SELECT sv.id FROM strategy_versions sv
                JOIN strategy_configs sc ON sv.strategy_id = sc.id
                WHERE sc.name = :name AND sv.version = '1.00'
            """), {"name": strategy}).fetchone()
            if not row:
                return HTMLResponse(f"<p style='color:#ef5350;'>未找到策略: {strategy}</p>")
            version_id = row[0]

            # Check existing running paper runs
            existing = conn.execute(text(
                "SELECT id FROM paper_runs WHERE status = 'running' AND strategy_id = (SELECT id FROM strategy_configs WHERE name = :name)"
            ), {"name": strategy}).fetchone()
            if existing:
                return HTMLResponse(f"<p style='color:#ffb74d;'>策略 {strategy} 已有运行中的模拟盘 (id={existing[0]})</p>")

            sd = start_date or str(date.today())
            ed = end_date or None

            conn.execute(text("""
                INSERT INTO paper_runs (strategy_id, version_id, start_date, end_date, initial_capital, status)
                VALUES ((SELECT id FROM strategy_configs WHERE name = :name), :vid, :sd, :ed, :cap, 'running')
            """), {"name": strategy, "vid": version_id, "sd": sd, "ed": ed, "cap": capital})
            conn.commit()

            return HTMLResponse(f"<p style='color:#66bb6a;'>模拟盘已创建: {strategy} v1.00, 初始资金 {capital:,.0f}, 起始日 {sd}</p>"
                               f"<p style='color:#999;font-size:12px;margin-top:8px;'>模拟盘引擎运行方式: "
                               f"<code>python -c \"from portfolio.paper_engine import PaperEngine; ...\"</code> "
                               f"或在后台脚本中调用 PaperEngine.run_daily()</p>")
    except Exception as e:
        return HTMLResponse(f"<p style='color:#ef5350;'>创建失败: {e}</p>")


@router.post("/sync/trigger")
async def trigger_sync():
    import subprocess, sys
    subprocess.Popen([sys.executable, "-m", "data.sync"],
                     cwd=os.path.join(os.path.dirname(__file__), "..", ".."))
    return HTMLResponse("<p style='color:#4fc3f7;'>同步已触发，正在后台运行... 刷新页面查看结果</p>")
