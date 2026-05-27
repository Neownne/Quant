import json

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
