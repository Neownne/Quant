"""ETF 三因子监测页面路由。"""
import json
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy import text
from data.db import get_engine
from web.templates_loader import templates

router = APIRouter()


@router.get("/etf", response_class=HTMLResponse)
async def etf_monitor_page(request: Request):
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
            d = dict(zip(
                ["code", "name", "close", "chg_pct", "vol_ratio", "vol_prob",
                 "dir_prob", "share_prob", "composite_prob", "signal_level", "date"], r))
            d["signal_color"] = {"high": "#c62828", "mid": "#e65100", "normal": "#2e7d32"}.get(
                d.get("signal_level", ""), "#666")
            results.append(d)
    except Exception:
        pass
    engine.dispose()

    # 排序：宽基ETF置顶 → 信号由强到弱 → 综合概率降序
    BROAD_BASE = ["沪深300", "上证50", "中证500", "中证1000", "创业板", "科创50", "上证指数", "深证"]
    def sort_key(r):
        name = r.get("name", "")
        is_broad = any(kw in name for kw in BROAD_BASE)
        sig_order = {"high": 0, "mid": 1, "normal": 2}.get(r.get("signal_level", ""), 3)
        prob = -(r.get("composite_prob") or 0)
        return (not is_broad, sig_order, prob)
    results.sort(key=sort_key)

    high_count = sum(1 for r in results if r.get("signal_level") == "high")
    mid_count = sum(1 for r in results if r.get("signal_level") == "mid")
    normal_count = sum(1 for r in results if r.get("signal_level") == "normal")

    return templates.TemplateResponse(request, "etf_monitor.html", {
        "active_page": "etf",
        "rows": results,
        "high_count": high_count,
        "mid_count": mid_count,
        "normal_count": normal_count,
    })


@router.get("/api/etf/kline/{code}")
async def etf_kline(code: str, days: int = 60):
    """返回 ETF K 线数据供 ECharts 渲染。"""
    engine = get_engine()
    try:
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT trade_date, open, high, low, close, volume
                FROM etf_daily WHERE code = :code
                ORDER BY trade_date DESC LIMIT :days
            """), {"code": code, "days": days}).fetchall()
            dates = [str(r[0]) for r in reversed(rows)]
            ohlc = [[float(r[1]), float(r[2]), float(r[3]), float(r[4])] for r in reversed(rows)]
            volumes = [int(r[5]) for r in reversed(rows)]
            name_row = conn.execute(text("SELECT name FROM etf_basic WHERE code = :code"),
                                    {"code": code}).fetchone()
            name = name_row[0] if name_row else code
    except Exception as e:
        from loguru import logger
        logger.warning(f"ETF kline error for {code}: {e}")
        dates, ohlc, volumes, name = [], [], [], code
    engine.dispose()
    return JSONResponse({"code": code, "name": name, "dates": dates, "ohlc": ohlc, "volumes": volumes})
