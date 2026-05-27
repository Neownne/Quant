import numpy as np
from data.db import get_engine
from sqlalchemy import text
from datetime import date
import json


def compute_factor_contribution(factor_values: dict, pnl_pct: float) -> dict:
    """按因子值的符号与大小分解收益归因"""
    if not factor_values:
        return {}
    total_abs = sum(abs(v) for v in factor_values.values())
    if total_abs == 0:
        return {k: 0.0 for k in factor_values}
    contrib = {}
    for factor, value in factor_values.items():
        contrib[factor] = round(pnl_pct * abs(value) / total_abs, 6)
    return contrib


def run_attribution(run_id: int, eval_date: date):
    """对模拟盘某日信号做归因分析"""
    engine = get_engine()
    try:
        with engine.connect() as conn:
            signals = conn.execute(text("""
                SELECT ps.id, ps.stock_code, ps.signal_date
                FROM paper_signals ps
                WHERE ps.run_id = :rid AND ps.signal_date <= :edate
            """), {"rid": run_id, "edate": eval_date}).fetchall()
    except Exception:
        return

    for signal_id, stock_code, signal_date in signals:
        try:
            with engine.connect() as conn:
                pos = conn.execute(text("""
                    SELECT pnl_pct FROM paper_positions
                    WHERE signal_id = :sid AND pnl_pct IS NOT NULL
                """), {"sid": signal_id}).fetchone()
        except Exception:
            continue

        if not pos:
            continue
        pnl_pct = pos[0]

        try:
            with engine.connect() as conn:
                factor_rows = conn.execute(text(
                    "SELECT factor_name, value FROM signal_factors WHERE signal_id = :sid"
                ), {"sid": signal_id}).fetchall()
        except Exception:
            continue

        factor_values = {r[0]: r[1] for r in factor_rows}
        contrib = compute_factor_contribution(factor_values, pnl_pct)
        days_held = (eval_date - signal_date).days

        try:
            with engine.begin() as conn:
                conn.execute(text("""
                    INSERT INTO signal_attribution (signal_id, eval_date, days_held, pnl_pct, factor_contrib_json)
                    VALUES (:sid, :edate, :days, :pnl, :contrib)
                    ON CONFLICT DO NOTHING
                """), {"sid": signal_id, "edate": eval_date, "days": days_held, "pnl": pnl_pct,
                       "contrib": json.dumps(contrib)})
        except Exception:
            pass
