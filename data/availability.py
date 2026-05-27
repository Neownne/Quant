"""因子可用性跟踪。

记录每个因子在每个交易日的数据就绪时间，避免回测/实盘中的
前视偏差（look-ahead bias）。
"""

from datetime import date, datetime

from sqlalchemy import text

from data.db import get_engine


def mark_ready(
    trade_date: date,
    factor_name: str,
    data_source: str = "computed",
    ready_at: datetime | None = None,
) -> None:
    """标记因子值在指定交易日已就绪（幂等：同日同因子存在则更新）。"""
    if ready_at is None:
        ready_at = datetime.now()

    # 以当天15:00为基准计算延迟（毫秒）
    base = datetime(ready_at.year, ready_at.month, ready_at.day, 15, 0, 0)
    latency_ms = max(int((ready_at - base).total_seconds() * 1000), 0)

    with get_engine().begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO factor_availability
                    (trade_date, factor_name, data_ready_at, data_source, latency_ms)
                VALUES (:date, :name, :ready, :src, :latency)
                ON CONFLICT (trade_date, factor_name) DO UPDATE SET
                    data_ready_at = EXCLUDED.data_ready_at,
                    latency_ms = EXCLUDED.latency_ms
                """
            ),
            {
                "date": trade_date,
                "name": factor_name,
                "ready": ready_at,
                "src": data_source,
                "latency": latency_ms,
            },
        )


def get_ready_factors(trade_date: date, before: datetime) -> list[str]:
    """返回指定交易日、指定时刻之前已就绪的所有因子名。"""
    with get_engine().connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT factor_name FROM factor_availability
                WHERE trade_date = :date AND data_ready_at <= :before
                """
            ),
            {"date": trade_date, "before": before},
        ).fetchall()
    return [r[0] for r in rows]
