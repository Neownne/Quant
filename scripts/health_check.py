from data.db import get_engine
from sqlalchemy import text
from datetime import date, timedelta


def check_strategy_health(strategy_id: int):
    """每日检查策略健康状况并更新 strategy_health 表"""
    engine = get_engine()
    today = date.today()

    # 获取7日IC均值
    try:
        with engine.connect() as conn:
            ic_row = conn.execute(text("""
                SELECT AVG(pnl_pct) FROM paper_positions
                WHERE run_id IN (SELECT id FROM paper_runs WHERE strategy_id = :sid)
                  AND entry_date >= :start
            """), {"sid": strategy_id, "start": today - timedelta(days=7)}).fetchone()
    except Exception:
        ic_row = None

    avg_pnl = ic_row[0] if ic_row and ic_row[0] else 0

    # Get initial capital for drawdown calc
    try:
        with engine.connect() as conn:
            cap_row = conn.execute(text("""
                SELECT initial_capital FROM paper_runs WHERE strategy_id = :sid ORDER BY created_at DESC LIMIT 1
            """), {"sid": strategy_id}).fetchone()
    except Exception:
        cap_row = None

    initial_capital = cap_row[0] if cap_row else 100000.0

    # 简化：用平均盈亏作为IC代理
    ic = avg_pnl
    dd_val = None

    # 7日回撤估算
    try:
        with engine.connect() as conn:
            dd_row = conn.execute(text("""
                WITH daily_pnl AS (
                    SELECT entry_date, SUM(COALESCE(pnl, 0)) AS day_pnl
                    FROM paper_positions
                    WHERE run_id IN (SELECT id FROM paper_runs WHERE strategy_id = :sid)
                      AND entry_date >= :start
                    GROUP BY entry_date
                )
                SELECT MIN(day_pnl) FROM daily_pnl
            """), {"sid": strategy_id, "start": today - timedelta(days=7)}).fetchone()
    except Exception:
        dd_row = None

    min_daily = dd_row[0] if dd_row else None
    if initial_capital > 0 and min_daily is not None:
        dd_val = abs(min(min_daily, 0)) / initial_capital

    # 判定状态
    if ic is not None and ic > 0 and (dd_val is None or dd_val < 0.10):
        status, action = "normal", "none"
    elif ic is not None and (ic < 0 or (dd_val and dd_val > 0.15)):
        status, action = "warning", "pause"
    elif ic is not None and ic < 0 and dd_val and dd_val > 0.25:
        status, action = "critical", "switch_backup"
    else:
        status, action = "normal", "none"

    # 简单regime判断（基于IC符号）
    regime = "bull" if ic > 0.01 else "bear" if ic < -0.01 else "range"

    try:
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO strategy_health (strategy_id, date, overall_ic, max_drawdown_7d, regime_tag, status, action_required)
                VALUES (:sid, :date, :ic, :dd, :regime, :status, :action)
                ON CONFLICT (strategy_id, date) DO UPDATE SET
                    overall_ic = EXCLUDED.overall_ic,
                    max_drawdown_7d = EXCLUDED.max_drawdown_7d,
                    regime_tag = EXCLUDED.regime_tag,
                    status = EXCLUDED.status,
                    action_required = EXCLUDED.action_required
            """), {"sid": strategy_id, "date": today, "ic": ic, "dd": dd_val,
                   "regime": regime, "status": status, "action": action})
    except Exception as e:
        print(f"写入健康记录失败: {e}")

    if status == "critical":
        print(f"策略{strategy_id}触发熔断，暂停所有运行中的模拟盘")
        try:
            with engine.begin() as conn:
                conn.execute(text(
                    "UPDATE paper_runs SET status = 'paused' WHERE strategy_id = :sid AND status = 'running'"
                ), {"sid": strategy_id})
        except Exception:
            pass

    return {"strategy_id": strategy_id, "status": status, "action": action, "ic": ic}
