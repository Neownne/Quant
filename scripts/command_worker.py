from data.db import get_engine
from sqlalchemy import text
from datetime import date
import json


def process_pending_commands():
    """消费 strategy_commands 队列中未执行的指令"""
    engine = get_engine()
    try:
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT id, strategy_id, command_type, payload_json, requested_by
                FROM strategy_commands
                WHERE executed_at IS NULL
                ORDER BY requested_at
            """)).fetchall()
    except Exception as e:
        print(f"读取指令队列失败: {e}")
        return

    for cmd_id, strategy_id, cmd_type, payload, requested_by in rows:
        payload = payload if isinstance(payload, dict) else json.loads(payload) if payload else {}
        try:
            if cmd_type == "adjust_weight":
                _exec_adjust_weight(strategy_id, payload)
            elif cmd_type == "pause":
                with engine.begin() as conn:
                    conn.execute(text(
                        "UPDATE paper_runs SET status = 'paused' WHERE strategy_id = :sid AND status = 'running'"
                    ), {"sid": strategy_id})
            elif cmd_type == "resume":
                with engine.begin() as conn:
                    conn.execute(text(
                        "UPDATE paper_runs SET status = 'running' WHERE strategy_id = :sid AND status = 'paused'"
                    ), {"sid": strategy_id})
            elif cmd_type == "rollback":
                _exec_rollback(strategy_id, payload)
            elif cmd_type == "retrain":
                import subprocess, sys
                subprocess.run([sys.executable, "scripts/run_ml_backtest.py", "--strategy-id", str(strategy_id)])
            with engine.begin() as conn:
                conn.execute(text(
                    "UPDATE strategy_commands SET executed_at = NOW(), execution_result = 'ok' WHERE id = :cid"
                ), {"cid": cmd_id})
            print(f"指令{cmd_id} ({cmd_type}) 执行成功")
        except Exception as e:
            try:
                with engine.begin() as conn:
                    conn.execute(text(
                        "UPDATE strategy_commands SET executed_at = NOW(), execution_result = :err WHERE id = :cid"
                    ), {"err": str(e), "cid": cmd_id})
            except Exception:
                pass
            print(f"指令{cmd_id}执行失败: {e}")


def _exec_adjust_weight(strategy_id, payload):
    factor_name = payload.get("factor_name", "")
    old_weight = float(payload.get("old_weight", 0))
    new_weight = float(payload.get("new_weight", 0))
    reason = payload.get("reason", "manual")
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO weight_adjustments (strategy_id, factor_name, old_weight, new_weight, source, reason)
            VALUES (:sid, :name, :old, :new, 'manual', :reason)
        """), {"sid": strategy_id, "name": factor_name, "old": old_weight, "new": new_weight, "reason": reason})
        conn.execute(text("""
            INSERT INTO factor_weights_history (strategy_id, factor_name, weight, effective_date, source, reason)
            VALUES (:sid, :name, :weight, :date, 'manual', :reason)
        """), {"sid": strategy_id, "name": factor_name, "weight": new_weight, "date": date.today(), "reason": reason})


def _exec_rollback(strategy_id, payload):
    target_version = payload.get("target_version")
    if not target_version:
        return
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("""
            UPDATE paper_runs SET version_id = (
                SELECT id FROM strategy_versions WHERE strategy_id = :sid AND version = :ver
            ) WHERE strategy_id = :sid AND status = 'paused'
        """), {"sid": strategy_id, "ver": target_version})
