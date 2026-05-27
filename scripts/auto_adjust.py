from data.db import get_engine
from sqlalchemy import text
from datetime import date, timedelta
from scipy import stats
import numpy as np


def check_and_adjust(strategy_id: int, lookback_periods: int = 10,
                     decay_threshold: float = 0.3, min_consecutive: int = 3,
                     pvalue_threshold: float = 0.05):
    """检查因子IC衰减，满足条件则自动降权"""
    engine = get_engine()
    try:
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT DISTINCT ON (factor_name) factor_name, weight, effective_date
                FROM factor_weights_history
                WHERE strategy_id = :sid
                ORDER BY factor_name, effective_date DESC
            """), {"sid": strategy_id}).fetchall()
    except Exception as e:
        print(f"策略{strategy_id}无权重历史，跳过: {e}")
        return

    if not rows:
        print(f"策略{strategy_id}无权重历史，跳过")
        return

    adjustments = []
    for factor_name, current_weight, _ in rows:
        # Get recent IC for this factor from strategy_health table
        try:
            with engine.connect() as conn:
                ic_rows = conn.execute(text("""
                    SELECT overall_ic FROM strategy_health
                    WHERE strategy_id = :sid AND date >= :start
                    ORDER BY date
                """), {"sid": strategy_id, "start": date.today() - timedelta(days=lookback_periods * 2)}).fetchall()
        except Exception:
            continue

        ic_series = [r[0] for r in ic_rows if r[0] is not None]
        if len(ic_series) < min_consecutive:
            continue

        recent = ic_series[-min_consecutive:]
        if all(x > 0 for x in recent):
            continue

        if ic_series[0] == 0:
            continue
        decay = (ic_series[0] - ic_series[-1]) / abs(ic_series[0])
        if decay < decay_threshold:
            continue

        t_stat, p_value = stats.ttest_1samp(recent, 0)
        p_value = p_value / 2 if t_stat < 0 else 1.0
        if p_value >= pvalue_threshold:
            print(f"{factor_name}: IC衰减{decay:.0%}但不显著(p={p_value:.3f})，跳过")
            continue

        new_weight = round(current_weight * 0.8, 6)
        confidence = 0.99 if p_value < 0.01 else 0.95
        reason = f"IC衰减{decay:.1%}, 连续{min_consecutive}期, p={p_value:.3f}"

        try:
            with engine.begin() as conn:
                conn.execute(text("""
                    INSERT INTO weight_adjustments (strategy_id, factor_name, old_weight, new_weight, confidence_level, source, reason)
                    VALUES (:sid, :name, :old, :new, :conf, 'auto', :reason)
                """), {"sid": strategy_id, "name": factor_name, "old": current_weight,
                       "new": new_weight, "conf": confidence, "reason": reason})
                conn.execute(text("""
                    INSERT INTO factor_weights_history (strategy_id, factor_name, weight, effective_date, source, reason)
                    VALUES (:sid, :name, :weight, :date, 'auto', :reason)
                """), {"sid": strategy_id, "name": factor_name, "weight": new_weight,
                       "date": date.today(), "reason": f"自降权: {reason}"})
        except Exception as e:
            print(f"写入调权记录失败: {e}")
            continue

        adjustments.append({"factor": factor_name, "old": current_weight, "new": new_weight})

    print(f"自动调参完成: {len(adjustments)}项调整")
    for a in adjustments:
        print(f"  {a['factor']}: {a['old']:.4f} -> {a['new']:.4f}")
    return adjustments
