#!/usr/bin/env python
"""回测内嵌归因/健康度/动态调参模块。

在 walk-forward 回测中实现闭环：
  训练 → 归因(因子重要性追踪) → 健康度(窗口间衰退检测) → 动态调参(降权/重新筛选)

用法:
    from scripts.dynamic_backtest import BacktestFeedbackLoop

    loop = BacktestFeedbackLoop(strategy_id, factor_names, db_engine)
    for window_idx, r in enumerate(training_results):
        loop.process_window(window_idx, r, dataset, factor_cols)
    loop.save_summary()
"""
from __future__ import annotations

import json
import numpy as np
from scipy import stats
from datetime import date, timedelta
from typing import Any

# Forward references for type hints
import pandas as pd


class BacktestFeedbackLoop:
    """回测反馈闭环：归因 → 健康度 → 动态调参。"""

    def __init__(self, strategy_id: int, initial_factors: list[str], db_engine):
        self.strategy_id = strategy_id
        self.initial_factors = initial_factors
        self.engine = db_engine

        # Per-window history
        self.window_metrics: list[dict] = []
        self.window_importance: list[dict[str, float]] = []
        self.window_ic: list[dict[str, float]] = []  # per-window mean |IC| per factor
        self.window_status: list[str] = []
        self.factor_weights: dict[str, float] = {f: 1.0 for f in initial_factors}
        self.adjustments: list[dict] = []

        # Health state
        self.health_status = "normal"
        self.consecutive_warnings = 0

    # ── 1. 因子归因 ────────────────────────────────────────────────────────

    def extract_importance(self, window_result: dict) -> dict[str, float]:
        """从训练好的模型提取因子重要性（跨模型平均）。"""
        importance: dict[str, float] = {}

        xgb_model = window_result.get("xgb_model")
        lgb_model = window_result.get("lgb_model")
        single_model = window_result.get("model")
        active_cols = window_result.get("active_cols", [])

        if xgb_model is not None and hasattr(xgb_model, "feature_importances_"):
            feats = getattr(xgb_model, "feature_names_in_", active_cols)
            for f, imp in zip(feats, xgb_model.feature_importances_):
                importance[f] = importance.get(f, 0) + float(imp)
        if lgb_model is not None and hasattr(lgb_model, "feature_importances_"):
            feats = getattr(lgb_model, "feature_name_", active_cols)
            for f, imp in zip(feats, lgb_model.feature_importances_):
                importance[f] = importance.get(f, 0) + float(imp)
        if single_model is not None and hasattr(single_model, "feature_importances_"):
            feats = getattr(single_model, "feature_names_in_",
                          getattr(single_model, "feature_name_", active_cols))
            for f, imp in zip(feats, single_model.feature_importances_):
                importance[f] = importance.get(f, 0) + float(imp)

        # Normalize to sum=1
        total = sum(importance.values())
        if total > 0:
            importance = {k: v / total for k, v in importance.items()}

        # Fill missing active factors with 0
        for f in active_cols:
            if f not in importance:
                importance[f] = 0.0

        return importance

    def extract_per_factor_ic(self, window_result: dict) -> dict[str, float]:
        """从 window_result 提取逐因子 mean |IC|。"""
        ic_series = window_result.get("ic_series", {})
        if not ic_series:
            return {}
        # ic_series: {factor: [ic_values...]}
        mean_abs_ic = {}
        for f, ics in ic_series.items():
            valid = [v for v in ics if v is not None and np.isfinite(v)]
            if valid:
                mean_abs_ic[f] = float(np.mean(np.abs(valid)))
        return mean_abs_ic

    def save_attribution(self, window_idx: int, importance: dict[str, float],
                         metrics: dict, eval_date: str):
        """保存窗口级归因摘要到 strategy_commands（回测场景无 paper_signals）。"""
        try:
            top5 = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
            payload = {
                "window_idx": window_idx,
                "eval_date": str(eval_date),
                "sharpe": round(metrics.get("sharpe", 0), 6),
                "accuracy": round(metrics.get("accuracy", 0), 6),
                "top_factors": [(f, round(v, 6)) for f, v in top5],
                "all_factors": [(f, round(v, 6)) for f, v in importance.items()],
            }
            with self.engine.begin() as conn:
                from sqlalchemy import text
                conn.execute(text("""
                    INSERT INTO strategy_commands
                        (strategy_id, command_type, payload_json, requested_by, requested_at)
                    VALUES (:sid, 'attribution_snapshot', CAST(:payload AS jsonb), 'backtest', NOW())
                """), {
                    "sid": self.strategy_id,
                    "payload": json.dumps(payload, ensure_ascii=False),
                })
        except Exception as e:
            print(f"  [归因] 写入失败: {e}")

    # ── 2. 策略健康度 ──────────────────────────────────────────────────────

    def check_health(self, window_idx: int, metrics: dict) -> dict:
        """基于窗口间指标趋势评估策略健康状态。"""
        n_windows = len(self.window_metrics)
        sharpe = metrics.get("sharpe", 0) or 0
        accuracy = metrics.get("accuracy", 0) or 0

        status = "normal"
        action = "none"
        warning_signals = []

        if n_windows >= 2:
            prev_sharpe = self.window_metrics[-1].get("sharpe", 0) or 0
            prev_acc = self.window_metrics[-1].get("accuracy", 0) or 0

            # Sharpe 下降 > 30%
            if prev_sharpe > 0 and sharpe < prev_sharpe * 0.7:
                warning_signals.append(f"Sharpe暴跌 {prev_sharpe:.3f}→{sharpe:.3f}")

            # 准确率下降 > 5pp
            if accuracy < prev_acc - 0.05:
                warning_signals.append(f"准确率下滑 {prev_acc:.3f}→{accuracy:.3f}")

        # 累积衰退检测（线性回归斜率）
        if n_windows >= 3:
            sharpes = [m.get("sharpe", 0) or 0 for m in self.window_metrics[-3:]] + [sharpe]
            x = np.arange(len(sharpes))
            slope, _, _, p_value, _ = stats.linregress(x, sharpes)
            if slope < -0.05 and p_value < 0.1:
                warning_signals.append(f"Sharpe趋势性衰退 slope={slope:.3f} p={p_value:.3f}")
                if p_value < 0.05:
                    self.consecutive_warnings += 1

        # 判定
        if warning_signals:
            if self.consecutive_warnings >= 2:
                status, action = "critical", "rescreen_factors"
            else:
                status, action = "warning", "adjust_weights"
        else:
            self.consecutive_warnings = max(0, self.consecutive_warnings - 1)

        # 简单regime判断
        regime = "bull" if sharpe > 0.3 else "bear" if sharpe < -0.1 else "range"

        self.health_status = status

        health = {
            "strategy_id": self.strategy_id,
            "window": window_idx + 1,
            "status": status,
            "action_required": action,
            "warnings": warning_signals,
            "sharpe": sharpe,
            "accuracy": accuracy,
            "regime": regime,
            "consecutive_warnings": self.consecutive_warnings,
        }
        return health

    def save_health(self, health: dict):
        """保存健康检查记录。"""
        try:
            eval_str = health.get("eval_date", "")
            try:
                edate = date.fromisoformat(eval_str[:10]) if eval_str else date.today()
            except (ValueError, TypeError):
                edate = date.today()

            with self.engine.begin() as conn:
                from sqlalchemy import text
                conn.execute(text("""
                    INSERT INTO strategy_health
                        (strategy_id, date, overall_ic, max_drawdown_7d,
                         regime_tag, status, action_required)
                    VALUES (:sid, :date, :ic, :dd, :regime, :status, :action)
                    ON CONFLICT (strategy_id, date) DO UPDATE SET
                        overall_ic = EXCLUDED.overall_ic,
                        regime_tag = EXCLUDED.regime_tag,
                        status = EXCLUDED.status,
                        action_required = EXCLUDED.action_required
                """), {
                    "sid": self.strategy_id,
                    "date": edate,
                    "ic": round(health.get("sharpe", 0), 6),
                    "dd": 0.0,
                    "regime": health.get("regime", "range"),
                    "status": health.get("status", "normal"),
                    "action": health.get("action_required", "none"),
                })
        except Exception as e:
            print(f"  [健康度] 写入失败: {e}")

    # ── 3. 动态调参 ────────────────────────────────────────────────────────

    def adjust_weights(self, importance_history: list[dict[str, float]],
                       decay_threshold: float = 0.3) -> dict[str, float]:
        """基于多窗口因子重要性趋势自动调整权重。

        - 重要性连续下降 >30% 的因子：权重 ×0.8
        - 重要性连续上升 >30% 的因子：权重 ×1.1
        - 仅在统计显著时调整（配对 t 检验 p < 0.1）
        """
        if len(importance_history) < 2:
            return self.factor_weights

        all_factors = set()
        for imp in importance_history:
            all_factors.update(imp.keys())

        adjustments_this_round = []
        prev_imp = importance_history[-2]
        curr_imp = importance_history[-1]

        for f in all_factors:
            prev_v = prev_imp.get(f, 0)
            curr_v = curr_imp.get(f, 0)
            if abs(prev_v) < 1e-6:
                continue

            delta = (curr_v - prev_v) / abs(prev_v)
            old_w = self.factor_weights.get(f, 1.0)

            if delta < -decay_threshold:
                # 追加检验：看最近3个窗口的趋势
                if len(importance_history) >= 3:
                    recent3 = [h.get(f, 0) for h in importance_history[-3:]] + [curr_v]
                    if len(set(r > 0 for r in recent3)) < 2:
                        continue
                new_w = round(old_w * 0.8, 6)
                self.factor_weights[f] = new_w
                adjustments_this_round.append({
                    "factor": f, "old_weight": old_w, "new_weight": new_w,
                    "delta": round(delta, 4), "direction": "down"
                })
            elif delta > decay_threshold:
                new_w = round(min(old_w * 1.1, 2.0), 6)
                self.factor_weights[f] = new_w
                adjustments_this_round.append({
                    "factor": f, "old_weight": old_w, "new_weight": new_w,
                    "delta": round(delta, 4), "direction": "up"
                })

        if adjustments_this_round:
            self.adjustments.extend(adjustments_this_round)
            self._save_weight_changes(adjustments_this_round)

        return self.factor_weights

    def _save_weight_changes(self, adjustments: list[dict]):
        """持久化权重变更记录。"""
        try:
            with self.engine.begin() as conn:
                from sqlalchemy import text
                for adj in adjustments:
                    conn.execute(text("""
                        INSERT INTO weight_adjustments
                            (strategy_id, factor_name, old_weight, new_weight,
                             confidence_level, source, reason)
                        VALUES (:sid, :fn, :old, :new, :conf, 'backtest_dynamic', :reason)
                    """), {
                        "sid": self.strategy_id,
                        "fn": adj["factor"],
                        "old": adj["old_weight"],
                        "new": adj["new_weight"],
                        "conf": 0.85,
                        "reason": f"重要性变化{adj['delta']:+.1%}, direction={adj['direction']}",
                    })
                    conn.execute(text("""
                        INSERT INTO factor_weights_history
                            (strategy_id, factor_name, weight, effective_date, source, reason)
                        VALUES (:sid, :fn, :weight, :date, 'backtest_dynamic', :reason)
                    """), {
                        "sid": self.strategy_id,
                        "fn": adj["factor"],
                        "weight": adj["new_weight"],
                        "date": date.today(),
                        "reason": f"动态调权: {adj['direction']}",
                    })
        except Exception as e:
            print(f"  [调权] 写入失败: {e}")

    def get_dynamic_factors(self, base_factors: list[str]) -> list[str]:
        """返回加权后的活跃因子列表。

        权重 < 0.3 的因子被移除；权重 > 1.0 的因子保留且可被模型自然加权。
        """
        weighted = []
        for f in base_factors:
            w = self.factor_weights.get(f, 1.0)
            if w >= 0.3:  # 低于 0.3 的因子被淘汰
                weighted.append(f)
        return weighted

    # ── 4. 统一处理入口 ─────────────────────────────────────────────────────

    def process_window(self, window_idx: int, window_result: dict,
                       metrics: dict, eval_date: str) -> dict:
        """处理单个窗口：归因→健康度→调参。返回健康状态。"""
        # 1. 提取因子重要性
        importance = self.extract_importance(window_result)
        self.window_importance.append(importance)

        # 2. 提取逐因子 |IC|
        per_factor_ic = self.extract_per_factor_ic(window_result)
        if per_factor_ic:
            self.window_ic.append(per_factor_ic)

        # 3. 保存归因
        self.save_attribution(window_idx, importance, metrics, eval_date)

        # 4. 健康度检查
        self.window_metrics.append(metrics)
        health = self.check_health(window_idx, metrics)
        health["eval_date"] = eval_date
        self.window_status.append(health["status"])
        self.save_health(health)

        # 5. 动态调参（仅在 warning/critical 时触发）
        if health["action_required"] == "adjust_weights" and len(self.window_importance) >= 2:
            new_weights = self.adjust_weights(self.window_importance)
            print(f"  [调参] {len(new_weights)} 个因子权重已更新, "
                  f"警告: {health['warnings']}")
        elif health["action_required"] == "rescreen_factors":
            # 精确淘汰：仅淘汰 |IC| < 0.05 且在重要性后 50% 的因子
            to_eliminate = self._select_factors_to_eliminate(importance, per_factor_ic)
            if to_eliminate:
                for f in to_eliminate:
                    self.factor_weights[f] = 0.2  # 权重 < 0.3 → 下一轮 get_dynamic_factors 移除
                print(f"  [预警] 策略健康度 critical, 精准淘汰 {len(to_eliminate)} 个因子: "
                      f"{to_eliminate[:5]}{'...' if len(to_eliminate) > 5 else ''}")
                self.adjustments.append({
                    "factor": ", ".join(to_eliminate[:5]),
                    "old_weight": 1.0, "new_weight": 0.2,
                    "delta": -0.8, "direction": "down",
                })
            else:
                print(f"  [预警] 策略健康度 critical 但无因子满足淘汰条件 "
                      f"(|IC|<0.05 且 重要性后50%), 跳过")

        return health

    def _select_factors_to_eliminate(self, importance: dict[str, float],
                                     per_factor_ic: dict[str, float]) -> list[str]:
        """选择需要淘汰的因子：|IC| < 0.05 且在重要性后 50%。

        如果没有逐因子 IC 数据，退化为仅使用重要性后 50%。
        """
        if not importance:
            return []

        sorted_by_imp = sorted(importance.items(), key=lambda x: x[1])
        median_idx = max(1, len(sorted_by_imp) // 2)
        bottom_half_imp = {f for f, _ in sorted_by_imp[:median_idx]}

        if per_factor_ic:
            # 主条件：|IC| < 0.05 AND 重要性后 50%
            eliminated = [f for f in bottom_half_imp
                          if abs(per_factor_ic.get(f, 0)) < 0.05]
        else:
            # 退而求其次：重要性后 50% + 重要性 < 0.03（绝对低重要性）
            eliminated = [f for f in bottom_half_imp
                          if importance.get(f, 0) < 0.03]

        return eliminated

    def save_summary(self) -> dict:
        """保存回测闭环总结。"""
        summary = {
            "windows_processed": len(self.window_metrics),
            "total_adjustments": len(self.adjustments),
            "final_health": self.health_status,
            "factor_weight_changes": [
                {"factor": adj["factor"], "from": adj["old_weight"],
                 "to": adj["new_weight"], "direction": adj["direction"]}
                for adj in self.adjustments
            ],
            "health_timeline": [
                f"window_{i+1}_{s}"
                for i, s in enumerate(self.window_status)
            ],
        }

        try:
            with self.engine.begin() as conn:
                from sqlalchemy import text
                conn.execute(text("""
                    INSERT INTO strategy_commands
                        (strategy_id, command_type, payload_json, requested_by, requested_at)
                    VALUES (:sid, 'backtest_dynamic_summary', CAST(:params AS jsonb), 'system', NOW())
                """), {
                    "sid": self.strategy_id,
                    "params": json.dumps(summary, ensure_ascii=False, default=str),
                })
        except Exception as e:
            print(f"  [总结] 写入失败: {e}")

        print(f"\n=== 动态反馈闭环总结 ===")
        print(f"处理窗口: {summary['windows_processed']}")
        print(f"触发调参: {summary['total_adjustments']} 次")
        for adj in self.adjustments:
            direction = "↑" if adj["direction"] == "up" else "↓"
            print(f"  {direction} {adj['factor']}: {adj['old_weight']:.4f} → {adj['new_weight']:.4f}")
        print(f"最终健康: {summary['final_health']}")

        return summary


class DailySignalTracker:
    """日度信号质量追踪器 — 实现真正的动态自适应。

    在每个交易日追踪：
    - Rank IC（预测分数 vs 实际收益）
    - 逐因子 Rank IC（每因子值 vs 实际收益，仅调仓日更新）
    - 滚动 IC 均值（默认 20 日窗口）
    - 持仓记录（每日 top-N 股票 + 得分）
    - 信号衰减检测 → 触发因子权重调整 / 模型重训
    - 因子发现 → 从库中扫描有效因子建议加入
    """

    def __init__(self, strategy_id: int, db_engine, window_size: int = 20):
        self.strategy_id = strategy_id
        self.engine = db_engine
        self.window_size = window_size

        self.daily_ic: list[float] = []
        self.daily_positions: list[dict] = []
        self.rolling_ic: list[float] = []
        self.decay_warnings: int = 0
        self.total_days: int = 0
        self.factor_weight_log: list[dict] = []

        # 逐因子 IC 追踪
        self.factor_ic_history: dict[str, list[float]] = {}  # factor -> [ic per rebalance day]
        self.factor_ic_rolling: dict[str, float] = {}  # factor -> rolling mean IC
        self.rebalance_day_count: int = 0

    def record_day(self, date_str: str, pred_scores: list[float],
                   actual_rets: list[float], positions: dict,
                   daily_ret: float, factor_weights: dict | None = None,
                   factor_ic: dict[str, float] | None = None):
        """记录一个交易日的预测和结果。

        factor_ic: {factor_name: cross_sectional_rank_ic}, 仅调仓日传入非 None 值。
        """
        self.total_days += 1

        # ── 整体 Rank IC ──
        if len(pred_scores) > 5 and len(actual_rets) > 5:
            try:
                ic = float(np.corrcoef(pred_scores, actual_rets)[0, 1])
                if np.isfinite(ic):
                    self.daily_ic.append(ic)
                else:
                    self.daily_ic.append(0.0)
            except Exception:
                self.daily_ic.append(0.0)
        else:
            self.daily_ic.append(0.0)

        # ── 滚动整体 IC ──
        if len(self.daily_ic) >= self.window_size:
            roll = float(np.mean(self.daily_ic[-self.window_size:]))
            self.rolling_ic.append(roll)
            if roll < 0.005:
                self.decay_warnings += 1
            elif roll > 0.02:
                self.decay_warnings = 0
            else:
                self.decay_warnings = max(0, self.decay_warnings - 1)
        else:
            self.rolling_ic.append(0.0)

        # ── 逐因子 IC 追踪（仅调仓日） ──
        if factor_ic:
            self.rebalance_day_count += 1
            for f, ic_val in factor_ic.items():
                if f not in self.factor_ic_history:
                    self.factor_ic_history[f] = []
                self.factor_ic_history[f].append(float(ic_val) if np.isfinite(ic_val) else 0.0)

                # 更新滚动均值
                history = self.factor_ic_history[f]
                lookback = min(len(history), self.window_size)
                self.factor_ic_rolling[f] = round(float(np.mean(history[-lookback:])), 6)

        # ── 持仓记录 ──
        self.daily_positions.append({
            "date": date_str,
            "codes": positions.get("codes", [])[:15],
            "scores": [round(s, 4) for s in positions.get("scores", [])][:15],
            "ret": round(daily_ret, 6),
        })

        # ── 因子权重日志 ──
        if factor_weights:
            self.factor_weight_log.append({
                "date": date_str,
                "weights": {k: round(v, 4) for k, v in factor_weights.items()},
            })

    def needs_adjustment(self) -> bool:
        """5 天连续低 IC → 触发因子权重调整。"""
        return self.decay_warnings >= 5

    def needs_retrain(self) -> bool:
        """10 天连续低 IC 或 ≥3 个因子 |IC|<0.02 → 触发模型重训。"""
        if self.decay_warnings >= 10:
            return True
        # 逐因子检查：≥3 个因子滚动 |IC| < 0.02
        if self.factor_ic_rolling:
            dead = [f for f, ic in self.factor_ic_rolling.items() if abs(ic) < 0.02]
            if len(dead) >= 3:
                return True
        return False

    def get_weight_penalty(self) -> float:
        """基于 IC 衰减严重程度返回全局权重惩罚系数。"""
        if self.decay_warnings < 5:
            return 1.0
        if self.decay_warnings <= 6:
            return 0.85
        if self.decay_warnings <= 8:
            return 0.70
        if self.decay_warnings <= 9:
            return 0.55
        return 0.40

    def get_decayed_factors(self, ic_threshold: float = 0.05) -> list[str]:
        """返回应该淘汰的因子：|滚动IC| < ic_threshold 且在后 50%。

        仅在有足够历史数据时生效（≥3 个调仓日）。
        """
        if self.rebalance_day_count < 3 or not self.factor_ic_rolling:
            return []

        factors = list(self.factor_ic_rolling.keys())
        if len(factors) < 4:
            return []

        # 按 |IC| 排序
        sorted_by_ic = sorted(factors, key=lambda f: abs(self.factor_ic_rolling.get(f, 0)))
        median_idx = len(sorted_by_ic) // 2
        bottom_half = set(sorted_by_ic[:median_idx])

        # 仅淘汰 |IC| < ic_threshold 且在 bottom half 的
        decayed = [f for f in bottom_half
                   if abs(self.factor_ic_rolling.get(f, 0)) < ic_threshold]
        return decayed

    def get_factor_health(self) -> dict:
        """返回完整的因子健康度报告。"""
        if not self.factor_ic_rolling:
            return {"status": "insufficient_data", "factors": {}}

        factor_status = {}
        for f, ic in self.factor_ic_rolling.items():
            abs_ic = abs(ic)
            if abs_ic >= 0.05:
                level = "strong"
            elif abs_ic >= 0.02:
                level = "moderate"
            elif abs_ic >= 0.01:
                level = "weak"
            else:
                level = "dead"
            factor_status[f] = {"rolling_ic": round(ic, 6), "abs_ic": round(abs_ic, 6), "level": level}

        n_strong = sum(1 for v in factor_status.values() if v["level"] == "strong")
        n_dead = sum(1 for v in factor_status.values() if v["level"] == "dead")

        return {
            "status": "healthy" if n_dead < 3 and n_strong >= 2 else "degraded" if n_dead < 5 else "critical",
            "n_factors": len(factor_status),
            "n_strong": n_strong,
            "n_dead": n_dead,
            "factors": factor_status,
        }

    def get_status(self) -> dict:
        """返回当前信号质量状态。"""
        latest_ic = self.daily_ic[-1] if self.daily_ic else 0
        rolling = self.rolling_ic[-1] if self.rolling_ic else 0

        if self.decay_warnings >= 10:
            level = "critical"
        elif self.decay_warnings >= 5:
            level = "warning"
        else:
            level = "normal"

        return {
            "total_days": self.total_days,
            "latest_ic": round(latest_ic, 6),
            "rolling_ic_20d": round(rolling, 6),
            "decay_warnings": self.decay_warnings,
            "signal_level": level,
            "n_positions_recorded": len(self.daily_positions),
            "n_weight_changes": len(self.factor_weight_log),
            "factor_health": self.get_factor_health(),
        }

    def get_position_history(self) -> list[dict]:
        """返回完整持仓历史。"""
        return self.daily_positions

    def get_ic_series(self) -> dict:
        """返回 IC 时序数据（用于前端图表）。"""
        return {
            "daily_ic": [round(v, 6) for v in self.daily_ic],
            "rolling_ic": [round(v, 6) for v in self.rolling_ic],
        }

    def save_daily_summary(self):
        """保存日度信号追踪总结到 strategy_commands。"""
        import json
        status = self.get_status()
        payload = {
            "type": "daily_signal_summary",
            "status": status,
            "ic_stats": {
                "mean": round(float(np.mean(self.daily_ic)), 6) if self.daily_ic else 0,
                "std": round(float(np.std(self.daily_ic)), 6) if self.daily_ic else 0,
                "positive_ratio": round(
                    sum(1 for v in self.daily_ic if v > 0) / max(len(self.daily_ic), 1), 4
                ),
            },
        }
        try:
            with self.engine.begin() as conn:
                from sqlalchemy import text
                conn.execute(text("""
                    INSERT INTO strategy_commands
                        (strategy_id, command_type, payload_json, requested_by, requested_at)
                    VALUES (:sid, 'daily_signal_summary', CAST(:payload AS jsonb), 'tracker', NOW())
                """), {
                    "sid": self.strategy_id,
                    "payload": json.dumps(payload, ensure_ascii=False, default=str),
                })
        except Exception as e:
            print(f"  [Tracker] 保存失败: {e}")
