"""ML 策略配置管理：CRUD + 内置预设。"""
from __future__ import annotations

import json
import pandas as pd
from sqlalchemy import text
from data.db import get_engine


def create_ml_config(name: str, **kwargs) -> int:
    """创建 ML 策略配置，返回 config_id。"""
    engine = get_engine()
    try:
        # Build INSERT from kwargs, handling JSONB columns
        allowed = {
            "name", "description", "factor_names", "ic_threshold", "t_threshold",
            "orthogonal_threshold", "label_mode", "forward_days", "train_years",
            "val_years", "model_type", "top_n", "rebalance_mode", "ndrop_n",
            "max_single", "max_industry", "stop_loss_pct", "atr_multiplier",
            "atr_period", "portfolio_dd_threshold", "portfolio_dd_reduce_to",
            "max_dd_limit", "stock_pool", "stock_count",
        }
        params = {"name": name}
        cols = ["name"]
        vals = [":name"]
        for k, v in kwargs.items():
            if k in allowed and v is not None:
                cols.append(k)
                if k == "factor_names":
                    vals.append("CAST(:factor_names AS jsonb)")
                    params["factor_names"] = json.dumps(v) if not isinstance(v, str) else v
                else:
                    vals.append(f":{k}")
                    params[k] = v

        with engine.begin() as conn:
            result = conn.execute(text(
                f"INSERT INTO ml_strategy_config ({', '.join(cols)}) "
                f"VALUES ({', '.join(vals)}) RETURNING id"
            ), params)
            config_id = result.scalar()
        return config_id
    finally:
        engine.dispose()


def get_ml_config(config_id: int) -> dict | None:
    """获取单个 ML 策略配置。"""
    engine = get_engine()
    try:
        with engine.connect() as conn:
            row = conn.execute(text(
                "SELECT * FROM ml_strategy_config WHERE id = :cid"
            ), {"cid": config_id}).fetchone()
        if row is None:
            return None
        keys = [
            "id", "name", "description", "factor_names", "ic_threshold",
            "t_threshold", "orthogonal_threshold", "label_mode", "forward_days",
            "train_years", "val_years", "model_type", "top_n", "rebalance_mode",
            "ndrop_n", "max_single", "max_industry", "stop_loss_pct",
            "atr_multiplier", "atr_period", "portfolio_dd_threshold",
            "portfolio_dd_reduce_to", "max_dd_limit", "stock_pool", "stock_count",
            "created_at", "updated_at",
        ]
        result = dict(zip(keys, row))
        fn = result.get("factor_names")
        if isinstance(fn, str):
            result["factor_names"] = json.loads(fn)
        elif fn is None:
            result["factor_names"] = []
        return result
    finally:
        engine.dispose()


def get_ml_config_by_name(name: str) -> dict | None:
    """按名称获取 ML 策略配置。"""
    engine = get_engine()
    try:
        with engine.connect() as conn:
            row = conn.execute(text(
                "SELECT id FROM ml_strategy_config WHERE name = :n"
            ), {"n": name}).fetchone()
        if row is None:
            return None
        return get_ml_config(row[0])
    finally:
        engine.dispose()


def list_ml_configs() -> pd.DataFrame:
    """列出所有 ML 策略配置。"""
    engine = get_engine()
    try:
        with engine.connect() as conn:
            df = pd.read_sql_query(text(
                "SELECT id, name, description, model_type, top_n, rebalance_mode, "
                "train_years, label_mode, created_at FROM ml_strategy_config ORDER BY id"
            ), conn)
    finally:
        engine.dispose()
    return df


def update_ml_config(config_id: int, **kwargs) -> None:
    """更新 ML 策略配置。"""
    if not kwargs:
        return
    sets = []
    params = {}
    allowed = {
        "name", "description", "factor_names", "ic_threshold", "t_threshold",
        "orthogonal_threshold", "label_mode", "forward_days", "train_years",
        "val_years", "model_type", "top_n", "rebalance_mode", "ndrop_n",
        "max_single", "max_industry", "stop_loss_pct", "atr_multiplier",
        "atr_period", "portfolio_dd_threshold", "portfolio_dd_reduce_to",
        "max_dd_limit", "stock_pool", "stock_count",
    }
    for k, v in kwargs.items():
        if k in allowed:
            if k == "factor_names":
                sets.append("factor_names = CAST(:factor_names AS jsonb)")
                params["factor_names"] = json.dumps(v) if not isinstance(v, str) else v
            else:
                sets.append(f"{k} = :{k}")
                params[k] = v
    sets.append("updated_at = NOW()")
    params["cid"] = config_id
    engine = get_engine()
    try:
        with engine.begin() as conn:
            conn.execute(text(
                f"UPDATE ml_strategy_config SET {', '.join(sets)} WHERE id = :cid"
            ), params)
    finally:
        engine.dispose()


def delete_ml_config(config_id: int) -> None:
    """删除 ML 策略配置。"""
    engine = get_engine()
    try:
        with engine.begin() as conn:
            conn.execute(text(
                "DELETE FROM ml_strategy_config WHERE id = :cid"
            ), {"cid": config_id})
    finally:
        engine.dispose()


def seed_builtin_configs() -> None:
    """插入 3 个内置 ML 策略预设（如已存在则跳过）。"""
    builtins = [
        {
            "name": "ML-默认集成",
            "description": "全因子XGBoost+LightGBM集成，NDrop增量调仓",
            "factor_names": [],
            "ic_threshold": 0.02,
            "t_threshold": 2.0,
            "orthogonal_threshold": 0.7,
            "label_mode": "binary",
            "forward_days": 1,
            "train_years": 3,
            "val_years": 1,
            "model_type": "ensemble",
            "top_n": 15,
            "rebalance_mode": "ndrop",
            "ndrop_n": 2,
        },
        {
            "name": "ML-动量精选",
            "description": "趋势类因子为主，5日收益预测，全量换仓",
            "factor_names": ["mom_20", "mom_60", "ema_ratio_5_20", "vwap_ratio",
                             "turnover_ret_corr", "obv_roc", "force_index"],
            "ic_threshold": 0.015,
            "t_threshold": 1.8,
            "orthogonal_threshold": 0.7,
            "label_mode": "binary",
            "forward_days": 5,
            "train_years": 3,
            "val_years": 1,
            "model_type": "ensemble",
            "top_n": 10,
            "rebalance_mode": "full",
            "ndrop_n": 2,
        },
        {
            "name": "ML-反转精选",
            "description": "反转+波动率因子，1日收益预测，NDrop调仓",
            "factor_names": ["rev_5", "rev_10", "rsi_7", "vol_20", "atr_14",
                             "down_vol_ratio", "lower_shadow", "intra_day_rev"],
            "ic_threshold": 0.02,
            "t_threshold": 2.0,
            "orthogonal_threshold": 0.65,
            "label_mode": "binary",
            "forward_days": 1,
            "train_years": 2,
            "val_years": 1,
            "model_type": "ensemble",
            "top_n": 15,
            "rebalance_mode": "ndrop",
            "ndrop_n": 3,
        },
    ]
    for cfg in builtins:
        name = cfg.pop("name")
        try:
            create_ml_config(name, **cfg)
        except Exception:
            pass  # 已存在则跳过
