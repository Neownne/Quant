"""统一模拟账户管理：创建、查询、更新、策略绑定。"""
from __future__ import annotations

import pandas as pd
from sqlalchemy import text

from data.db import get_engine
from config.settings import AccountConfig


def create_account(
    name: str,
    initial_capital: float = AccountConfig.DEFAULT_CASH,
    strategy_type: str = "ml",
    strategy_name: str = "",
    strategy_params: dict | None = None,
    commission_rate: float = AccountConfig.DEFAULT_COMMISSION,
    stamp_duty_rate: float = AccountConfig.DEFAULT_STAMP_DUTY,
    slippage: float = AccountConfig.DEFAULT_SLIPPAGE,
    use_market_filter: bool = True,
) -> int:
    """创建模拟账户，返回 account_id。"""
    import json
    engine = get_engine()
    try:
        with engine.begin() as conn:
            result = conn.execute(text("""
                INSERT INTO paper_account (name, initial_capital, cash,
                    strategy_type, strategy_name, strategy_params,
                    commission_rate, stamp_duty_rate, slippage, use_market_filter)
                VALUES (:name, :cap, :cap,
                    :stype, :sname, CAST(:sparams AS jsonb),
                    :comm, :stamp, :slip, :market)
                RETURNING id
            """), {
                "name": name,
                "cap": initial_capital,
                "stype": strategy_type,
                "sname": strategy_name,
                "sparams": json.dumps(strategy_params or {}),
                "comm": commission_rate,
                "stamp": stamp_duty_rate,
                "slip": slippage,
                "market": use_market_filter,
            })
            account_id = result.scalar()
        return account_id
    finally:
        engine.dispose()


def get_account(account_id: int) -> dict | None:
    """获取账户完整配置。"""
    engine = get_engine()
    try:
        with engine.connect() as conn:
            row = conn.execute(text(
                "SELECT * FROM paper_account WHERE id = :aid"
            ), {"aid": account_id}).fetchone()
        if row is None:
            return None
        keys = [
            "id", "name", "initial_capital", "cash", "created_at",
            "strategy_type", "strategy_name", "strategy_params",
            "commission_rate", "stamp_duty_rate", "slippage", "use_market_filter",
        ]
        result = dict(zip(keys, row))
        # Convert strategy_params from string/JSONB to dict
        sp = result.get("strategy_params")
        if isinstance(sp, str):
            import json
            result["strategy_params"] = json.loads(sp)
        elif sp is None:
            result["strategy_params"] = {}
        return result
    finally:
        engine.dispose()


def update_account_config(account_id: int, **kwargs) -> None:
    """更新账户策略/费率配置。"""
    if not kwargs:
        return
    import json
    sets = []
    params = {}
    for k, v in kwargs.items():
        safe_k = k
        if safe_k == "strategy_params":
            params["sparams"] = json.dumps(v) if not isinstance(v, str) else v
            sets.append("strategy_params = CAST(:sparams AS jsonb)")
        else:
            params[safe_k] = v
            sets.append(f"{safe_k} = :{safe_k}")
    params["aid"] = account_id
    engine = get_engine()
    try:
        with engine.begin() as conn:
            conn.execute(text(
                f"UPDATE paper_account SET {', '.join(sets)} WHERE id = :aid"
            ), params)
    finally:
        engine.dispose()


def list_accounts() -> pd.DataFrame:
    """列出所有模拟账户。"""
    engine = get_engine()
    try:
        with engine.connect() as conn:
            df = pd.read_sql_query(text(
                "SELECT id, name, initial_capital, cash, strategy_type, "
                "strategy_name, created_at FROM paper_account ORDER BY id"
            ), conn)
    finally:
        engine.dispose()
    return df


def promote_strategy_to_account(
    strategy_type: str,
    strategy_name: str,
    strategy_params: dict,
    account_name: str = "",
    initial_capital: float = AccountConfig.DEFAULT_CASH,
) -> int:
    """将回测策略升级为模拟账户，返回 account_id。"""
    name = account_name or f"{strategy_name}_{pd.Timestamp.now().strftime('%m%d_%H%M')}"
    return create_account(
        name=name,
        initial_capital=initial_capital,
        strategy_type=strategy_type,
        strategy_name=strategy_name,
        strategy_params=strategy_params,
    )
