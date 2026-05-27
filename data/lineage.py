"""因子血缘跟踪。

记录每个因子依赖的原始字段和计算公式哈希，当源数据变更时
可快速定位需要重新计算的因子列表。
"""

import hashlib

from sqlalchemy import text

from data.db import get_engine


def register_lineage(
    factor_name: str,
    source_fields: list[str],
    computation_formula: str,
    upstream_factors: list[str] | None = None,
) -> None:
    """登记因子血缘信息（幂等：同名因子存在则更新）。"""
    formula_hash = hashlib.sha256(computation_formula.encode()).hexdigest()[:16]
    with get_engine().begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO factor_lineage
                    (factor_name, source_fields, computation_formula_hash, upstream_factors)
                VALUES (:name, :fields, :hash, :upstream)
                ON CONFLICT (factor_name) DO UPDATE SET
                    source_fields = EXCLUDED.source_fields,
                    computation_formula_hash = EXCLUDED.computation_formula_hash,
                    upstream_factors = EXCLUDED.upstream_factors,
                    last_validated_at = NOW()
                """
            ),
            {
                "name": factor_name,
                "fields": source_fields,
                "hash": formula_hash,
                "upstream": upstream_factors or [],
            },
        )


def find_dirty_factors(changed_field: str) -> list[str]:
    """给定一个变更的原始字段名，返回所有依赖该字段的因子名列表。"""
    with get_engine().connect() as conn:
        rows = conn.execute(
            text("SELECT factor_name, source_fields FROM factor_lineage")
        ).fetchall()

    dirty: list[str] = []
    for factor_name, source_fields in rows:
        if changed_field in (source_fields or []):
            dirty.append(factor_name)
    return dirty
