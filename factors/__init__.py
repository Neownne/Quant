"""因子计算模块。

用法:
    from factors import FactorEngine, ALL_FACTORS
    engine = FactorEngine(factor_names=["rsi_14", "mom_20"])
    result = engine.compute(df_ohlcv)
"""
from factors.engine import FactorEngine

# 因子注册表将在后续任务中填充
# 此处先定义空注册表，确保模块可导入
ALL_FACTORS: dict = {}

__all__ = ["FactorEngine", "ALL_FACTORS"]
