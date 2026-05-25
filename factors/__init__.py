"""因子计算模块。

用法:
    from factors import FactorEngine, ALL_FACTORS
    engine = FactorEngine(factor_names=["rsi_14", "mom_20"])
    result = engine.compute(df_ohlcv)
"""
from factors.engine import FactorEngine
from factors.alpha101 import ALPHA101_FUNCTIONS

# ALL_FACTORS 将在后续任务中加入 alpha191 和 custom
ALL_FACTORS: dict = dict(ALPHA101_FUNCTIONS)

__all__ = ["FactorEngine", "ALL_FACTORS"]
