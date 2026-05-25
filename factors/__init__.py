"""因子计算模块。

用法:
    from factors import FactorEngine, ALL_FACTORS
    engine = FactorEngine(factor_names=["rsi_14", "mom_20"])
    result = engine.compute(df_ohlcv)
"""
from factors.engine import FactorEngine
from factors.alpha101 import ALPHA101_FUNCTIONS
from factors.custom import CUSTOM_FACTORS
from factors.alpha191_turnover import ALPHA191_TURNOVER
from factors.alpha191_intraday import ALPHA191_INTRADAY

ALL_FACTORS: dict = {
    **ALPHA101_FUNCTIONS,
    **CUSTOM_FACTORS,
    **ALPHA191_TURNOVER,
    **ALPHA191_INTRADAY,
}

__all__ = ["FactorEngine", "ALL_FACTORS"]
