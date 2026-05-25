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
from factors.alpha191_flow import ALPHA191_FLOW
from factors.alpha191_gap import ALPHA191_GAP

ALL_FACTORS: dict = {
    **ALPHA101_FUNCTIONS,
    **CUSTOM_FACTORS,
    **ALPHA191_TURNOVER,
    **ALPHA191_INTRADAY,
    **ALPHA191_FLOW,
    **ALPHA191_GAP,
}

__all__ = ["FactorEngine", "ALL_FACTORS"]
