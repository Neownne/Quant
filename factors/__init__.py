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
from factors.alpha191_vol import ALPHA191_VOL
from factors.alpha191_liquidity import ALPHA191_LIQUIDITY
from factors.fundamental import FUNDAMENTAL_FACTORS
from factors.intraday_minute import INTRADAY_MINUTE_FACTORS
from factors.market_breadth import MARKET_BREADTH_FACTORS
from factors.limit_up import LIMIT_UP_FACTORS

ALL_FACTORS: dict = {
    **ALPHA101_FUNCTIONS,
    **CUSTOM_FACTORS,
    **ALPHA191_TURNOVER,
    **ALPHA191_INTRADAY,
    **ALPHA191_FLOW,
    **ALPHA191_GAP,
    **ALPHA191_VOL,
    **ALPHA191_LIQUIDITY,
    **FUNDAMENTAL_FACTORS,
    **INTRADAY_MINUTE_FACTORS,
    **MARKET_BREADTH_FACTORS,
    **LIMIT_UP_FACTORS,
}

__all__ = ["FactorEngine", "ALL_FACTORS"]
