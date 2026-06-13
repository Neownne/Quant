"""涨停策略核心逻辑单元测试。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import pytest

from strategies.limit_up.base import LimitUpParams, run_screening
from strategies.limit_up.execution import _calc_trade_cost
from config.settings import TradingConfig


def _make_daily(codes, dates, closes):
    """构造日线 DataFrame。"""
    rows = []
    for code in codes:
        for i, d in enumerate(dates):
            rows.append({"code": code, "trade_date": d, "open": closes[code][i] * 0.99,
                         "close": closes[code][i]})
    df = pd.DataFrame(rows)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values(["code", "trade_date"]).reset_index(drop=True)
    df["ret"] = df.groupby("code")["close"].pct_change()
    df["ma5"] = df.groupby("code")["close"].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df["ma10"] = df.groupby("code")["close"].transform(lambda x: x.rolling(10, min_periods=1).mean())
    return df


def _make_extra(codes, dates, mcaps):
    rows = []
    for code in codes:
        for d in dates:
            rows.append({"code": code, "trade_date": d, "market_cap": mcaps[code]})
    df = pd.DataFrame(rows)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


def test_run_screening_basic():
    dates = pd.date_range("2026-06-01", periods=25, freq="B")
    closes = {
        "000001": [10.0] * 25,
        "000002": [10.0] * 25,
        "300001": [10.0] * 25,
        "688001": [10.0] * 25,
    }
    # 让 000002 在最后 10 天持续上涨，并出现两次涨停（>1 次）
    for i in range(15, 25):
        closes["000002"][i] = 10.0 + (i - 14) * 0.25
    closes["000002"][19] = closes["000002"][18] * 1.09
    closes["000002"][22] = closes["000002"][21] * 1.09

    daily = _make_daily(list(closes.keys()), dates, closes)
    mcaps = {"000001": 50, "000002": 50, "300001": 50, "688001": 50}
    extra = _make_extra(list(closes.keys()), dates, mcaps)
    code_set = set(closes.keys())

    params = LimitUpParams(
        mcap_min=30, mcap_max=500, price_min=5, price_max=63,
        lu_pct=0.09, lu_lookback=20, lu_count=1, min_conditions=4,
    )
    signals = run_screening(dates[-1], daily, extra, code_set, params)
    codes = [s[0] for s in signals]
    assert "000002" in codes
    assert len(signals) == 1


def test_run_screening_price_filter():
    dates = pd.date_range("2026-06-01", periods=25, freq="B")
    closes = {
        "000001": [100.0] * 25,   # 股价过高
        "000002": [2.0] * 25,     # 股价过低
        "300001": [10.0] * 25,    # 正常
    }
    # 让 300001 在最后 10 天上涨并涨停两次
    for i in range(15, 25):
        closes["300001"][i] = 10.0 + (i - 14) * 0.25
    closes["300001"][19] = closes["300001"][18] * 1.09
    closes["300001"][22] = closes["300001"][21] * 1.09

    daily = _make_daily(list(closes.keys()), dates, closes)
    mcaps = {"000001": 50, "000002": 50, "300001": 50}
    extra = _make_extra(list(closes.keys()), dates, mcaps)
    code_set = set(closes.keys())

    params = LimitUpParams(
        mcap_min=30, mcap_max=500, price_min=5, price_max=63,
        lu_pct=0.09, lu_lookback=20, lu_count=1, min_conditions=4,
    )
    signals = run_screening(dates[-1], daily, extra, code_set, params)
    codes = [s[0] for s in signals]
    assert "000001" not in codes
    assert "000002" not in codes
    assert "300001" in codes


def test_run_screening_ma_filter():
    dates = pd.date_range("2026-06-01", periods=25, freq="B")
    closes = {
        "000001": [10.0] * 25,
    }
    # 让价格持续下跌，MA5 < MA10
    for i in range(15, 25):
        closes["000001"][i] = 10.0 - (i - 14) * 0.2
    # 涨停一次
    closes["000001"][23] = closes["000001"][22] * 1.09

    daily = _make_daily(["000001"], dates, closes)
    mcaps = {"000001": 50}
    extra = _make_extra(["000001"], dates, mcaps)

    params = LimitUpParams(
        mcap_min=30, mcap_max=500, price_min=5, price_max=63,
        lu_pct=0.09, lu_lookback=20, lu_count=1, min_conditions=4,
    )
    signals = run_screening(dates[-1], daily, extra, {"000001"}, params)
    # MA5 < MA10，不应入选
    assert len(signals) == 0


def test_trade_cost():
    assert _calc_trade_cost(100000, "BUY") == pytest.approx(100000 * (TradingConfig.COMMISSION + TradingConfig.SLIPPAGE))
    assert _calc_trade_cost(100000, "SELL") == pytest.approx(
        100000 * (TradingConfig.COMMISSION + TradingConfig.STAMP_DUTY + TradingConfig.SLIPPAGE)
    )


def test_limit_up_params_defaults():
    p = LimitUpParams()
    assert p.lu_pct == TradingConfig.LIMIT_UP_PCT
    assert p.mcap_min == 30.0
    assert p.min_conditions == 4
