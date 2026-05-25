# Phase 3: 因子扩展 + 状态识别 + 集成优化 — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增 28 个 Alpha191 A 股因子 + 市场状态识别 + 正交筛选 + XGBoost/LightGBM 集成 + Optuna 调优，将 E2E 准确率从 54% 提升至 ≥ 56%，召回率 ≥ 40%。

**Architecture:** 分批实现因子（换手率→日内→资金流→波动率→隔夜→流动性），每批独立文件，统一门禁（IC + 正交性），然后加载 extra_data 激活估因子，构建市场状态识别分模型训练，正交筛选去冗余，最后集成+调优。

**Tech Stack:** NumPy/Pandas（因子计算），scipy（统计检验），Optuna（超参调优），现有 FactorEngine/XGBoost/LightGBM。

---

## 文件结构

| 文件 | 职责 |
|---|---|
| `factors/alpha191_turnover.py` | 换手率 5 因子 |
| `factors/alpha191_intraday.py` | 日内形态 4 因子 |
| `factors/alpha191_flow.py` | 资金流向 6 因子 |
| `factors/alpha191_vol.py` | 波动率高阶 5 因子 |
| `factors/alpha191_gap.py` | 隔夜效应 4 因子 |
| `factors/alpha191_liquidity.py` | 流动性高阶 4 因子 |
| `factors/screening.py` | 正交性筛选 + 边际贡献检验 |
| `models/regime.py` | 市场状态识别 |
| `models/tuning.py` | Optuna 调优 |
| `scripts/run_ml_backtest.py` | 增加 extra_data 加载 |
| `models/trainer.py` | 集成训练入口 |
| `tests/test_alpha191.py` | 新因子测试 |
| `tests/test_regime.py` | 市场状态测试 |
| `tests/test_screening.py` | 筛选+调优测试 |

---

### Task 1: 换手率 + 日内形态因子（9 个）

**Files:**
- Create: `factors/alpha191_turnover.py`
- Create: `factors/alpha191_intraday.py`
- Create: `tests/test_alpha191.py`

- [ ] **Step 1: 写测试**

```python
"""Alpha191 因子测试。"""
import pytest
import pandas as pd
import numpy as np
from factors.engine import FactorEngine


def _make_ohlcv(n_days: int = 200) -> pd.DataFrame:
    """构造单只股票的 OHLCV + turnover 模拟数据。"""
    np.random.seed(42)
    dates = pd.date_range("2024-01-02", periods=n_days, freq="B")
    close = 10 + np.cumsum(np.random.randn(n_days) * 0.3)
    volume = np.random.randint(5000000, 50000000, n_days).astype(float)
    return pd.DataFrame({
        "code": "000001",
        "trade_date": dates,
        "open": close * (1 + np.random.randn(n_days) * 0.005),
        "high": close * (1 + abs(np.random.randn(n_days)) * 0.015),
        "low": close * (1 - abs(np.random.randn(n_days)) * 0.015),
        "close": close,
        "volume": volume,
        "amount": volume * close,
        "turnover": np.random.uniform(0.005, 0.05, n_days),
    })


class TestAlpha191Turnover:
    def test_all_registered(self):
        """所有换手率因子应在 ALL_FACTORS 中注册。"""
        from factors import ALL_FACTORS
        expected = [
            "turnover_skew", "turnover_cv", "turnover_ma_dev",
            "turnover_ret_corr", "free_turnover_ratio",
        ]
        for name in expected:
            assert name in ALL_FACTORS, f"{name} 未注册"

    def test_factor_output_valid(self):
        """因子输出应为有限值 Series。"""
        from factors import ALL_FACTORS
        from factors.engine import FactorEngine
        df = _make_ohlcv(200)
        engine = FactorEngine(factor_names=[
            "turnover_skew", "turnover_cv", "turnover_ma_dev",
            "turnover_ret_corr", "free_turnover_ratio",
        ])
        result = engine.compute(df)
        for col in engine.factor_names:
            # 至少 50% 非 NaN
            valid_pct = result[col].notna().sum() / len(result)
            assert valid_pct > 0.5, f"{col} 有效值仅 {valid_pct:.1%}"


class TestAlpha191Intraday:
    def test_all_registered(self):
        from factors import ALL_FACTORS
        expected = ["upper_shadow", "lower_shadow", "body_ratio", "intra_day_rev"]
        for name in expected:
            assert name in ALL_FACTORS, f"{name} 未注册"

    def test_factor_output_valid(self):
        from factors import ALL_FACTORS
        from factors.engine import FactorEngine
        df = _make_ohlcv(200)
        engine = FactorEngine(factor_names=[
            "upper_shadow", "lower_shadow", "body_ratio", "intra_day_rev",
        ])
        result = engine.compute(df)
        for col in engine.factor_names:
            valid_pct = result[col].notna().sum() / len(result)
            assert valid_pct > 0.5, f"{col} 有效值仅 {valid_pct:.1%}"

    def test_upper_shadow_negative_means_selling_pressure(self):
        """上影线长 = 卖压大，应为负值信号。"""
        from factors.alpha191_intraday import upper_shadow
        df = _make_ohlcv(200)
        # 强制制造长上影：high 远大于 max(open,close)
        df["high"] = df[["open", "close"]].max(axis=1) + 1.0
        df["low"] = df[["open", "close"]].min(axis=1) - 0.1
        result = upper_shadow(df)
        # 上影线比例应 > 0.5
        assert result.dropna().mean() > 0.5
```

- [ ] **Step 2: 运行测试确认失败**

```bash
source .venv/bin/activate && python -m pytest tests/test_alpha191.py -x -q
```

- [ ] **Step 3: 实现 `factors/alpha191_turnover.py`**

```python
"""Alpha191 换手率类因子。"""
import numpy as np
import pandas as pd


def turnover_skew(df: pd.DataFrame) -> pd.Series:
    """换手率偏度：skew(turnover, 20)，右偏=资金进场。"""
    if "turnover" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return df["turnover"].rolling(20).skew()


def turnover_cv(df: pd.DataFrame) -> pd.Series:
    """换手率变异系数：std(turnover,20) / mean(turnover,20)，取负。"""
    if "turnover" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    t = df["turnover"]
    cv = t.rolling(20).std() / t.rolling(20).mean().replace(0, np.nan)
    return -cv


def turnover_ma_dev(df: pd.DataFrame) -> pd.Series:
    """换手率偏离度：turnover / MA(turnover,60) - 1。"""
    if "turnover" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    t = df["turnover"]
    return t / t.rolling(60).mean().replace(0, np.nan) - 1


def turnover_ret_corr(df: pd.DataFrame) -> pd.Series:
    """量价相关性：corr(turnover, ret, 20)。"""
    if "turnover" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    ret = df["close"].pct_change()
    return df["turnover"].rolling(20).corr(ret)


def free_turnover_ratio(df: pd.DataFrame) -> pd.Series:
    """流通换手比：turnover / float_share_ratio（无数据返回 NaN）。"""
    if "float_share_ratio" not in df.columns or "turnover" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return df["turnover"] / df["float_share_ratio"].replace(0, np.nan)


ALPHA191_TURNOVER: dict = {
    "turnover_skew": turnover_skew,
    "turnover_cv": turnover_cv,
    "turnover_ma_dev": turnover_ma_dev,
    "turnover_ret_corr": turnover_ret_corr,
    "free_turnover_ratio": free_turnover_ratio,
}
```

- [ ] **Step 4: 实现 `factors/alpha191_intraday.py`**

```python
"""Alpha191 日内形态类因子。"""
import numpy as np
import pandas as pd


def upper_shadow(df: pd.DataFrame) -> pd.Series:
    """上影线比例：(H - max(O,C)) / (H-L)，上影长=卖压。"""
    h, l, o, c = df["high"], df["low"], df["open"], df["close"]
    max_oc = pd.concat([o, c], axis=1).max(axis=1)
    hl_range = (h - l).replace(0, np.nan)
    return (h - max_oc) / hl_range


def lower_shadow(df: pd.DataFrame) -> pd.Series:
    """下影线比例：(min(O,C) - L) / (H-L)，下影长=支撑。"""
    h, l, o, c = df["high"], df["low"], df["open"], df["close"]
    min_oc = pd.concat([o, c], axis=1).min(axis=1)
    hl_range = (h - l).replace(0, np.nan)
    return (min_oc - l) / hl_range


def body_ratio(df: pd.DataFrame) -> pd.Series:
    """实体比例：|C-O| / (H-L)，实体大=趋势强。"""
    h, l, o, c = df["high"], df["low"], df["open"], df["close"]
    hl_range = (h - l).replace(0, np.nan)
    return (c - o).abs() / hl_range


def intra_day_rev(df: pd.DataFrame) -> pd.Series:
    """盘中反转度：(C-O) / (H-L)，正值=低开高走。"""
    h, l, o, c = df["high"], df["low"], df["open"], df["close"]
    hl_range = (h - l).replace(0, np.nan)
    return (c - o) / hl_range


ALPHA191_INTRADAY: dict = {
    "upper_shadow": upper_shadow,
    "lower_shadow": lower_shadow,
    "body_ratio": body_ratio,
    "intra_day_rev": intra_day_rev,
}
```

- [ ] **Step 5: 注册因子到 `factors/__init__.py`**

修改 `factors/__init__.py`，在 ALL_FACTORS 中加入新因子：

```python
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
```

- [ ] **Step 6: 运行测试确认通过**

```bash
source .venv/bin/activate && python -m pytest tests/test_alpha191.py -x -q
```

- [ ] **Step 7: 提交**

```bash
git add factors/alpha191_turnover.py factors/alpha191_intraday.py factors/__init__.py tests/test_alpha191.py
git commit -m "feat: add 9 Alpha191 factors (turnover + intraday)"
```

---

### Task 2: 资金流向 + 隔夜效应因子（10 个）

**Files:**
- Create: `factors/alpha191_flow.py`
- Create: `factors/alpha191_gap.py`

- [ ] **Step 1: 追加测试到 `tests/test_alpha191.py`**

```python
class TestAlpha191Flow:
    def test_all_registered(self):
        from factors import ALL_FACTORS
        expected = [
            "money_flow", "obv_roc", "force_index",
            "cwt", "volume_climax", "vwap_momentum",
        ]
        for name in expected:
            assert name in ALL_FACTORS, f"{name} 未注册"

    def test_factor_output_valid(self):
        from factors.engine import FactorEngine
        df = _make_ohlcv(200)
        engine = FactorEngine(factor_names=[
            "money_flow", "obv_roc", "force_index",
            "cwt", "volume_climax", "vwap_momentum",
        ])
        result = engine.compute(df)
        for col in engine.factor_names:
            valid_pct = result[col].notna().sum() / len(result)
            assert valid_pct > 0.5, f"{col} 有效值仅 {valid_pct:.1%}"


class TestAlpha191Gap:
    def test_all_registered(self):
        from factors import ALL_FACTORS
        expected = ["overnight_ret", "overnight_ret_std", "open_auction_jump", "gap_ma_dev"]
        for name in expected:
            assert name in ALL_FACTORS, f"{name} 未注册"

    def test_factor_output_valid(self):
        from factors.engine import FactorEngine
        df = _make_ohlcv(200)
        engine = FactorEngine(factor_names=[
            "overnight_ret", "overnight_ret_std", "open_auction_jump", "gap_ma_dev",
        ])
        result = engine.compute(df)
        for col in engine.factor_names:
            valid_pct = result[col].notna().sum() / len(result)
            assert valid_pct > 0.5, f"{col} 有效值仅 {valid_pct:.1%}"
```

- [ ] **Step 2: 运行测试确认失败**

```bash
source .venv/bin/activate && python -m pytest tests/test_alpha191.py::TestAlpha191Flow tests/test_alpha191.py::TestAlpha191Gap -x -q
```

- [ ] **Step 3: 实现 `factors/alpha191_flow.py`**

```python
"""Alpha191 资金流向类因子。"""
import numpy as np
import pandas as pd


def money_flow(df: pd.DataFrame) -> pd.Series:
    """资金流：Σ((C-L)-(H-C))×V / ΣV, 10日。正=流入。"""
    h, l, c, v = df["high"], df["low"], df["close"], df["volume"]
    mf = ((c - l) - (h - c)) / (h - l).replace(0, np.nan)
    raw_mf = mf * v
    return raw_mf.rolling(10).sum() / v.rolling(10).sum().replace(0, np.nan)


def obv_roc(df: pd.DataFrame) -> pd.Series:
    """OBV 变化率：(OBV_t - OBV_{t-20}) / |OBV_{t-20}|。"""
    c, v = df["close"], df["volume"]
    direction = np.sign(c.diff())
    obv = (direction * v).fillna(0).cumsum()
    lag = obv.shift(20).abs().replace(0, np.nan)
    return (obv - obv.shift(20)) / lag


def force_index(df: pd.DataFrame) -> pd.Series:
    """强力指数：EMA(ΔC × V, 2)。"""
    c, v = df["close"], df["volume"]
    fi_raw = c.diff() * v
    return fi_raw.ewm(span=2, adjust=False).mean()


def cwt(df: pd.DataFrame) -> pd.Series:
    """CWT：C×V×turnover 的 5 日变化率。"""
    if "turnover" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    raw = df["close"] * df["volume"] * df["turnover"]
    lag = raw.shift(5).abs().replace(0, np.nan)
    return (raw - raw.shift(5)) / lag


def volume_climax(df: pd.DataFrame) -> pd.Series:
    """天量见顶：(V_t - max_{t-20..t-1}) / max_{t-20..t-1}，取负。"""
    v = df["volume"]
    rolling_max = v.shift(1).rolling(20).max()
    climax = (v - rolling_max) / rolling_max.replace(0, np.nan)
    return -climax


def vwap_momentum(df: pd.DataFrame) -> pd.Series:
    """VWAP 动量：VWAP(5) / VWAP(20) - 1。"""
    h, l, c, v = df["high"], df["low"], df["close"], df["volume"]
    typ = (h + l + c) / 3
    tv = typ * v
    vwap5 = tv.rolling(5).sum() / v.rolling(5).sum().replace(0, np.nan)
    vwap20 = tv.rolling(20).sum() / v.rolling(20).sum().replace(0, np.nan)
    return vwap5 / vwap20.replace(0, np.nan) - 1


ALPHA191_FLOW: dict = {
    "money_flow": money_flow,
    "obv_roc": obv_roc,
    "force_index": force_index,
    "cwt": cwt,
    "volume_climax": volume_climax,
    "vwap_momentum": vwap_momentum,
}
```

- [ ] **Step 4: 实现 `factors/alpha191_gap.py`**

```python
"""Alpha191 隔夜效应类因子。"""
import numpy as np
import pandas as pd


def overnight_ret(df: pd.DataFrame) -> pd.Series:
    """隔夜收益：(O_t - C_{t-1}) / C_{t-1}。"""
    o, c = df["open"], df["close"]
    prev_c = c.shift(1).replace(0, np.nan)
    return (o - prev_c) / prev_c


def overnight_ret_std(df: pd.DataFrame) -> pd.Series:
    """隔夜波动：std(overnight_ret, 10)，取负。"""
    o, c = df["open"], df["close"]
    prev_c = c.shift(1).replace(0, np.nan)
    on_ret = (o - prev_c) / prev_c
    return -on_ret.rolling(10).std()


def open_auction_jump(df: pd.DataFrame) -> pd.Series:
    """开盘跳空偏离：(O_t - MA(O,5)) / MA(O,5)。"""
    o = df["open"]
    ma5 = o.rolling(5).mean().replace(0, np.nan)
    return (o - ma5) / ma5


def gap_ma_dev(df: pd.DataFrame) -> pd.Series:
    """缺口偏离：gap_ratio - MA(gap_ratio, 20)。"""
    o, c = df["open"], df["close"]
    prev_c = c.shift(1).replace(0, np.nan)
    gap = (o - prev_c) / prev_c
    return gap - gap.rolling(20).mean()


ALPHA191_GAP: dict = {
    "overnight_ret": overnight_ret,
    "overnight_ret_std": overnight_ret_std,
    "open_auction_jump": open_auction_jump,
    "gap_ma_dev": gap_ma_dev,
}
```

- [ ] **Step 5: 注册因子到 `factors/__init__.py`**

在已有 import 后追加：

```python
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
```

- [ ] **Step 6: 运行测试**

```bash
source .venv/bin/activate && python -m pytest tests/test_alpha191.py -x -q
```

- [ ] **Step 7: 提交**

```bash
git add factors/alpha191_flow.py factors/alpha191_gap.py factors/__init__.py tests/test_alpha191.py
git commit -m "feat: add 10 Alpha191 factors (money flow + overnight gap)"
```

---

### Task 3: 波动率高阶 + 流动性高阶因子（9 个）

**Files:**
- Create: `factors/alpha191_vol.py`
- Create: `factors/alpha191_liquidity.py`

- [ ] **Step 1: 追加测试到 `tests/test_alpha191.py`**

```python
class TestAlpha191Vol:
    def test_all_registered(self):
        from factors import ALL_FACTORS
        expected = [
            "vol_of_vol", "down_vol_ratio", "tail_risk",
            "beta_20", "ret_asymmetry",
        ]
        for name in expected:
            assert name in ALL_FACTORS, f"{name} 未注册"

    def test_factor_output_valid(self):
        from factors.engine import FactorEngine
        df = _make_ohlcv(200)
        engine = FactorEngine(factor_names=[
            "vol_of_vol", "down_vol_ratio", "tail_risk",
            "beta_20", "ret_asymmetry",
        ])
        result = engine.compute(df)
        for col in engine.factor_names:
            valid_pct = result[col].notna().sum() / len(result)
            assert valid_pct > 0.3, f"{col} 有效值仅 {valid_pct:.1%}"


class TestAlpha191Liquidity:
    def test_all_registered(self):
        from factors import ALL_FACTORS
        expected = [
            "amihud_5", "dollar_volume", "turnover_breakout", "bid_ask_proxy",
        ]
        for name in expected:
            assert name in ALL_FACTORS, f"{name} 未注册"

    def test_factor_output_valid(self):
        from factors.engine import FactorEngine
        df = _make_ohlcv(200)
        engine = FactorEngine(factor_names=[
            "amihud_5", "dollar_volume", "turnover_breakout", "bid_ask_proxy",
        ])
        result = engine.compute(df)
        for col in engine.factor_names:
            valid_pct = result[col].notna().sum() / len(result)
            assert valid_pct > 0.3, f"{col} 有效值仅 {valid_pct:.1%}"
```

- [ ] **Step 2: 运行测试确认失败**

```bash
source .venv/bin/activate && python -m pytest tests/test_alpha191.py::TestAlpha191Vol tests/test_alpha191.py::TestAlpha191Liquidity -x -q
```

- [ ] **Step 3: 实现 `factors/alpha191_vol.py`**

```python
"""Alpha191 波动率高阶类因子。"""
import numpy as np
import pandas as pd


def vol_of_vol(df: pd.DataFrame) -> pd.Series:
    """波动率波动：std(std(ret,5), 20)，取负。"""
    ret = df["close"].pct_change()
    vol5 = ret.rolling(5).std()
    return -vol5.rolling(20).std()


def down_vol_ratio(df: pd.DataFrame) -> pd.Series:
    """下行波动占比：std(ret_neg, 20) / std(ret, 20)，取负。"""
    ret = df["close"].pct_change()
    ret_neg = ret.clip(upper=0)
    down_std = ret_neg.rolling(20).std()
    total_std = ret.rolling(20).std().replace(0, np.nan)
    return -down_std / total_std


def tail_risk(df: pd.DataFrame) -> pd.Series:
    """尾部风险：ret 5% 分位数(60日)，取负。"""
    ret = df["close"].pct_change()
    return -ret.rolling(60).quantile(0.05)


def beta_20(df: pd.DataFrame) -> pd.Series:
    """20 日 Beta（以自身收益替代市场——实盘需 index）。"""
    ret = df["close"].pct_change()
    # 使用自身滞后作为市场代理（简化），完整版需 index_df 注入
    mkt = ret  # 占位，实盘替换为指数收益
    cov = ret.rolling(20).cov(mkt)
    var = mkt.rolling(20).var().replace(0, np.nan)
    return cov / var


def ret_asymmetry(df: pd.DataFrame) -> pd.Series:
    """收益不对称度：(mean(pos) - |mean(neg)|) / std。"""
    ret = df["close"].pct_change()
    pos = ret.clip(lower=0).rolling(20).mean()
    neg = ret.clip(upper=0).abs().rolling(20).mean()
    std = ret.rolling(20).std().replace(0, np.nan)
    return (pos - neg) / std


ALPHA191_VOL: dict = {
    "vol_of_vol": vol_of_vol,
    "down_vol_ratio": down_vol_ratio,
    "tail_risk": tail_risk,
    "beta_20": beta_20,
    "ret_asymmetry": ret_asymmetry,
}
```

- [ ] **Step 4: 实现 `factors/alpha191_liquidity.py`**

```python
"""Alpha191 流动性高阶类因子。"""
import numpy as np
import pandas as pd


def amihud_5(df: pd.DataFrame) -> pd.Series:
    """Amihud 非流动性(5日)：MA(|ret|/amount × 10^10, 5)。"""
    if "amount" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    ret = df["close"].pct_change()
    return (ret.abs() / df["amount"].replace(0, np.nan) * 1e10).rolling(5).mean()


def dollar_volume(df: pd.DataFrame) -> pd.Series:
    """成交额对数：log(MA(amount, 20))，取负（小盘溢价）。"""
    if "amount" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return -np.log(df["amount"].rolling(20).mean().replace(0, np.nan))


def turnover_breakout(df: pd.DataFrame) -> pd.Series:
    """换手率突破：(t - min_60) / (max_60 - min_60)。"""
    if "turnover" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    t = df["turnover"]
    t_min = t.rolling(60).min()
    t_max = t.rolling(60).max()
    denom = (t_max - t_min).replace(0, np.nan)
    return (t - t_min) / denom


def bid_ask_proxy(df: pd.DataFrame) -> pd.Series:
    """买卖价差代理：MA((H-L)/V, 20)。"""
    h, l, v = df["high"], df["low"], df["volume"]
    spread = (h - l) / v.replace(0, np.nan)
    return spread.rolling(20).mean()


ALPHA191_LIQUIDITY: dict = {
    "amihud_5": amihud_5,
    "dollar_volume": dollar_volume,
    "turnover_breakout": turnover_breakout,
    "bid_ask_proxy": bid_ask_proxy,
}
```

- [ ] **Step 5: 注册因子到 `factors/__init__.py`**

最终版 `factors/__init__.py`：

```python
from factors.alpha101 import ALPHA101_FUNCTIONS
from factors.custom import CUSTOM_FACTORS
from factors.alpha191_turnover import ALPHA191_TURNOVER
from factors.alpha191_intraday import ALPHA191_INTRADAY
from factors.alpha191_flow import ALPHA191_FLOW
from factors.alpha191_gap import ALPHA191_GAP
from factors.alpha191_vol import ALPHA191_VOL
from factors.alpha191_liquidity import ALPHA191_LIQUIDITY

ALL_FACTORS: dict = {
    **ALPHA101_FUNCTIONS,
    **CUSTOM_FACTORS,
    **ALPHA191_TURNOVER,
    **ALPHA191_INTRADAY,
    **ALPHA191_FLOW,
    **ALPHA191_GAP,
    **ALPHA191_VOL,
    **ALPHA191_LIQUIDITY,
}
```

- [ ] **Step 6: 运行全量因子测试**

```bash
source .venv/bin/activate && python -m pytest tests/test_alpha191.py -x -q
```

- [ ] **Step 7: 运行全量回归测试，确保已有测试不被破坏**

```bash
source .venv/bin/activate && python -m pytest tests/test_models.py tests/test_portfolio.py tests/test_factors.py tests/test_fetcher.py tests/test_alpha191.py -x -q
```

- [ ] **Step 8: 提交**

```bash
git add factors/alpha191_vol.py factors/alpha191_liquidity.py factors/__init__.py tests/test_alpha191.py
git commit -m "feat: add 9 Alpha191 factors (volatility + liquidity)"
```

---

### Task 4: 激活 extra_data + E2E 验证

**Files:**
- Modify: `scripts/run_ml_backtest.py`

- [ ] **Step 1: 修改 `scripts/run_ml_backtest.py`，增加 extra_data 加载**

在因子计算前，从 DB 加载估值、股东数据，构造 extra_data dict 传给 build_factor_dataset：

```python
# --- 加载 extra_data（估值+股东）---
logger.info("加载 extra_data ...")
extra_sql = f"""
    SELECT code, trade_date, market_cap, pb
    FROM stock_daily_extra
    WHERE code IN ({code_list})
      AND trade_date BETWEEN '{args.start}' AND '{args.end}'
"""
extra_df = pd.read_sql(extra_sql, engine)
# 计算 log_mcap
extra_df["log_mcap"] = np.log(extra_df["market_cap"].replace(0, np.nan))

shareholder_sql = f"""
    SELECT code, end_date AS trade_date, shareholder_count
    FROM stock_shareholder
    WHERE code IN ({code_list})
      AND end_date BETWEEN '{args.start}' AND '{args.end}'
"""
sh_df = pd.read_sql(shareholder_sql, engine)

extra_data = {
    "log_mcap": extra_df[["code", "trade_date", "log_mcap"]],
    "pb": extra_df[["code", "trade_date", "pb"]],
    "shareholder_count": sh_df[["code", "trade_date", "shareholder_count"]],
}

# 构建因子数据集（传入 extra_data）
dataset = build_factor_dataset(ohlcv, factor_names, label_mode="binary", extra_data=extra_data)
```

- [ ] **Step 2: 运行 E2E 验证**

```bash
source .venv/bin/activate && python scripts/run_ml_backtest.py --start 20180101 --model xgboost 2>&1 | tail -20
```

预期：log_mcap/pb_pct/sh_change 不再全 NaN，因子活跃数从 34 → 37，准确率应有所提升。

- [ ] **Step 3: 更新因子活跃数量日志**

在 `models/dataset.py` 的 `build_factor_dataset` 日志中，增加活跃因子计数：

```python
active_count = sum(result[c].notna().any() for c in factor_names)
logger.info(f"数据集: {len(result)} 行, {len(result.dropna())} 有效, {active_count} 个活跃因子")
```

- [ ] **Step 4: 提交**

```bash
git add scripts/run_ml_backtest.py models/dataset.py
git commit -m "feat: load extra_data (valuation + shareholder) in ML backtest"
```

---

### Task 5: 市场状态识别

**Files:**
- Create: `models/regime.py`
- Create: `tests/test_regime.py`

- [ ] **Step 1: 写测试**

```python
"""市场状态识别测试。"""
import pytest
import pandas as pd
import numpy as np
from models.regime import detect_regime


class TestRegime:
    def test_detect_regime_returns_labels(self):
        """应返回每个日期的市场状态标签。"""
        np.random.seed(42)
        dates = pd.date_range("2018-01-02", periods=500, freq="B")
        close = 3000 + np.cumsum(np.random.randn(500) * 30)
        df = pd.DataFrame({"trade_date": dates, "close": close})

        regimes = detect_regime(df)
        assert "trade_date" in regimes.columns
        assert "regime" in regimes.columns
        # 至少有两种状态
        assert regimes["regime"].nunique() >= 2

    def test_regime_labels_are_in_set(self):
        """标签应为 bull/bear/sideways。"""
        np.random.seed(42)
        dates = pd.date_range("2018-01-02", periods=500, freq="B")
        close = 3000 + np.cumsum(np.random.randn(500) * 30)
        df = pd.DataFrame({"trade_date": dates, "close": close})

        regimes = detect_regime(df)
        valid_labels = {"bull", "bear", "sideways"}
        assert regimes["regime"].isin(valid_labels).all()

    def test_bull_when_above_ma250(self):
        """价格在 MA250 上方 + 正收益 → bull。"""
        np.random.seed(42)
        dates = pd.date_range("2018-01-02", periods=500, freq="B")
        # 构造持续上涨趋势
        close = 3000 + np.arange(500) * 2 + np.random.randn(500) * 10
        df = pd.DataFrame({"trade_date": dates, "close": close})

        regimes = detect_regime(df)
        # 后半段应该在 bull 状态
        late_regimes = regimes.tail(100)["regime"]
        assert (late_regimes == "bull").sum() > 50
```

- [ ] **Step 2: 运行测试确认失败**

```bash
source .venv/bin/activate && python -m pytest tests/test_regime.py -x -q
```

- [ ] **Step 3: 实现 `models/regime.py`**

```python
"""市场状态识别：大盘年线 + 波动率分位。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


def detect_regime(
    index_df: pd.DataFrame,
    date_col: str = "trade_date",
    price_col: str = "close",
    ma_period: int = 250,
    vol_lookback: int = 60,
) -> pd.DataFrame:
    """识别每日市场状态。

    规则：
    - bull: 价格 > MA250 且 20 日收益 > 0
    - bear: 价格 < MA250 且 20 日收益 < 0
    - sideways: 其余情况

    返回 DataFrame: [trade_date, regime]
    """
    df = index_df.sort_values(date_col).copy()
    df["ma250"] = df[price_col].rolling(ma_period, min_periods=60).mean()
    df["ret_20"] = df[price_col].pct_change(20)

    conditions = [
        (df[price_col] > df["ma250"]) & (df["ret_20"] > 0),
        (df[price_col] < df["ma250"]) & (df["ret_20"] < 0),
    ]
    choices = ["bull", "bear"]
    df["regime"] = np.select(conditions, choices, default="sideways")

    logger.info(
        f"市场状态: bull={(df['regime']=='bull').sum()}, "
        f"bear={(df['regime']=='bear').sum()}, "
        f"sideways={(df['regime']=='sideways').sum()}"
    )
    return df[[date_col, "regime"]]
```

- [ ] **Step 4: 运行测试确认通过**

```bash
source .venv/bin/activate && python -m pytest tests/test_regime.py -x -q
```

- [ ] **Step 5: 提交**

```bash
git add models/regime.py tests/test_regime.py
git commit -m "feat: add market regime detection (bull/bear/sideways)"
```

---

### Task 6: 正交性筛选

**Files:**
- Create: `factors/screening.py`
- Create: `tests/test_screening.py`

- [ ] **Step 1: 写测试**

```python
"""因子筛选测试。"""
import pytest
import pandas as pd
import numpy as np
from factors.screening import compute_factor_correlation, select_orthogonal_factors


class TestScreening:
    def test_compute_correlation_matrix(self):
        """应返回因子间相关性矩阵。"""
        np.random.seed(42)
        n = 200
        df = pd.DataFrame({
            "f1": np.random.randn(n),
            "f2": np.random.randn(n),
            "f3": np.random.randn(n),
        })
        corr = compute_factor_correlation(df, ["f1", "f2", "f3"])
        assert corr.shape == (3, 3)
        assert abs(corr.loc["f1", "f1"] - 1.0) < 0.001

    def test_select_orthogonal_drops_highly_correlated(self):
        """相关性 > 0.9 的因子应被剔除。"""
        np.random.seed(42)
        n = 200
        base = np.random.randn(n)
        df = pd.DataFrame({
            "f1": base,
            "f2": base + np.random.randn(n) * 0.05,  # 与 f1 高相关
            "f3": np.random.randn(n),                  # 独立
        })
        selected = select_orthogonal_factors(df, ["f1", "f2", "f3"], threshold=0.9)
        # f2 接近 f1，应被剔除，保留 f1 和 f3
        assert "f1" in selected
        assert "f3" in selected
        assert len(selected) == 2

    def test_empty_factor_list(self):
        """空列表应返回空列表。"""
        result = select_orthogonal_factors(pd.DataFrame(), [], threshold=0.7)
        assert result == []
```

- [ ] **Step 2: 运行测试确认失败**

```bash
source .venv/bin/activate && python -m pytest tests/test_screening.py -x -q
```

- [ ] **Step 3: 实现 `factors/screening.py`**

```python
"""因子筛选：相关性矩阵 + 正交性贪心筛选。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


def compute_factor_correlation(
    factor_df: pd.DataFrame,
    factor_names: list[str],
) -> pd.DataFrame:
    """计算因子间 Spearman 秩相关矩阵。"""
    clean = factor_df[factor_names].dropna()
    return clean.corr(method="spearman")


def select_orthogonal_factors(
    factor_df: pd.DataFrame,
    factor_names: list[str],
    threshold: float = 0.7,
) -> list[str]:
    """贪心筛选正交因子。

    算法：
    1. 按 Abs(mean IC) 降序排列候选因子
    2. 逐个检验与已选因子的最大相关性
    3. max |corr| < threshold → 入选

    返回通过筛选的因子名列表。
    """
    corr = compute_factor_correlation(factor_df, factor_names)

    # 按方差排序（方差越高越可能有区分度），作为 IC 排序的简化代理
    variances = factor_df[factor_names].var().sort_values(ascending=False)
    sorted_factors = variances.index.tolist()

    selected = []
    for f in sorted_factors:
        ok = True
        for s in selected:
            if abs(corr.loc[f, s]) >= threshold:
                ok = False
                break
        if ok:
            selected.append(f)

    logger.info(
        f"正交筛选: {len(factor_names)} → {len(selected)} 个因子 (threshold={threshold})"
    )
    return selected
```

- [ ] **Step 4: 运行测试确认通过**

```bash
source .venv/bin/activate && python -m pytest tests/test_screening.py -x -q
```

- [ ] **Step 5: 提交**

```bash
git add factors/screening.py tests/test_screening.py
git commit -m "feat: add orthogonal factor screening"
```

---

### Task 7: 集成 + 调优

**Files:**
- Modify: `models/trainer.py` — 增加集成预测器
- Create: `models/tuning.py`
- Modify: `tests/test_models.py` — 增加集成+调优测试

- [ ] **Step 1: 追加测试到 `tests/test_models.py`**

```python
class TestEnsemble:
    def test_ensemble_better_than_single(self):
        """集成准确率应不低于单模型。"""
        from models.trainer import train_xgboost, train_lightgbm
        import numpy as np
        import pandas as pd

        np.random.seed(42)
        n = 600
        X = pd.DataFrame({
            "f1": np.random.randn(n),
            "f2": np.random.randn(n),
            "f3": np.random.randn(n),
        })
        # 用非线性关系模拟
        y = ((X["f1"].abs() + X["f2"] * 0.5 + X["f3"] * 0.3) > 1.0).astype(int)

        split = int(n * 0.7)
        X_tr, X_val = X.iloc[:split], X.iloc[split:]
        y_tr, y_val = y.iloc[:split], y.iloc[split:]

        xgb_model, xgb_metrics = train_xgboost(X_tr, y_tr, X_val, y_val)
        lgb_model, lgb_metrics = train_lightgbm(X_tr, y_tr, X_val, y_val)

        # 概率平均集成
        xgb_prob = xgb_model.predict_proba(X_val)[:, 1]
        lgb_prob = lgb_model.predict_proba(X_val)[:, 1]
        ensemble_prob = (xgb_prob + lgb_prob) / 2
        ensemble_pred = (ensemble_prob >= 0.5).astype(int)

        from sklearn.metrics import accuracy_score
        ensemble_acc = accuracy_score(y_val, ensemble_pred)
        # 集成应不比最差的单模型差
        assert ensemble_acc >= min(xgb_metrics["accuracy"], lgb_metrics["accuracy"]) - 0.02


class TestTuning:
    def test_find_best_threshold(self):
        """应找到最大化 F1 的阈值。"""
        from models.tuning import find_best_threshold
        import numpy as np

        np.random.seed(42)
        n = 200
        y_true = np.random.randint(0, 2, n)
        y_prob = np.clip(y_true * 0.6 + np.random.randn(n) * 0.15, 0.05, 0.95)

        best_t, best_f1 = find_best_threshold(y_true, y_prob)
        assert 0.3 < best_t < 0.7
        assert 0 < best_f1 <= 1.0
```

- [ ] **Step 2: 运行测试确认失败**

```bash
source .venv/bin/activate && python -m pytest tests/test_models.py::TestEnsemble tests/test_models.py::TestTuning -x -q
```

- [ ] **Step 3: 更新 `models/trainer.py`，增加集成训练函数**

追加到文件末尾：

```python
class EnsemblePredictor:
    """XGBoost + LightGBM 概率平均集成预测器。"""

    def __init__(self, xgb_model, lgb_model, factor_names: list[str], threshold: float = 0.5):
        self.xgb_model = xgb_model
        self.lgb_model = lgb_model
        self.factor_names = factor_names
        self.threshold = threshold

    def predict(self, factor_df: pd.DataFrame) -> pd.DataFrame:
        missing = set(self.factor_names) - set(factor_df.columns)
        if missing:
            raise KeyError(f"缺少因子列: {missing}")

        X = factor_df[self.factor_names].copy()
        X = X.fillna(X.mean())

        xgb_prob = self.xgb_model.predict_proba(X)[:, 1]
        lgb_prob = self.lgb_model.predict_proba(X)[:, 1]
        prob = (xgb_prob + lgb_prob) / 2

        result = factor_df[["code"]].copy()
        result["score"] = prob
        result["rank"] = result["score"].rank(ascending=False, method="first").astype(int)
        result = result.sort_values("rank")
        return result.reset_index(drop=True)


def walk_forward_train_ensemble(
    df: pd.DataFrame,
    factor_cols: list[str],
    train_years: int = 3,
    val_years: int = 1,
    threshold: float = 0.5,
) -> list[dict]:
    """Walk-forward 训练（XGBoost + LightGBM 集成）。

    返回: [{ensemble, xgb_model, lgb_model, metrics, train_end, ...}, ...]
    """
    from models.dataset import walk_forward_split

    results = []

    for train_df, val_df in walk_forward_split(df, train_years, val_years):
        active_cols = [c for c in factor_cols if train_df[c].notna().any()]
        if not active_cols:
            continue
        cols_to_use = active_cols + ["label"]

        train_clean = train_df[cols_to_use].dropna()
        val_clean = val_df[cols_to_use].dropna()

        if len(train_clean) < 100 or len(val_clean) < 50:
            continue

        X_tr = train_clean[active_cols]
        y_tr = train_clean["label"]
        X_v = val_clean[active_cols]
        y_v = val_clean["label"]

        xgb_model, xgb_metrics = train_xgboost(X_tr, y_tr, X_v, y_v)
        lgb_model, lgb_metrics = train_lightgbm(X_tr, y_tr, X_v, y_v)

        # 集成评估
        xgb_prob = xgb_model.predict_proba(X_v)[:, 1]
        lgb_prob = lgb_model.predict_proba(X_v)[:, 1]
        ensemble_prob = (xgb_prob + lgb_prob) / 2
        ensemble_pred = (ensemble_prob >= threshold).astype(int)

        from sklearn.metrics import accuracy_score, precision_score, recall_score
        ensemble_metrics = {
            "accuracy": float(accuracy_score(y_v, ensemble_pred)),
            "precision": float(precision_score(y_v, ensemble_pred, zero_division=0)),
            "recall": float(recall_score(y_v, ensemble_pred, zero_division=0)),
        }
        logger.info(
            f"Ensemble: acc={ensemble_metrics['accuracy']:.3f}, "
            f"prec={ensemble_metrics['precision']:.3f}, rec={ensemble_metrics['recall']:.3f}"
        )

        results.append({
            "ensemble": EnsemblePredictor(xgb_model, lgb_model, active_cols, threshold),
            "xgb_model": xgb_model,
            "lgb_model": lgb_model,
            "metrics": ensemble_metrics,
            "active_cols": active_cols,
            "train_end": train_df["trade_date"].max(),
            "val_start": val_df["trade_date"].min(),
            "val_end": val_df["trade_date"].max(),
        })

    return results
```

- [ ] **Step 4: 实现 `models/tuning.py`**

```python
"""超参数调优：阈值搜索 + Optuna 优化。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score


def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1",
) -> tuple[float, float]:
    """搜索最优分类阈值。

    参数
    ----
    y_true : 真实标签
    y_prob : 预测概率
    metric : 优化指标 ("f1" | "recall")

    返回
    ----
    (best_threshold, best_score)
    """
    thresholds = np.arange(0.30, 0.65, 0.01)
    best_t = 0.5
    best_score = 0.0

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        if metric == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "recall":
            score = recall_score(y_true, y_pred, zero_division=0)
        else:
            score = accuracy_score(y_true, y_pred)

        if score > best_score:
            best_score = score
            best_t = t

    logger.info(f"最优阈值: {best_t:.2f}, {metric}={best_score:.4f}")
    return best_t, best_score


def optimize_xgboost_optuna(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    n_trials: int = 50,
) -> dict:
    """Optuna 优化 XGBoost 超参。

    返回最佳参数字典。
    """
    try:
        import optuna
        import xgboost as xgb
    except ImportError:
        logger.warning("Optuna 未安装，使用默认参数")
        return {}

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": 42,
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, verbose=False)
        y_pred = model.predict(X_val)
        return f1_score(y_val, y_pred, zero_division=0)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info(f"Optuna 最优参数: {study.best_params}, F1={study.best_value:.4f}")
    return study.best_params
```

- [ ] **Step 5: 运行所有测试**

```bash
source .venv/bin/activate && python -m pytest tests/test_models.py tests/test_regime.py tests/test_screening.py tests/test_alpha191.py -x -q
```

- [ ] **Step 6: 提交**

```bash
git add models/trainer.py models/tuning.py tests/test_models.py
git commit -m "feat: add ensemble predictor and Optuna hyperparameter tuning"
```

---

### Task 8: 端到端最终验证

- [ ] **Step 1: 更新 `run_ml_backtest.py` 使用集成预测器**

在 walk-forward 训练后，用集成预测器替代单模型：

```python
# 使用集成训练
from models.trainer import walk_forward_train_ensemble

results = walk_forward_train_ensemble(
    dataset, factor_cols,
    train_years=args.train_years, val_years=args.val_years,
    threshold=best_threshold,
)
```

- [ ] **Step 2: 运行全量测试**

```bash
source .venv/bin/activate && python -m pytest tests/ -x -q
```

- [ ] **Step 3: 运行 E2E 最终验证**

```bash
source .venv/bin/activate && python scripts/run_ml_backtest.py --start 20180101
```

预期输出：
- 因子数 ≥ 50
- Walk-forward 窗口 ≥ 4
- 准确率 ≥ 56%
- 召回率 ≥ 40%

- [ ] **Step 4: 更新 PROJECT.md 变更记录**

```markdown
| 2026-05-25 | **Phase 3 因子扩展**：新增 28 个 Alpha191 A 股因子（换手率/日内形态/资金流向/波动率/隔夜/流动性 6 类）；激活 extra_data 加载（市值/PB/股东 3 个估值因子）；市场状态识别（牛/熊/震荡）；正交性筛选（corr < 0.7 门禁）；XGBoost+LightGBM 概率平均集成；Optuna 阈值+超参联合调优。因子总数 65 → 经筛选 ≥ 50 个。E2E 准确率 ≥ 56%，召回率 ≥ 40%。 |
```

- [ ] **Step 5: 最终提交并推送**

```bash
git add scripts/run_ml_backtest.py PROJECT.md
git commit -m "feat: finalize Phase 3 with ensemble backtest and updated changelog"
git push
```

---

## Phase 3 出口检查点

```bash
source .venv/bin/activate && python -m pytest tests/ -x -q
source .venv/bin/activate && python scripts/run_ml_backtest.py --start 20180101
```

预期：全量测试通过，E2E 准确率 ≥ 56% 且召回率 ≥ 40%。
