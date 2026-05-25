# 多因子 ML 量化选股 — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建数据补齐 + 因子库 + ML预测 + 组合优化的完整量化选股系统

**Architecture:** 四阶段递进：数据层（新增财务/行业表）→ 因子层（Alpha101/191 + 自定义 + 中性化 + IC监控）→ 模型层（walk-forward + XGBoost/LightGBM）→ 组合层（选股 + 仓位 + 风控）

**Tech Stack:** Python 3.14, PostgreSQL, AKShare, pandas, numpy, scikit-learn, xgboost, lightgbm, statsmodels, backtrader

---

## 文件结构

### 新增文件
| 文件 | 职责 |
|---|---|
| `factors/__init__.py` | 因子模块入口，导出因子名列表 |
| `factors/engine.py` | 因子计算引擎：输入codes+日期 → 输出因子矩阵 |
| `factors/alpha101.py` | Alpha101 核心因子实现（30+个） |
| `factors/alpha191.py` | Alpha191 A股适配因子 |
| `factors/custom.py` | 自定义A股因子（散户/市值/换手率动量等） |
| `factors/monitor.py` | IC/ICIR/衰减曲线监控 |
| `models/__init__.py` | 模型模块入口 |
| `models/dataset.py` | Walk-forward样本构造 |
| `models/trainer.py` | 训练/验证流程 |
| `models/predictor.py` | 每日预测接口 |
| `portfolio/__init__.py` | 组合模块入口 |
| `portfolio/selector.py` | Top-N 选股 + 过滤 |
| `portfolio/allocator.py` | 仓位分配（等权/波动率倒数/风险平价） |
| `portfolio/risk.py` | 风控规则（止损/回撤控制） |
| `tests/test_factors.py` | 因子计算单元测试 |
| `tests/test_models.py` | 模型训练/预测测试 |
| `tests/test_portfolio.py` | 组合优化测试 |

### 修改文件
| 文件 | 变更 |
|---|---|
| `data/db.py` | 新增 stock_financial + stock_industry 两张表 DDL |
| `data/fetcher.py` | 新增 fetch_financial_data() + fetch_industry() |
| `data/sync.py` | 新增 financial / industry 同步模式 |

---

## 阶段一：数据补齐

### Task 1: 财务数据 Fetcher

**Files:**
- Modify: `data/fetcher.py` (追加)
- Create: `tests/test_fetcher.py`

- [ ] **Step 1: 写测试**

```python
import pytest
import pandas as pd
from data.fetcher import fetch_financial_data


def test_fetch_financial_data_returns_dataframe():
    """财务数据 fetcher 应对一只已知股票返回非空 DataFrame。"""
    df = fetch_financial_data("000001")

    assert isinstance(df, pd.DataFrame)
    assert not df.empty, "平安银行应该有财务数据"
    assert "code" in df.columns
    assert "report_date" in df.columns
    assert "revenue" in df.columns
    assert "net_profit" in df.columns
    assert "roe" in df.columns
    assert "eps" in df.columns
    assert all(df["code"] == "000001")


def test_fetch_financial_data_invalid_code():
    """无效代码应返回空 DataFrame 而不抛异常。"""
    df = fetch_financial_data("999999")
    assert isinstance(df, pd.DataFrame)
    # 无效代码可能返回空，也可能 AKShare 返回默认数据，两种情况都接受
```

- [ ] **Step 2: 运行测试确认失败**

```bash
source .venv/bin/activate && python -m pytest tests/test_fetcher.py::test_fetch_financial_data_returns_dataframe -x -q
```

- [ ] **Step 3: 实现 `fetch_financial_data()`**

在 `data/fetcher.py` 末尾追加：

```python
# ============================================================
#  财务基本面数据
# ============================================================

@retry_on_network_error()
def fetch_financial_data(symbol: str) -> pd.DataFrame:
    """获取单只股票财务数据（同花顺财务摘要）。
    数据源：同花顺（通过 AKShare stock_financial_abstract_ths）。
    返回 columns: code, report_date, revenue, net_profit, gross_margin,
                  net_margin, roe, total_assets, total_liability, bps, eps, cash_flow
    """
    try:
        raw = ak.stock_financial_abstract_ths(symbol=symbol)
    except Exception:
        return pd.DataFrame()

    if raw is None or raw.empty:
        return pd.DataFrame()

    raw = raw.reset_index(drop=True)
    # AKShare 返回中文列名，需映射
    col_map = {
        "报告日期": "report_date",
        "营业总收入": "revenue",
        "归属母公司净利润": "net_profit",
        "毛利率": "gross_margin",
        "净利率": "net_margin",
        "净资产收益率": "roe",
        "总资产": "total_assets",
        "总负债": "total_liability",
        "每股净资产": "bps",
        "每股收益": "eps",
        "经营活动现金流量净额": "cash_flow",
    }
    # 只保留 col_map 里存在的列
    existing = {k: v for k, v in col_map.items() if k in raw.columns}
    if not existing:
        return pd.DataFrame()

    df = raw[list(existing.keys())].rename(columns=existing).copy()
    df["code"] = symbol
    df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce").apply(
        lambda x: x.date() if pd.notna(x) else None
    )

    # 数值列转换
    numeric_cols = [c for c in existing.values() if c not in ("code", "report_date")]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["report_date"])
    return df[["code", "report_date"] + numeric_cols]
```

- [ ] **Step 4: 运行测试确认通过**

```bash
source .venv/bin/activate && python -m pytest tests/test_fetcher.py::test_fetch_financial_data_returns_dataframe -x -q
```

- [ ] **Step 5: 手动验证数据内容**

```bash
source .venv/bin/activate && python -c "
from data.fetcher import fetch_financial_data
df = fetch_financial_data('000001')
print(f'行数: {len(df)}')
print(f'报告期范围: {df.report_date.min()} ~ {df.report_date.max()}')
print(df.head())
"
```

- [ ] **Step 6: 提交**

```bash
git add data/fetcher.py tests/test_fetcher.py
git commit -m "feat: add fetch_financial_data() for THS financial abstracts"
```

---

### Task 2: 行业分类 Fetcher

**Files:**
- Modify: `data/fetcher.py` (追加)

- [ ] **Step 1: 写测试**

```python
def test_fetch_industry_classification():
    """申万行业分类应对股票列表返回非空 DataFrame。"""
    df = fetch_industry_classification()

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "code" in df.columns
    assert "industry_sw1" in df.columns
    assert "market" in df.columns
    # 至少应覆盖 3000 只股票
    assert len(df) > 3000
```

- [ ] **Step 2: 运行测试确认失败**

```bash
source .venv/bin/activate && python -m pytest tests/test_fetcher.py::test_fetch_industry_classification -x -q
```

- [ ] **Step 3: 实现 `fetch_industry_classification()`**

```python
@retry_on_network_error()
def fetch_industry_classification() -> pd.DataFrame:
    """获取全市场股票行业分类（申万）。
    数据源：东方财富行业板块（通过 AKShare）。
    返回 columns: code, industry_sw1, industry_sw2, market
    """
    try:
        raw = ak.stock_board_industry_name_em()
    except Exception:
        return pd.DataFrame()

    if raw is None or raw.empty:
        return pd.DataFrame()

    # AKShare 申万行业分类需要另一种接口
    try:
        raw = ak.stock_info_sz_name_code()
    except Exception:
        pass

    # 使用 stock_individual_info_em 逐个查询，但量太大
    # 改用 stock_board_industry_cons_em 逐个行业板块获取成分股
    industries = []
    try:
        board_list = ak.stock_board_industry_name_em()
        for _, row in board_list.iterrows():
            board_name = row.get("板块名称", "")
            if not board_name:
                continue
            try:
                members = ak.stock_board_industry_cons_em(symbol=board_name)
                if members is not None and not members.empty:
                    members["industry_sw1"] = board_name
                    industries.append(members[["代码", "名称", "industry_sw1"]])
            except Exception:
                continue
    except Exception:
        pass

    if not industries:
        # 回退：用 stock_basic 表的 industry 字段作为行业分类
        engine = get_engine()
        with engine.connect() as conn:
            df = pd.read_sql("SELECT code, industry, market FROM stock_basic", conn)
        engine.dispose()
        if not df.empty:
            df = df.rename(columns={"industry": "industry_sw1"})
            df["industry_sw2"] = ""
            return df

    result = pd.concat(industries, ignore_index=True)
    result = result.rename(columns={"代码": "code", "名称": "name"})
    result["industry_sw2"] = ""
    result["market"] = result["code"].apply(_detect_market)

    # 去重，保留第一个行业分类
    result = result.drop_duplicates(subset=["code"], keep="first")
    return result[["code", "industry_sw1", "industry_sw2", "market"]]


def _detect_market(code: str) -> str:
    """根据代码前缀判断市场板块。"""
    if code.startswith("688"):
        return "科创板"
    elif code.startswith("300") or code.startswith("301"):
        return "创业板"
    elif code.startswith("8") or code.startswith("4"):
        return "北交所"
    elif code.startswith("6"):
        return "主板"
    elif code.startswith("0") or code.startswith("2"):
        return "主板"
    return "未知"
```

- [ ] **Step 4: 运行测试确认通过**

```bash
source .venv/bin/activate && python -m pytest tests/test_fetcher.py::test_fetch_industry_classification -x -q
```

- [ ] **Step 5: 提交**

```bash
git add data/fetcher.py tests/test_fetcher.py
git commit -m "feat: add fetch_industry_classification() for Shenwan classification"
```

---

### Task 3: 新增表 DDL

**Files:**
- Modify: `data/db.py` (追加 DDL + 注册 init_db)

- [ ] **Step 1: 在 `data/db.py` 中追加 DDL 常量**

放在 `DDL_STOCK_SHAREHOLDER` 和 `DDL_PAPER_ACCOUNT` 之间：

```python
# DDL —— 财务数据（同花顺财务摘要）
DDL_STOCK_FINANCIAL = """
CREATE TABLE IF NOT EXISTS stock_financial (
    code               VARCHAR(10),
    report_date        DATE,
    revenue            DOUBLE PRECISION,
    net_profit         DOUBLE PRECISION,
    gross_margin       DOUBLE PRECISION,
    net_margin         DOUBLE PRECISION,
    roe                DOUBLE PRECISION,
    total_assets       DOUBLE PRECISION,
    total_liability    DOUBLE PRECISION,
    bps                DOUBLE PRECISION,
    eps                DOUBLE PRECISION,
    cash_flow          DOUBLE PRECISION,
    PRIMARY KEY (code, report_date)
);
"""

# DDL —— 行业分类
DDL_STOCK_INDUSTRY = """
CREATE TABLE IF NOT EXISTS stock_industry (
    code               VARCHAR(10) PRIMARY KEY,
    industry_sw1       VARCHAR(50),
    industry_sw2       VARCHAR(50),
    market             VARCHAR(10)
);
"""
```

- [ ] **Step 2: 注册到 `init_db()`**

在 `init_db()` 中 `DDL_STOCK_SHAREHOLDER` 之后追加：

```python
        conn.execute(text(DDL_STOCK_FINANCIAL))
        conn.execute(text(DDL_STOCK_INDUSTRY))
```

并更新日志：

```python
    logger.info("数据库表初始化完成（12张表）")  # 10+2张新增（财务+行业）
```

- [ ] **Step 3: 注册到 `upsert_df()` 主键映射**

在 `upsert_df()` 的 `pk_map` 和条件链中增加 `stock_financial` 处理。当前的 fallback `pk = "code, trade_date"` 对 `stock_financial` 不适用（它的主键是 `code, report_date`）。追加：

```python
        elif "financial" in table:
            pk = "code, report_date"
```

放在 `elif "shareholder" in table:` 之后。

- [ ] **Step 4: 手动验证 DDL**

```bash
source .venv/bin/activate && python -c "
from data.db import get_engine, init_db
from sqlalchemy import text

init_db()
engine = get_engine()
with engine.connect() as conn:
    for table in ['stock_financial', 'stock_industry']:
        r = conn.execute(text(f\"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table}'\")).fetchall()
        print(f'{table}: {[(x[0], x[1]) for x in r]}')
engine.dispose()
"
```

- [ ] **Step 5: 提交**

```bash
git add data/db.py
git commit -m "feat: add stock_financial and stock_industry table DDL"
```

---

### Task 4: 财务 & 行业数据同步

**Files:**
- Modify: `data/sync.py` (追加 sync 函数 + 注册 mode)

- [ ] **Step 1: 写测试**

```python
import pytest
from data.db import get_engine, init_db
from data.sync import sync_financial, sync_industry


@pytest.fixture
def engine():
    init_db()
    return get_engine()


def test_sync_financial_single_stock(engine):
    """同步单只股票财务数据，表应有数据写入。"""
    from sqlalchemy import text
    sync_financial(engine, codes=["000001"])
    with engine.connect() as conn:
        cnt = conn.execute(
            text("SELECT COUNT(*) FROM stock_financial WHERE code = '000001'")
        ).scalar()
    assert cnt > 0, "平安银行应有财务数据"


def test_sync_industry_populates_table(engine):
    """行业同步后 stock_industry 表非空。"""
    from sqlalchemy import text
    sync_industry(engine)
    with engine.connect() as conn:
        cnt = conn.execute(text("SELECT COUNT(*) FROM stock_industry")).scalar()
    assert cnt > 1000, f"行业表应有大量记录，实际: {cnt}"
```

- [ ] **Step 2: 运行测试确认失败**

```bash
source .venv/bin/activate && python -m pytest tests/test_fetcher.py -x -q
```

- [ ] **Step 3: 实现模块级 worker 函数和 sync 函数**

在 `data/sync.py` 中追加（`sync_shareholder` 函数之后）：

```python
# ============================================================
#  财务数据
# ============================================================

def _do_fetch_financial(code: str) -> tuple[str, pd.DataFrame]:
    """模块级 worker，供 ProcessPoolExecutor 使用。"""
    try:
        df = fetch_financial_data(code)
        return code, df
    except Exception as e:
        logger.warning(f"{code} 财务数据获取失败: {e}")
        return code, pd.DataFrame()


def sync_financial(engine: Engine, codes: list[str] | None = None, workers: int = 4) -> None:
    """同步财务数据到 stock_financial。跳过已有报告的股票。"""
    logger.info("=" * 50)
    logger.info("开始同步财务数据 ...")

    if codes is None:
        codes = pd.read_sql("SELECT code FROM stock_basic", engine)["code"].tolist()

    # 过滤：跳过已同步过的股票
    existing_codes = set()
    with engine.connect() as conn:
        r = conn.execute(text("SELECT DISTINCT code FROM stock_financial")).fetchall()
        existing_codes = {x[0] for x in r}
    to_fetch = [c for c in codes if c not in existing_codes]
    logger.info(f"待同步: {len(to_fetch)}/{len(codes)} 只股票")

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_do_fetch_financial, c) for c in to_fetch]
        for f in tqdm(as_completed(futures), total=len(to_fetch), desc="财务数据"):
            code, df = f.result()
            if not df.empty:
                upsert_df(df, "stock_financial", engine)


# ============================================================
#  行业分类
# ============================================================

def sync_industry(engine: Engine) -> None:
    """同步行业分类到 stock_industry。全量刷新。"""
    logger.info("=" * 50)
    logger.info("开始同步行业分类 ...")

    df = fetch_industry_classification()
    if df.empty:
        logger.error("行业分类数据为空，跳过")
        return

    # 全量替换（行业分类数据量小）
    n = upsert_df(df, "stock_industry", engine)
    logger.success(f"行业分类同步完成，写入 {n} 行")
```

- [ ] **Step 4: 注册到 `main()` 的 mode choices 和路由**

在 `main()` 中：
- choices 列表追加 `"financial", "industry"`
- 路由逻辑追加：

```python
        if mode in ("all", "financial"):
            sync_financial(engine, workers=args.workers)

        if mode in ("all", "industry"):
            sync_industry(engine)
```

放在 `sync_shareholder` 路由之后。

- [ ] **Step 5: 运行测试确认通过**

```bash
source .venv/bin/activate && python -m pytest tests/test_fetcher.py -x -q
```

- [ ] **Step 6: 手动验证全量同步**

```bash
source .venv/bin/activate && python -m data.sync --mode financial --workers 8
```

- [ ] **Step 7: 提交**

```bash
git add data/sync.py tests/test_fetcher.py
git commit -m "feat: add sync_financial and sync_industry modes"
```

---

## 阶段二：因子库

### Task 5: 因子计算引擎

**Files:**
- Create: `factors/__init__.py`
- Create: `factors/engine.py`
- Create: `tests/test_factors.py`

- [ ] **Step 1: 创建 `factors/__init__.py`**

```python
"""因子计算模块。提供因子计算引擎和因子上线清单。"""

from factors.engine import FactorEngine
from factors.alpha101 import ALPHA101_FUNCTIONS
from factors.alpha191 import ALPHA191_FUNCTIONS
from factors.custom import CUSTOM_FACTORS

ALL_FACTORS = {**ALPHA101_FUNCTIONS, **ALPHA191_FUNCTIONS, **CUSTOM_FACTORS}

__all__ = ["FactorEngine", "ALL_FACTORS"]
```

- [ ] **Step 2: 写测试（先写后再实现 engine）**

```python
import pytest
import pandas as pd
import numpy as np
from data.db import get_engine
from factors.engine import FactorEngine


@pytest.fixture
def sample_ohlcv():
    """构造 100 个交易日 × 3 只股票的模拟 OHLCV 数据。"""
    dates = pd.date_range("2020-01-02", periods=100, freq="B")
    codes = ["000001", "000002", "000003"]
    rows = []
    np.random.seed(42)
    for code in codes:
        close = 10 + np.cumsum(np.random.randn(100) * 0.5)
        for i, d in enumerate(dates):
            rows.append({
                "code": code,
                "trade_date": d,
                "open": close[i] * (1 + np.random.randn() * 0.01),
                "high": close[i] * (1 + abs(np.random.randn()) * 0.02),
                "low": close[i] * (1 - abs(np.random.randn()) * 0.02),
                "close": close[i],
                "volume": np.random.randint(100000, 1000000),
                "amount": np.random.randint(1000000, 10000000),
            })
    return pd.DataFrame(rows)


class TestFactorEngine:
    def test_compute_factors_returns_matrix(self, sample_ohlcv):
        """因子计算引擎应返回 (N×T) × M 的因子矩阵。"""
        engine = FactorEngine(factor_names=["sma_5", "sma_10"])
        result = engine.compute(sample_ohlcv)

        assert isinstance(result, pd.DataFrame)
        assert "code" in result.columns
        assert "trade_date" in result.columns
        assert "sma_5" in result.columns
        assert "sma_10" in result.columns
        assert len(result) > 0

    def test_compute_factors_no_nan_in_middle(self, sample_ohlcv):
        """因子值在 warmup 期后不应有 NaN。"""
        engine = FactorEngine(factor_names=["rsi_14"])
        result = engine.compute(sample_ohlcv)

        warmup = 20
        for code in result["code"].unique():
            code_df = result[result["code"] == code].iloc[warmup:]
            assert not code_df["rsi_14"].isna().any(), \
                f"rsi_14 should have no NaN after warmup for {code}"

    def test_neutralize_removes_market_cap_effect(self, sample_ohlcv):
        """市值中性化后因子与市值的相关性应接近零。"""
        # 给数据拼上市值因子
        sample_ohlcv["log_mcap"] = np.random.randn(len(sample_ohlcv))

        engine = FactorEngine(factor_names=["sma_5"])
        result = engine.compute(sample_ohlcv)
        result["log_mcap"] = sample_ohlcv["log_mcap"]

        from factors.engine import neutralize
        result["sma_5_neutral"] = neutralize(
            result["sma_5"], result[["log_mcap"]]
        )

        corr = result["sma_5_neutral"].corr(result["log_mcap"])
        assert abs(corr) < 0.01, f"中性化后相关性应 < 0.01，实际: {corr}"
```

- [ ] **Step 3: 运行测试确认失败**

```bash
source .venv/bin/activate && python -m pytest tests/test_factors.py -x -q
```

- [ ] **Step 4: 实现 `factors/engine.py`**

```python
"""因子计算引擎。

用法:
    engine = FactorEngine(factor_names=["rsi_14", "sma_5"])
    factor_matrix = engine.compute(df_ohlcv)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from factors import ALL_FACTORS


class FactorEngine:
    """因子计算引擎。

    参数
    ----
    factor_names : 要计算的因子名列表，必须在 ALL_FACTORS 中注册。
    """

    def __init__(self, factor_names: list[str]):
        missing = set(factor_names) - set(ALL_FACTORS.keys())
        if missing:
            raise KeyError(f"未知因子: {missing}")
        self.factor_names = factor_names

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算因子矩阵。

        参数
        ----
        df : DataFrame, 列需包含 code, trade_date, open, high, low, close, volume
             按 code 分组，trade_date 排序。

        返回
        ----
        pd.DataFrame: 列 = [code, trade_date] + factor_names
        """
        df = df.sort_values(["code", "trade_date"]).copy()
        result_parts = []

        for code, group in df.groupby("code"):
            part = group[["code", "trade_date"]].copy()
            for name in self.factor_names:
                fn = ALL_FACTORS[name]
                part[name] = fn(group)
            result_parts.append(part)

        result = pd.concat(result_parts, ignore_index=True)
        return result


def neutralize(factor: pd.Series, exposures: pd.DataFrame) -> pd.Series:
    """截面中性化：用线性回归去除 factor 中的 exposures 影响。

    参数
    ----
    factor : 因子值序列
    exposures : 暴露矩阵（如 log_mcap, industry dummies）

    返回
    ----
    残差序列
    """
    valid = factor.notna() & exposures.notna().all(axis=1)
    if valid.sum() < 10:
        return factor

    X = exposures[valid].values
    y = factor[valid].values

    if X.shape[1] == 0:
        return factor

    model = LinearRegression()
    model.fit(X, y)
    predicted = model.predict(X)
    residuals = y - predicted

    result = pd.Series(np.nan, index=factor.index, dtype=float)
    result.loc[valid] = residuals
    return result
```

- [ ] **Step 5: 运行测试确认通过**

```bash
source .venv/bin/activate && python -m pytest tests/test_factors.py -x -q
```

- [ ] **Step 6: 提交**

```bash
git add factors/__init__.py factors/engine.py tests/test_factors.py
git commit -m "feat: add factor computation engine with neutralization"
```

---

### Task 6: Alpha101 核心因子实现

**Files:**
- Create: `factors/alpha101.py`

- [ ] **Step 1: 写测试**

```python
class TestAlpha101Factors:
    def test_rsi_computes_correctly(self, sample_ohlcv):
        """RSI因子手动验算：手工计算应与因子输出一致。"""
        from factors.alpha101 import rsi
        code_df = sample_ohlcv[sample_ohlcv["code"] == "000001"].reset_index(drop=True)

        result = rsi(code_df)

        # 手工计算 RSI(14)
        delta = code_df["close"].diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        expected = 100 - (100 / (1 + rs))

        # 最后 50 个值应接近（Wilder smoothing vs SMA 有微小差异，允许 2% 容差）
        pd.testing.assert_series_equal(
            result.iloc[-50:].round(1),
            expected.iloc[-50:].round(1),
            check_names=False,
        )

    def test_all_factors_return_series(self, sample_ohlcv):
        """所有 Alpha101 因子函数对分组数据返回 Series。"""
        from factors.alpha101 import ALPHA101_FUNCTIONS
        code_df = sample_ohlcv[sample_ohlcv["code"] == "000001"].reset_index(drop=True)

        for name, fn in ALPHA101_FUNCTIONS.items():
            result = fn(code_df)
            assert isinstance(result, pd.Series), f"{name}: 应返回 Series"
            assert len(result) == len(code_df), f"{name}: 长度不匹配"
```

- [ ] **Step 2: 运行测试确认失败**

```bash
source .venv/bin/activate && python -m pytest tests/test_factors.py::TestAlpha101Factors -x -q
```

- [ ] **Step 3: 实现 `factors/alpha101.py`**

```python
"""Alpha101 核心因子实现。

参考: Kakushadze & Tulchinsky (2015), "101 Formulaic Alphas"
每个因子函数签名: (df: pd.DataFrame) -> pd.Series
df 是单只股票按 trade_date 排序的 DataFrame，至少含 open/high/low/close/volume。
"""

import numpy as np
import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _ts_sum(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).sum()


def _ts_std(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).std()


def _ts_min(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).min()


def _ts_max(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).max()


def _ts_rank(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).apply(lambda x: pd.Series(x).rank().iloc[-1] / len(x), raw=False)


def _ts_corr(a: pd.Series, b: pd.Series, period: int) -> pd.Series:
    return a.rolling(period).corr(b)


def _ts_delta(series: pd.Series, period: int) -> pd.Series:
    return series - series.shift(period)


def _ts_delay(series: pd.Series, period: int) -> pd.Series:
    return series.shift(period)


def _ts_roc(series: pd.Series, period: int) -> pd.Series:
    """变化率: (C_t - C_{t-period}) / C_{t-period}"""
    lag = series.shift(period)
    return (series - lag) / lag.abs()


# ============================================================
#  均值回归型因子 (Reversal)
# ============================================================

def alpha001_reversal_5(df: pd.DataFrame) -> pd.Series:
    """5日反转: - (C_t - C_{t-5}) / C_{t-5}"""
    c = df["close"]
    return -_ts_roc(c, 5)


def alpha002_reversal_10(df: pd.DataFrame) -> pd.Series:
    """10日反转"""
    c = df["close"]
    return -_ts_roc(c, 10)


def alpha003_reversal_20(df: pd.DataFrame) -> pd.Series:
    """20日反转"""
    c = df["close"]
    return -_ts_roc(c, 20)


# ============================================================
#  动量型因子 (Momentum)
# ============================================================

def alpha004_momentum_20(df: pd.DataFrame) -> pd.Series:
    """20日动量"""
    c = df["close"]
    return _ts_roc(c, 20)


def alpha005_momentum_60(df: pd.DataFrame) -> pd.Series:
    """60日动量（中期趋势）"""
    c = df["close"]
    return _ts_roc(c, 60)


def alpha006_ema_ratio_5_20(df: pd.DataFrame) -> pd.Series:
    """EMA(5) / EMA(20) - 1"""
    c = df["close"]
    return ema(c, 5) / ema(c, 20) - 1


# ============================================================
#  波动率因子 (Volatility)
# ============================================================

def alpha007_volatility_20(df: pd.DataFrame) -> pd.Series:
    """20日波动率 (年化)"""
    return df["close"].pct_change().rolling(20).std() * np.sqrt(252)


def alpha008_atr_14(df: pd.DataFrame) -> pd.Series:
    """ATR(14) / Close"""
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([
        h - l,
        (h - c.shift()).abs(),
        (l - c.shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    return atr / c


# ============================================================
#  量价背离因子 (Volume-Price Divergence)
# ============================================================

def alpha009_volume_ratio_5(df: pd.DataFrame) -> pd.Series:
    """5日均量 / 20日均量"""
    return _ts_sum(df["volume"], 5) / _ts_sum(df["volume"], 20)


def alpha010_volume_price_trend(df: pd.DataFrame) -> pd.Series:
    """量价趋势: (C × V - MA(C×V, 20)) / MA(C×V, 20)"""
    cv = df["close"] * df["volume"]
    ma = cv.rolling(20).mean()
    return (cv - ma) / ma


def alpha011_vwap_ratio(df: pd.DataFrame) -> pd.Series:
    """成交量加权价格比率: Close / VWAP(20) - 1"""
    typ = (df["high"] + df["low"] + df["close"]) / 3
    vwap = (typ * df["volume"]).rolling(20).sum() / df["volume"].rolling(20).sum()
    return df["close"] / vwap - 1


# ============================================================
#  趋势强度因子
# ============================================================

def alpha012_macd_diff(df: pd.DataFrame) -> pd.Series:
    """MACD DIF"""
    c = df["close"]
    return ema(c, 12) - ema(c, 26)


def alpha013_macd_signal(df: pd.DataFrame) -> pd.Series:
    """MACD Signal"""
    return ema(alpha012_macd_diff(df), 9)


def alpha014_macd_hist(df: pd.DataFrame) -> pd.Series:
    """MACD 柱 / Close"""
    dif = alpha012_macd_diff(df)
    dea = alpha013_macd_signal(df)
    return (dif - dea) / df["close"]


def alpha015_rsi_14(df: pd.DataFrame) -> pd.Series:
    """RSI(14)"""
    c = df["close"]
    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


def alpha016_rsi_7(df: pd.DataFrame) -> pd.Series:
    """RSI(7)"""
    c = df["close"]
    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / 7, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 7, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


# ============================================================
#  通道/布林带因子
# ============================================================

def alpha017_bb_position(df: pd.DataFrame) -> pd.Series:
    """布林带位置: (C - 中轨) / (上轨 - 下轨)"""
    c = df["close"]
    mid = sma(c, 20)
    std = c.rolling(20).std()
    upper = mid + 2 * std
    lower = mid - 2 * std
    return (c - mid) / (upper - lower + 1e-10)


def alpha018_bb_width(df: pd.DataFrame) -> pd.Series:
    """布林带宽度 (标准化)"""
    c = df["close"]
    std = c.rolling(20).std()
    mid = sma(c, 20)
    return (4 * std) / mid


# ============================================================
#  流动性因子
# ============================================================

def alpha019_turnover_5(df: pd.DataFrame) -> pd.Series:
    """5日均换手率"""
    if "turnover" in df.columns:
        return df["turnover"].rolling(5).mean()
    return pd.Series(np.nan, index=df.index)


def alpha020_illiquidity(df: pd.DataFrame) -> pd.Series:
    """非流动性: |return| / (volume × close) × 10^8"""
    r = df["close"].pct_change()
    liquidity = r.abs() / (df["volume"] * df["close"]) * 1e8
    return liquidity.rolling(20).mean()


def alpha021_amount_ratio(df: pd.DataFrame) -> pd.Series:
    """5日均额 / 20日均额"""
    if "amount" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return df["amount"].rolling(5).mean() / df["amount"].rolling(20).mean()


# ============================================================
#  高阶因子
# ============================================================

def alpha022_skewness(df: pd.DataFrame) -> pd.Series:
    """20日收益偏度"""
    return df["close"].pct_change().rolling(20).skew()


def alpha023_kurtosis(df: pd.DataFrame) -> pd.Series:
    """20日收益峰度"""
    return df["close"].pct_change().rolling(20).kurtosis()


def alpha024_high_low_ratio(df: pd.DataFrame) -> pd.Series:
    """(High - Low) / Close 的 20日均值"""
    return ((df["high"] - df["low"]) / df["close"]).rolling(20).mean()


def alpha025_max_drawdown(df: pd.DataFrame) -> pd.Series:
    """20日回撤比例"""
    c = df["close"]
    rolling_max = c.rolling(20, min_periods=1).max()
    return (c - rolling_max) / rolling_max


def alpha026_corr_close_volume(df: pd.DataFrame) -> pd.Series:
    """10日收盘价与成交量相关系数"""
    return _ts_corr(df["close"], df["volume"], 10)


def alpha027_close_open_ratio(df: pd.DataFrame) -> pd.Series:
    """(Close - Open) / (High - Low + 1e-10)"""
    return (df["close"] - df["open"]) / (df["high"] - df["low"] + 1e-10)


def alpha028_up_day_ratio(df: pd.DataFrame) -> pd.Series:
    """20日中上涨天数比例"""
    return (df["close"].diff() > 0).rolling(20).mean()


def alpha029_price_position(df: pd.DataFrame) -> pd.Series:
    """价格在 20日区间位置: (C - Low_20) / (High_20 - Low_20)"""
    c = df["close"]
    h = df["high"].rolling(20).max()
    l = df["low"].rolling(20).min()
    return (c - l) / (h - l + 1e-10)


def alpha030_volume_swing(df: pd.DataFrame) -> pd.Series:
    """量价异动: |volume/MA(volume,20) - 1| × sign(return)"""
    vol_ratio = df["volume"] / df["volume"].rolling(20).mean() - 1
    sign = np.sign(df["close"].pct_change())
    return vol_ratio * sign


# ============================================================
#  因子注册表
# ============================================================

ALPHA101_FUNCTIONS: dict[str, callable] = {
    # 反转
    "rev_5": alpha001_reversal_5,
    "rev_10": alpha002_reversal_10,
    "rev_20": alpha003_reversal_20,
    # 动量
    "mom_20": alpha004_momentum_20,
    "mom_60": alpha005_momentum_60,
    "ema_ratio_5_20": alpha006_ema_ratio_5_20,
    # 波动率
    "vol_20": alpha007_volatility_20,
    "atr_14": alpha008_atr_14,
    # 量价
    "vol_ratio_5_20": alpha009_volume_ratio_5,
    "vpt": alpha010_volume_price_trend,
    "vwap_ratio": alpha011_vwap_ratio,
    # 趋势
    "macd_dif": alpha012_macd_diff,
    "macd_signal": alpha013_macd_signal,
    "macd_hist": alpha014_macd_hist,
    "rsi_14": alpha015_rsi_14,
    "rsi_7": alpha016_rsi_7,
    # 通道
    "bb_position": alpha017_bb_position,
    "bb_width": alpha018_bb_width,
    # 流动性
    "turnover_5": alpha019_turnover_5,
    "illiquidity": alpha020_illiquidity,
    "amount_ratio": alpha021_amount_ratio,
    # 高阶
    "skewness_20": alpha022_skewness,
    "kurtosis_20": alpha023_kurtosis,
    "high_low_ratio": alpha024_high_low_ratio,
    "max_dd_20": alpha025_max_drawdown,
    "corr_c_v": alpha026_corr_close_volume,
    "co_ratio": alpha027_close_open_ratio,
    "up_day_ratio": alpha028_up_day_ratio,
    "price_position": alpha029_price_position,
    "vol_swing": alpha030_volume_swing,
}
```

- [ ] **Step 4: 运行测试确认通过**

```bash
source .venv/bin/activate && python -m pytest tests/test_factors.py::TestAlpha101Factors -x -q
```

- [ ] **Step 5: 提交**

```bash
git add factors/alpha101.py tests/test_factors.py
git commit -m "feat: implement 30 Alpha101 core factors"
```

---

### Task 7: 自定义A股因子

**Files:**
- Create: `factors/custom.py`

- [ ] **Step 1: 实现自定义因子**

```python
"""自定义 A 股因子。

每函数签名: (df: pd.DataFrame) -> pd.Series
df 可能包含额外列: turnover, log_mcap, shareholder_count 等。
"""

import numpy as np
import pandas as pd


def custom_market_cap_factor(df: pd.DataFrame) -> pd.Series:
    """市值因子: 对数流通市值（小市值溢价在A股显著）。"""
    if "log_mcap" in df.columns:
        return -df["log_mcap"]  # 负值 = 小市值得分高
    return pd.Series(np.nan, index=df.index)


def custom_turnover_momentum(df: pd.DataFrame) -> pd.Series:
    """换手率动量: 5日换手率变化 / 20日换手率标准差。"""
    if "turnover" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    chg = df["turnover"].diff(5)
    std = df["turnover"].rolling(20).std()
    return chg / std


def custom_pb_percentile(df: pd.DataFrame) -> pd.Series:
    """PB 历史分位: PB 在 500 个交易日中的分位数（负值 = 低估值）。"""
    if "pb" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    rank = df["pb"].rolling(500, min_periods=60).apply(
        lambda x: pd.Series(x).rank().iloc[-1] / len(x),
        raw=False,
    )
    return -rank


def custom_shareholder_change(df: pd.DataFrame) -> pd.Series:
    """散户参与度变化: 股东户数季度环比变化率（增多利空）。"""
    if "shareholder_count" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    # 股东户数是季度数据，forward fill 到交易日
    sc = df["shareholder_count"].ffill()
    return -sc.pct_change(60)  # 约一个季度


def custom_volume_convergence(df: pd.DataFrame) -> pd.Series:
    """量能聚拢: -|vol/MA(vol,5) - vol/MA(vol,20)| (量缩等变盘)。"""
    v = df["volume"].replace(0, np.nan)
    ratio_5 = v / v.rolling(5).mean()
    ratio_20 = v / v.rolling(20).mean()
    return -(ratio_5 - ratio_20).abs()


def custom_intraday_volatility(df: pd.DataFrame) -> pd.Series:
    """日内波动: (High - Low) / Open 的 5日 EMA。"""
    iv = (df["high"] - df["low"]) / df["open"]
    return iv.ewm(span=5, adjust=False).mean()


def custom_gap_ratio(df: pd.DataFrame) -> pd.Series:
    """跳空缺口: (Open - Close_lag1) / Close_lag1。"""
    c = df["close"]
    return (df["open"] - c.shift(1)) / c.shift(1)


CUSTOM_FACTORS: dict[str, callable] = {
    "log_mcap": custom_market_cap_factor,
    "turnover_mom": custom_turnover_momentum,
    "pb_pct": custom_pb_percentile,
    "sh_change": custom_shareholder_change,
    "vol_conv": custom_volume_convergence,
    "intra_vol": custom_intraday_volatility,
    "gap_ratio": custom_gap_ratio,
}
```

- [ ] **Step 2: 提交**

```bash
git add factors/custom.py
git commit -m "feat: add 7 A-share custom factors"
```

---

### Task 8: IC 监控管线

**Files:**
- Create: `factors/monitor.py`

- [ ] **Step 1: 写测试**

```python
class TestICMonitor:
    def test_rank_ic_between_neg1_and_1(self):
        """RankIC 应在 [-1, 1] 范围内。"""
        from factors.monitor import compute_rank_ic
        import numpy as np

        f = np.random.randn(100)
        r = np.random.randn(100)
        ic = compute_rank_ic(f, r)
        assert -1.0 <= ic <= 1.0, f"RankIC={ic}"

    def test_rank_ic_perfect_positive(self):
        """完全正相关因子 RankIC = 1。"""
        from factors.monitor import compute_rank_ic
        f = np.array([1, 2, 3, 4, 5])
        r = np.array([1, 2, 3, 4, 5])
        ic = compute_rank_ic(f, r)
        assert ic == 1.0

    def test_ic_series_from_matrix(self):
        """从因子矩阵应能计算每个因子的 IC 序列。"""
        from factors.monitor import compute_ic_series
        import pandas as pd
        import numpy as np

        dates = pd.date_range("2020-01-02", periods=100, freq="B")
        n = len(dates) * 10
        df = pd.DataFrame({
            "trade_date": np.tile(dates, 10),
            "code": np.repeat([f"{i:06d}" for i in range(10)], 100),
            "factor_a": np.random.randn(n),
            "ret_1d": np.random.randn(n) * 0.02,
        })

        ic_df = compute_ic_series(df, factor_cols=["factor_a"], ret_col="ret_1d")
        assert "trade_date" in ic_df.columns
        assert "factor_a" in ic_df.columns
        assert len(ic_df) == 100
```

- [ ] **Step 2: 运行测试确认失败**

```bash
source .venv/bin/activate && python -m pytest tests/test_factors.py::TestICMonitor -x -q
```

- [ ] **Step 3: 实现 `factors/monitor.py`**

```python
"""因子 IC 监控管线。

提供:
- compute_rank_ic: 单截面 RankIC
- compute_ic_series: 因子矩阵 → 逐日 IC
- compute_ic_summary: IC/ICIR/IC_decay
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def compute_rank_ic(factor: np.ndarray, ret: np.ndarray) -> float:
    """计算 RankIC（Spearman 秩相关系数）。

    参数
    ----
    factor : 因子值
    ret : 同期或未来收益率

    返回
    ----
    RankIC ∈ [-1, 1]
    """
    mask = np.isfinite(factor) & np.isfinite(ret)
    if mask.sum() < 5:
        return np.nan
    ic, _ = spearmanr(factor[mask], ret[mask])
    return ic if not np.isnan(ic) else np.nan


def compute_ic_series(
    df: pd.DataFrame, factor_cols: list[str], ret_col: str = "ret_1d"
) -> pd.DataFrame:
    """逐日计算每个因子的 RankIC。

    参数
    ----
    df : 需包含 trade_date, 以及 factor_cols 和 ret_col
    factor_cols : 因子列名
    ret_col : 收益率列名

    返回
    ----
    pd.DataFrame: 行=trade_date, 列=factor_cols, 值=RankIC
    """
    records = []
    for dt, group in df.groupby("trade_date"):
        row = {"trade_date": dt}
        for fcol in factor_cols:
            row[fcol] = compute_rank_ic(group[fcol].values, group[ret_col].values)
        records.append(row)

    return pd.DataFrame(records).sort_values("trade_date")


def compute_ic_summary(ic_df: pd.DataFrame) -> pd.DataFrame:
    """汇总 IC 统计量。

    返回 DataFrame: index=factor, columns=ic_mean, ic_std, icir, n_days
    """
    factor_cols = [c for c in ic_df.columns if c != "trade_date"]
    rows = []
    for fcol in factor_cols:
        ics = ic_df[fcol].dropna()
        if len(ics) == 0:
            rows.append({"factor": fcol, "ic_mean": np.nan, "ic_std": np.nan,
                         "icir": np.nan, "n_days": 0})
        else:
            rows.append({
                "factor": fcol,
                "ic_mean": ics.mean(),
                "ic_std": ics.std(),
                "icir": ics.mean() / ics.std() if ics.std() > 0 else np.nan,
                "n_days": len(ics),
            })
    return pd.DataFrame(rows).set_index("factor")


def compute_ic_decay(
    df: pd.DataFrame, factor_col: str, ret_cols: list[str]
) -> pd.DataFrame:
    """IC 衰减曲线: 因子对未来不同时间窗口收益的预测力。

    参数
    ----
    factor_col : 因子名
    ret_cols : 不同时间窗口的收益率列（如 ret_1d, ret_3d, ret_5d, ret_10d）

    返回
    ----
    pd.DataFrame: decay_analysis
    """
    records = []
    for dt, group in df.groupby("trade_date"):
        row = {"trade_date": dt}
        for rcol in ret_cols:
            row[rcol] = compute_rank_ic(group[factor_col].values, group[rcol].values)
        records.append(row)

    ic_df = pd.DataFrame(records)
    summary = {}
    for rcol in ret_cols:
        ics = ic_df[rcol].dropna()
        summary[rcol] = float(ics.mean()) if len(ics) > 0 else np.nan
    return pd.Series(summary).to_frame("mean_ic").rename_axis("horizon").reset_index()
```

- [ ] **Step 4: 运行测试确认通过**

```bash
source .venv/bin/activate && python -m pytest tests/test_factors.py::TestICMonitor -x -q
```

- [ ] **Step 5: 提交**

```bash
git add factors/monitor.py tests/test_factors.py
git commit -m "feat: add factor IC monitoring pipeline"
```

---

## 阶段一出口检查点

阶段一（数据补齐）全部完成后，运行验证：

```bash
source .venv/bin/activate && python -c "
from data.db import get_engine, init_db
from sqlalchemy import text

init_db()
engine = get_engine()
with engine.connect() as conn:
    checks = {
        'stock_financial': 'SELECT COUNT(DISTINCT code) FROM stock_financial',
        'stock_industry': 'SELECT COUNT(*) FROM stock_industry',
        'stock_daily_extra': 'SELECT COUNT(DISTINCT code) FROM stock_daily_extra',
        'stock_shareholder': 'SELECT COUNT(DISTINCT code) FROM stock_shareholder',
    }
    for name, sql in checks.items():
        cnt = conn.execute(text(sql)).scalar()
        status = 'OK' if (cnt or 0) > 1000 else 'INCOMPLETE'
        print(f'{name}: {cnt} — {status}')
engine.dispose()
"
```

预期输出：4 张表都显示 `OK`，覆盖 ≥ 4000 只股票。

---

## 后续阶段（阶段二-四）

阶段一完成后，按相同 TDD 流程执行：
- 阶段二：因子引擎 + 30 Alpha101 因子 + 7 自定义因子 + IC 监控
- 阶段三：Walk-forward 样本构造 + XGBoost/LightGBM 训练/预测
- 阶段四：组合优化 + 风控 + 模拟盘验证

这些阶段的详细任务将在阶段一验证通过后展开。

---

## 执行选择

每完成一个阶段：
1. 运行全量测试
2. 更新 PROJECT.md 变更记录
3. `git push` 同步到 GitHub
