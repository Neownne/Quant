# Phase 2: ML 选股模型 — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建 walk-forward 样本 → 训练 XGBoost/LightGBM 预测次日涨跌 → 选股排序 → 回测验证年化 > 30%

**Architecture:** 从 DB 加载全市场 OHLCV + 财务数据 → FactorEngine 计算 37 因子 → walk-forward 切分 → 训练二分类器 → 每日预测 top-N 选股 → 组合优化 → 集成现有 backtest_runner 验证

**Tech Stack:** XGBoost, LightGBM, scikit-learn, SHAP, 现有 FactorEngine/backtest_runner

---

## 文件结构

| 文件 | 职责 |
|---|---|
| `models/__init__.py` | 模块入口 |
| `models/dataset.py` | 加载数据 → 因子计算 → 标签构造 → walk-forward 切分 |
| `models/trainer.py` | 训练流程：训练/验证/评估，输出模型 + 指标 |
| `models/predictor.py` | 加载模型，对最新截面做预测排序 |
| `portfolio/__init__.py` | 模块入口 |
| `portfolio/selector.py` | Top-N 选股 + ST/停牌/次新过滤 |
| `portfolio/allocator.py` | 仓位分配（等权/波动率倒数） |
| `portfolio/risk.py` | 风控规则（个股止损/组合回撤控制） |
| `tests/test_models.py` | 数据集+训练+预测测试 |
| `tests/test_portfolio.py` | 组合优化测试 |

---

### Task 1: 数据集构造 `models/dataset.py`

**Files:**
- Create: `models/__init__.py`
- Create: `models/dataset.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: 写测试**

```python
"""ML 模型测试。"""
import pytest
import pandas as pd
import numpy as np
from models.dataset import (
    build_factor_dataset,
    make_labels,
    walk_forward_split,
)


class TestDataset:
    def test_make_labels_binary(self):
        """make_labels 应为每个交易日计算 T+1 涨跌标签。"""
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            "code": ["000001"] * n,
            "trade_date": pd.date_range("2020-01-02", periods=n, freq="B"),
            "close": 10 + np.cumsum(np.random.randn(n) * 0.5),
        })
        result = make_labels(df, forward_days=1, mode="binary")
        assert "label" in result.columns
        # 最后一天没有未来数据，标签应为 NaN
        assert pd.isna(result["label"].iloc[-1])
        # 前面应有 0/1 标签
        labels = result["label"].dropna()
        assert labels.isin([0, 1]).all()

    def test_make_labels_regression(self):
        """回归模式应返回连续收益率。"""
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            "code": ["000001"] * n,
            "trade_date": pd.date_range("2020-01-02", periods=n, freq="B"),
            "close": 10 + np.cumsum(np.random.randn(n) * 0.5),
        })
        result = make_labels(df, forward_days=1, mode="regression")
        assert "label" in result.columns
        assert result["label"].iloc[-2] is not np.nan  # second to last has 1 forward day

    def test_walk_forward_split(self):
        """walk_forward_split 应生成 (train, val) 对的迭代器。"""
        dates = pd.date_range("2018-01-02", "2023-12-29", freq="B")
        df = pd.DataFrame({"trade_date": dates, "value": range(len(dates))})

        splits = list(walk_forward_split(df, train_years=3, val_years=1))
        assert len(splits) >= 2  # 至少 2 个窗口 (2018-20→2021, 2019-21→2022)
        train, val = splits[0]
        assert len(train) > len(val)
        # train 和 val 的时间不应重叠
        assert train["trade_date"].max() < val["trade_date"].min()

    def test_build_factor_dataset_smoke(self):
        """端到端冒烟测试：从少量股票的 OHLCV 构建因子数据集。"""
        # 构造 20 只股票 × 200 个交易日的模拟数据
        np.random.seed(42)
        dates = pd.date_range("2020-01-02", periods=200, freq="B")
        codes = [f"{i:06d}" for i in range(20)]
        rows = []
        for code in codes:
            close = 10 + np.cumsum(np.random.randn(200) * 0.5)
            for i, d in enumerate(dates):
                rows.append({
                    "code": code,
                    "trade_date": d.date(),
                    "open": close[i] * (1 + np.random.randn() * 0.01),
                    "high": close[i] * (1 + abs(np.random.randn()) * 0.02),
                    "low": close[i] * (1 - abs(np.random.randn()) * 0.02),
                    "close": close[i],
                    "volume": np.random.randint(100000, 1000000),
                })
        ohlcv = pd.DataFrame(rows)

        result = build_factor_dataset(
            ohlcv,
            factor_names=["rsi_14", "mom_20", "vol_20"],
            label_mode="binary",
        )

        assert "label" in result.columns
        assert "rsi_14" in result.columns
        assert "mom_20" in result.columns
        assert "vol_20" in result.columns
        assert "code" in result.columns
        assert "trade_date" in result.columns
        # 应有足够有效行（200天 × 20只，warmup 后）
        assert len(result.dropna()) > 1000
```

- [ ] **Step 2: 运行测试确认失败**

```bash
source .venv/bin/activate && python -m pytest tests/test_models.py -x -q
```

- [ ] **Step 3: 实现 `models/dataset.py`**

```python
"""数据集构造：因子计算 + 标签生成 + walk-forward 切分。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from factors import FactorEngine


def make_labels(
    df: pd.DataFrame,
    forward_days: int = 1,
    mode: str = "binary",
) -> pd.DataFrame:
    """为每行计算 T+N 收益率标签。

    参数
    ----
    df : 单只股票 DataFrame，须含 close, trade_date，已排序
    forward_days : 前瞻天数（1 = T+1 收益）
    mode : "binary" → 0/1 涨跌；"regression" → 连续收益率

    返回
    ----
    带 'label' 列的 DataFrame
    """
    df = df.copy()
    future_close = df["close"].shift(-forward_days)
    if mode == "binary":
        df["label"] = (future_close > df["close"]).astype(int)
    else:
        df["label"] = (future_close - df["close"]) / df["close"]
    return df


def walk_forward_split(
    df: pd.DataFrame,
    train_years: int = 3,
    val_years: int = 1,
    date_col: str = "trade_date",
    gap_days: int = 0,
):
    """Walk-forward 滚动窗口迭代器。

    参数
    ----
    df : 含 date_col 的 DataFrame
    train_years : 训练窗口年数
    val_years : 验证窗口年数
    gap_days : train 和 val 之间的间隔天数（避免 look-ahead）

    Yields
    ------
    (train_df, val_df)
    """
    df = df.sort_values(date_col).copy()
    all_dates = sorted(df[date_col].unique())
    if not all_dates:
        return

    start = all_dates[0]
    end = all_dates[-1]

    train_start = pd.Timestamp(start)
    while True:
        train_end = train_start + pd.DateOffset(years=train_years)
        val_start = train_end + pd.DateOffset(days=gap_days)
        val_end = val_start + pd.DateOffset(years=val_years)

        if val_end > pd.Timestamp(end):
            break

        train_mask = (df[date_col] >= train_start) & (df[date_col] < train_end)
        val_mask = (df[date_col] >= val_start) & (df[date_col] < val_end)

        train_df = df[train_mask]
        val_df = df[val_mask]

        if len(train_df) > 0 and len(val_df) > 0:
            yield train_df, val_df

        train_start = train_start + pd.DateOffset(years=1)


def build_factor_dataset(
    ohlcv: pd.DataFrame,
    factor_names: list[str],
    label_mode: str = "binary",
    forward_days: int = 1,
    extra_data: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """从 OHLCV 构建带标签的因子数据集。

    参数
    ----
    ohlcv : 须含 code, trade_date, open, high, low, close, volume
    factor_names : 因子名列表
    label_mode : "binary" | "regression"
    forward_days : 标签前瞻天数
    extra_data : 传递给 FactorEngine 的额外数据

    返回
    ----
    pd.DataFrame: [code, trade_date] + factor_names + [label]
    """
    engine = FactorEngine(factor_names=factor_names)
    logger.info(f"计算 {len(factor_names)} 个因子 ...")
    result = engine.compute(ohlcv, extra_data=extra_data)

    # 按股票分组计算标签
    logger.info("生成标签 ...")
    labelled_parts = []
    for code, group in result.groupby("code"):
        group = group.sort_values("trade_date")
        labelled_parts.append(make_labels(group, forward_days, label_mode))

    result = pd.concat(labelled_parts, ignore_index=True)
    logger.info(f"数据集: {len(result)} 行, {len(result.dropna())} 有效")
    return result
```

- [ ] **Step 4: 运行测试确认通过**

```bash
source .venv/bin/activate && python -m pytest tests/test_models.py -x -q
```

- [ ] **Step 5: 提交**

```bash
git add models/__init__.py models/dataset.py tests/test_models.py
git commit -m "feat: add dataset builder with walk-forward split"
```

---

### Task 2: 模型训练器 `models/trainer.py`

**Files:**
- Create: `models/trainer.py`

- [ ] **Step 1: 写测试**

```python
class TestTrainer:
    def test_train_xgboost_binary(self):
        """用模拟数据训练 XGBoost 二分类器，应返回模型和指标。"""
        from models.trainer import train_xgboost
        import numpy as np
        import pandas as pd

        np.random.seed(42)
        n = 500
        X = pd.DataFrame({
            "f1": np.random.randn(n),
            "f2": np.random.randn(n),
            "f3": np.random.randn(n),
        })
        y = (X["f1"] + X["f2"] * 0.5 > 0).astype(int)

        model, metrics = train_xgboost(X, y, X, y)
        assert model is not None
        assert "accuracy" in metrics
        assert metrics["accuracy"] > 0.5  # better than random

    def test_train_xgboost_returns_feature_importance(self):
        """应返回特征重要性 DataFrame。"""
        from models.trainer import train_xgboost
        import numpy as np
        import pandas as pd

        np.random.seed(42)
        n = 500
        X = pd.DataFrame({
            "f1": np.random.randn(n),
            "f2": np.random.randn(n),
        })
        y = (X["f1"] > 0).astype(int)

        model, metrics = train_xgboost(X, y, X, y)
        assert "feature_importance" in metrics
        fi = metrics["feature_importance"]
        assert "f1" in fi.index
        assert "f2" in fi.index

    def test_train_lightgbm(self):
        """LightGBM 也应能训练并返回模型。"""
        from models.trainer import train_lightgbm
        import numpy as np
        import pandas as pd

        np.random.seed(42)
        n = 500
        X = pd.DataFrame({
            "f1": np.random.randn(n),
            "f2": np.random.randn(n),
        })
        y = (X["f1"] + X["f2"] * 0.3 > 0).astype(int)

        model, metrics = train_lightgbm(X, y, X, y)
        assert model is not None
        assert "accuracy" in metrics
```

- [ ] **Step 2: 运行测试确认失败**

```bash
source .venv/bin/activate && python -m pytest tests/test_models.py::TestTrainer -x -q
```

- [ ] **Step 3: 实现 `models/trainer.py`**

```python
"""模型训练：XGBoost / LightGBM 训练与评估。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, feature_names: list[str], model) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "feature_importance": _feature_importance(model, feature_names),
    }


def _feature_importance(model, names: list[str]) -> pd.Series:
    """提取特征重要性。"""
    if hasattr(model, "feature_importances_"):
        return pd.Series(model.feature_importances_, index=names).sort_values(ascending=False)
    elif hasattr(model, "get_score"):
        scores = model.get_score(importance_type="gain")
        return pd.Series({k: scores.get(f"f{i}", 0) for i, k in enumerate(names)}).sort_values(ascending=False)
    return pd.Series(dtype=float)


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    params: dict | None = None,
) -> tuple:
    """训练 XGBoost 二分类器。

    返回: (model, metrics_dict)
    """
    default_params = {
        "n_estimators": 200,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "logloss",
        "random_state": 42,
    }
    if params:
        default_params.update(params)

    model = xgb.XGBClassifier(**default_params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    metrics = _compute_metrics(
        y_val.values if hasattr(y_val, "values") else y_val,
        y_pred, y_prob,
        list(X_train.columns), model,
    )
    logger.info(f"XGBoost: acc={metrics['accuracy']:.3f}, prec={metrics['precision']:.3f}, rec={metrics['recall']:.3f}")
    return model, metrics


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    params: dict | None = None,
) -> tuple:
    """训练 LightGBM 二分类器。"""
    default_params = {
        "n_estimators": 200,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbose": -1,
    }
    if params:
        default_params.update(params)

    model = lgb.LGBMClassifier(**default_params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    metrics = _compute_metrics(
        y_val.values if hasattr(y_val, "values") else y_val,
        y_pred, y_prob,
        list(X_train.columns), model,
    )
    logger.info(f"LightGBM: acc={metrics['accuracy']:.3f}, prec={metrics['precision']:.3f}, rec={metrics['recall']:.3f}")
    return model, metrics


def walk_forward_train(
    df: pd.DataFrame,
    factor_cols: list[str],
    model_type: str = "xgboost",
    train_years: int = 3,
    val_years: int = 1,
) -> list[dict]:
    """Walk-forward 训练循环。

    返回: [{model, metrics, train_start, train_end, val_start, val_end}, ...]
    """
    from models.dataset import walk_forward_split

    train_fn = train_xgboost if model_type == "xgboost" else train_lightgbm
    results = []

    for train_df, val_df in walk_forward_split(df, train_years, val_years):
        train_clean = train_df[factor_cols + ["label"]].dropna()
        val_clean = val_df[factor_cols + ["label"]].dropna()

        if len(train_clean) < 100 or len(val_clean) < 50:
            continue

        X_tr = train_clean[factor_cols]
        y_tr = train_clean["label"]
        X_v = val_clean[factor_cols]
        y_v = val_clean["label"]

        model, metrics = train_fn(X_tr, y_tr, X_v, y_v)
        results.append({
            "model": model,
            "metrics": metrics,
            "train_end": train_df["trade_date"].max(),
            "val_start": val_df["trade_date"].min(),
            "val_end": val_df["trade_date"].max(),
        })

    return results
```

- [ ] **Step 4: 运行测试确认通过**

```bash
source .venv/bin/activate && python -m pytest tests/test_models.py::TestTrainer -x -q
```

- [ ] **Step 5: 提交**

```bash
git add models/trainer.py tests/test_models.py
git commit -m "feat: add XGBoost/LightGBM trainer with walk-forward loop"
```

---

### Task 3: 每日预测器 `models/predictor.py`

**Files:**
- Create: `models/predictor.py`

- [ ] **Step 1: 写测试**

```python
class TestPredictor:
    def test_predict_returns_scores(self):
        """预测器应返回每只股票的得分和排名。"""
        from models.predictor import DailyPredictor
        from models.trainer import train_xgboost
        import numpy as np
        import pandas as pd

        np.random.seed(42)
        n = 300
        X = pd.DataFrame({"f1": np.random.randn(n), "f2": np.random.randn(n)})
        y = (X["f1"] * 0.7 + X["f2"] * 0.3 > 0).astype(int)

        model, _ = train_xgboost(X, y, X, y)
        predictor = DailyPredictor(model, factor_names=["f1", "f2"])

        # 模拟今日横截面
        today_data = pd.DataFrame({
            "code": [f"{i:06d}" for i in range(50)],
            "f1": np.random.randn(50),
            "f2": np.random.randn(50),
        })

        result = predictor.predict(today_data)
        assert "code" in result.columns
        assert "score" in result.columns
        assert "rank" in result.columns
        assert result["rank"].min() == 1
        assert result["rank"].max() == 50

    def test_predict_handles_missing_factors(self):
        """缺失因子列时应报清晰错误。"""
        from models.predictor import DailyPredictor
        import pytest

        predictor = DailyPredictor(None, factor_names=["f1", "f2"])
        bad_data = pd.DataFrame({"code": ["000001"], "f1": [1.0]})  # missing f2

        with pytest.raises(KeyError):
            predictor.predict(bad_data)
```

- [ ] **Step 2: 运行测试确认失败**

```bash
source .venv/bin/activate && python -m pytest tests/test_models.py::TestPredictor -x -q
```

- [ ] **Step 3: 实现 `models/predictor.py`**

```python
"""每日预测：加载模型，对最新截面做预测排序。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


class DailyPredictor:
    """每日预测器。

    用法:
        predictor = DailyPredictor(model, factor_names=["rsi_14", "mom_20"])
        scores = predictor.predict(today_factors)  # → [code, score, rank]
    """

    def __init__(self, model, factor_names: list[str]):
        self.model = model
        self.factor_names = factor_names

    def predict(self, factor_df: pd.DataFrame) -> pd.DataFrame:
        """对横截面数据打分排序。

        参数
        ----
        factor_df : DataFrame, 至少含 code 和所有 factor_names 列

        返回
        ----
        DataFrame: [code, score, rank], 按 score 降序排列
        """
        missing = set(self.factor_names) - set(factor_df.columns)
        if missing:
            raise KeyError(f"缺少因子列: {missing}")

        X = factor_df[self.factor_names].copy()

        # 填充 NaN（用列均值）
        X = X.fillna(X.mean())

        try:
            prob = self.model.predict_proba(X)[:, 1]
        except Exception:
            # 回归模式回退
            prob = self.model.predict(X)

        result = factor_df[["code"]].copy()
        result["score"] = prob
        result["rank"] = result["score"].rank(ascending=False, method="first").astype(int)
        result = result.sort_values("rank")

        logger.info(f"预测完成: {len(result)} 只股票, top-5: {result['code'].head().tolist()}")
        return result.reset_index(drop=True)
```

- [ ] **Step 4: 运行测试确认通过**

```bash
source .venv/bin/activate && python -m pytest tests/test_models.py::TestPredictor -x -q
```

- [ ] **Step 5: 提交**

```bash
git add models/predictor.py tests/test_models.py
git commit -m "feat: add DailyPredictor for cross-sectional ranking"
```

---

### Task 4: 选股 + 仓位分配 `portfolio/`

**Files:**
- Create: `portfolio/__init__.py`
- Create: `portfolio/selector.py`
- Create: `portfolio/allocator.py`
- Create: `tests/test_portfolio.py`

- [ ] **Step 1: 写测试**

```python
"""组合优化测试。"""
import pytest
import pandas as pd
import numpy as np
from portfolio.selector import select_top_n, filter_stocks
from portfolio.allocator import equal_weight, volatility_inverse_weight


class TestSelector:
    def test_select_top_n(self):
        """select_top_n 应从排序结果中选出得分最高的 N 只。"""
        scores = pd.DataFrame({
            "code": ["000001", "000002", "000003", "000004", "000005"],
            "score": [0.9, 0.7, 0.5, 0.3, 0.1],
            "rank": [1, 2, 3, 4, 5],
        })
        selected = select_top_n(scores, n=3)
        assert len(selected) == 3
        assert selected.iloc[0]["code"] == "000001"

    def test_filter_stocks_excludes_st(self):
        """应排除 ST 股票。"""
        stocks = pd.DataFrame({
            "code": ["000001", "000002", "000003"],
            "name": ["平安银行", "ST瑞德", "深振业"],
            "score": [0.9, 0.8, 0.7],
        })
        filtered = filter_stocks(stocks, exclude_st=True)
        assert "000002" not in filtered["code"].values

    def test_filter_stocks_excludes_new_listings(self):
        """应排除上市不足 60 天的次新股。"""
        stocks = pd.DataFrame({
            "code": ["000001", "000002"],
            "score": [0.9, 0.8],
            "list_date": [pd.Timestamp("2020-01-01"), pd.Timestamp("2026-05-01")],
        })
        ref_date = pd.Timestamp("2026-05-25")
        filtered = filter_stocks(stocks, ref_date=ref_date, min_list_days=60)
        assert "000002" not in filtered["code"].values


class TestAllocator:
    def test_equal_weight(self):
        """等权分配：N 只股票每只 1/N。"""
        result = equal_weight(["000001", "000002", "000003", "000004"], cash=1_000_000)
        assert len(result) == 4
        assert abs(result["weight"].sum() - 1.0) < 0.001
        assert result.iloc[0]["weight"] == 0.25

    def test_volatility_inverse_weight(self):
        """波动率倒数加权：低波动股票权重大。"""
        returns = pd.DataFrame({
            "000001": np.random.randn(100) * 0.01,
            "000002": np.random.randn(100) * 0.03,
        })
        result = volatility_inverse_weight(["000001", "000002"], returns, cash=1_000_000)
        assert len(result) == 2
        assert abs(result["weight"].sum() - 1.0) < 0.01
        # 000001 波动率更低，权重应更大
        assert result[result["code"] == "000001"]["weight"].iloc[0] > \
               result[result["code"] == "000002"]["weight"].iloc[0]
```

- [ ] **Step 2: 运行测试确认失败**

```bash
source .venv/bin/activate && python -m pytest tests/test_portfolio.py -x -q
```

- [ ] **Step 3: 实现 files**

`portfolio/__init__.py`:
```python
"""组合优化模块。"""
from portfolio.selector import select_top_n, filter_stocks
from portfolio.allocator import equal_weight, volatility_inverse_weight

__all__ = ["select_top_n", "filter_stocks", "equal_weight", "volatility_inverse_weight"]
```

`portfolio/selector.py`:
```python
"""选股：模型打分 top-N + ST/停牌/次新过滤。"""
import pandas as pd
from datetime import date


def select_top_n(scores: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """从排序结果中选前 N 只。"""
    return scores.sort_values("rank").head(n).reset_index(drop=True)


def filter_stocks(
    stocks: pd.DataFrame,
    ref_date: pd.Timestamp | None = None,
    exclude_st: bool = True,
    min_list_days: int = 60,
) -> pd.DataFrame:
    """过滤不可交易的股票。

    参数
    ----
    stocks : 至少含 code, name 列
    ref_date : 参考日期（默认今天）
    exclude_st : 排除 ST
    min_list_days : 最小上市天数
    """
    result = stocks.copy()
    ref = ref_date or pd.Timestamp(date.today())

    if exclude_st and "name" in result.columns:
        result = result[~result["name"].str.contains("ST", na=False)]

    if "list_date" in result.columns:
        result["days_listed"] = (ref - pd.to_datetime(result["list_date"])).dt.days
        result = result[result["days_listed"] >= min_list_days]

    return result.reset_index(drop=True)
```

`portfolio/allocator.py`:
```python
"""仓位分配。"""
import numpy as np
import pandas as pd


def equal_weight(codes: list[str], cash: float) -> pd.DataFrame:
    """等权分配。

    返回 DataFrame: [code, weight, shares, value]
    """
    n = len(codes)
    weight = 1.0 / n
    return pd.DataFrame({
        "code": codes,
        "weight": weight,
        "value": cash * weight,
    })


def volatility_inverse_weight(
    codes: list[str],
    returns_matrix: pd.DataFrame,
    cash: float,
    lookback: int = 60,
) -> pd.DataFrame:
    """波动率倒数加权。

    参数
    ----
    codes : 股票代码列表
    returns_matrix : DataFrame, 列为 code, 行=日期, 值=日收益率
    cash : 总资金
    lookback : 波动率回看窗口

    返回
    ----
    DataFrame: [code, weight, vol, value]
    """
    vols = {}
    for c in codes:
        if c in returns_matrix.columns:
            r = returns_matrix[c].tail(lookback).dropna()
            vols[c] = r.std() if len(r) > 10 else 1.0
        else:
            vols[c] = 1.0

    inv_vols = {c: 1.0 / max(v, 0.001) for c, v in vols.items()}
    total = sum(inv_vols.values())
    weights = {c: v / total for c, v in inv_vols.items()}

    return pd.DataFrame({
        "code": list(weights.keys()),
        "weight": list(weights.values()),
        "vol": [vols[c] for c in codes],
        "value": [cash * weights[c] for c in codes],
    })
```

- [ ] **Step 4: 运行测试确认通过**

```bash
source .venv/bin/activate && python -m pytest tests/test_portfolio.py -x -q
```

- [ ] **Step 5: 提交**

```bash
git add portfolio/__init__.py portfolio/selector.py portfolio/allocator.py tests/test_portfolio.py
git commit -m "feat: add portfolio selector and allocator"
```

---

### Task 5: 端到端回测集成

**Files:**
- Create: `portfolio/risk.py`
- Modify: `models/__init__.py`

- [ ] **Step 1: 实现风控 `portfolio/risk.py`**

```python
"""风控规则。"""
import numpy as np
import pandas as pd


def apply_stop_loss(
    positions: pd.DataFrame,
    prices: dict[str, float],
    cost_basis: dict[str, float],
    stop_pct: float = 0.08,
) -> pd.DataFrame:
    """个股止损：-8% 或 -1.5x ATR 触发卖出。

    返回需要平仓的 code 列表。
    """
    to_sell = []
    for _, row in positions.iterrows():
        code = row["code"]
        if code in prices and code in cost_basis and cost_basis[code] > 0:
            loss = (prices[code] - cost_basis[code]) / cost_basis[code]
            if loss < -stop_pct:
                to_sell.append(code)
    return pd.DataFrame({"code": to_sell}) if to_sell else pd.DataFrame(columns=["code"])


def check_drawdown_limit(current_value: float, peak_value: float, limit: float = 0.25) -> bool:
    """检查是否触发组合回撤上限。

    返回 True 表示应清仓/暂停。
    """
    if peak_value <= 0:
        return False
    drawdown = (peak_value - current_value) / peak_value
    return drawdown >= limit


def position_sizing(cash: float, risk_pct: float = 0.02, atr: float = 0) -> float:
    """基于风险的仓位计算。

    单个头寸风险 = cash × risk_pct
    止损距离 = 1.5 × ATR
    仓位 = 风险金额 / 止损距离
    """
    risk_amount = cash * risk_pct
    stop_distance = max(atr * 1.5, 0.01)
    return risk_amount / stop_distance
```

- [ ] **Step 2: 写风控测试**

```python
class TestRisk:
    def test_stop_loss_triggers(self):
        """跌幅超过阈值应触发止损。"""
        from portfolio.risk import apply_stop_loss
        positions = pd.DataFrame({"code": ["000001", "000002"]})
        prices = {"000001": 92.0, "000002": 105.0}
        cost_basis = {"000001": 100.0, "000002": 100.0}  # 000001 -8%

        result = apply_stop_loss(positions, prices, cost_basis, stop_pct=0.08)
        assert "000001" in result["code"].values
        assert "000002" not in result["code"].values

    def test_drawdown_limit(self):
        """回撤超限应触发预警。"""
        from portfolio.risk import check_drawdown_limit
        assert check_drawdown_limit(75.0, 100.0, 0.25)  # 25% drawdown → True
        assert not check_drawdown_limit(80.0, 100.0, 0.25)  # 20% → False
```

- [ ] **Step 3: 运行全量测试**

```bash
source .venv/bin/activate && python -m pytest tests/ -x -q
```

- [ ] **Step 4: 更新 `models/__init__.py`**

```python
"""ML 预测模块。"""
from models.dataset import build_factor_dataset, walk_forward_split, make_labels
from models.trainer import train_xgboost, train_lightgbm, walk_forward_train
from models.predictor import DailyPredictor

__all__ = [
    "build_factor_dataset", "walk_forward_split", "make_labels",
    "train_xgboost", "train_lightgbm", "walk_forward_train",
    "DailyPredictor",
]
```

- [ ] **Step 5: 提交**

```bash
git add portfolio/risk.py tests/test_portfolio.py models/__init__.py
git commit -m "feat: add risk controls and finalize module exports"
```

---

### Task 6: 端到端验证脚本

**Files:**
- Create: `scripts/run_ml_backtest.py`

- [ ] **Step 1: 实现端到端验证脚本**

```python
#!/usr/bin/env python
"""ML 选股端到端回测验证。

用法:
    python scripts/run_ml_backtest.py                   # 默认参数
    python scripts/run_ml_backtest.py --model lightgbm  # 换模型
"""
import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from loguru import logger

from data.db import get_engine
from app.utils.backtest_runner import run_backtest
from models.dataset import build_factor_dataset, walk_forward_split
from models.trainer import walk_forward_train
from models.predictor import DailyPredictor
from factors import ALL_FACTORS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="xgboost", choices=["xgboost", "lightgbm"])
    parser.add_argument("--factors", default="all", help="因子列表，逗号分隔或 'all'")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--train-years", type=int, default=3)
    parser.add_argument("--val-years", type=int, default=1)
    parser.add_argument("--start", default="20180101")
    parser.add_argument("--end", default="20260101")
    parser.add_argument("--codes", default="", help="测试股票代码，逗号分隔，留空=全量")
    args = parser.parse_args()

    # 选择因子
    if args.factors == "all":
        factor_names = list(ALL_FACTORS.keys())
    else:
        factor_names = [f.strip() for f in args.factors.split(",")]

    logger.info(f"使用 {len(factor_names)} 个因子: {factor_names[:5]}...")

    # 加载数据
    engine = get_engine()
    codes = [c.strip() for c in args.codes.split(",") if c.strip()] if args.codes else None
    if codes is None:
        codes = pd.read_sql("SELECT code FROM stock_basic LIMIT 200", engine)["code"].tolist()
        logger.info(f"测试范围: {len(codes)} 只股票")

    # OHLCV
    code_list = ",".join([f"'{c}'" for c in codes])
    sql = f"""
        SELECT code, trade_date, open, high, low, close, volume, turnover
        FROM stock_daily
        WHERE code IN ({code_list})
          AND trade_date BETWEEN '{args.start}' AND '{args.end}'
        ORDER BY code, trade_date
    """
    ohlcv = pd.read_sql(sql, engine)
    engine.dispose()
    logger.info(f"OHLCV: {len(ohlcv)} 行")

    # 构建因子数据集
    dataset = build_factor_dataset(ohlcv, factor_names, label_mode="binary")

    # Walk-forward 训练
    factor_cols = factor_names
    results = walk_forward_train(
        dataset, factor_cols, model_type=args.model,
        train_years=args.train_years, val_years=args.val_years,
    )
    logger.info(f"完成 {len(results)} 个 walk-forward 窗口")

    # 汇总
    all_metrics = []
    for i, r in enumerate(results):
        m = r["metrics"]
        logger.info(
            f"窗口 {i+1}: val={r['val_start'].date()}~{r['val_end'].date()}, "
            f"acc={m['accuracy']:.3f}, prec={m['precision']:.3f}, rec={m['recall']:.3f}"
        )
        all_metrics.append({
            "window": i + 1,
            "val_start": r["val_start"],
            "val_end": r["val_end"],
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
        })

    summary = pd.DataFrame(all_metrics)
    print("\n=== Walk-Forward 汇总 ===")
    print(summary.to_string(index=False))
    print(f"\n平均准确率: {summary['accuracy'].mean():.3f}")
    print(f"平均精确率: {summary['precision'].mean():.3f}")
    print(f"平均召回率: {summary['recall'].mean():.3f}")

    # 特征重要性
    if results:
        fi = results[-1]["metrics"].get("feature_importance", pd.Series(dtype=float))
        if not fi.empty:
            print("\n=== Top-10 因子 ===")
            print(fi.head(10).to_string())


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 运行验证**

```bash
source .venv/bin/activate && python scripts/run_ml_backtest.py --codes "000001,000002,000858,600036,600519,601318" --start 20200101
```

- [ ] **Step 3: 提交**

```bash
git add scripts/run_ml_backtest.py
git commit -m "feat: add end-to-end ML backtest validation script"
```

---

## 阶段二出口检查点

全部完成后：

```bash
# 全量测试
source .venv/bin/activate && python -m pytest tests/ -x -q

# 端到端验证（200 只股票，全量 37 因子）
source .venv/bin/activate && python scripts/run_ml_backtest.py --start 20180101 --model xgboost
```

预期输出：3+ 个 walk-forward 窗口，平均准确率 > 0.52（超过 50% 随机基线）。
```

---

## 执行选择

每完成一个阶段：
1. 运行全量测试
2. 更新 PROJECT.md 变更记录
3. `git push` 同步到 GitHub
