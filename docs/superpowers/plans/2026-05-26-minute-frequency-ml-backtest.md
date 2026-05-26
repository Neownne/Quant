# 分钟频 ML 回测优化 — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 ML 回测数据源从日频升级为分钟频（60min），同步历史数据并通过交叉验证确认有效性。

**Architecture:** 4 个独立任务：数据同步 → 数据校验 → ML 引擎 freq 适配 → 回测 UI。日频逻辑不受影响，通过 `freq` 参数切换。分钟因子计算后在 `build_factor_dataset` 内聚合成日频用于标签生成。

**Tech Stack:** Python 3, akshare, pandas, SQLAlchemy, Streamlit, backtrader

---

## File Structure

| 文件 | 职责 |
|------|------|
| `scripts/sync_minute_data.py` (NEW) | 同步 60min K 线，排除 ST/<180天/北交所 |
| `scripts/validate_minute_data.py` (NEW) | 分钟聚合日 close vs stock_daily close 偏差检查 |
| `app/utils/ml_backtest.py` (MODIFY) | 新增 `_get_minute_ohlcv`, `load_price_data`; `run_ml_backtest` 使用 `freq` 参数 |
| `models/dataset.py` (MODIFY) | `build_factor_dataset` 新增 `bar_per_day`，分钟频因子值聚合为日频 |
| `factors/engine.py` (MODIFY) | `FactorEngine.__init__` 新增 `bar_per_day` 参数 |
| `app/pages/3_🧪_Backtest.py` (MODIFY) | ML 策略面板新增 freq 下拉框 |

---

### Task 1: 分钟数据同步脚本

**Files:**
- Create: `scripts/sync_minute_data.py`

**目的:** 批量下载 60 分钟 K 线写入 `stock_minute` 表。

- [ ] **Step 1: 编写同步脚本**

```python
#!/usr/bin/env python3
"""同步 60 分钟 K 线数据到 stock_minute 表。

排除: ST / 上市 <180 天 / 北交所(4开头) / 科创板(8开头，无 Sina 分钟数据)。
速率: 每只间隔 0.3s，单只失败重试 3 次。
"""
import sys
import time
sys.path.insert(0, ".")

import pandas as pd
from data.db import get_engine, upsert_df, init_db
from data.fetcher import fetch_minute_data

PERIOD = "60"
SLEEP = 0.3
MAX_RETRIES = 3


def get_eligible_codes() -> list[str]:
    engine = get_engine()
    sql = """
        SELECT DISTINCT b.code FROM stock_basic b
        INNER JOIN stock_daily d ON b.code = d.code
        WHERE b.is_st = FALSE
          AND b.list_date <= CURRENT_DATE - INTERVAL '180 days'
          AND b.code NOT LIKE '4%'
          AND b.code NOT LIKE '8%'
        ORDER BY b.code
    """
    codes = pd.read_sql(sql, engine)["code"].tolist()
    engine.dispose()
    return codes


def main():
    init_db()
    codes = get_eligible_codes()
    print(f"共 {len(codes)} 只股票待同步")

    success = 0
    fail = 0
    total_bars = 0

    for i, code in enumerate(codes):
        ok = False
        for attempt in range(MAX_RETRIES):
            try:
                df = fetch_minute_data(code, period=PERIOD, adjust="qfq")
                if df.empty:
                    ok = True
                    break
                upsert_df(df, "stock_minute")
                n = len(df)
                total_bars += n
                success += 1
                ok = True
                print(f"[{i+1}/{len(codes)}] {code} OK ({n} bars)")
                break
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2)
                else:
                    fail += 1
                    print(f"[{i+1}/{len(codes)}] {code} FAIL: {e}")
        time.sleep(SLEEP)

    print(f"\n完成: 成功 {success}, 失败 {fail}, 总 bar {total_bars}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 验证单只股票同步**

Run:
```bash
python3 -c "
import sys; sys.path.insert(0, '.')
from data.db import get_engine, upsert_df
from data.fetcher import fetch_minute_data

df = fetch_minute_data('600036', period='60', adjust='qfq')
print(f'rows={len(df)}, cols={list(df.columns)}')
print(df.head(2))
upsert_df(df, 'stock_minute')
print('upsert OK')

# verify
engine = get_engine()
import pandas as pd
n = pd.read_sql(\"SELECT COUNT(*) FROM stock_minute WHERE code='600036' AND period='60'\", engine).iloc[0,0]
print(f'stored rows: {n}')
engine.dispose()
"
```

Expected: `rows > 0`, `upsert OK`, `stored rows > 0`

- [ ] **Step 3: 提交**

```bash
git add scripts/sync_minute_data.py
git commit -m "feat: add minute K-line sync script (60min, Sina)"
```

---

### Task 2: 数据校验脚本

**Files:**
- Create: `scripts/validate_minute_data.py`

**目的:** 对比分钟数据聚合 close vs 日线 close，确保偏差 < 1%。

- [ ] **Step 1: 编写校验脚本**

```python
#!/usr/bin/env python3
"""校验分钟数据：聚合日频 close 对比 stock_daily close。

用法:
    python3 scripts/validate_minute_data.py --sample 50
"""
import sys
sys.path.insert(0, ".")

import pandas as pd
from data.db import get_engine


def validate(codes: list[str]) -> dict:
    engine = get_engine()
    code_list = ",".join([f"'{c}'" for c in codes])

    # 分钟取每日最后 bar close
    minute_sql = f"""
        SELECT code, trade_time::date AS trade_date, close
        FROM stock_minute
        WHERE code IN ({code_list}) AND period = '60'
        ORDER BY code, trade_time
    """
    minute_df = pd.read_sql(minute_sql, engine)
    if minute_df.empty:
        engine.dispose()
        return {"error": "无分钟数据，请先运行 sync_minute_data.py"}

    minute_daily = minute_df.groupby(["code", "trade_date"])["close"].last().reset_index()
    minute_daily["trade_date"] = pd.to_datetime(minute_daily["trade_date"])

    # 日线 close
    daily_df = pd.read_sql(
        f"SELECT code, trade_date, close FROM stock_daily "
        f"WHERE code IN ({code_list}) ORDER BY code, trade_date",
        engine,
    )
    engine.dispose()
    daily_df["trade_date"] = pd.to_datetime(daily_df["trade_date"])

    merged = minute_daily.merge(daily_df, on=["code", "trade_date"], suffixes=("_m", "_d"))
    if merged.empty:
        return {"error": "无重叠日期"}

    merged["deviation"] = (merged["close_m"] - merged["close_d"]).abs() / merged["close_d"]
    bad = merged[merged["deviation"] > 0.01]

    return {
        "n_overlap": len(merged),
        "max_deviation": merged["deviation"].max(),
        "mean_deviation": merged["deviation"].mean(),
        "n_bad": len(bad),
        "bad_samples": bad[["code", "trade_date", "close_m", "close_d", "deviation"]].head(20).to_string(),
    }


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=50)
    args = p.parse_args()

    engine = get_engine()
    codes = pd.read_sql(
        f"SELECT DISTINCT code FROM stock_minute WHERE period='60' LIMIT {args.sample}", engine
    )["code"].tolist()
    engine.dispose()

    if not codes:
        print("无分钟数据，请先运行 sync_minute_data.py")
        return

    r = validate(codes)
    if "error" in r:
        print(f"ERROR: {r['error']}")
        return

    print(f"样本: {len(codes)} 只, 重叠日: {r['n_overlap']}")
    print(f"Close偏差: max={r['max_deviation']:.4%}, mean={r['mean_deviation']:.4%}, >1%= {r['n_bad']} 条")
    if r["n_bad"] > 0:
        print("异常:")
        print(r["bad_samples"])
    else:
        print("校验通过 ✓")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 提交**

```bash
git add scripts/validate_minute_data.py
git commit -m "feat: add minute-vs-daily data validation script"
```

---

### Task 3: ML 回测引擎 freq 适配

**Files:**
- Modify: `app/utils/ml_backtest.py` (lines 1-17 新增 helper, lines 44-63 替换 OHLCV 加载)
- Modify: `models/dataset.py` (lines 90-141, build_factor_dataset 新增 bar_per_day + 分钟频标签适配)
- Modify: `factors/engine.py` (line 24, FactorEngine 新增 bar_per_day 参数)

**目的:** `run_ml_backtest` 支持 `freq="60min"`，分钟频因子值聚合成日频后与日频标签对齐。

- [ ] **Step 1: FactorEngine 新增 bar_per_day 参数**

修改 `factors/engine.py:24`:

```python
class FactorEngine:
    """因子计算引擎。

    参数
    ----
    factor_names : 要计算的因子名列表，必须在 ALL_FACTORS 中注册。
    bar_per_day  : 每日 bar 数，用于 window 自适应（60min=4, daily=1）。
    """

    def __init__(self, factor_names: list[str], bar_per_day: int = 1):
        if factor_names:
            missing = set(factor_names) - set(factors.ALL_FACTORS.keys())
            if missing:
                raise KeyError(f"未知因子: {missing}")
        self.factor_names = factor_names
        self.bar_per_day = bar_per_day
```

- [ ] **Step 2: 在 ml_backtest.py 中添加数据加载函数**

在 `app/utils/ml_backtest.py` 第 16 行之后（`from portfolio.selector ...` 之后），添加：

```python
from data.db import get_engine


def _get_daily_ohlcv(codes: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """从 stock_daily 加载日线 OHLCV。"""
    engine = get_engine()
    code_list = ",".join([f"'{c}'" for c in codes])
    try:
        df = pd.read_sql(
            f"SELECT code, trade_date, open, high, low, close, volume, amount, turnover "
            f"FROM stock_daily WHERE code IN ({code_list}) "
            f"AND trade_date BETWEEN '{start_date}' AND '{end_date}' "
            f"ORDER BY code, trade_date",
            engine,
        )
    finally:
        engine.dispose()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


def _get_minute_ohlcv(codes: list[str], start_date: str, end_date: str, period: str = "60") -> pd.DataFrame:
    """从 stock_minute 加载分钟 K 线。返回列包含 trade_date（date 类型）。"""
    engine = get_engine()
    code_list = ",".join([f"'{c}'" for c in codes])
    try:
        df = pd.read_sql(
            f"SELECT code, trade_time, trade_time::date AS trade_date, "
            f"open, high, low, close, volume, amount "
            f"FROM stock_minute WHERE code IN ({code_list}) "
            f"AND trade_time >= '{start_date}' AND trade_time < '{end_date}235959' "
            f"AND period = '{period}' "
            f"ORDER BY code, trade_time",
            engine,
        )
    finally:
        engine.dispose()
    if not df.empty:
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df["turnover"] = None
    return df


def load_price_data(codes: list[str], start_date: str, end_date: str,
                    freq: str = "daily") -> pd.DataFrame:
    """统一价格数据加载入口。
    
    参数
    ----
    freq : "daily" | "60min"
    """
    if freq == "daily":
        return _get_daily_ohlcv(codes, start_date, end_date)
    elif freq == "60min":
        return _get_minute_ohlcv(codes, start_date, end_date, period="60")
    else:
        raise ValueError(f"不支持的频率: {freq}")
```

- [ ] **Step 3: 替换 run_ml_backtest 中 OHLCV 加载逻辑**

将原 lines 44-64（从 `# 2. 加载 OHLCV` 到 `ohlcv["trade_date"] = pd.to_datetime(ohlcv["trade_date"])`）替换为：

```python
    # 2. 加载 OHLCV
    if progress_callback:
        progress_callback("加载OHLCV数据...", 0.05)

    freq = config.get("freq", "daily")
    ohlcv = load_price_data(codes, start_date, end_date, freq=freq)

    if ohlcv.empty:
        return {"error": "无OHLCV数据"}
```

同时删除原 `from data.db import get_engine` import（line 8）和后续的 `engine = get_engine()` / `engine.dispose()` 调用（因为它们已在 helper 函数内部处理）。

注意：第 51 行仍用 `engine = get_engine()` 做 `pd.read_sql`，但已在 Step 3 移除了这段代码。确认删除干净即可。

- [ ] **Step 4: 修改 build_factor_dataset 添加 bar_per_day 和分钟频标签适配**

修改 `models/dataset.py` 中 `build_factor_dataset` 函数签名和实现：

```python
def build_factor_dataset(
    ohlcv: pd.DataFrame,
    factor_names: list[str],
    label_mode: str = "binary",
    forward_days: int = 1,
    extra_data: dict[str, pd.DataFrame] | None = None,
    bar_per_day: int = 1,
) -> pd.DataFrame:
    """从 OHLCV 构建带标签的因子数据集。

    参数
    ----
    ohlcv : 须含 code, trade_date, open, high, low, close, volume
    factor_names : 因子名列表
    label_mode : "binary" | "regression"
    forward_days : 标签前瞻天数
    extra_data : 传递给 FactorEngine 的额外数据
    bar_per_day : 每日 bar 数（60min=4, daily=1）。

    返回
    ----
    pd.DataFrame: [code, trade_date] + factor_names + [label]
    """
    engine = FactorEngine(factor_names=factor_names, bar_per_day=bar_per_day)
    logger.info(f"计算 {len(factor_names)} 个因子 ...")
    result = engine.compute(ohlcv, extra_data=extra_data)

    result["trade_date"] = pd.to_datetime(result["trade_date"])

    # 分钟频：因子值聚合成日频（取每日最后一根 bar 的因子值）
    if bar_per_day > 1:
        group_cols = ["code", "trade_date"]
        factor_vals = result[group_cols + factor_names].copy()
        result = factor_vals.groupby(group_cols, as_index=False).last()

    # 构建日频 close 用于标签
    ohlcv_sub = ohlcv[["code", "trade_date", "close"]].copy()
    ohlcv_sub["trade_date"] = pd.to_datetime(ohlcv_sub["trade_date"])
    if bar_per_day > 1:
        ohlcv_sub = ohlcv_sub.groupby(["code", "trade_date"], as_index=False).last()

    result = result.merge(ohlcv_sub, on=["code", "trade_date"], how="left")

    # 按股票分组计算标签
    logger.info("生成标签 ...")
    labelled_parts = []
    for code, group in result.groupby("code"):
        group = group.sort_values("trade_date")
        labelled_parts.append(make_labels(group, forward_days, label_mode))

    result = pd.concat(labelled_parts, ignore_index=True)

    # 计算 T+1 连续收益率（供 IC 计算用）
    ret_parts = []
    for code, group in result.groupby("code"):
        group = group.sort_values("trade_date")
        group["ret_1d"] = group["close"].pct_change().shift(-1)
        ret_parts.append(group)
    result = pd.concat(ret_parts, ignore_index=True)

    result = result.drop(columns=["close"])
    logger.info(f"数据集: {len(result)} 行, {len(result.dropna())} 有效")
    return result
```

- [ ] **Step 5: run_ml_backtest 调用 build_factor_dataset 时传入 bar_per_day**

修改 `app/utils/ml_backtest.py` 中因子构建调用处：

```python
    # 3. 构建因子数据集
    if progress_callback:
        progress_callback("计算因子...", 0.10)

    bar_per_day = 4 if freq == "60min" else 1
    try:
        dataset = build_factor_dataset(ohlcv, factor_names,
                                        label_mode=config.get("label_mode", "binary"),
                                        bar_per_day=bar_per_day)
    except Exception as e:
        return {"error": f"因子计算失败: {e}"}
```

- [ ] **Step 6: 提交**

```bash
git add app/utils/ml_backtest.py models/dataset.py factors/engine.py
git commit -m "feat: add freq parameter to ML backtest for minute-frequency support"
```

---

### Task 4: 回测页面 freq 选择器

**Files:**
- Modify: `app/pages/3_🧪_Backtest.py` (ML 策略参数区)

**目的:** ML 策略面板新增数据频率下拉框。

- [ ] **Step 1: 在回测页面 ML 参数区添加 freq 选择**

找到 ML 策略参数面板区域（应在 ML 策略选中后展开的参数列），添加：

```python
# 在 ML 策略参数展开区域中添加
freq = st.selectbox(
    "数据频率",
    ["daily", "60min"],
    index=0,
    key="ml_freq",
    help="60分钟线使用前复权 Sina 数据，回测耗时更长",
)
if freq == "60min":
    st.caption("分钟频回测耗时较长，建议先用少量股票测试")
```

此 `freq` 值在调用 `run_ml_backtest` 前写入 config：

```python
config["freq"] = freq
```

- [ ] **Step 2: 提交**

```bash
git add app/pages/3_🧪_Backtest.py
git commit -m "feat: add frequency selector (daily/60min) to ML backtest UI"
```

---

## 验证

### Task 1+2 验证

```bash
# 先同步少量数据确认跑通
python3 scripts/sync_minute_data.py
# 校验
python3 scripts/validate_minute_data.py --sample 100
```

### Task 3+4 验证

```bash
# 启动 Streamlit
streamlit run app/main.py
#  → 回测页 → 选 ML 策略 → freq 选 60min → 运行
#  → 确认权益曲线显示、指标合理
```

### 交叉验证（完整流程）

用同一 ML 配置分别跑 daily 和 60min（相同 stock pool、时间区间），对比年化收益和夏普，偏差应 < 30%。

---

## 执行顺序

```
Task 1 (sync) → Task 2 (validate) → Task 3 (engine freq) → Task 4 (UI)
```

Task 1/2 可在一个 implementer 中连续执行，Task 3/4 各独立。
