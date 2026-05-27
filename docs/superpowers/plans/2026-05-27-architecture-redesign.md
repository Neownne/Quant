# 架构重构实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将项目从 Streamlit 单体重构为 FastAPI + HTMX 监控面板 + 后台策略研究引擎，新增数据质量/防过拟合/归因反馈闭环。

**Architecture:** Web 层（FastAPI + Jinja2 + HTMX + Alpine.js + ECharts CDN）只读查询 PostgreSQL，后台脚本（数据同步/因子计算/回测/归因）读写数据库，策略代码不进 Web 进程。

**Tech Stack:** FastAPI, Jinja2, HTMX, Alpine.js, ECharts CDN, PostgreSQL, Python (pandas, numpy, scipy, xgboost, lightgbm)

---

## 文件结构总览

```
quant/
├── web/                        # NEW: FastAPI app (替代 app/)
│   ├── __init__.py
│   ├── main.py                 # FastAPI entry
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── dashboard.py        # GET /
│   │   ├── backtest.py         # GET /backtest
│   │   ├── paper.py            # GET /paper
│   │   ├── data_status.py      # GET /data
│   │   ├── factors.py          # GET /factors
│   │   └── api.py              # HTMX JSON/HTML endpoints
│   ├── templates/
│   │   ├── base.html
│   │   ├── dashboard.html
│   │   ├── backtest.html
│   │   ├── paper.html
│   │   ├── data.html
│   │   ├── factors.html
│   │   └── partials/
│   │       ├── kline.html
│   │       ├── quote_table.html
│   │       ├── backtest_table.html
│   │       ├── equity_curve.html
│   │       ├── paper_positions.html
│   │       └── sync_log.html
│   └── static/
│       └── app.js
├── data/
│   ├── quality.py              # NEW: 数据质量校验
│   ├── lineage.py              # NEW: 因子血缘跟踪
│   └── availability.py         # NEW: 因子可用性时间戳
├── scripts/
│   ├── overfit_check.py        # NEW: 回测防过拟合验证
│   ├── attribution.py          # NEW: 信号归因分析
│   ├── auto_adjust.py          # NEW: 自动权重调整
│   ├── health_check.py         # NEW: 策略健康监控
│   └── command_worker.py       # NEW: 策略指令队列执行器
├── data/db.py                  # MODIFY: 新增13张表DDL
├── data/sync.py                # MODIFY: 加质量校验关卡
├── factors/engine.py           # MODIFY: 加lineage/availability记录
├── portfolio/paper_engine.py   # MODIFY: 加signal_factors写入
├── scripts/run_ml_backtest.py  # MODIFY: 加防过拟合+结果入库
├── app/                        # DELETE: 整个目录
├── README.md                   # MODIFY: 更新架构/目录/工作流
└── docs/project-learning-guide.md  # MODIFY: 更新系统架构章节
```

---

### Task 1: 数据库DDL —— 新增13张表

**Files:**
- Modify: `data/db.py`

- [ ] **Step 1: 在 db.py 末尾追加新表DDL**

在 `data/db.py` 的 `create_tables()` 函数中现有建表语句之后追加以下DDL：

```python
# ========== 策略管理 ==========

DB.execute("""
CREATE TABLE IF NOT EXISTS strategy_configs (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    type VARCHAR(20) NOT NULL CHECK (type IN ('ml', 'static')),
    description TEXT DEFAULT '',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
""")

DB.execute("""
CREATE TABLE IF NOT EXISTS strategy_versions (
    id SERIAL PRIMARY KEY,
    strategy_id INT NOT NULL REFERENCES strategy_configs(id),
    version VARCHAR(20) NOT NULL,
    algorithm_type VARCHAR(50) NOT NULL,
    feature_list_version VARCHAR(20) NOT NULL,
    model_file_path TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (strategy_id, version)
);
""")

DB.execute("""
CREATE TABLE IF NOT EXISTS factor_weights_history (
    id SERIAL PRIMARY KEY,
    strategy_id INT NOT NULL REFERENCES strategy_configs(id),
    factor_name VARCHAR(100) NOT NULL,
    weight DOUBLE PRECISION NOT NULL,
    effective_date DATE NOT NULL,
    source VARCHAR(10) NOT NULL CHECK (source IN ('auto', 'manual')),
    reason TEXT DEFAULT '',
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_fwh_strategy_date
    ON factor_weights_history (strategy_id, effective_date);
""")

# ========== 因子元数据 ==========

DB.execute("""
CREATE TABLE IF NOT EXISTS factor_lineage (
    id SERIAL PRIMARY KEY,
    factor_name VARCHAR(100) NOT NULL UNIQUE,
    source_fields TEXT[] NOT NULL,
    computation_formula_hash VARCHAR(64) NOT NULL,
    upstream_factors TEXT[] DEFAULT '{}',
    last_validated_at TIMESTAMPTZ DEFAULT NOW()
);
""")

DB.execute("""
CREATE TABLE IF NOT EXISTS factor_availability (
    id SERIAL PRIMARY KEY,
    trade_date DATE NOT NULL,
    factor_name VARCHAR(100) NOT NULL,
    data_ready_at TIMESTAMPTZ NOT NULL,
    data_source VARCHAR(50) DEFAULT '',
    latency_ms INT DEFAULT 0,
    UNIQUE (trade_date, factor_name)
);
CREATE INDEX IF NOT EXISTS idx_fa_date ON factor_availability (trade_date);
""")

# ========== 回测结果（扩展） ==========

DB.execute("""
CREATE TABLE IF NOT EXISTS backtest_results (
    id SERIAL PRIMARY KEY,
    version_id INT NOT NULL REFERENCES strategy_versions(id),
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    quality VARCHAR(10) NOT NULL DEFAULT 'valid'
        CHECK (quality IN ('valid', 'suspect', 'invalid')),
    quality_flags TEXT[] DEFAULT '{}',
    metrics_json JSONB NOT NULL DEFAULT '{}',
    equity_curve_json JSONB NOT NULL DEFAULT '{}',
    daily_returns_json JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_br_version ON backtest_results (version_id);
""")

# ========== 模拟盘 ==========

DB.execute("""
CREATE TABLE IF NOT EXISTS paper_runs (
    id SERIAL PRIMARY KEY,
    strategy_id INT NOT NULL REFERENCES strategy_configs(id),
    version_id INT NOT NULL REFERENCES strategy_versions(id),
    start_date DATE NOT NULL,
    end_date DATE,
    initial_capital DOUBLE PRECISION NOT NULL,
    status VARCHAR(10) NOT NULL DEFAULT 'running'
        CHECK (status IN ('running', 'paused', 'stopped')),
    created_at TIMESTAMPTZ DEFAULT NOW()
);
""")

DB.execute("""
CREATE TABLE IF NOT EXISTS paper_signals (
    id SERIAL PRIMARY KEY,
    run_id INT NOT NULL REFERENCES paper_runs(id),
    signal_date DATE NOT NULL,
    stock_code VARCHAR(10) NOT NULL,
    predicted_score DOUBLE PRECISION,
    rank INT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_ps_run_date ON paper_signals (run_id, signal_date);
""")

DB.execute("""
CREATE TABLE IF NOT EXISTS signal_factors (
    id SERIAL PRIMARY KEY,
    signal_id INT NOT NULL REFERENCES paper_signals(id) ON DELETE CASCADE,
    factor_name VARCHAR(100) NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    UNIQUE (signal_id, factor_name)
);
CREATE INDEX IF NOT EXISTS idx_sf_signal ON signal_factors (signal_id);
CREATE INDEX IF NOT EXISTS idx_sf_factor_date ON signal_factors (factor_name);
""")

DB.execute("""
CREATE TABLE IF NOT EXISTS paper_positions (
    id SERIAL PRIMARY KEY,
    run_id INT NOT NULL REFERENCES paper_runs(id),
    signal_id INT REFERENCES paper_signals(id),
    stock_code VARCHAR(10) NOT NULL,
    entry_date DATE NOT NULL,
    entry_price DOUBLE PRECISION NOT NULL,
    exit_date DATE,
    exit_price DOUBLE PRECISION,
    quantity INT NOT NULL DEFAULT 100,
    pnl DOUBLE PRECISION DEFAULT 0,
    pnl_pct DOUBLE PRECISION DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_pp_run ON paper_positions (run_id);
""")

# ========== 归因分析 ==========

DB.execute("""
CREATE TABLE IF NOT EXISTS signal_attribution (
    id SERIAL PRIMARY KEY,
    signal_id INT NOT NULL REFERENCES paper_signals(id),
    eval_date DATE NOT NULL,
    days_held INT NOT NULL DEFAULT 1,
    pnl DOUBLE PRECISION DEFAULT 0,
    pnl_pct DOUBLE PRECISION DEFAULT 0,
    factor_contrib_json JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_sa_signal ON signal_attribution (signal_id);
""")

# ========== 权重调整记录 ==========

DB.execute("""
CREATE TABLE IF NOT EXISTS weight_adjustments (
    id SERIAL PRIMARY KEY,
    strategy_id INT NOT NULL REFERENCES strategy_configs(id),
    factor_name VARCHAR(100) NOT NULL,
    old_weight DOUBLE PRECISION NOT NULL,
    new_weight DOUBLE PRECISION NOT NULL,
    confidence_level DOUBLE PRECISION DEFAULT 0.95,
    source VARCHAR(10) NOT NULL CHECK (source IN ('auto', 'manual')),
    reason TEXT DEFAULT '',
    created_at TIMESTAMPTZ DEFAULT NOW()
);
""")

# ========== 策略健康 ==========

DB.execute("""
CREATE TABLE IF NOT EXISTS strategy_health (
    id SERIAL PRIMARY KEY,
    strategy_id INT NOT NULL REFERENCES strategy_configs(id),
    date DATE NOT NULL DEFAULT CURRENT_DATE,
    overall_ic DOUBLE PRECISION,
    max_drawdown_7d DOUBLE PRECISION,
    regime_tag VARCHAR(10) DEFAULT 'unknown'
        CHECK (regime_tag IN ('bull', 'bear', 'range', 'unknown')),
    status VARCHAR(10) NOT NULL DEFAULT 'normal'
        CHECK (status IN ('normal', 'warning', 'critical')),
    action_required VARCHAR(20) DEFAULT 'none',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (strategy_id, date)
);
""")

# ========== 指令队列 ==========

DB.execute("""
CREATE TABLE IF NOT EXISTS strategy_commands (
    id SERIAL PRIMARY KEY,
    strategy_id INT NOT NULL REFERENCES strategy_configs(id),
    command_type VARCHAR(30) NOT NULL
        CHECK (command_type IN ('adjust_weight', 'pause', 'resume', 'rollback', 'retrain')),
    payload_json JSONB NOT NULL DEFAULT '{}',
    requested_by VARCHAR(50) DEFAULT 'user',
    requested_at TIMESTAMPTZ DEFAULT NOW(),
    executed_at TIMESTAMPTZ,
    execution_result TEXT DEFAULT '',
    rolled_back_by INT REFERENCES strategy_commands(id)
);
""")

# ========== 数据质量 ==========

DB.execute("""
CREATE TABLE IF NOT EXISTS data_quality_log (
    id SERIAL PRIMARY KEY,
    trade_date DATE NOT NULL,
    check_name VARCHAR(50) NOT NULL,
    expected_value TEXT,
    actual_value TEXT,
    passed BOOLEAN NOT NULL DEFAULT FALSE,
    detail TEXT DEFAULT '',
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_dql_date ON data_quality_log (trade_date);
""")
```

- [ ] **Step 2: 运行建表脚本验证**

```bash
cd /Users/chenwan/Documents/quant && python -c "
from data.db import create_tables, DB
create_tables()
tables = DB.fetch_all(\"SELECT table_name FROM information_schema.tables WHERE table_schema='public' ORDER BY table_name\")
for t in tables:
    print(t[0])
"
```

预计输出：新增13张表，总表数从16张增至29张。

- [ ] **Step 3: 提交**

```bash
git add data/db.py
git commit -m "feat: add 13 new tables for strategy lifecycle management"
```

---

### Task 2: 数据质量校验模块

**Files:**
- Create: `data/quality.py`
- Test: `tests/test_quality.py`

- [ ] **Step 1: 编写测试**

```bash
mkdir -p /Users/chenwan/Documents/quant/tests
```

```python
# tests/test_quality.py
import pytest
import pandas as pd
from data.quality import DataQualityChecker


class TestCoverageCheck:
    def test_coverage_pass_when_all_stocks_present(self):
        checker = DataQualityChecker(expected_stock_count=5000)
        df = pd.DataFrame({
            "code": [f"00000{i}" for i in range(5000)],
            "close": [10.0] * 5000,
            "volume": [1e6] * 5000,
            "change_pct": [1.0] * 5000,
        })
        result = checker.check_coverage(df, "2024-01-15")
        assert result["passed"] is True

    def test_coverage_fail_when_coverage_low(self):
        checker = DataQualityChecker(expected_stock_count=5000)
        df = pd.DataFrame({
            "code": ["000001"],
            "close": [10.0],
            "volume": [1e6],
            "change_pct": [1.0],
        })
        result = checker.check_coverage(df, "2024-01-15")
        assert result["passed"] is False

    def test_null_rate_detects_missing_close(self):
        checker = DataQualityChecker(expected_stock_count=100)
        df = pd.DataFrame({
            "code": [f"00000{i}" for i in range(100)],
            "close": [10.0] * 99 + [None],
            "volume": [1e6] * 100,
            "change_pct": [1.0] * 100,
        })
        result = checker.check_null_rate(df, "2024-01-15")
        assert result["passed"] is False

    def test_limit_freeze_detected(self):
        checker = DataQualityChecker(expected_stock_count=100)
        df = pd.DataFrame({
            "code": [f"00000{i}" for i in range(100)],
            "close": [10.0] * 100,
            "volume": [1e6] * 80 + [0] * 20,
            "change_pct": [1.0] * 100,
        })
        result = checker.check_frozen(df, "2024-01-15")
        assert not result["passed"]
```

- [ ] **Step 2: 运行测试验证失败**

```bash
cd /Users/chenwan/Documents/quant && python -m pytest tests/test_quality.py -v
```
Expected: 全部 FAIL（模块不存在）

- [ ] **Step 3: 实现 DataQualityChecker**

```python
# data/quality.py
import pandas as pd
import numpy as np
from datetime import date


class DataQualityChecker:
    def __init__(self, expected_stock_count: int = 5000, coverage_threshold: float = 0.95):
        self.expected_stock_count = expected_stock_count
        self.coverage_threshold = coverage_threshold

    def run_all(self, df: pd.DataFrame, trade_date: date) -> list[dict]:
        results = []
        for method in [self.check_coverage, self.check_null_rate, self.check_frozen, self.check_jumps]:
            results.append(method(df, trade_date))
        return results

    def check_coverage(self, df: pd.DataFrame, trade_date: date) -> dict:
        actual = len(df)
        ratio = actual / self.expected_stock_count if self.expected_stock_count > 0 else 1.0
        passed = ratio >= self.coverage_threshold
        return {
            "check_name": "coverage",
            "trade_date": trade_date,
            "expected": str(self.expected_stock_count),
            "actual": str(actual),
            "passed": passed,
            "detail": f"覆盖率 {ratio:.2%}，阈值 {self.coverage_threshold:.0%}",
        }

    def check_null_rate(self, df: pd.DataFrame, trade_date: date) -> dict:
        null_close = df["close"].isna().sum()
        null_vol = df["volume"].isna().sum()
        bad = null_close + null_vol
        passed = bad == 0
        return {
            "check_name": "null_rate",
            "trade_date": trade_date,
            "expected": "0",
            "actual": str(bad),
            "passed": passed,
            "detail": f"close空{null_close}行, volume空{null_vol}行",
        }

    def check_frozen(self, df: pd.DataFrame, trade_date: date) -> dict:
        zero_vol = (df["volume"] == 0).sum()
        ratio = zero_vol / len(df) if len(df) > 0 else 0
        passed = ratio < 0.05
        return {
            "check_name": "frozen",
            "trade_date": trade_date,
            "expected": "<5%",
            "actual": f"{ratio:.1%}",
            "passed": passed,
            "detail": f"零成交量{zero_vol}只 ({ratio:.2%})",
        }

    def check_jumps(self, df: pd.DataFrame, trade_date: date) -> dict:
        if "change_pct" not in df.columns or df["change_pct"].dropna().empty:
            return {"check_name": "jumps", "trade_date": trade_date,
                    "expected": "no_extreme", "actual": "no_data", "passed": True, "detail": ""}
        extreme = (df["change_pct"].abs() > 20).sum()
        passed = extreme == 0
        return {
            "check_name": "jumps",
            "trade_date": trade_date,
            "expected": "0",
            "actual": str(extreme),
            "passed": passed,
            "detail": f"涨跌幅超20%的{extreme}只",
        }
```

- [ ] **Step 4: 运行测试验证通过**

```bash
cd /Users/chenwan/Documents/quant && python -m pytest tests/test_quality.py -v
```

- [ ] **Step 5: 提交**

```bash
git add data/quality.py tests/test_quality.py
git commit -m "feat: add data quality checker with coverage/null/frozen/jump checks"
```

---

### Task 3: 将质量校验集成到数据同步流程

**Files:**
- Modify: `data/sync.py`

- [ ] **Step 1: 在 sync.py 的每日同步完成后加入质量关卡**

在 `data/sync.py` 中找到每日同步结束的位置，添加校验逻辑。在文件顶部加 import，在同步模式执行完的汇总位置加校验调用：

```python
# data/sync.py 顶部添加
from data.quality import DataQualityChecker
from data.db import DB
from datetime import date

# 在 sync 主函数末尾、所有模式执行完后添加:
def _run_quality_gate(trade_date: date):
    """同步后质量校验，不通过则记录告警"""
    checker = DataQualityChecker(expected_stock_count=5000)
    df = DB.fetch_df("SELECT code, close, volume, change_pct FROM stock_daily WHERE trade_date = %s", (trade_date,))

    if df.empty:
        print(f"[QUALITY] 无数据，跳过校验")
        return False

    results = checker.run_all(df, trade_date)
    all_pass = True
    for r in results:
        DB.execute("""
            INSERT INTO data_quality_log (trade_date, check_name, expected_value, actual_value, passed, detail)
            VALUES (%(trade_date)s, %(check_name)s, %(expected)s, %(actual)s, %(passed)s, %(detail)s)
        """, r)
        status = "PASS" if r["passed"] else "FAIL"
        if not r["passed"]:
            all_pass = False
        print(f"[QUALITY] {r['check_name']}: {status} — {r['detail']}")

    if not all_pass:
        print("[QUALITY] 质量校验未通过，下游因子计算已阻断。请检查数据源后手动重跑。")
    return all_pass
```

> 注：具体插入位置需根据 sync.py 实际代码结构调整。核心逻辑是在所有同步模式完成后调用 `_run_quality_gate()`，若返回 False 则不触发后续因子计算。

- [ ] **Step 2: 提交**

```bash
git add data/sync.py
git commit -m "feat: integrate quality gate into daily sync pipeline"
```

---

### Task 4: 因子血缘与可用性跟踪

**Files:**
- Create: `data/lineage.py`
- Create: `data/availability.py`
- Modify: `factors/engine.py`

- [ ] **Step 1: 实现因子血缘记录**

```python
# data/lineage.py
import hashlib
from data.db import DB


def register_lineage(factor_name: str, source_fields: list[str],
                     computation_formula: str, upstream_factors: list[str] = None):
    formula_hash = hashlib.sha256(computation_formula.encode()).hexdigest()[:16]
    DB.execute("""
        INSERT INTO factor_lineage (factor_name, source_fields, computation_formula_hash, upstream_factors)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (factor_name) DO UPDATE SET
            source_fields = EXCLUDED.source_fields,
            computation_formula_hash = EXCLUDED.computation_formula_hash,
            upstream_factors = EXCLUDED.upstream_factors,
            last_validated_at = NOW()
    """, (factor_name, source_fields, formula_hash,
          upstream_factors or []))


def find_dirty_factors(changed_field: str) -> list[str]:
    """当某原始字段发生变化时，返回所有需要重算的因子名"""
    rows = DB.fetch_all("SELECT factor_name, source_fields FROM factor_lineage", ())
    dirty = []
    for factor_name, source_fields in rows:
        if changed_field in source_fields:
            dirty.append(factor_name)
    return dirty
```

- [ ] **Step 2: 实现因子可用性时间戳**

```python
# data/availability.py
from datetime import datetime, date
from data.db import DB


def mark_ready(trade_date: date, factor_name: str, data_source: str = "computed",
               ready_at: datetime = None):
    if ready_at is None:
        ready_at = datetime.now()
    latency_ms = int((ready_at - datetime(ready_at.year, ready_at.month, ready_at.day, 15, 0)).total_seconds() * 1000)
    DB.execute("""
        INSERT INTO factor_availability (trade_date, factor_name, data_ready_at, data_source, latency_ms)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (trade_date, factor_name) DO UPDATE SET
            data_ready_at = EXCLUDED.data_ready_at,
            latency_ms = EXCLUDED.latency_ms
    """, (trade_date, factor_name, ready_at, data_source, max(latency_ms, 0)))


def get_ready_factors(trade_date: date, before: datetime) -> list[str]:
    """返回在before时刻之前已就绪的因子列表，用于防未来函数"""
    rows = DB.fetch_all("""
        SELECT factor_name FROM factor_availability
        WHERE trade_date = %s AND data_ready_at <= %s
    """, (trade_date, before))
    return [r[0] for r in rows]
```

- [ ] **Step 3: 在 FactorEngine.compute_all 中集成**

在 `factors/engine.py` 的 `compute_all` 方法末尾添加：

```python
from data.lineage import register_lineage
from data.availability import mark_ready
from datetime import date

# 在 compute_all 返回前，对每个因子记录血缘和可用时间
for f in ALL_FACTORS:
    register_lineage(
        factor_name=f.name,
        source_fields=f.source_fields,       # 需在因子定义中添加此属性
        computation_formula=f.formula,        # 需在因子定义中添加此属性
        upstream_factors=getattr(f, 'upstream_factors', []),
    )
    mark_ready(trade_date=current_date, factor_name=f.name, data_source="computed")
```

> 注：需要在因子基类中添加 `source_fields` 和 `formula` 属性，先对自定义因子和 alpha101 因子做补充，alpha191 因子逐步补。

- [ ] **Step 4: 提交**

```bash
git add data/lineage.py data/availability.py factors/engine.py
git commit -m "feat: add factor lineage tracking and availability timestamps"
```

---

### Task 5: FastAPI 基础框架

**Files:**
- Create: `web/__init__.py`
- Create: `web/main.py`
- Create: `web/routes/__init__.py`
- Create: `web/templates/base.html`
- Create: `web/static/app.js`

- [ ] **Step 1: 创建 FastAPI 入口**

```python
# web/__init__.py
```

```python
# web/main.py
import sys
import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from web.routes import dashboard, backtest, paper, data_status, factors, api

app = FastAPI(title="Quant Monitor")

app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

app.include_router(dashboard.router)
app.include_router(backtest.router)
app.include_router(paper.router)
app.include_router(data_status.router)
app.include_router(factors.router)
app.include_router(api.router)
```

- [ ] **Step 2: 创建基础模板**

```html
<!-- web/templates/base.html -->
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}Quant Monitor{% endblock %}</title>
    <script src="https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"></script>
    <script src="https://unpkg.com/htmx.org@1.9.10"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3.13.5/dist/cdn.min.js"></script>
    <script src="/static/app.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0f1923; color: #e0e0e0; }
        .nav { display: flex; gap: 0; background: #1a2a3a; border-bottom: 1px solid #2a3a4a; padding: 0 20px; }
        .nav a { color: #8899aa; text-decoration: none; padding: 14px 20px; font-size: 14px; border-bottom: 2px solid transparent; transition: all 0.2s; }
        .nav a:hover, .nav a.active { color: #4fc3f7; border-bottom-color: #4fc3f7; }
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
        table { width: 100%; border-collapse: collapse; font-size: 13px; }
        th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid #1a2a3a; }
        th { color: #8899aa; font-weight: 600; font-size: 12px; text-transform: uppercase; }
        tr:hover { background: rgba(79,195,247,0.05); }
        .card { background: #1a2a3a; border-radius: 8px; padding: 20px; margin-bottom: 20px; border: 1px solid #2a3a4a; }
        .card h3 { font-size: 16px; margin-bottom: 12px; color: #ccc; }
        .badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }
        .badge-valid { background: #1b5e20; color: #66bb6a; }
        .badge-suspect { background: #e65100; color: #ffb74d; }
        .badge-invalid { background: #b71c1c; color: #ef5350; }
        .badge-normal { background: #1b5e20; color: #66bb6a; }
        .badge-warning { background: #e65100; color: #ffb74d; }
        .badge-critical { background: #b71c1c; color: #ef5350; }
        .up { color: #ef5350; }
        .down { color: #26a69a; }
    </style>
    {% block head %}{% endblock %}
</head>
<body>
    <nav class="nav">
        <a href="/" {% if active_page == 'dashboard' %}class="active"{% endif %}>行情看板</a>
        <a href="/backtest" {% if active_page == 'backtest' %}class="active"{% endif %}>回测对比</a>
        <a href="/paper" {% if active_page == 'paper' %}class="active"{% endif %}>模拟盘</a>
        <a href="/data" {% if active_page == 'data' %}class="active"{% endif %}>数据状态</a>
        <a href="/factors" {% if active_page == 'factors' %}class="active"{% endif %}>因子监控</a>
    </nav>
    <div class="container">
        {% block content %}{% endblock %}
    </div>
</body>
</html>
```

- [ ] **Step 3: 创建 app.js（ECharts 工具 + Alpine 组件）**

```javascript
// web/static/app.js
function initChart(elId, option) {
    const el = document.getElementById(elId);
    if (!el) return;
    const chart = echarts.init(el, 'dark');
    chart.setOption(option);
    window.addEventListener('resize', () => chart.resize());
    return chart;
}

function htmxAfterSettle(evt) {
    // HTMX 替换后重新绑定图表
    document.querySelectorAll('[data-chart]').forEach(el => {
        const id = el.id;
        const spec = JSON.parse(el.dataset.chart);
        initChart(id, spec);
    });
}
document.body.addEventListener('htmx:afterSettle', htmxAfterSettle);
```

- [ ] **Step 4: 创建占位路由**

```python
# web/routes/__init__.py
```

```python
# web/routes/dashboard.py
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
router = APIRouter(tags=["dashboard"])

@router.get("/", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    from fastapi.templating import Jinja2Templates
    import os
    templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "..", "templates"))
    return templates.TemplateResponse("dashboard.html", {"request": request, "active_page": "dashboard"})
```

```python
# web/routes/backtest.py, web/routes/paper.py, web/routes/data_status.py, web/routes/factors.py
# 结构同上，各自返回对应模板，active_page 分别设为 "backtest", "paper", "data", "factors"
```

```python
# web/routes/api.py
from fastapi import APIRouter
router = APIRouter(prefix="/api", tags=["api"])

@router.get("/ping")
async def ping():
    return {"status": "ok"}
```

- [ ] **Step 5: 创建占位页面模板**

```bash
# 为每个页面创建占位模板
for page in dashboard backtest paper data factors; do
cat > web/templates/${page}.html << 'TMPL'
{% extends "base.html" %}
{% block title %}Quant Monitor{% endblock %}
{% block content %}
<div class="card"><h3>开发中...</h3></div>
{% endblock %}
TMPL
done
mkdir -p web/templates/partials
```

- [ ] **Step 6: 启动验证**

```bash
cd /Users/chenwan/Documents/quant && pip install fastapi uvicorn jinja2 && python -m uvicorn web.main:app --host 127.0.0.1 --port 8000 --reload &
```

访问 http://127.0.0.1:8000 确认导航栏和5个页面切换正常。

- [ ] **Step 7: 提交**

```bash
git add web/ requirements.txt
git commit -m "feat: add FastAPI skeleton with 5-page navigation and base template"
```

---

### Task 6: 行情看板页面

**Files:**
- Modify: `web/routes/dashboard.py`
- Create: `web/templates/dashboard.html`
- Create: `web/templates/partials/kline.html`
- Create: `web/templates/partials/quote_table.html`
- Modify: `web/routes/api.py`

- [ ] **Step 1: 实现行情API端点**

```python
# web/routes/api.py 添加
from data.db import DB

@router.get("/api/quotes/{group}")
async def get_quotes(group: str = "default"):
    """获取自选股分组报价"""
    codes = _get_watchlist_codes(group)
    rows = DB.fetch_all("""
        SELECT code, name, close, change_pct, volume, turnover, high, low, open, preclose
        FROM stock_daily
        WHERE code = ANY(%s) AND trade_date = (SELECT MAX(trade_date) FROM stock_daily)
        ORDER BY change_pct DESC
    """, (codes,))
    return {"quotes": [dict(zip(["code","name","close","change_pct","volume","turnover","high","low","open","preclose"], r)) for r in rows]}

def _get_watchlist_codes(group: str) -> list:
    # 临时：从配置文件或环境变量读取自选股分组
    # 后续可改为从DB watchlist表读取
    WATCHLIST = {
        "default": ["000001","000002","600519","300750","002415"],
        "tech": ["300750","002415","002475","688981","300124"],
    }
    return WATCHLIST.get(group, WATCHLIST["default"])
```

- [ ] **Step 2: 实现行情看板模板**

```html
<!-- web/templates/dashboard.html -->
{% extends "base.html" %}
{% block content %}
<div x-data="{ group: 'default', selectedCode: '' }">
    <div class="card" style="display:flex; gap:12px; align-items:center;">
        <h3 style="margin:0;">行情看板</h3>
        <select x-model="group" @change="fetch(`/api/quotes/${group}`).then(r=>r.json()).then(d=>$refs.table.innerHTML=d.html)">
            <option value="default">默认自选</option>
            <option value="tech">科技</option>
        </select>
    </div>

    <div style="display:grid; grid-template-columns: 1fr 1fr; gap:20px;">
        <!-- 左侧：报价表 -->
        <div class="card" hx-get="/api/quotes/default" hx-trigger="load" hx-swap="innerHTML">
            <div hx-get="/api/quotes/default" hx-trigger="every 30s" hx-swap="outerHTML">
                {% include "partials/quote_table.html" %}
            </div>
        </div>

        <!-- 右侧：K线图 -->
        <div class="card" id="kline-panel">
            <div id="kline-chart" style="width:100%;height:500px;" data-chart='{}'></div>
        </div>
    </div>
</div>
{% endblock %}
```

```html
<!-- web/templates/partials/quote_table.html -->
<table>
    <thead><tr>
        <th>代码</th><th>名称</th><th>现价</th><th>涨跌幅</th><th>成交量</th><th>换手率</th>
    </tr></thead>
    <tbody>
    {% for q in quotes %}
    <tr style="cursor:pointer"
        hx-get="/api/kline/{{ q.code }}"
        hx-target="#kline-panel"
        hx-swap="innerHTML">
        <td>{{ q.code }}</td>
        <td>{{ q.name }}</td>
        <td>{{ "%.2f"|format(q.close) }}</td>
        <td class="{% if q.change_pct > 0 %}up{% else %}down{% endif %}">
            {{ "%+.2f%%"|format(q.change_pct) }}
        </td>
        <td>{{ "%.0f"|format(q.volume) }}</td>
        <td>{{ "%.2f%%"|format(q.turnover) if q.turnover else '-' }}</td>
    </tr>
    {% endfor %}
    </tbody>
</table>
```

```html
<!-- web/templates/partials/kline.html -->
<div id="kline-chart" style="width:100%;height:500px;"
     data-chart='{{ chart_json | tojson }}'></div>
```

- [ ] **Step 3: 添加K线图API端点**

```python
# web/routes/api.py 添加
import json

@router.get("/api/kline/{code}")
async def get_kline(code: str):
    rows = DB.fetch_all("""
        SELECT trade_date, open, high, low, close, volume
        FROM stock_daily WHERE code = %s
        ORDER BY trade_date DESC LIMIT 250
    """, (code,))
    rows = list(reversed(rows))
    dates = [str(r[0]) for r in rows]
    ohlc = [[r[1], r[2], r[3], r[4]] for r in rows]
    volumes = [r[5] for r in rows]

    option = {
        "grid": [{"left": "8%", "right": "2%", "top": "5%", "height": "65%"},
                 {"left": "8%", "right": "2%", "top": "75%", "height": "20%"}],
        "xAxis": [{"data": dates, "axisLabel": {"show": False}}, {"data": dates, "axisLabel": {"rotate": 30}}],
        "yAxis": [{"scale": True}, {"scale": True}],
        "series": [
            {"type": "candlestick", "data": ohlc, "xAxisIndex": 0, "yAxisIndex": 0,
             "itemStyle": {"color": "#ef5350", "color0": "#26a69a", "borderColor": "#ef5350", "borderColor0": "#26a69a"}},
            {"type": "bar", "data": volumes, "xAxisIndex": 1, "yAxisIndex": 1},
        ],
        "tooltip": {"trigger": "axis"},
    }
    from fastapi.templating import Jinja2Templates
    import os
    templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "..", "templates"))
    return templates.TemplateResponse("partials/kline.html", {
        "request": None, "code": code, "chart_json": option,
    })
```

- [ ] **Step 4: 提交**

```bash
git add web/
git commit -m "feat: implement dashboard page with live quotes and K-line chart"
```

---

### Task 7: 回测对比页面

**Files:**
- Modify: `web/routes/backtest.py`
- Create: `web/templates/backtest.html`
- Create: `web/templates/partials/backtest_table.html`
- Create: `web/templates/partials/equity_curve.html`

- [ ] **Step 1: 实现回测数据查询API**

```python
# web/routes/api.py 添加
@router.get("/api/backtest-list")
async def list_backtests(strategy: str = None, quality: str = None):
    where = ["1=1"]
    params = []
    if strategy:
        where.append("sc.name = %s")
        params.append(strategy)
    if quality:
        where.append("br.quality = %s")
        params.append(quality)
    rows = DB.fetch_all(f"""
        SELECT sc.name, sv.version, br.start_date, br.end_date, br.quality,
               br.quality_flags, br.metrics_json, br.id
        FROM backtest_results br
        JOIN strategy_versions sv ON br.version_id = sv.id
        JOIN strategy_configs sc ON sv.strategy_id = sc.id
        WHERE {' AND '.join(where)}
        ORDER BY br.created_at DESC
    """, params)
    results = []
    for r in rows:
        metrics = r[6] if isinstance(r[6], dict) else json.loads(r[6])
        results.append({
            "name": r[0], "version": r[1], "start": str(r[2]), "end": str(r[3]),
            "quality": r[4], "flags": r[5] or [],
            "annual_return": metrics.get("annual_return", 0),
            "max_drawdown": metrics.get("max_drawdown", 0),
            "sharpe": metrics.get("sharpe", 0),
            "win_rate": metrics.get("win_rate", 0),
            "profit_factor": metrics.get("profit_factor", 0),
            "id": r[7],
        })
    return {"backtests": results}


@router.get("/api/backtest-equity/{backtest_id}")
async def get_equity_curve(backtest_id: int):
    row = DB.fetch_one(
        "SELECT equity_curve_json FROM backtest_results WHERE id = %s", (backtest_id,))
    if not row:
        return {"error": "not found"}
    curve = row[0] if isinstance(row[0], dict) else json.loads(row[0])
    return {"equity": curve}
```

- [ ] **Step 2: 实现回测对比页面模板**

```html
<!-- web/templates/backtest.html -->
{% extends "base.html" %}
{% block content %}
<div x-data="{ selected: [] }" class="card">
    <h3>回测对比</h3>
    <div style="display:flex; gap:12px; margin-bottom:16px;">
        <input placeholder="策略名称" hx-get="/api/backtest-list" hx-trigger="keyup changed delay:500ms"
               hx-target="#bt-table" hx-swap="innerHTML" name="strategy">
        <select name="quality" hx-get="/api/backtest-list" hx-trigger="change"
                hx-target="#bt-table" hx-swap="innerHTML">
            <option value="">全部质量</option>
            <option value="valid">有效</option>
            <option value="suspect">可疑</option>
            <option value="invalid">无效</option>
        </select>
    </div>

    <div id="bt-table" hx-get="/api/backtest-list" hx-trigger="load"></div>

    <div id="equity-compare" style="width:100%;height:400px;margin-top:20px;"></div>
</div>
{% endblock %}
```

```html
<!-- web/templates/partials/backtest_table.html -->
<table>
    <thead><tr>
        <th>策略</th><th>版本</th><th>区间</th><th>年化收益</th><th>最大回撤</th><th>夏普</th><th>胜率</th><th>盈亏比</th><th>质量</th><th>操作</th>
    </tr></thead>
    <tbody>
    {% for bt in backtests %}
    <tr>
        <td>{{ bt.name }}</td><td>{{ bt.version }}</td>
        <td>{{ bt.start }} ~ {{ bt.end }}</td>
        <td class="{% if bt.annual_return > 0 %}up{% else %}down{% endif %}">{{ "%.1f%%"|format(bt.annual_return*100) }}</td>
        <td>{{ "%.1f%%"|format(bt.max_drawdown*100) }}</td>
        <td>{{ "%.2f"|format(bt.sharpe) }}</td>
        <td>{{ "%.0f%%"|format(bt.win_rate*100) }}</td>
        <td>{{ "%.2f"|format(bt.profit_factor) }}</td>
        <td><span class="badge badge-{{ bt.quality }}">{{ bt.quality }}</span></td>
        <td><button hx-get="/api/backtest-equity/{{ bt.id }}" hx-target="#equity-compare" hx-swap="innerHTML">查看</button></td>
    </tr>
    {% endfor %}
    </tbody>
</table>
```

- [ ] **Step 3: 提交**

```bash
git add web/
git commit -m "feat: implement backtest comparison page with filtering and equity curve"
```

---

### Task 8: 模拟盘监控页面

**Files:**
- Modify: `web/routes/paper.py`
- Create: `web/templates/paper.html`
- Create: `web/templates/partials/paper_positions.html`

- [ ] **Step 1: 实现模拟盘数据查询API**

```python
# web/routes/api.py 添加
@router.get("/api/paper-runs")
async def list_paper_runs():
    rows = DB.fetch_all("""
        SELECT pr.id, sc.name, sv.version, pr.start_date, pr.end_date,
               pr.initial_capital, pr.status
        FROM paper_runs pr
        JOIN strategy_configs sc ON pr.strategy_id = sc.id
        JOIN strategy_versions sv ON pr.version_id = sv.id
        ORDER BY pr.created_at DESC
    """)
    return {"runs": [dict(zip(["id","strategy","version","start","end","capital","status"], r)) for r in rows]}


@router.get("/api/paper-run/{run_id}")
async def get_paper_run_detail(run_id: int):
    positions = DB.fetch_all("""
        SELECT stock_code, entry_date, entry_price, exit_date, exit_price,
               quantity, pnl, pnl_pct
        FROM paper_positions WHERE run_id = %s ORDER BY entry_date DESC
    """, (run_id,))
    signals = DB.fetch_all("""
        SELECT signal_date, stock_code, predicted_score, rank
        FROM paper_signals WHERE run_id = %s ORDER BY signal_date DESC LIMIT 50
    """, (run_id,))
    return {
        "positions": [dict(zip(["code","entry_date","entry","exit_date","exit","qty","pnl","pnl_pct"], p)) for p in positions],
        "signals": [dict(zip(["date","code","score","rank"], s)) for s in signals],
        "total_pnl": sum(p[6] or 0 for p in positions),
        "win_count": sum(1 for p in positions if (p[7] or 0) > 0),
        "total_count": len([p for p in positions if p[6] is not None]),
    }
```

- [ ] **Step 2: 实现模拟盘监控页面模板**

```html
<!-- web/templates/paper.html -->
{% extends "base.html" %}
{% block content %}
<div class="card">
    <h3>模拟盘监控</h3>
    <div hx-get="/api/paper-runs" hx-trigger="load" hx-swap="innerHTML">
        <p>加载中...</p>
    </div>
</div>
<div id="paper-detail" class="card" style="display:none;"></div>
{% endblock %}
```

提交合并在 Task 8 末尾。

- [ ] **Step 3: 提交**

```bash
git add web/
git commit -m "feat: implement paper trading monitor page"
```

---

### Task 9: 数据状态与因子监控页面

**Files:**
- Modify: `web/routes/data_status.py`
- Modify: `web/routes/factors.py`
- Create: `web/templates/data.html`
- Create: `web/templates/factors.html`
- Create: `web/templates/partials/sync_log.html`

- [ ] **Step 1: 数据状态页API + 因子监控页API**

```python
# web/routes/api.py 添加

@router.get("/api/data-status")
async def get_data_status():
    tables = ["stock_daily", "stock_basic", "index_daily", "financial", "shareholder",
              "daily_extra"]  # 现有主要表
    status = []
    for t in tables:
        row = DB.fetch_one(f"SELECT MAX(trade_date), COUNT(*) FROM {t}" if t != "stock_basic"
                           else f"SELECT NULL, COUNT(*) FROM {t}")
        status.append({"table": t, "latest_date": str(row[0]) if row and row[0] else "无数据", "rows": row[1] if row else 0})
    quality = DB.fetch_all("SELECT * FROM data_quality_log ORDER BY trade_date DESC, check_name LIMIT 20")
    return {"tables": status, "quality": [dict(zip(["id","trade_date","check","expected","actual","passed","detail","created_at"], q)) for q in quality]}


@router.get("/api/factor-ic")
async def get_factor_ic():
    """返回最近一期IC序列"""
    from factors.monitor import get_latest_ic
    return get_latest_ic()
```

- [ ] **Step 2: 数据状态页模板**

```html
<!-- web/templates/data.html -->
{% extends "base.html" %}
{% block content %}
<div class="card">
    <h3>数据状态</h3>
    <button hx-post="/api/sync/trigger" hx-target="#sync-log" hx-swap="innerHTML">手动触发同步</button>
    <div id="sync-log"></div>
</div>
<div class="card" hx-get="/api/data-status" hx-trigger="load" hx-swap="innerHTML">
    <p>加载中...</p>
</div>
{% endblock %}
```

```html
<!-- web/templates/factors.html -->
{% extends "base.html" %}
{% block content %}
<div class="card">
    <h3>因子IC监控</h3>
    <div id="ic-chart" style="width:100%;height:400px;" hx-get="/api/factor-ic" hx-trigger="load"></div>
</div>
{% endblock %}
```

- [ ] **Step 3: 添加手动同步触发端点**

```python
# web/routes/api.py
@router.post("/api/sync/trigger")
async def trigger_sync():
    import subprocess, sys
    subprocess.Popen([sys.executable, "-m", "data.sync"],
                     cwd=os.path.join(os.path.dirname(__file__), "..", ".."))
    return HTMLResponse("<p>同步已触发，正在后台运行...</p>")
```

- [ ] **Step 4: 提交**

```bash
git add web/
git commit -m "feat: implement data status and factor monitoring pages"
```

---

### Task 10: 回测防过拟合验证模块

**Files:**
- Create: `scripts/overfit_check.py`
- Test: `tests/test_overfit_check.py`

- [ ] **Step 1: 编写测试**

```python
# tests/test_overfit_check.py
import pytest
import numpy as np
from scripts.overfit_check import OverfitChecker


def make_metrics(**overrides):
    defaults = {
        "train_sharpe": 1.5, "val_sharpe": 1.2, "test_sharpe": 0.8,
        "annual_return": 0.35, "max_drawdown": 0.18,
        "n_trades": 80, "n_params": 10,
    }
    defaults.update(overrides)
    return defaults


class TestOverfitCheck:
    def test_valid_strategy_passes(self):
        checker = OverfitChecker()
        result = checker.check(make_metrics(), regime_count=2, sensitivity_stable=True)
        assert result["quality"] == "valid"
        assert len(result["flags"]) == 0

    def test_low_sample_ratio_fails(self):
        checker = OverfitChecker()
        m = make_metrics(train_sharpe=2.5, val_sharpe=0.3)
        result = checker.check(m, regime_count=2, sensitivity_stable=True)
        assert result["quality"] == "suspect"
        assert any("样本外一致性" in f for f in result["flags"])

    def test_few_trades_triggers_warning(self):
        checker = OverfitChecker()
        result = checker.check(make_metrics(n_trades=15), regime_count=2, sensitivity_stable=True)
        assert result["quality"] == "suspect"
        assert any("交易次数" in f for f in result["flags"])

    def test_single_regime_flagged(self):
        checker = OverfitChecker()
        result = checker.check(make_metrics(), regime_count=1, sensitivity_stable=True)
        assert result["quality"] == "suspect"
        assert any("时段" in f for f in result["flags"])

    def test_param_sensitivity_flagged(self):
        checker = OverfitChecker()
        result = checker.check(make_metrics(), regime_count=2, sensitivity_stable=False)
        assert result["quality"] == "suspect"
        assert any("参数敏感" in f for f in result["flags"])

    def test_adjusted_sharpe_in_metrics(self):
        checker = OverfitChecker()
        result = checker.check(make_metrics(), regime_count=2, sensitivity_stable=True)
        assert "adjusted_sharpe" in result
        assert result["adjusted_sharpe"] < result.get("raw_sharpe", 999)
```

- [ ] **Step 2: 实现 OverfitChecker**

```python
# scripts/overfit_check.py
import numpy as np


class OverfitChecker:
    def __init__(self, min_trades: int = 30, min_regimes: int = 2,
                 oos_ratio_threshold: float = 0.3):
        self.min_trades = min_trades
        self.min_regimes = min_regimes
        self.oos_ratio_threshold = oos_ratio_threshold

    def check(self, metrics: dict, regime_count: int = 0,
              sensitivity_stable: bool = True) -> dict:
        flags = []
        train_sr = metrics.get("train_sharpe", 0)
        val_sr = metrics.get("val_sharpe", 0)
        test_sr = metrics.get("test_sharpe", 0)
        n_trades = metrics.get("n_trades", 0)
        n_params = metrics.get("n_params", 1)

        if val_sr > 0 and train_sr > 0 and (val_sr / train_sr) < self.oos_ratio_threshold:
            flags.append(f"样本外一致性严重不足: val/train夏普比={val_sr/train_sr:.2f}")
        if n_trades < self.min_trades:
            flags.append(f"交易次数不足: {n_trades}笔 < {self.min_trades}笔")
        if regime_count < self.min_regimes:
            flags.append(f"覆盖市场时段不足: {regime_count}种 < {self.min_regimes}种")
        if not sensitivity_stable:
            flags.append("参数敏感性过高: ±10%→结果波动>20%")

        adjusted_sharpe = test_sr * np.sqrt(n_trades / (n_params + 1))

        quality = "valid"
        if len(flags) >= 2 or any("严重" in f for f in flags):
            quality = "suspect"
        if test_sr < 0:
            quality = "invalid"

        return {
            "quality": quality,
            "flags": flags,
            "adjusted_sharpe": round(adjusted_sharpe, 4),
        }
```

- [ ] **Step 3: 运行测试**

```bash
cd /Users/chenwan/Documents/quant && python -m pytest tests/test_overfit_check.py -v
```

- [ ] **Step 4: 提交**

```bash
git add scripts/overfit_check.py tests/test_overfit_check.py
git commit -m "feat: add backtest overfitting prevention checker"
```

---

### Task 11: 防过拟合检查集成到回测流程

**Files:**
- Modify: `scripts/run_ml_backtest.py`

- [ ] **Step 1: 在回测脚本中集成防过拟合检查 + 结果入库**

在 `scripts/run_ml_backtest.py` 的结果汇总部分添加：

```python
from scripts.overfit_check import OverfitChecker
from data.db import DB
import json


def _validate_and_save(version_id: int, metrics: dict, equity_curve: dict,
                       daily_returns: dict, regime_count: int,
                       sensitivity_stable: bool = True):
    checker = OverfitChecker(min_trades=30, min_regimes=2)
    result = checker.check(metrics, regime_count=regime_count,
                           sensitivity_stable=sensitivity_stable)

    DB.execute("""
        INSERT INTO backtest_results
            (version_id, start_date, end_date, quality, quality_flags,
             metrics_json, equity_curve_json, daily_returns_json)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        version_id,
        metrics.get("start_date"),
        metrics.get("end_date"),
        result["quality"],
        result["flags"],
        json.dumps({**metrics, "adjusted_sharpe": result["adjusted_sharpe"]}, default=str),
        json.dumps(equity_curve, default=str),
        json.dumps(daily_returns, default=str),
    ))

    print(f"\n回测结果已写入 backtest_results(id={version_id}, quality={result['quality']})")
    if result["flags"]:
        print("警告标记:")
        for f in result["flags"]:
            print(f"  ⚠ {f}")
```

> 注：实际集成时需根据 run_ml_backtest.py 的现有输出结构调整插入位置。

- [ ] **Step 2: 提交**

```bash
git add scripts/run_ml_backtest.py
git commit -m "feat: integrate overfitting check into ML backtest pipeline"
```

---

### Task 12: 信号归因分析模块

**Files:**
- Create: `scripts/attribution.py`
- Test: `tests/test_attribution.py`

- [ ] **Step 1: 编写测试**

```python
# tests/test_attribution.py
import pytest
import numpy as np
from scripts.attribution import compute_factor_contribution


class TestFactorContribution:
    def test_positive_contribution_for_winning_signal(self):
        factor_values = {"momentum": 0.05, "volatility": -0.02, "liquidity": 0.01}
        # 模拟：该信号选股后涨了3%
        result = compute_factor_contribution(factor_values, pnl_pct=3.0)
        assert result["momentum"] > 0
        assert isinstance(result["volatility"], float)

    def test_negative_pnl_gives_negative_sign(self):
        factor_values = {"momentum": 0.05, "volatility": -0.02}
        result = compute_factor_contribution(factor_values, pnl_pct=-2.0)
        # 因子贡献符号应与pnl同向
        for v in result.values():
            assert v <= 0

    def test_empty_factors_returns_empty(self):
        result = compute_factor_contribution({}, pnl_pct=5.0)
        assert result == {}
```

- [ ] **Step 2: 实现归因分析**

```python
# scripts/attribution.py
import numpy as np
from data.db import DB
from datetime import date


def compute_factor_contribution(factor_values: dict, pnl_pct: float) -> dict:
    """按因子值的符号与大小分解收益归因"""
    if not factor_values:
        return {}
    total_abs = sum(abs(v) for v in factor_values.values())
    if total_abs == 0:
        return {k: 0.0 for k in factor_values}
    contrib = {}
    for factor, value in factor_values.items():
        contrib[factor] = round(pnl_pct * abs(value) / total_abs * (1 if value > 0 else -1), 6)
    return contrib


def run_attribution(run_id: int, eval_date: date):
    """对模拟盘某日信号做归因分析"""
    signals = DB.fetch_all("""
        SELECT ps.id, ps.stock_code, ps.signal_date
        FROM paper_signals ps
        WHERE ps.run_id = %s AND ps.signal_date <= %s
    """, (run_id, eval_date))

    for signal_id, stock_code, signal_date in signals:
        # 获取该信号选股后N日收益（简化：查paper_positions）
        pos = DB.fetch_one("""
            SELECT pnl_pct FROM paper_positions
            WHERE signal_id = %s AND pnl_pct IS NOT NULL
        """, (signal_id,))
        if not pos:
            continue
        pnl_pct = pos[0]

        # 获取该信号的因子值
        factor_rows = DB.fetch_all(
            "SELECT factor_name, value FROM signal_factors WHERE signal_id = %s", (signal_id,))
        factor_values = {r[0]: r[1] for r in factor_rows}

        contrib = compute_factor_contribution(factor_values, pnl_pct)
        days_held = (eval_date - signal_date).days

        DB.execute("""
            INSERT INTO signal_attribution (signal_id, eval_date, days_held, pnl_pct, factor_contrib_json)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING
        """, (signal_id, eval_date, days_held, pnl_pct,
              __import__('json').dumps(contrib)))
```

- [ ] **Step 3: 运行测试**

```bash
cd /Users/chenwan/Documents/quant && python -m pytest tests/test_attribution.py -v
```

- [ ] **Step 4: 提交**

```bash
git add scripts/attribution.py tests/test_attribution.py
git commit -m "feat: add signal attribution analysis module"
```

---

### Task 13: 自动权重调整与策略健康监控

**Files:**
- Create: `scripts/auto_adjust.py`
- Create: `scripts/health_check.py`

- [ ] **Step 1: 实现自动权重调整**

```python
# scripts/auto_adjust.py
import numpy as np
from scipy import stats
from data.db import DB
from datetime import date


def check_and_adjust(strategy_id: int, lookback_periods: int = 10,
                     decay_threshold: float = 0.3, min_consecutive: int = 3,
                     pvalue_threshold: float = 0.05):
    """检查因子IC衰减，满足条件则自动降权"""
    rows = DB.fetch_all("""
        SELECT factor_name, weight, effective_date
        FROM factor_weights_history
        WHERE strategy_id = %s
        ORDER BY factor_name, effective_date DESC
    """, (strategy_id,))

    if not rows:
        print(f"策略{strategy_id}无权重历史，跳过")
        return

    from factors.monitor import get_factor_ic_series

    adjustments = []
    seen = set()
    for factor_name, current_weight, _ in rows:
        if factor_name in seen:
            continue
        seen.add(factor_name)

        ic_series = get_factor_ic_series(factor_name, lookback_periods)
        if len(ic_series) < min_consecutive:
            continue

        recent = ic_series[-min_consecutive:]
        if all(x > 0 for x in ic_series[-min_consecutive:]):
            continue

        decay = (ic_series[0] - ic_series[-1]) / abs(ic_series[0]) if ic_series[0] != 0 else 0
        if decay < decay_threshold:
            continue

        t_stat, p_value = stats.ttest_1samp(recent, 0)
        p_value = p_value / 2
        if p_value >= pvalue_threshold:
            print(f"{factor_name}: IC衰减{decay:.0%}但不显著(p={p_value:.3f})，跳过")
            continue

        new_weight = round(current_weight * 0.8, 6)
        confidence = 0.99 if p_value < 0.01 else 0.95
        DB.execute("""
            INSERT INTO weight_adjustments (strategy_id, factor_name, old_weight, new_weight, confidence_level, source, reason)
            VALUES (%s, %s, %s, %s, %s, 'auto', %s)
        """, (strategy_id, factor_name, current_weight, new_weight, confidence,
              f"IC衰减{decay:.1%}, 连续{min_consecutive}期, p={p_value:.3f}"))
        DB.execute("""
            INSERT INTO factor_weights_history (strategy_id, factor_name, weight, effective_date, source, reason)
            VALUES (%s, %s, %s, %s, 'auto', %s)
        """, (strategy_id, factor_name, new_weight, date.today(),
              f"自降权: IC衰减{decay:.1%}"))
        adjustments.append({"factor": factor_name, "old": current_weight, "new": new_weight})

    print(f"自动调参完成: {len(adjustments)}项调整")
    for a in adjustments:
        print(f"  {a['factor']}: {a['old']:.4f} → {a['new']:.4f}")
    return adjustments
```

- [ ] **Step 2: 实现策略健康监控**

```python
# scripts/health_check.py
from data.db import DB
from datetime import date, timedelta


def check_strategy_health(strategy_id: int):
    """每日检查策略健康状况并更新 strategy_health 表"""
    today = date.today()

    recent = DB.fetch_one("""
        SELECT AVG(pnl_pct), BOOL_AND(pnl > 0)
        FROM paper_positions
        WHERE run_id IN (SELECT id FROM paper_runs WHERE strategy_id = %s)
          AND entry_date >= %s
    """, (strategy_id, today - timedelta(days=7)))

    avg_pnl = recent[0] or 0 if recent else 0

    dd_row = DB.fetch_one("""
        WITH daily_pnl AS (
            SELECT entry_date, SUM(COALESCE(pnl, 0)) AS day_pnl
            FROM paper_positions
            WHERE run_id IN (SELECT id FROM paper_runs WHERE strategy_id = %s)
              AND entry_date >= %s
            GROUP BY entry_date
        ),
        cumulative AS (
            SELECT entry_date, SUM(day_pnl) OVER (ORDER BY entry_date) AS cum_pnl
            FROM daily_pnl
        ),
        running_max AS (
            SELECT entry_date, cum_pnl,
                   MAX(cum_pnl) OVER (ORDER BY entry_date) AS peak
            FROM cumulative
        )
        SELECT MAX(peak - cum_pnl) / NULLIF(AVG(cum_pnl + %s), 0) FROM running_max
    """, (strategy_id, today - timedelta(days=7), recent_row[2] if recent_row else 100000.0))
    dd_val = dd_row[0] if dd_row else None

    ic = _get_7day_ic(strategy_id)

    if ic is not None and ic > 0 and (dd_val < 0.10 if dd_val else True):
        status, action = "normal", "none"
    elif ic is not None and ic < 0 or (dd_val and dd_val > 0.15):
        status, action = "warning", "pause"
    elif ic is not None and ic < 0 and (dd_val and dd_val > 0.25):
        status, action = "critical", "switch_backup"
    else:
        status, action = "normal", "none"

    DB.execute("""
        INSERT INTO strategy_health (strategy_id, date, overall_ic, max_drawdown_7d, status, action_required)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (strategy_id, date) DO UPDATE SET
            overall_ic = EXCLUDED.overall_ic,
            max_drawdown_7d = EXCLUDED.max_drawdown_7d,
            status = EXCLUDED.status,
            action_required = EXCLUDED.action_required
    """, (strategy_id, today, ic, dd_val, status, action))

    if status == "critical":
        _switch_to_backup(strategy_id)

    return {"status": status, "action": action, "ic": ic}


def _get_7day_ic(strategy_id: int) -> float:
    row = DB.fetch_one("SELECT overall_ic FROM strategy_health WHERE strategy_id = %s ORDER BY date DESC LIMIT 1",
                       (strategy_id,))
    return row[0] if row else None


def _switch_to_backup(strategy_id: int):
    print(f"策略{strategy_id}触发熔断，切换至等权组合")
    DB.execute("UPDATE paper_runs SET status = 'paused' WHERE strategy_id = %s AND status = 'running'",
               (strategy_id,))
```

- [ ] **Step 3: 提交**

```bash
git add scripts/auto_adjust.py scripts/health_check.py
git commit -m "feat: add auto weight adjustment and strategy health monitoring"
```

---

### Task 14: 策略指令队列执行器

**Files:**
- Create: `scripts/command_worker.py`

- [ ] **Step 1: 实现指令执行器**

```python
# scripts/command_worker.py
from data.db import DB
from datetime import date
import json


def process_pending_commands():
    """消费 strategy_commands 队列中未执行的指令"""
    rows = DB.fetch_all("""
        SELECT id, strategy_id, command_type, payload_json, requested_by
        FROM strategy_commands
        WHERE executed_at IS NULL
        ORDER BY requested_at
    """)

    for cmd_id, strategy_id, cmd_type, payload, requested_by in rows:
        payload = payload if isinstance(payload, dict) else json.loads(payload)
        try:
            if cmd_type == "adjust_weight":
                _exec_adjust_weight(strategy_id, payload)
            elif cmd_type == "pause":
                DB.execute("UPDATE paper_runs SET status = 'paused' WHERE strategy_id = %s AND status = 'running'",
                           (strategy_id,))
            elif cmd_type == "resume":
                DB.execute("UPDATE paper_runs SET status = 'running' WHERE strategy_id = %s AND status = 'paused'",
                           (strategy_id,))
            elif cmd_type == "rollback":
                _exec_rollback(strategy_id, payload)
            elif cmd_type == "retrain":
                _exec_retrain(strategy_id)
            DB.execute("UPDATE strategy_commands SET executed_at = NOW(), execution_result = 'ok' WHERE id = %s",
                       (cmd_id,))
        except Exception as e:
            DB.execute("UPDATE strategy_commands SET executed_at = NOW(), execution_result = %s WHERE id = %s",
                       (str(e), cmd_id))
            print(f"指令{cmd_id}执行失败: {e}")


def _exec_adjust_weight(strategy_id: int, payload: dict):
    factor_name = payload["factor_name"]
    old_weight = float(payload["old_weight"])
    new_weight = float(payload["new_weight"])
    reason = payload.get("reason", "manual")
    DB.execute("""
        INSERT INTO weight_adjustments (strategy_id, factor_name, old_weight, new_weight, source, reason)
        VALUES (%s, %s, %s, %s, 'manual', %s)
    """, (strategy_id, factor_name, old_weight, new_weight, reason))
    DB.execute("""
        INSERT INTO factor_weights_history (strategy_id, factor_name, weight, effective_date, source, reason)
        VALUES (%s, %s, %s, %s, 'manual', %s)
    """, (strategy_id, factor_name, new_weight, date.today(), reason))


def _exec_rollback(strategy_id: int, payload: dict):
    target_version = payload.get("target_version")
    DB.execute("""
        UPDATE paper_runs SET version_id = (SELECT id FROM strategy_versions WHERE strategy_id = %s AND version = %s)
        WHERE strategy_id = %s AND status = 'paused'
    """, (strategy_id, target_version, strategy_id))


def _exec_retrain(strategy_id: int):
    import subprocess, sys
    subprocess.run([sys.executable, "scripts/run_ml_backtest.py", "--strategy-id", str(strategy_id)])
```

- [ ] **Step 2: 提交**

```bash
git add scripts/command_worker.py
git commit -m "feat: add strategy command queue worker"
```

---

### Task 15: 模拟盘引擎集成 signal_factors 写入

**Files:**
- Modify: `portfolio/paper_engine.py`

- [ ] **Step 1: 在 PaperEngine 选股逻辑中添加 signal_factors 写入**

在 `portfolio/paper_engine.py` 中生成信号并写入 `paper_signals` 后添加：

```python
# 在写入 paper_signals 之后
for signal_id, factor_values in signal_factors_map.items():
    for factor_name, value in factor_values.items():
        DB.execute("""
            INSERT INTO signal_factors (signal_id, factor_name, value)
            VALUES (%s, %s, %s)
            ON CONFLICT (signal_id, factor_name) DO NOTHING
        """, (signal_id, factor_name, float(value)))
```

> 注：`signal_factors_map` 需在选股循环中构建：`{signal_id: {"momentum": 0.03, ...}, ...}`

- [ ] **Step 2: 提交**

```bash
git add portfolio/paper_engine.py
git commit -m "feat: write factor values per-signal for attribution analysis"
```

---

### Task 16: launchd 定时任务配置

**Files:**
- Create: `~/Library/LaunchAgents/com.quant.sync.plist`

- [ ] **Step 1: 创建 plist 配置**

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.quant.sync</string>
    <key>ProgramArguments</key>
    <array>
        <string>/Users/chenwan/Documents/quant/.venv/bin/python</string>
        <string>-m</string>
        <string>data.sync</string>
    </array>
    <key>WorkingDirectory</key>
    <string>/Users/chenwan/Documents/quant</string>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>15</integer>
        <key>Minute</key>
        <integer>30</integer>
    </dict>
    <key>Weekday</key>
    <integer>1-5</integer>
    <key>StandardOutPath</key>
    <string>/Users/chenwan/Documents/quant/output/sync.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/chenwan/Documents/quant/output/sync.err</string>
</dict>
</plist>
```

- [ ] **Step 2: 加载到 launchd**

```bash
cp ~/Library/LaunchAgents/com.quant.sync.plist ~/Library/LaunchAgents/com.quant.sync.plist.bak 2>/dev/null
# 实际部署时:
# launchctl load ~/Library/LaunchAgents/com.quant.sync.plist
```

- [ ] **Step 3: 提交（plist 放入项目目录管理）**

```bash
mkdir -p /Users/chenwan/Documents/quant/config
cp ~/Library/LaunchAgents/com.quant.sync.plist /Users/chenwan/Documents/quant/config/com.quant.sync.plist
git add config/com.quant.sync.plist
git commit -m "feat: add launchd plist for automated daily data sync"
```

---

### Task 17: 清理旧 Streamlit 代码并删除废弃文件

**Files:**
- Delete: `app/` (entire directory)

- [ ] **Step 1: 删除 app 目录**

```bash
cd /Users/chenwan/Documents/quant
rm -rf app/
```

- [ ] **Step 2: 确认 Web 端无残留引用**

```bash
cd /Users/chenwan/Documents/quant
grep -r "from app\." --include="*.py" . 2>/dev/null || echo "无残留引用"
grep -r "import streamlit" --include="*.py" . 2>/dev/null || echo "无streamlit引用"
```

- [ ] **Step 3: 更新 requirements.txt**

从 `requirements.txt` 中删除 `streamlit`，添加 `fastapi`, `uvicorn`, `jinja2`。

- [ ] **Step 4: 提交**

```bash
git add -A
git commit -m "refactor: remove Streamlit app, replace with FastAPI web layer"
```

---

### Task 18: 更新 README 和文档

**Files:**
- Modify: `README.md`
- Modify: `docs/project-learning-guide.md`

- [ ] **Step 1: 更新 README.md**

更新以下章节：
- 架构图：用新的三层架构（Web监控层 → PostgreSQL → 后台引擎）
- 目录结构：添加 `web/`，删除 `app/`，添加新增脚本
- 核心工作流：从7步（策略编辑器 → 回测）改为5步（因子计算 → 训练 → 回测(防过拟合) → 模拟盘 → 归因反馈）
- 数据流：加入质量校验关卡
- Web 页面：从9页改为5页

- [ ] **Step 2: 更新 docs/project-learning-guide.md**

更新系统架构章节，删除 Streamlit 和策略编辑器的说明，添加 FastAPI + HTMX 技术栈介绍。

- [ ] **Step 3: 提交**

```bash
git add README.md docs/project-learning-guide.md
git commit -m "docs: update README and learning guide for new architecture"
```

---

### Task 19: 端到端验证

- [ ] **Step 1: 启动 FastAPI 服务**

```bash
cd /Users/chenwan/Documents/quant && python -m uvicorn web.main:app --host 127.0.0.1 --port 8000 &
```

- [ ] **Step 2: 验证所有页面可访问**

```bash
curl -s http://127.0.0.1:8000/ | head -c 200
curl -s http://127.0.0.1:8000/backtest | head -c 200
curl -s http://127.0.0.1:8000/paper | head -c 200
curl -s http://127.0.0.1:8000/data | head -c 200
curl -s http://127.0.0.1:8000/factors | head -c 200
```

- [ ] **Step 3: 验证 API 端点**

```bash
curl -s http://127.0.0.1:8000/api/ping
curl -s http://127.0.0.1:8000/api/data-status
```

- [ ] **Step 4: 运行全部测试**

```bash
cd /Users/chenwan/Documents/quant && python -m pytest tests/ -v
```

- [ ] **Step 5: 验证数据库新表**

```bash
cd /Users/chenwan/Documents/quant && python -c "
from data.db import create_tables, DB
create_tables()
tables = DB.fetch_all(\"SELECT table_name FROM information_schema.tables WHERE table_schema='public'\")
print(f'总表数: {len(tables)}')
for t in tables:
    print(f'  {t[0]}')
"
```

- [ ] **Step 6: 提交最终状态**

```bash
git add -A
git status
git commit -m "chore: final verification after architecture refactoring"
```
