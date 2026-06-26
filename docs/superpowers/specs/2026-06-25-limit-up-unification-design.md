# 涨停判断统一 & Bug 修复 — 设计规格

> 日期: 2026-06-25 | 状态: 待审批

---

## 一、目标

1. 消灭 `ret >= _get_limit(code) * X` 模式（4 个 buggy 文件）
2. 涨停判断唯一入口：`TradingConfig.is_at_limit_up(close, prev_close, code, tolerance)`
3. 附带修复：熔断对齐、资金分配、死代码清理、板别 bug

---

## 二、接口定义

### 唯一入口

```python
# config/settings.py — TradingConfig（已有，需加 tolerance 参数）

@staticmethod
def is_at_limit_up(close: float, prev_close: float, code: str, tolerance: float = 1.0) -> bool:
    """涨停判断。tolerance=0.98 表示 9.7%+ 近涨停。"""
    limit_price = TradingConfig.calc_limit_price(prev_close, code, is_up=True)
    return close >= limit_price * tolerance

@staticmethod
def get_limit_multiplier(code: str) -> float:
    """板别感知涨跌停乘数。"""
```

### 禁止模式

- `_LIMIT_MULT = {...}` 字典自建
- `ret >= _get_limit(code) * X`
- `ret >= multiplier * factor`
- 硬编码 `1.09899` / `1.19899` / `1.29899`

---

## 三、文件变更

### P0 — ret-vs-multiplier bug

| 文件 | 变更 |
|------|------|
| `config/settings.py` | `is_at_limit_up` 加 `tolerance` 参数（默认 1.0） |
| `scripts/screen_bull.py:112` | 删 `_LIMIT_MULT` + `_get_limit`；`ret >= mult*0.98` → `TradingConfig.is_at_limit_up(close, prev_close, code, 0.98)` |
| `scripts/scan_intraday.py:107` | 同上；`ret% >= mult*98` → `TradingConfig.is_at_limit_up(price, prev_close, code, 0.98)` |
| `scripts/run_arsenal.py:73,135,141` | 同上（3处） |
| `scripts/export_limit_up_history.py:77` | 同上 |
| `factors/limit_up.py:542,590` | `_get_limit()` → `_get_multiplier()`（该文件不定义 `_get_limit`，当前是 NameError） |

### P0 — `_is_limit_up` 板别 bug

| 文件 | 变更 |
|------|------|
| `factors/limit_up.py:71` | `code = df["code"].iloc[0]` → 逐行 `apply(_get_multiplier)` |

当前所有调用方都通过 FactorEngine（逐只股票 `groupby`），不会触发，但防御性修复。

### P1 — 资金分配 + 熔断

| 文件 | 变更 |
|------|------|
| `scripts/bt_label_ocr.py:365` | `args.cash / args.top_n` → `cash / max(available, 1)` |
| `scripts/bt_small_cap.py:120-130` | 加 `frozen_days` 计数器，60 天解冻，对齐 bt_yaogu.py/bt_label_ocr.py |

### P2 — 死代码

| 文件 | 变更 |
|------|------|
| `scripts/run_daily_signals.py:136` | `_wait_until_morning_close` 补 `time.sleep(wait_seconds + 5)` |
| `scripts/run_daily_signals.py:191` | 删 `return` 后的 `time.sleep(wait_seconds + 5)` |

---

## 四、CLAUDE.md 新增铁律

```
| 19 | **涨停判断唯一入口** `TradingConfig.is_at_limit_up(close, prev_close, code)` | 禁止手写 ret>=mult、自建 _LIMIT_MULT |
```

---

## 五、不改的

- 其他 7 个定义 `_LIMIT_MULT` 但**使用正确**的文件（gen_*.py, analyze_daban.py 等）— 下次自然迭代时替换
- COC 现金跟踪 — 已有三层防护
- 涨停池"不检查当日涨停" — CLAUDE.md 铁律写死

---

## 六、验证

| # | 命令 | 检查点 |
|---|------|--------|
| 1 | `python scripts/screen_bull.py --date 2026-06-20` | `lu_60d` 列不再全为 0 |
| 2 | `python scripts/scan_intraday.py` | 涨停股数量 > 0 |
| 3 | `python scripts/run_arsenal.py` | 涨停统计正常 |
| 4 | `python scripts/bt_label_ocr.py --top-n 5` | 资金分配使用动态 cash/available |
| 5 | `python scripts/bt_small_cap.py --start 2025-01-01 --top-n 10` | 熔断有 60 天冻结 |
| 6 | `python scripts/run_daily_signals.py --help` | 无 import 错误 |
