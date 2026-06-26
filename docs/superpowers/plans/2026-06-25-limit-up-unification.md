# 涨停判断统一 & Bug 修复 — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 消灭 4 个 buggy 文件中 `ret >= _get_limit(code) * X` 的错误涨停判断，统一到 `TradingConfig.is_at_limit_up` 入口，附带修复熔断/资金分配/死代码。

**Architecture:** 在 `TradingConfig.is_at_limit_up` 新增 `tolerance` 参数（默认 1.0=精确），6 个调用方切换至此接口；`_is_limit_up` 修复板别 bug；`bt_label_ocr.py`/`bt_small_cap.py` 各改一行。

**Tech Stack:** Python 3.12, pandas, PostgreSQL

---

### Task 1: TradingConfig.is_at_limit_up 加 tolerance 参数

**Files:**
- Modify: `config/settings.py:104-109`

- [ ] **Step 1: 修改 is_at_limit_up 签名**

```python
@staticmethod
def is_at_limit_up(close: float, prev_close: float, code: str, tolerance: float = 1.0) -> bool:
    """判断是否涨停封板（无法买入）。tolerance<1.0 放宽到近涨停区，如 0.98=涨9.7%+即算。"""
    if prev_close <= 0 or close <= 0:
        return False
    limit_price = TradingConfig.calc_limit_price(prev_close, code, is_up=True)
    return close >= limit_price * tolerance
```

> 注意：现有调用方（bt_backtest.py, bt_label_ocr.py, bt_yaogu.py）不传 tolerance，默认 1.0=精确比较，行为不变。

- [ ] **Step 2: 验证无 syntax error**

```bash
python -c "from config.settings import TradingConfig; print(TradingConfig.is_at_limit_up(11.0, 10.0, '600000'))"
```
Expected: `True`

- [ ] **Step 3: Commit**

```bash
git add config/settings.py
git commit -m "feat: TradingConfig.is_at_limit_up 加 tolerance 参数"
```

---

### Task 2: 修复 screen_bull.py 涨停判断

**Files:**
- Modify: `scripts/screen_bull.py:31-38, 111-113`

- [ ] **Step 1: 删 _LIMIT_MULT + _get_limit，替换 is_lu 行**

删除第 31-38 行（`_LIMIT_MULT` 字典 + `_get_limit` 函数）。

第 111-113 行，将：
```python
daily['is_lu'] = daily.apply(
    lambda r: 1 if pd.notna(r['ret']) and r['ret'] >= _get_limit(str(r['code'])) * 0.98 else 0, axis=1
)
```
改为：
```python
# 涨停标记（统一入口，tolerance=0.98 近涨停区）
from config.settings import TradingConfig
daily['is_lu'] = daily.apply(
    lambda r: 1 if pd.notna(r.get('prev_close')) and r['prev_close'] > 0
              and TradingConfig.is_at_limit_up(r['close'], r['prev_close'], str(r['code']), tolerance=0.98)
              else 0, axis=1
)
```

注意：`screen_bull.py` 独立模式下需要在 `screen()` 函数内计算 `prev_close`（目前已有 `daily['ret'] = daily.groupby('code')['close'].pct_change()` 在第 98 行，需确保 `prev_close` 在 `is_lu` 计算前可用）。检查发现第 98 行只有 `ret`，需在第 98 行后加上：
```python
daily['prev_close'] = daily.groupby('code')['close'].shift(1)
```

而在集成模式下（`daily_df is not None`），调用方 `run_daily_signals.py` 已经预计算了 `prev_close` 和 `is_lu`，不会进入这个分支。

- [ ] **Step 2: 验证**

```bash
python scripts/screen_bull.py --date 2026-06-20 --top 5
```
Expected: `近期涨停次` 列不再全为 0（如果目标日有涨停股）

- [ ] **Step 3: Commit**

```bash
git add scripts/screen_bull.py
git commit -m "fix: screen_bull.py 涨停判断走 TradingConfig.is_at_limit_up"
```

---

### Task 3: 修复 scan_intraday.py 涨停判断

**Files:**
- Modify: `scripts/scan_intraday.py:24-33, 106-108`

- [ ] **Step 1: 删 _LIMIT_MULT + _get_limit，替换 is_lu 行**

删除第 24-33 行。

第 106-108 行，将：
```python
df['is_lu'] = df.apply(
    lambda r: 1 if r['ret'] >= _get_limit(r['code']) * 98 else 0, axis=1
)
```
改为：
```python
from config.settings import TradingConfig
df['is_lu'] = df.apply(
    lambda r: 1 if r['prev_close'] > 0 and TradingConfig.is_at_limit_up(
        r['price'], r['prev_close'], r['code'], tolerance=0.98) else 0, axis=1
)
```

注意：腾讯行情数据字段名是 `price` 不是 `close`，`prev_close` 叫 `prev_close`（第 82 行）。

- [ ] **Step 2: 验证**

```bash
python scripts/scan_intraday.py --top 5 2>&1 | head -30
```
Expected: 涨停统计数 > 0

- [ ] **Step 3: Commit**

```bash
git add scripts/scan_intraday.py
git commit -m "fix: scan_intraday.py 涨停判断走 TradingConfig.is_at_limit_up"
```

---

### Task 4: 修复 run_arsenal.py 涨停判断（3 处）

**Files:**
- Modify: `scripts/run_arsenal.py:24-31, 72-73, 133-135, 141`

- [ ] **Step 1: 删 _LIMIT_MULT + _get_limit，替换热力图 is_lu**

删除第 24-31 行。

第 70 行后加 `prev_close`（已有 `df = df.merge(prev, on='code', how='left')`，`prev_close` 列已存在）。第 72-73 行，将：
```python
df['is_lu'] = df.apply(
    lambda r: 1 if pd.notna(r['ret']) and r['ret'] >= _get_limit(str(r['code'])) * 0.98 else 0, axis=1
)
```
改为：
```python
from config.settings import TradingConfig
df['is_lu'] = df.apply(
    lambda r: 1 if pd.notna(r.get('prev_close')) and r['prev_close'] > 0
              and TradingConfig.is_at_limit_up(r['close'], r['prev_close'], str(r['code']), tolerance=0.98)
              else 0, axis=1
)
```

- [ ] **Step 2: 替换妖股扫描 is_lu 判断（第 133-135 行）**

将：
```python
limit = _get_limit(code)
is_lu = r['ret'] >= limit * 0.98 if pd.notna(r['ret']) else False
```
改为：
```python
is_lu = (pd.notna(r.get('prev_close')) and r['prev_close'] > 0
         and TradingConfig.is_at_limit_up(r['close'], r['prev_close'], code, tolerance=0.98))
```

- [ ] **Step 3: 替换妖股扫描连板计算（第 141 行）**

将：
```python
is_lu_series = (cdata['ret'] >= limit * 0.98)
```
改为：
```python
mult = TradingConfig.get_limit_multiplier(code)
limit_pxs = (cdata['prev_close'] * mult).round(4)
is_lu_series = cdata['close'] >= limit_pxs * 0.98
```

- [ ] **Step 4: 验证**

```bash
python scripts/run_arsenal.py 2>&1 | head -30
```
Expected: 涨停统计数 > 0，无 AttributeError/KeyError

- [ ] **Step 5: Commit**

```bash
git add scripts/run_arsenal.py
git commit -m "fix: run_arsenal.py 涨停判断走 TradingConfig.is_at_limit_up"
```

---

### Task 5: 修复 export_limit_up_history.py 涨停判断

**Files:**
- Modify: `scripts/export_limit_up_history.py:25-32, 77`

- [ ] **Step 1: 删 _LIMIT_MULT + _get_limit，替换 is_lu 行**

删除第 25-32 行。第 77 行，将：
```python
lambda r: 1 if pd.notna(r['ret']) and r['ret'] >= _get_limit(str(r['code'])) * 0.98 else 0,
```
改为：
```python
lambda r: 1 if (pd.notna(r.get('prev_close')) and r['prev_close'] > 0
                and __import__('config.settings').TradingConfig.is_at_limit_up(
                    r['close'], r['prev_close'], str(r['code']), tolerance=0.98)) else 0,
```

注意：此 lambda 在 `daily.apply(..., axis=1)` 内，`prev_close` 在第 74 行已通过 `shift(1)` 计算。

- [ ] **Step 2: 验证**

```bash
python -c "
import pandas as pd
from scripts.export_limit_up_history import main
" 2>&1 | head -5
```
Expected: 无 import 错误

- [ ] **Step 3: Commit**

```bash
git add scripts/export_limit_up_history.py
git commit -m "fix: export_limit_up_history.py 涨停判断走 TradingConfig.is_at_limit_up"
```

---

### Task 6: 修复 factors/limit_up.py

**Files:**
- Modify: `factors/limit_up.py:61-73, 542, 590`

- [ ] **Step 1: 修复 _is_limit_up 板别 bug（第 71-72 行）**

将：
```python
code = df["code"].iloc[0] if "code" in df.columns else ""
mult = _get_multiplier(str(code))
return df["close"] >= round(df["prev_close"] * mult, 2)
```
改为：
```python
mults = df["code"].astype(str).apply(_get_multiplier) if "code" in df.columns else _DEFAULT_MULT
return df["close"] >= round(df["prev_close"] * mults, 2)
```

- [ ] **Step 2: 修复 sector/market 函数中的 _get_limit（第 542、590 行）**

将两处：
```python
df["is_lu"] = df.apply(
    lambda r: r["ret"] >= _get_limit(str(r["code"])) * 0.98
    if pd.notna(r["ret"]) else False, axis=1
)
```
改为：
```python
df["is_lu"] = df.apply(
    lambda r: (pd.notna(r.get("prev_close")) and r["prev_close"] > 0
               and r["close"] >= round(r["prev_close"] * _get_multiplier(str(r["code"])), 4) * 0.98)
    if pd.notna(r["ret"]) else False, axis=1
)
```

- [ ] **Step 3: 验证**

```bash
python -c "from factors.limit_up import _is_limit_up, compute_sector_lu_factors, compute_market_lu_extra; print('OK')"
```
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add factors/limit_up.py
git commit -m "fix: _is_limit_up 板别感知修复 + sector/market 函数涨停判断修正"
```

---

### Task 7: 修复 bt_label_ocr.py 资金分配

**Files:**
- Modify: `scripts/bt_label_ocr.py:365`

- [ ] **Step 1: 改资金分配公式**

将：
```python
shares = int(args.cash / args.top_n / px / 100) * 100
```
改为：
```python
shares = int(cash / max(available, 1) / px / 100) * 100
```

- [ ] **Step 2: 验证**

```bash
python scripts/bt_label_ocr.py --top-n 5 2>&1 | tail -10
```
Expected: 回测正常完成，交割单中后期买入股数不再固定

- [ ] **Step 3: Commit**

```bash
git add scripts/bt_label_ocr.py
git commit -m "fix: bt_label_ocr 资金分配使用动态剩余资金"
```

---

### Task 8: 修复 bt_small_cap.py 熔断逻辑

**Files:**
- Modify: `scripts/bt_small_cap.py:98, 122-129`

- [ ] **Step 1: 引入 frozen_days 计数器**

第 98 行，将：
```python
frozen = False              # 组合熔断标记
```
改为：
```python
frozen = False              # 组合熔断标记
frozen_days = 0             # 冻结天数计数器
```

第 122-129 行，将：
```python
if dd > args.portfolio_dd_stop:
    frozen = True
    logger.info(f"  [{td.strftime('%Y-%m-%d')}] 组合回撤 {dd:.1%} > {args.portfolio_dd_stop:.0%}，熔断")
if dd < args.portfolio_dd_stop * 0.8:
    frozen = False
```
改为：
```python
if dd > args.portfolio_dd_stop and not frozen:
    frozen, frozen_days = True, 0
    logger.info(f"  [{td.strftime('%Y-%m-%d')}] 组合回撤 {dd:.1%} > {args.portfolio_dd_stop:.0%}，熔断")
if frozen:
    frozen_days += 1
if frozen and frozen_days > 60:
    frozen, peak_value = False, total
    logger.info(f"  [{td.strftime('%Y-%m-%d')}] 熔断 60 天期满，解除")
```

- [ ] **Step 2: 验证**

```bash
python scripts/bt_small_cap.py --start 2025-01-01 --top-n 10 2>&1 | tail -10
```
Expected: 回测正常完成，如有熔断则 60 天后显示解除日志

- [ ] **Step 3: Commit**

```bash
git add scripts/bt_small_cap.py
git commit -m "fix: bt_small_cap 熔断加 60 天最低冻结期"
```

---

### Task 9: 清理 run_daily_signals.py 死代码

**Files:**
- Modify: `scripts/run_daily_signals.py:135-136, 191`

- [ ] **Step 1: _wait_until_morning_close 加 time.sleep**

第 136 行后加：
```python
    time.sleep(wait_seconds + 5)
```

- [ ] **Step 2: 删除 _is_eod_report_missing 中 return 后的死代码**

删除第 191 行：
```python
    time.sleep(wait_seconds + 5)
```

- [ ] **Step 3: Commit**

```bash
git add scripts/run_daily_signals.py
git commit -m "fix: 清理 _wait_until_morning_close 和 _is_eod_report_missing 死代码"
```

---

### Task 10: 更新 CLAUDE.md 铁律

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: 在铁律表中新增第 19 条**

在铁律表格末尾（第 #19 行之前的最末条规则之后）插入：

```
| 19 | **涨停判断唯一入口** `TradingConfig.is_at_limit_up(close, prev_close, code)` | 禁止手写 `ret >= mult`/`ret >= _get_limit()`、禁止自建 `_LIMIT_MULT` 字典 |
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: CLAUDE.md 新增铁律 #19 涨停判断唯一入口"
```

---

### Task 11: 全量回归验证

- [ ] **Step 1: 牛股筛选器**

```bash
python scripts/screen_bull.py --date 2026-06-20 --top 5
```
Check: `近期涨停次` 列值正常，无全 0。

- [ ] **Step 2: 午盘扫描**

```bash
python scripts/scan_intraday.py --top 5 2>&1 | head -15
```
Check: 涨停股数 > 0。

- [ ] **Step 3: 武器库**

```bash
python scripts/run_arsenal.py 2>&1 | head -15
```
Check: 涨停统计正常。

- [ ] **Step 4: 因子库**

```bash
python -c "from factors import ALL_FACTORS; print(f'{len(ALL_FACTORS)} factors OK')"
```
Expected: `123 factors OK`

- [ ] **Step 5: 标签烙印回测**

```bash
python scripts/bt_label_ocr.py --top-n 5 2>&1 | tail -10
```
Check: 回测正常完成。

- [ ] **Step 6: 小市值回测**

```bash
python scripts/bt_small_cap.py --start 2025-01-01 --top-n 10 2>&1 | tail -10
```
Check: 回测正常完成，对比修复前后收益率无显著偏差。

- [ ] **Step 7: 日终信号**

```bash
python -c "from scripts.run_daily_signals import sync_all, screen_limit_up, screen_yaogu; print('import OK')"
```
Expected: `import OK`

---

## Self-Review

1. **Spec coverage**: ✅ 所有 6 类修复都有对应 Task（ret-vs-multiplier ×4 → T2-T6, _is_limit_up → T6, 资金分配 → T7, 熔断 → T8, 死代码 → T9, CLAUDE.md → T10）
2. **Placeholder scan**: ✅ 无 TBD/TODO，每步有完整代码
3. **Type consistency**: ✅ `is_at_limit_up(close, prev_close, code, tolerance)` 签名在各 Task 中一致
