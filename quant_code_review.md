# Quant 项目代码审查报告

> 审查对象: https://github.com/Neownne/Quant.git
> 审查日期: 2026-06-25
> 版本: v7.6 (hotpoint OCR + 自进化因子挖掘 + 标签烙印策略)
> 审查范围: 核心策略脚本、回测引擎、数据同步、因子库、OCR分析模块

---

## 一、代码级 Bug & 逻辑错误

### 严重 Bug（可能导致回测失真或运行时崩溃）

#### 1.1 bt_small_cap.py -- 组合熔断逻辑完全错误

**位置**: bt_small_cap.py 第 95-101 行

```python
# 组合熔断检查
dd = (peak_value - total) / peak_value if peak_value > 0 else 0
if dd > args.portfolio_dd_stop:
    frozen = True
if dd < args.portfolio_dd_stop * 0.8:
    frozen = False
```

**问题**: 熔断解除条件 `dd < args.portfolio_dd_stop * 0.8` 在 **同一天内** 就可能触发。
当日触发熔断时 `dd = 0.25 > 0.25` -> `frozen = True`，但下一秒如果价格回升，`dd = 0.24 < 0.20` -> `frozen = False`。
这导致熔断形同虚设--**日内波动即可解除**，完全违背了组合回撤保护的设计意图。

**修复建议**:
```python
if dd > args.portfolio_dd_stop and not frozen:
    frozen = True
    frozen_trigger_date = td  # 记录触发日
# 熔断后至少冻结 N 天，或需要人工确认
if frozen and (td - frozen_trigger_date).days >= args.frozen_days:
    frozen = False
```

---

#### 1.2 run_daily_signals.py -- `_wait_until_morning_close()` 函数死代码

**位置**: 第 157-164 行

```python
def _wait_until_morning_close():
    """等到当日 11:30 上午收盘。"""
    now = datetime.now()
    close_time = now.replace(hour=11, minute=30, second=0, microsecond=0)
    if now >= close_time:
        return
    wait_seconds = (close_time - now).total_seconds()
    logger.info(f"距上午收盘还有 {wait_seconds/60:.0f} 分钟，等待中...")
```

**问题**: 函数声明了等待逻辑，但**没有执行 `time.sleep()`**，调用后立即返回。
更离谱的是，这个函数在 `_is_data_stale()` 之后被调用（第 183 行附近），但 `_is_data_stale` 内部已经处理了等待逻辑，导致 `_wait_until_morning_close` 成为**完全无用的死代码**。

**修复建议**: 删除此函数，或补充 `time.sleep(wait_seconds + 5)`。

---

#### 1.3 run_daily_signals.py -- 涨停池筛选条件自相矛盾

**位置**: `screen_limit_up()` 函数

```python
# 4条件筛选（不要求当日涨停，只要近20日涨停2-4次）
mask = (
    (today['mcap'].between(30, 500)) &
    (today['close'].between(5, 100)) &
    (today['ma5'] > today['ma10']) &
    (today['lu_20d'] >= 2) & (today['lu_20d'] <= 4) &
    (today['close'] > 0)
)
```

**问题**: 注释说"不要求当日涨停"，但函数名是 `screen_limit_up`（涨停池），README 也说"输出当日涨停且满足筛选的股票"。
实际代码**不检查当日是否涨停**，导致选出的可能是 20 日内有过涨停但当日并未涨停的股票。
这与"涨停池"的语义完全不符，是一个**命名与实现不一致**的严重逻辑错误。

**修复建议**: 补充当日涨停判断：`mask = mask & (today['is_lu'] == 1)`

---

#### 1.4 bt_yaogu.py -- `wait_for_buyable` 买入窗口逻辑有漏洞

**位置**: 第 108-130 行

```python
for offset in range(1, 11):
    nxt = idx + offset
    if nxt >= len(all_dates):
        break
    nd = all_dates[nxt]
    ndf = daily_by_date.get(nd)
    if ndf is None or code not in ndf.index:
        continue
    r = ndf.loc[code]
    px, prev_c = r["close"], r.get("prev_close")
    if pd.notna(prev_c) and prev_c > 0:
        if TradingConfig.is_at_limit_up(px, prev_c, code):
            continue
        if TradingConfig.is_at_limit_down(px, prev_c, code):
            continue
        buy_signals.append({...})
        bought = True
        break
```

**问题**:
1. **T+1 制度未考虑**: A 股是 T+1，信号日买入后次日才能卖出，但这里 T 日信号 -> T+1 日买入，买入后**当天无法卖出**，而止损逻辑是按日检查收盘价。如果 T+1 日买入后当天跌停，次日继续跌停，止损无法执行。
2. **连续涨停跳过**: 如果一只股票连续 10 天涨停，这个循环会全部跳过，**永远无法买入**。对于妖股策略来说，这正是最应该买入的情况（一字板后的首次开板）。
3. **买入价用收盘价**: 实际交易中，T+1 日开盘价和收盘价差异可能很大，用收盘价模拟买入严重失真。

---

#### 1.5 bt_backtest.py -- backtrader `coc` 模式下手工现金跟踪与 broker 不同步

**位置**: `LuStrategy.next()` 中大量使用 `self._running_cash` 和 `self._running_total`

**问题**: 代码启用了 `coc=True`（Cheat-On-Close），这意味着订单在当日收盘价成交。但 backtrader 的 broker 在 `next()` 期间**不会立即更新现金和持仓**（要等 bar 结束后的结算）。
作者用 `_running_cash` 和 `_running_total` 手工跟踪，但存在以下问题：

1. **卖出回款计算错误**: `sell_proceeds = pos.size * d.close[0] * NET_SELL` 中 `NET_SELL` 已经扣除了佣金和印花税，但 backtrader 的 `getoperationcost` 在 `CNStockComm` 中是**按笔计算**的，两者计算口径不一致。
2. **金字塔加仓的现金扣除**: `available_cash -= py_cost` 后，如果 backtrader 实际成交数量不同（如资金不足导致部分成交），手工值和 broker 值会**永久分叉**。
3. **日终 `_validate()` 校验**: 校验 `cash >= -0.01` 和 `drift < 0.01`，但 `coc` 模式下 broker 的现金是**上日结算后**的值，校验逻辑本身就有问题。

---

#### 1.6 screen_bull.py -- 涨停判断使用收益率而非涨停价

**位置**: 第 80-82 行

```python
daily['is_lu'] = daily.apply(
    lambda r: 1 if pd.notna(r['ret']) and r['ret'] >= _get_limit(str(r['code'])) * 0.98 else 0, axis=1
)
```

**问题**: `_get_limit(code)` 返回的是乘数（如 1.09899），乘以 0.98 后约为 1.077。
但 `r['ret']` 是收益率（如 0.10 表示 10%），比较 `0.10 >= 1.077` 永远为 False。
**正确写法应该是**: `r['ret'] >= (_get_limit(str(r['code'])) - 1) * 0.98`

这导致牛股筛选器中的 `lu_60d`（60 日内涨停次数）**永远为 0**，"60 日内涨停 < 2 次"这个条件形同虚设。

---

#### 1.7 scan_intraday.py -- 涨停判断同样错误

**位置**: 第 95-97 行

```python
df['is_lu'] = df.apply(
    lambda r: 1 if r['ret'] >= _get_limit(r['code']) * 98 else 0, axis=1
)
```

**问题**: `_get_limit(r['code'])` 返回乘数（如 1.09899），乘以 98 后约为 107.7。
但 `r['ret']` 是百分比（如 9.8），比较 `9.8 >= 107.7` 永远为 False。
**正确写法**: `r['ret'] >= (_get_limit(r['code']) - 1) * 100 * 0.98`

这导致午盘扫描中的涨停统计**完全错误**。

---

#### 1.8 factors/limit_up.py -- `_is_limit_up` 函数硬编码单只股票

**位置**: 第 55-62 行

```python
def _is_limit_up(df: pd.DataFrame) -> pd.Series:
    if "prev_close" not in df.columns:
        df = df.copy()
        df["prev_close"] = df.groupby("code")["close"].shift(1)
    code = df["code"].iloc[0] if "code" in df.columns else ""  # 只取第一只股票的代码！
    mult = _get_multiplier(str(code))
    return df["close"] >= round(df["prev_close"] * mult, 2)
```

**问题**: 当 `df` 包含多只股票时，`df["code"].iloc[0]` 只取第一只的代码，然后用这个代码的乘数去判断**所有股票**是否涨停。
如果第一只是主板（10%），后面有创业板（20%），创业板的涨停会被**漏判**。

**修复建议**: 使用 `apply` 或向量化按代码分组判断。

---

#### 1.9 analyze_hotpoint.py -- `_build_recommendation_table` 中涨停池检查硬编码主板乘数

**位置**: 第 680-690 行

```python
daily["limit_price"] = daily.apply(
    lambda r: round(r["prev_close"] * 1.09899, 4)
    if pd.notna(r["prev_close"]) and r["prev_close"] > 0 else 0, axis=1)
daily["is_lu"] = (daily["close"] >= daily["limit_price"]).astype(int)
```

**问题**: 硬编码 `1.09899`，未区分创业板/科创板/北交所。
虽然函数 `_get_limit_pool_codes` 只在标签烙印推荐中使用，但如果 OCR 数据包含非主板股票，涨停判断会出错。

---

#### 1.10 bt_label_ocr.py -- 买入资金分配逻辑错误

**位置**: 第 290-295 行

```python
shares = int(args.cash / args.top_n / px / 100) * 100
if shares <= 0:
    continue
cost = shares * px * BUY_COST
if cost > cash:
    continue
```

**问题**:
1. 使用 `args.cash`（初始本金 100 万）而非 `cash`（当前可用现金）计算每只股票分配金额。随着持仓增加，可用现金减少，但分配金额不变，导致**后期资金不足时大量信号被跳过**。
2. 没有考虑已有持仓，可能出现 `available = args.top_n - len(positions)` 为 0 时仍然尝试买入（虽然外层有判断，但逻辑混乱）。

**修复建议**: 使用 `cash / available / px / 100` 计算。

---

#### 1.11 run_daily_signals.py -- `screen_yaogu()` 中 `low_vol_streak` 计算逻辑错误

**位置**: 第 520-540 行

**问题**:
1. `vol_mean` 计算的是近 20 日 `vol_ma20` 的均值，但 `vol_ma20` 本身就是 20 日均量，再取均值没有意义。
2. `streak` 统计的是**连续缩量天数**，但妖股评分规则说"缩量整理 >= 1 天"，这里只要最后一天缩量就满足条件，但循环会统计最长连续缩量天数。
实际上代码返回的是**最后一个连续缩量 streak**，如果中间有放量再缩量，streak 会重置。这个逻辑与"缩量整理"的语义不完全匹配。

---

#### 1.12 data/sync.py -- `sync_stock_daily` 中 `existing_map` 构建后未正确使用

**位置**: 第 170-200 行

**问题**: `existing_map` 只包含 `trade_date >= start_date` 的数据。如果某只股票在 `start_date` 之前就有数据，但 `start_date` 之后没有，会被误判为"没有任何 >= start_date 的数据"，从而**全量重新拉取**。
这虽然不会导致数据错误，但会造成大量不必要的网络请求。

---

#### 1.13 evolve_factors.py -- 相关性矩阵使用**估计值**而非真实计算

**位置**: 第 280-295 行

```python
def compute_correlation_matrix(db):
    factors = [f for f in db["factors"] if f.get("ic_samples", 0) > 0]
    if len(factors) < 2:
        return {}
    corr_matrix = {}
    for i, f1 in enumerate(factors):
        for f2 in factors[i + 1:]:
            if f1.get("category") == f2.get("category"):
                corr_matrix[f"{f1['name']}_{f2['name']}"] = 0.75  # 硬编码估计！
            elif f1.get("operator") == f2.get("operator"):
                corr_matrix[f"{f1['name']}_{f2['name']}"] = 0.60
            else:
                corr_matrix[f"{f1['name']}_{f2['name']}"] = 0.15
    return corr_matrix
```

**问题**: 相关性矩阵没有**实际计算**因子 IC 序列的相关性，而是根据类别和算子**硬编码估计值**。
这导致"去重"功能完全失效--两个实际上相关性 0.95 的因子可能因为类别不同而被认为只有 0.15 的相关性，从而**都保留下来**。

---

### 中等 Bug（影响结果但不致命）

#### 2.1 bt_small_cap.py -- 回看期收益计算使用固定天数而非交易日

```python
lb_start = td - timedelta(days=args.lookback + 5)
lb_df = daily[(daily["trade_date"] >= lb_start) & (daily["trade_date"] <= td)]
```

**问题**: `lookback=20` 天，但 `timedelta(days=25)` 可能只覆盖 15-18 个交易日（遇到节假日）。
导致"近 20 日跌最多"的实际计算窗口**不固定**，回测结果不可复现。

#### 2.2 bt_yaogu.py -- 硬止损使用当日收盘价而非最低价

**问题**: 用收盘价判断止损，但实际交易中如果当日最低价已经触发止损，应该在**盘中**执行。用收盘价意味着**错过了盘中止损点**，可能亏损更大。

#### 2.3 run_daily_signals.py -- 日内模式市值使用前一日填充

**问题**: 午盘扫描时，市值数据用 T-1 日填充到 T 日。如果某只股票当日发生**除权除息**或**增发**，市值可能变化巨大，导致涨停池的市值筛选条件失真。

#### 2.4 analyze_hotpoint.py -- OCR 缓存格式不兼容导致重复 OCR

**问题**: 旧缓存格式（纯字符串）和新格式（dict）混存，如果用户从 Tesseract 切换到 Surya，旧缓存会被**误判为需要重新 OCR**，导致大量重复工作。

#### 2.5 run_daily_signals.py -- 邮件配置缺少 `EMAIL_FROM` 回退

**问题**: 如果 `.env` 中只配置了 `SMTP_USER` 但没有 `EMAIL_FROM`，`from_addr` 为空字符串，可能导致邮件发送失败。

#### 2.6 bt_backtest.py -- `version_id` 硬编码为 49

**问题**: 如果数据库中 `strategy_versions` 表没有 id=49 的记录，回测结果写入会失败。这是**临时 hack**，应该在 CLI 中强制要求 `--variant-label` 或 `--version-id`。

---

### 代码异味（不影响功能但增加维护成本）

#### 3.1 多处重复定义涨停乘数字典

以下文件中都有独立的 `_LIMIT_MULT` / `_DEFAULT_MULT`:
- config/settings.py
- run_daily_signals.py
- scan_intraday.py
- factors/limit_up.py
- analyze_hotpoint.py
- screen_bull.py
- bt_label_ocr.py

**问题**: 违反 DRY 原则。如果某类股票涨跌幅规则调整（如北交所从 30% 改为 20%），需要修改 7+ 个文件。

#### 3.2 run_daily_signals.py 中公共因子预计算代码重复

`load_and_precompute()` 和 `load_intraday_data()` 中，因子预计算代码**几乎完全相同**（均线、量能、涨停统计、连板数等），但分别写在两个函数中。任何修改需要改两处。

#### 3.3 bt_yaogu.py 和 bt_label_ocr.py 回测引擎代码高度重复

两个文件的回测循环（持仓更新、止损、买入、卖出、熔断）**代码结构几乎一致**，只是信号来源不同。应该抽象为统一的回测引擎基类。

#### 3.4 gen_signals.py 中 `compute_streak_map` 时间复杂度 O(n^2)

**问题**: 对每只股票、每个日期都做内层循环，时间复杂度 O(n^2)。对于 5000 只股票 x 2000 交易日 = 1000 万条数据，这个循环会非常慢。可以用**滑动窗口**优化到 O(n)。

#### 3.5 evolve_factors.py 中 `sector_relative` 模板 lambda 嵌套过深

**问题**: 三重 lambda 嵌套，可读性极差。且 `r.groupby(daily["sector"]).transform("mean")` 中 `r` 和 `daily["sector"]` 长度可能不一致，存在**隐式索引对齐风险**。

---

## 二、架构级缺陷（与之前结论整合）

### 2.1 数据源单一且脆弱

| 模块 | 数据源 | 风险 |
|------|--------|------|
| 日终同步 | Tushare / akshare | akshare 接口不稳定，经常变更 |
| 午盘扫描 | 腾讯实时 API | 无 SLA，可能限流或改接口 |
| 市值数据 | 新浪财经 (proxy) | 需要额外解析 HTML，易失效 |
| OCR | Surya VLM / Tesseract | Surya 首次加载 ~3h，Tesseract 准确率差 |

**建议**: 增加数据源降级机制（如腾讯失败 -> 东方财富 -> 雪球），并增加数据校验层。

### 2.2 回测过拟合嫌疑

- **小市值反转 2025 年 +87%**: 在 2025 年小盘股暴跌的市场环境下，这个业绩需要严格审计。可能原因：
  - 使用了**未来数据**（如市值数据在 T 日收盘后才公布，但回测在 T 日开盘就用）
  - **幸存者偏差**（退市股票未纳入回测）
  - **参数调优过拟合**（在 2025 年数据上反复调参）

- **妖股规则 33.7% 大涨率**: 基于 2020 年数据，但 README 承认 2025 年几乎不出信号。说明规则是**时期依赖型**的，不是稳健因子。

### 2.3 实盘与回测的鸿沟

| 回测假设 | 实盘现实 |
|----------|----------|
| 滑点 0.1% | 涨停股实际滑点可能 > 5%（买不到或买到即炸板） |
| 收盘价成交 | 实际需排队，成交率 < 100% |
| T+0 止损 | A 股 T+1，当日买入无法止损 |
| 无限流动性 | 小市值股票日均成交额可能 < 1000 万 |

### 2.4 策略逻辑混杂，缺乏分层

仓库中同时存在：
- **小市值反转**（均值回归，量化因子型）
- **涨停/妖股/牛股**（动量延续，事件驱动型）
- **OCR 关键词**（另类数据型）
- **自进化因子**（自动挖掘型）

这些策略的**底层假设互相矛盾**，但没有资金分配和策略权重框架。

### 2.5 ML 部分基本失败但还留着

README 明确写了：XGBoost PR-AUC=0.19、R^2=0.04。这些模型文件、训练脚本、特征工程代码还全在仓库里，增加维护负担。

### 2.6 缺乏风险管理和仓位控制

- 回测本金固定 100 万，每只策略都是"等权买入 Top-N"
- 没有**波动率目标**、**最大回撤控制**、**策略间相关性管理**
- 妖股策略 33.7% 大涨率 != 高期望收益（盈亏比未知）

---

## 三、修复优先级建议

| 优先级 | 问题 | 影响 | 修复难度 |
|--------|------|------|----------|
| P0 | 涨停判断收益率 vs 乘数混淆（screen_bull.py, scan_intraday.py） | 涨停统计完全错误 | 低 |
| P0 | 组合熔断逻辑错误（bt_small_cap.py） | 回撤保护失效 | 低 |
| P0 | 涨停池不检查当日涨停（run_daily_signals.py） | 池子定义错误 | 低 |
| P1 | backtrader coc 模式现金跟踪不同步 | 回测结果失真 | 高 |
| P1 | wait_for_buyable 连续涨停跳过 | 错失最佳买入时机 | 中 |
| P1 | _is_limit_up 硬编码单只股票代码 | 创业板/科创板漏判 | 低 |
| P1 | bt_label_ocr.py 资金分配用初始本金 | 后期资金利用率低 | 低 |
| P2 | 涨停乘数多处重复定义 | 维护困难 | 低 |
| P2 | evolve_factors.py 相关性矩阵用估计值 | 去重失效 | 中 |
| P2 | 回测过拟合审计 | 策略可信度 | 高 |
| P3 | ML 残骸清理 | 代码整洁度 | 低 |
| P3 | 策略分层 & 资金分配框架 | 组合管理 | 高 |

---

## 四、代码质量评分

| 维度 | 评分 | 说明 |
|------|------|------|
| 功能完整性 | 4/5 | 覆盖数据同步、回测、信号生成、OCR、邮件推送 |
| 代码正确性 | 2/5 | 多处核心逻辑错误（涨停判断、熔断、资金分配） |
| 可维护性 | 2/5 | 大量重复代码、硬编码、DRY 原则违反严重 |
| 可测试性 | 1/5 | 无单元测试，无 CI，验证靠手动运行 |
| 文档完整性 | 4/5 | README 详尽，但代码注释不足 |
| 工程化程度 | 2/5 | 缺少配置管理、日志分级、异常处理不完善 |

**总体评价**: 这是一个**功能丰富但工程化不足**的个人研究项目。核心策略逻辑存在多处低级错误（尤其是涨停判断相关的代码），回测结果可信度存疑。建议在实盘前进行全面的代码审计和数据校验。

---

*报告生成时间: 2026-06-25*
*审查范围: 8 个核心文件，约 3000+ 行代码*
