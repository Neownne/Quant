# 分钟频 ML 回测优化 — 设计文档

> 日期：2026-05-26
> 状态：待审阅

---

## 一、目标

将 ML 回测的数据源从日频升级为分钟频（60 分钟 K 线），通过更细粒度数据提升信号精度和策略收益。采用 Plan A（日内优化优先）+ Plan C（分钟数据同步后台化）架构。

---

## 二、数据管道

### 2.1 数据源

使用 Sina API（`ak.stock_zh_a_minute`），东方财富接口（`stock_zh_a_hist_min_em`）不可用。

- **API**: `ak.stock_zh_a_minute(symbol=sina_symbol, period="60", adjust="qfq")`
- **覆盖**: 约 2 年数据，前复权
- **支持周期**: 1/5/15/30/60 分钟

### 2.2 存储

数据写入已有 `stock_minute` 表（DDL 已存在于 `data/db.py`）：

```sql
CREATE TABLE IF NOT EXISTS stock_minute (
    code       VARCHAR(10) NOT NULL,
    trade_time TIMESTAMP   NOT NULL,
    period     VARCHAR(5)  NOT NULL,
    open       DOUBLE PRECISION,
    high       DOUBLE PRECISION,
    low        DOUBLE PRECISION,
    close      DOUBLE PRECISION,
    volume     DOUBLE PRECISION,
    amount     DOUBLE PRECISION,
    PRIMARY KEY (code, trade_time, period)
);
```

使用 `upsert_df()` 做冲突感知插入。

### 2.3 数据同步脚本（`scripts/sync_minute_data.py`）

单次运行脚本，逻辑：

1. 从 `stock_daily` 获取符合条件的所有股票代码
2. 排除规则：ST 股、上市不足 180 天、北交所（BJ）
3. 对每只股票调用 `fetch_minute_data()`，成功则 `upsert_df()` 写入
4. 限速：requests 间隔 0.3s 防封
5. 完成后打印统计：成功数 / 失败数 / 总 bar 数

### 2.4 数据完整性校验（`scripts/validate_minute_data.py`）

对比分钟线日频聚合 vs 日线原始数据，确保 `close` 偏差 < 1%。

---

## 三、ML 回测引擎适配

### 3.1 数据加载层 —— freq 开关

新增 `freq` 参数，`app/utils/ml_backtest.py` 中数据加载函数走不同路径：

```python
def load_price_data(codes, start, end, freq="daily"):
    if freq == "daily":
        return get_daily_ohlcv(codes, start, end)
    else:
        return get_minute_ohlcv(codes, start, end, period=freq)
```

分钟数据额外处理：对齐时间戳，所有股票共有的 minute bar 才纳入。

### 3.2 因子计算层 —— window 自适应

`models/dataset.py` 的 `build_factor_dataset()` 新增 `bar_per_day` 参数：

- `freq="daily"`: `bar_per_day=1`，window 不变
- `freq="60min"`: `bar_per_day=4`，所有 rolling window × 4

```python
bar_per_day = 4 if freq == "60min" else 1
window_map = {
    "ma5": 5 * bar_per_day,
    "ma10": 10 * bar_per_day,
    "ma20": 20 * bar_per_day,
    "volatility_20": 20 * bar_per_day,
}
```

### 3.3 标签生成与 Walk-Forward

**不改**。标签仍是日频（取每日最后一根 bar 的 close），walk-forward 仍按自然日切分 train/val 窗口。

### 3.4 交易模拟

**第一版不改**。信号仍是日频（每日一个截面得分），调仓仍是日频（T+1 开盘交易）。分钟数据仅提供更丰富的 bar 内信息用于因子计算。

后续可做盘中止损优化（bar 级别检查止损条件），但不在第一版范围内。

---

## 四、回测页面改动

`app/pages/3_📊_Backtest.py` 中 ML 策略参数面板新增：

- 数据频率选择：下拉框 `daily | 60min`
- `freq="60min"` 时显示提醒："分钟频回测耗时较长，建议先用默认 ML 配置测试"

静态策略不受影响。

---

## 五、验证策略

### 5.1 数据完整性

运行 `scripts/validate_minute_data.py` 对比分钟聚合 close vs 日线 close，偏差应 < 1%。

### 5.2 回测交叉验证

用同一 ML 配置分别跑日频和分钟频回测（相同 stock pool、时间区间）：

| 维度 | 期望 |
|------|------|
| 权益曲线趋势 | 大方向一致 |
| 年化收益率 | 偏差 30% 以内 |
| 最大回撤 | 分钟频可能更深 |

若差异巨大（如 +15% vs -20%），说明因子计算或数据对齐有问题，需排查。

### 5.3 回退策略

通过 `freq` 参数控制，`freq="daily"` 时逻辑与现在完全一致，分钟频出问题可随时回退。

---

## 六、执行顺序

```
sync_minute_data.py → validate_minute_data.py → ml_backtest.py freq 适配 → 回测页面 UI → 交叉验证
```
