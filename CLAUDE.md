# 项目规范

## Claude Code 八荣八耻

1. 以认真查询为荣，以猜测接口为耻。
2. 以寻求确认为荣，以模糊执行为耻。
3. 以人类确认为荣，以臆想业务为耻。
4. 以复用现有为荣，以新造接口为耻。
5. 以主动验证为荣，以跳过测试为耻。
6. 以遵循规范为荣，以破坏架构为耻。
7. 以诚实承认无知为荣，以假装理解为耻。
8. 以谨慎重构为荣，以盲目修改为耻。

## 快速参考

```bash
# 每日模拟盘（自动化）
python scripts/run_daily_paper_auto.py                  # 自动同步+跑涨停+大小票
python scripts/run_daily_paper_auto.py --dry-run         # 试运行
python scripts/run_daily_paper_auto.py --strategy lu     # 只跑涨停
python scripts/run_daily_paper_auto.py --daemon          # 守护模式(每天20:00)
python scripts/run_daily_paper_lu.py --no-sync           # 手动跑涨停

# 涨停策略
python scripts/scan_limit_up_strategy.py               # 每日筛选
python scripts/backtest_limit_up_strategy.py --start 2020-01-01 --mcap-proxy --top-n 5 --min-conditions 5 --exit-stop 0.08  # E3回测

# 大小票切换
python small_cap/switch_backtest.py                     # v1.0 原始
python small_cap/switch_backtest_v2.py                  # v4.0 涨停替代

# 小票/大票ML回测
python scripts/run_small_cap_backtest.py                 # 小市值alpha
python scripts/run_ml_backtest.py --strategy 舞          # 舞 v1.85

# ETF监控
python scripts/run_etf_monitor.py                        # 三因子信号扫描

# 数据同步
python -c "
from data.db import get_engine
from data.sync import sync_stock_daily, sync_daily_extra
e = get_engine()
sync_stock_daily(e, start_date='2026-06-09', workers=8)
sync_daily_extra(e, start_date='2026-06-09', workers=8)
"

# Web
python -m uvicorn web.main:app --host 0.0.0.0 --port 8899
# → http://localhost:8899/paper     模拟盘
# → http://localhost:8899/backtest   回测
# → http://localhost:8899/etf        ETF监控

# 数据库
pg_ctl -D /opt/homebrew/var/postgresql@18 start
pg_ctl -D /opt/homebrew/var/postgresql@18 stop
```

## 架构总览

```
┌─ 数据层 ─────────────────────────────────────────────┐
│  PostgreSQL (localhost:5432)                          │
│  stock_daily (5524只, 2015~今)                        │
│  stock_daily_extra (市值/PE/PB, API仅支持1年)          │
│  stock_minute (1472只, 2024-03~今)                    │
│  index_constituent (CSI300成分股)                      │
│  paper_* (模拟盘: account/signals/positions/daily_pnl) │
└──────────────────────────────────────────────────────┘
          │                    ▲
          ▼                    │
┌─ 因子层 ─────────────────────────────────────────────┐
│  FactorEngine → IC Gate → Orthogonal Screening        │
│  86+ 因子 (alpha101/191/custom/fundamental/intraday)   │
└──────────────────────────────────────────────────────┘
          │
          ▼
┌─ 模型层 ─────────────────────────────────────────────┐
│  Walk-Forward (3yr train / 1yr val, annual step)      │
│  XGBoost + LightGBM → EnsemblePredictor                │
│  RegimeAwareEnsemble (5状态自适应)                      │
└──────────────────────────────────────────────────────┘
          │
          ▼
┌─ 组合层 ─────────────────────────────────────────────┐
│  select_topk_ndrop(K=15,N=2) → 等权分配                │
│  风控: 个股-8% / 组合-20%减仓 / -25%清仓               │
├──────────────────────────────────────────────────────┤
│  规则筛选层（涨停策略）                                  │
│  5条件筛选(5选4): 市值50-300亿/股价5-50元              │
│  MA5>MA10/近月涨停>1次/近10日无跌停                     │
│  日频调仓 | Top-20等权                                  │
└──────────────────────────────────────────────────────┘
          │
          ▼
┌─ 模拟盘层 ──────────────────────────────────────────┐
│  PaperEngine: 信号→T+1执行→DB写入                     │
│  每日流程: 同步数据→训练→预测→执行信号→保存新信号     │
├──────────────────────────────────────────────────────┤
│  大小票平滑分配 (switch_backtest_v2.py)                │
│  CSI1000 vs MA60 → 动态分配小票/涨停权重               │
│  强势→涨停加码(上限70%) | 弱势→小票防御                │
│  回撤>20%→涨停减半 | 小票周频+涨停日频                 │
└──────────────────────────────────────────────────────┘
```

## 模拟盘数据流

```
Day T 收盘后:
  1. sync_stock_daily (增量, 8并发)
  2. load_data (OHLCV + extra + 分钟 + 日内特征)
  3. build_factor_dataset → IC筛选 → 正交筛选
  4. walk_forward_train_by_regime (XGBoost+LGB per regime)
  5. T+1执行: 找到 signal_date < today 且无对应 position 的信号 → run_daily()
  6. 预测今日: ensemble.predict() → 取Top-15 → 保存为待执行信号
  7. 首次无持仓时自动建仓

Day T+1:
  脚本再次运行 → 步骤5执行T日的信号 → NDrop调仓 → 生成T+1的新信号
```

## 模拟盘数据库表

| 表 | 说明 | 关键列 |
|----|------|--------|
| paper_account | 账户现金 | id, cash, initial_capital |
| paper_runs | 策略运行记录 | id, strategy_id, version_id |
| paper_signals | 每日信号 | run_id, signal_date, stock_code, score, rank |
| paper_positions | 逐笔持仓 | run_id, stock_code, entry_date, entry_price, exit_date, pnl |
| paper_daily_pnl | 每日估值 | account_id, trade_date, total_value, daily_return |
| paper_orders | 订单审计 | account_id, code, direction, price, volume |

**v1.8**: account_id=15, run_id=2 (universe=1000)
**v1.7**: account_id=15, run_id=1 (universe=500, 旧)
**小市值**: account_id=16, run_id=3
**涨停策略**: 独立脚本，未接入模拟盘

## 策略全景

| 策略 | 类型 | 候选池 | 选股方式 | 持仓 | 调频 | Sharpe(2020-26) |
|------|------|--------|----------|------|------|:---:|
| 舞 v1.85 | ML大票 | Top-1000 | 83因子ML | 5-15只 | 自适应 | 0.72 |
| 小市值 alpha v2.0 | ML小票 | 1000-3000 | 11因子ML | 15只 | 周度 | 0.93 |
| 涨停策略 Baseline | 规则中票 | Top-1000 | 5条件筛选 | 5只 | 日频 | 3.65 |
| **涨停策略 E3** | **规则中票** | **Top-1000** | **5条件+8%止损** | **5只** | **日频** | **3.75** |
| 大小票 v1.0 | 切换 | 双池 | mom_20反转+ML | 动态 | 周度 | 1.16 |
| 大小票 v4.0 | 切换 | 双池 | 涨停+ML | 动态 | 混合 | 1.39 |

## 涨停策略详解

### 5条件筛选（5选4即可通过）

| # | 条件 | 参数 | 说明 |
|---|------|------|------|
| 1 | 市值 | 50–300 亿 | 中盘股，避开微盘庄股和千亿大象 |
| 2 | 股价 | 5–50 元 | 过滤仙股和高价股 |
| 3 | 均线 | MA5 > MA10 | 短线多头排列 |
| 4 | 涨停次数 | 近20日 >1 次 | 日收益 ≥10%，捕获动量 |
| 5 | 跌停检查 | 近10日无跌停 | 排雷，日收益 ≤ -10% |

### 关键发现

- **市值条件是生死线**：不加市值过滤，2020-2024 亏 89.8%
- **必须日频调仓**：3日频/周频都大幅跑输。动量窗口仅3-5个交易日
- **5/5严格 >> 4/5宽松**：严格模式 Sharpe 2.03 vs 宽松 0.53
- **市值数据仅 1 年**（stock_daily_extra 从 2025-05 起），长区间需用 `--mcap-proxy`（隐含股本=最新市值/当日股价）

### 大小票 v4.0 分配逻辑

```
CSI1000 vs MA60 偏离度 → 动态权重：
  强势（CSI1000 > MA60）→ 涨停权重 30%-70%（追动量）
  弱势（CSI1000 < MA60）→ 小票权重 30%-90%（防御反转）
  回撤 > 20%           → 涨停权重减半 + 总仓位 ≤70%

小票侧：11因子ML | 周频 | 1000-3000名
涨停侧：5条件规则 | 日频 | Top-1000
```

## Web 模拟盘页面结构

```
http://localhost:8899/paper
┌──────────────────────────────┐
│  舞 v1.8  (universe=1000)    │  ← /api/paper-run/{run_id}?account_id={aid}
│  [汇总] [每日估值] [持仓表]    │
│  [权益+基准] [每日盈亏]        │
│  [待执行信号] [行业饼图]       │
└──────────────────────────────┘
```

## 策略配置

`config/paper_strategies.py` 定义每日跑数的策略参数：

```python
PAPER_STRATEGIES = [
    {"name": "舞", "version": "v1.8", "account_id": 15, "run_id": 2,
     "universe_size": 1000, "factor_mode": "all", ...},
]
```

- `factor_mode: "all"` → 全部因子（~83个，含日内+市场宽度）
- `universe_size: 1000` → v1.8 优化（v1.7 为 500），跨周期验证 Sharpe +23%

## v1.7 vs v1.8 差异

| | v1.7 | v1.8 |
|------|------|------|
| Universe | 500 | **1000** |
| Sharpe (2015-2025) | 1.19 | **1.46** |
| 最大回撤 | 17.4% | 24.7% |
| 其他参数 | 相同 | 相同 |

## 验证检查清单

每次修改后必须验证：

- [ ] `python scripts/run_daily_paper.py --dry-run --no-sync` 不报错
- [ ] Web 页面 v1.8 区加载正常
- [ ] 待执行信号(T+1)区域有数据
- [ ] 持仓表显示 entry_price ≠ 当前价（有浮盈浮亏）
- [ ] 权益曲线显示多天数据
- [ ] 每日盈亏柱状图有红绿柱
- [ ] 行业分布饼图有数据
- [ ] 汇总区显示总资产/日收益/累计/回撤
- [ ] v1.8 持仓表有数据，entry_price ≠ 当前价
- [ ] `python -m pytest tests/ -q --ignore=tests/test_overfit_check.py --ignore=tests/test_regime.py` 全绿

## 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| PostgreSQL 连不上 | 电脑重启后 PG 没启动 | `pg_ctl -D /opt/homebrew/var/postgresql@18 start` |
| 待执行信号为空 | 脚本跑了多次，信号被提前执行 | 删掉当日信号重跑 |
| 纸面资产腰斩 | paper_daily_pnl 的日收益算错 | 用累计盈亏法重建 |
| 数据同步太慢 | 全量 5524 只 | 8 并发，每天只差 1 天，2-3 分钟 |
| 两个策略持仓一样 | 用了同一个模型 | 确保 factor_mode 不同 |
| 市值数据不足1年 | AKShare Baidu API 限制 | 长区间回测用 `--mcap-proxy`（隐含股本估算） |
| 涨停侧没信号 | 市场弱势涨停股少 | 放宽 `--min-conditions 3` 或等市场回暖 |

## 回测统一参数

所有回测必须引用 `config/settings.py:TradingConfig`，不得硬编码：

| 参数 | 值 | 说明 |
|---|---|---|
| UNIVERSE_SIZE | 1000 | v1.8 优化 (v1.7=500)，跨周期 Sharpe +23% |
| INITIAL_CASH | 1,000,000 | 100万本金 |
| COMMISSION | 0.00009 | 万0.9 佣金（买卖双向） |
| STAMP_DUTY | 0.0005 | 万5 印花税（卖出单向） |
| SLIPPAGE | 0.001 | 0.1% 滑点 |
| NDROP_N | 2 | NDrop 每次替换最差2只 |
| STOP_LOSS_PCT | 0.08 | 个股止损-8% |
| MAX_DD_LIMIT | 0.25 | 组合最大回撤-25%清仓 |

选股管线：预测打分 → 排雷过滤(8项检查，允许≤3项违规) → NDrop剔除最差 → Top-N选股 → 等权持有

风控管线：个股止损-8% → 组合回撤-20%减仓 → 组合回撤-25%清仓 → 指数15日跌超12%空仓
