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
# 每日模拟盘
python scripts/run_daily_paper.py                    # 数据同步 + v1.8 策略执行
python scripts/run_daily_paper.py --no-sync           # 跳过同步
python scripts/run_daily_paper.py --strategies v1.5   # 只跑指定版本

# 回测
python scripts/run_ml_backtest.py --strategy 舞        # v1.8 (universe=1000)
python scripts/run_ml_backtest.py --strategy 舞 --universe-size 500

# Web
python -m uvicorn web.main:app --host 0.0.0.0 --port 8899
# → http://localhost:8899/paper   模拟盘
# → http://localhost:8899/backtest 回测

# 数据库
pg_ctl -D /opt/homebrew/var/postgresql@18 -l /opt/homebrew/var/log/postgresql.log start
pg_ctl -D /opt/homebrew/var/postgresql@18 stop
```

## 架构总览

```
┌─ 数据层 ─────────────────────────────────────────────┐
│  PostgreSQL (localhost:5432)                          │
│  stock_daily (5524只, 2015~今)                        │
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
└──────────────────────────────────────────────────────┘
          │
          ▼
┌─ 模拟盘层 ──────────────────────────────────────────┐
│  PaperEngine: 信号→T+1执行→DB写入                     │
│  每日流程: 同步数据→训练→预测→执行信号→保存新信号     │
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
