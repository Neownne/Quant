# 项目规范

## Superpowers 工作流

在每次会话开始时，你必须：
1. 检查当前任务是否适用任何 skill（即使只有 1% 的可能性）
2. 如果适用，先加载 skill，再执行任何操作
3. 遵循 skill 中的检查清单和流程

## 可用 Skills 目录

项目 skills 位于 `.claude/skills/` 目录下：
- `using-superpowers` - 元技能（必须先加载）
- `brainstorming` - 需求澄清
- `writing-plans` - 制定计划
- `executing-plans` - 执行计划
- `test-driven-development` - TDD 流程
- `systematic-debugging` - 系统调试

## Skill 加载指令

当需要加载 skill 时，使用 Skill 工具调用对应的 skill 名称。

## 回测统一参数

所有回测必须引用 `config/settings.py:TradingConfig`，不得硬编码：

| 参数 | 值 | 说明 |
|---|---|---|
| INITIAL_CASH | 1,000,000 | 100万本金 |
| COMMISSION | 0.00009 | 万0.9 佣金（买卖双向） |
| STAMP_DUTY | 0.0005 | 万5 印花税（卖出单向） |
| SLIPPAGE | 0.001 | 0.1% 滑点 |
| REBALANCE_FREQ | 5 | 默认周度调仓 |
| NDROP_N | 2 | NDrop 每次替换最差2只 |
| STOP_LOSS_PCT | 0.08 | 个股止损-8% |
| MAX_DD_LIMIT | 0.25 | 组合最大回撤-25%清仓 |

选股管线：预测打分 → 排雷过滤(8项检查，允许≤3项违规) → NDrop剔除最差(保留K-N，替换N) → Top-N选股 → 等权持有

风控管线：每日检查持仓 → 个股止损-8%(硬止损) → 组合回撤-20%减仓 → 组合回撤-25%清仓(10天冷静期+重置peak) → 指数15日跌超12%空仓

## 策略

舞 v1.5 — 5状态自适应ML日频策略。详见 VERSION.md。

## 项目结构

| 目录 | 说明 |
|------|------|
| `config/` | 配置: settings.py(交易参数), sector_map.py(板块分类) |
| `data/` | 数据: db.py(表定义), sync.py(同步), fetcher.py(抓取) |
| `factors/` | 因子: engine.py, alpha101/191, market_breadth, sector_fear_greed 等 |
| `models/` | 模型: trainer.py(XGBoost+LGB), regime.py, sector_model.py 等 |
| `portfolio/` | 组合: selector.py(NDrop), risk.py(风控), paper_engine.py(模拟盘) |
| `rl/` | 强化学习: environment, models, predictor, sector_env(实验) |
| `scripts/` | 脚本: run_ml_backtest.py(回测), run_daily_paper.py(每日模拟) |
| `web/` | Web: FastAPI(routes/), Jinja2(templates/), ECharts(app.js) |
| `tests/` | 测试: pytest |
| `output/` | 产出: backtests/, etf/, reports/ |
| `docs/` | 文档 |

## 版本历史

### v1.5 (2026-06-01)
- 日内因子引入: 尾盘集合竞价强度、下午收益率、日内波动偏度
- IC筛选偶然选中更好因子组合，CAGR 提升至 53.2%
- 板块分类设施: CSI300成分股(index_constituent表)、红利/大盘/小盘分类
- 板块宽度特征: sector_breadth.py(15特征)、sector_fear_greed.py(恐贪指数)
- 实验性RL板块打分: rl/ 模块(MPS GPU可用)
- 回测 CAGR 53.2%, Sharpe 1.59, MaxDD 20.6%

### v1.4 (2026-05-30)
- 跌停顺延(防实盘跌停无法卖出) + 去OHLCV污染
- 模拟盘上线: run_daily_paper.py + 小市值 run_small_cap_paper.py
- 回测 CAGR 46.1%, Sharpe 1.40, MaxDD 24.8%

### v1.3 (2026-05-30)
- 5状态市场检测 + Regime自适应调仓/仓位/止损
- 新增纯价格动量因子(price_mom_5/10/accel)
- 回测 CAGR 44.6%, Sharpe 1.34, MaxDD 25.8%
