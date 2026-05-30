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

舞 v1.3 — 5状态自适应ML日频策略。详见 VERSION.md。

## 版本历史

### v1.3 (2026-05-30)
- 5状态市场检测 + Regime自适应调仓/仓位/止损
- 新增纯价格动量因子(price_mom_5/10/accel)
- 模拟盘V2: run_daily_paper.py 每日驱动
- 回测 CAGR 44.6%, Sharpe 1.34, MaxDD 25.8%

### v1.10 (2026-05-28)
- 股票池从 `ORDER BY code LIMIT 200` 改为全市场非ST选股（修复仅选深市股票偏差）
- 新增 `--universe-size N` 参数控制候选池大小（0=全市场，默认）
- 集成组合级风控：-20%减仓、-25%清仓（含10天冷静期）、指数大跌空仓
- 指数数据无条件加载，不再依赖 `--regime` 开关
- 排雷过滤（8项检查，允许≤3项违规）
- NDrop 增量调仓（K=15, N=2）
- 止损事件和风控事件记录到 backtest_results.metrics_json
- 修复 run_all_backtests.py 和 verify_paper_trading.py 中相同的股票池偏差
