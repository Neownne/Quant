# 项目规范

## 铁律 —— 绝对禁止的操作

| # | 规则 | 原因 |
|---|------|------|
| 1 | **跑回测必须用管线** `python scripts/run_backtest_pipeline.py --start X --top-n N` | 管线自动处理信号深度（`max(N*4,20)`），手动调 gen_signals 的 --top-n 会导致候选池太浅、全涨停日买不到票 |
| 2 | **不得手动拼接 gen_signals + bt_backtest** | 参数传递容易出错（--top-n 深度、--mcap-proxy、--exec-close），管线已封装 |
| 3 | **不得修改 bt_backtest.py 交易逻辑** | coc 现金跟踪公式已验证正确，重复"修复"会引入新 bug。卖出回款公式 `available_cash = broker.getcash() + sell_proceeds` 是 coc 下唯一正确做法 |
| 4 | **策略实验用 lab** `python scripts/run_lab_forever.py` | 不要手动逐个跑变体。lab 负责：搜索→变体生成→批量回测→评分→报告 |
| 5 | **新增变体只改 variant JSON** | 不要直接改 bt_backtest.py 参数。在 `lab/variants/` 里加 JSON，lab 会自动跑 |

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
# 一键回测管线（信号生成 → 回测 → CSV → Web）
python scripts/run_backtest_pipeline.py --start 2025-01-01 --top-n 5

# 生成涨停信号（供回测用，市值数据已固化到 DB，无需 --mcap-proxy）
python scripts/gen_signals.py --start 2020-01-01 --end 2026-06-12 --top-n 5

# 涨停策略回测
python scripts/bt_backtest.py --start 2025-01-01 --end 2026-06-12 --top-n 5 \
    --signals data/signals/bt_signals.csv --exec-close

# 每日自动化（数据同步 + 涨停模拟盘 + ETF监控）
python scripts/run_daily_paper_auto.py

# 单独跑涨停模拟盘
python scripts/run_daily_paper_lu.py --date 2026-06-12 --no-sync

# 数据质量检查
python scripts/check_data_integrity.py

# Web
python -m uvicorn web.main:app --host 0.0.0.0 --port 8899
# → http://localhost:8899/paper     涨停模拟盘
# → http://localhost:8899/backtest  涨停回测
# → http://localhost:8899/etf       ETF监控

# 数据库
pg_ctl -D /opt/homebrew/var/postgresql@18 start
pg_ctl -D /opt/homebrew/var/postgresql@18 stop
```

## 数据覆盖

| 数据 | 起始 | 最新 | 说明 |
|------|------|------|------|
| stock_daily | 2015-01-05 | 至今 | 2779 个交易日，日均 3922 只 |
| stock_daily_extra | 2015-01-05 | 至今 | 1155 万行市值（含隐含股本估算） |
| stock_mcap_proxy | — | — | 5505 只股票的隐含股本，用于估算历史市值 |

## 项目范围

当前项目仅维护 **涨停策略 + ETF监控**。ML选股、大小票切换、小市值 alpha、RL 等非涨停策略已移入 `archive/` 冷宫，未来如需恢复请阅读 [archive/README.md](archive/README.md)。

## 涨停策略 E4（当前最优）

### 4条件筛选（全部通过）

| # | 条件 | 参数 | 说明 |
|---|------|------|------|
| 1 | 市值 | 30–500 亿 | 中小盘，宽区间 |
| 2 | 股价 | 5–63 元 | 过滤仙股和高价股 |
| 3 | 均线 | MA5 > MA10 | 短线多头排列 |
| 4 | 涨停次数 | 近20日 >1 次 | 日收益 ≥9%，捕获动量 |

**去跌停**: 删除「近10日无跌停」条件。回测验证: Sharpe 6.36 vs 含跌停 5.28，5/7年更优。

### 关键研究发现

| 实验 | 结论 |
|------|------|
| 跌停过滤 (含 vs 去) | **去跌停更好**。仅在高波动年(vol>18%)含跌停更优，多数年份去跌停大幅跑赢 |
| 自适应跌停 (波动率阈值) | **不可行**。波动率是滞后指标，总是在崩盘时切到保守模式、反弹时切回激进，whipsaw 导致回撤+10pp |
| Gap 开盘过滤 | **损害收益**。A股动量股常低开高走，跳过低开票错过盘中反弹 |
| 逃顶方案 (移动止盈/MA5/持有天数) | **全部降低收益**。策略盈利靠少数大赢家充分奔跑，中途截断=自断财路 |
| 黑名单 | **已停用**。去跌停 + 8%止损已足够排雷 |

### 模块位置

| 模块 | 文件 | 说明 |
|------|------|------|
| 数据加载 | `data/loader.py` | 日线/市值/价格查询，含隐含股本自动回退 |
| 选股 | `strategies/limit_up/base.py` | `LimitUpParams` + `run_screening` |
| 执行 | `strategies/limit_up/execution.py` | T日收盘主路径 + T+1开盘顺延（涨跌停封板时） |
| 净值 | `strategies/limit_up/pnl.py` | 日净值、回撤、现金 |
| 信号生成 | `scripts/gen_signals.py` | 批量生成 CSV 信号，含评分增强 |
| 回测引擎 | `scripts/bt_backtest.py` | backtrader + coc=True，T日收盘执行，10项日终校验 |
| 回测管线 | `scripts/run_backtest_pipeline.py` | 一键：信号→回测→CSV→Web |
| 模拟盘 | `scripts/run_daily_paper_lu.py` | 每日模拟盘入口 |
| 数据完整性 | `scripts/check_data_integrity.py` | 各表最新日期/覆盖率/缺失检查 |

## 模拟盘 run_id 分配

| run_id | account_id | 策略 | 脚本 |
|--------|-----------|------|------|
| 1 | 1 | 涨停 Top-5 (4条件去跌停) | run_daily_paper_lu.py |

## 回测统一参数

| 参数 | 值 | 说明 |
|---|---|---|
| INITIAL_CASH | 1,000,000 | 100万本金 |
| COMMISSION | 0.00009 | 万0.9 佣金（买卖双向） |
| STAMP_DUTY | 0.0005 | 万5 印花税（卖出单向） |
| SLIPPAGE | 0.001 | 0.1% 滑点 |
| STOP_LOSS_PCT | 0.08 | 个股止损-8% |
| LIMIT_UP_PCT | 0.09 | 涨停阈值 |
| LIMIT_DOWN_PCT | -0.09 | 跌停阈值 |

选股管线：4条件筛选 → 按涨停次数排序 → 排名顺延（涨停跳过，下一位补上）
风控管线：个股止损-8% → 跌停封死顺延 → 日终 10 项校验
执行模型：
  - 主路径 T 日收盘价成交（coc=True，模拟 14:50 实时行情→收盘下单）
  - 涨跌停封板顺延到 T+1 开盘
  - 候选 Top-5 买不到则顺延到 rank 6/7/8…直到填满
回测输出：固定格式 CSV（含股数/卖出原因/每日持仓快照）+ Web 展示

## CSV 信号导出

每日模拟盘运行后自动导出到 `data/paper_signals/signals_YYYY-MM-DD.csv`：
列: 数据日期, 排名, 股票代码, 股票名称, 评分, 收盘价, 近交易日涨跌幅(%)

## 代码规范

- 所有 SQL 必须参数化，禁止字符串拼接 `IN (...)`。PostgreSQL 中统一用 `code = ANY(:codes)`。
- 涨停策略公共逻辑必须放在 `strategies/limit_up/` 中，避免 `run_daily_paper_lu.py` 和 `gen_signals.py` 重复实现。
- 交易成本必须计入模拟盘现金更新。
- 新增功能前优先复用 `data/loader.py` 中的工具函数。

## 策略实验室 (lab/)

```bash
# 持续优化循环（搜索→回测→评判→循环）
python scripts/run_lab_forever.py --start 2020-01-01 --parallel 2

# 列出所有变体
python scripts/run_lab.py list

# 生成排名报告
python scripts/run_lab.py report
```

三管线：涨停策略变体（每轮）+ 行业轮动（每3轮）+ ML因子优化（每5轮）

新增策略变体：编辑 `lab/variants/` 下的 JSON 文件，不要改 bt_backtest.py。

## 已知问题（不要修）

| 问题 | 原因 | 为什么不动 |
|------|------|-----------|
| 交割单跨日总资产跳变 | coc 模式下 broker 持仓归零但现金 T+1 才到账 | 这是 backtrader coc 的固有行为，改成手工跟踪会让现金漂移更严重 |
| 买入当天现金偶尔为负（-500 以内） | NET_SELL 公式与 broker 实际扣费差 <0.02% | 不影响策略判断，修了会引入更大的漂移 bug |
| gen_signals 的 `--lu-lookback` vs `--limit-up-lookback` 参数名不一致 | gen_signals 用 DEFAULTS 的 `limit_up_lookback` 自动生成参数名 | 已在 `lab/variant.py` 的 `key_map` 中映射，不要改 gen_signals 的参数名 |

| 问题 | 原因 | 解决 |
|------|------|------|
| PostgreSQL 连不上 | PG 没启动 | `pg_ctl -D /opt/homebrew/var/postgresql@18 start` |
| 待执行信号为空 | 脚本跑了多次 | 删掉当日信号重跑 |
| 当天买卖同一只 | T+0 fallback bug (已修复) | 现在无次日数据会跳过执行 |
| 市值数据不足1年 | AKShare API限制 | 长区间回测用 `--mcap-proxy` |
| 模拟盘无盈利 | 两周太短 + 大盘下跌 | 策略靠长时间运行中抓翻倍票盈利 |
