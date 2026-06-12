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
python scripts/run_daily_paper_auto.py                  # 全策略: 涨停+ML+大小票
python scripts/run_daily_paper_auto.py --strategy lu     # 只跑涨停

# 单独跑某个模拟盘
python scripts/run_daily_paper_lu.py --date 2026-06-11 --no-sync    # 涨停策略
python scripts/run_daily_paper_ml.py --date 2026-06-11 --no-sync    # ML智能选股
python scripts/run_daily_paper_switch.py --date 2026-06-11 --no-sync # 大小票v4.0

# 涨停策略回测
python scripts/backtest_limit_up_strategy.py --start 2020-01-01 --mcap-proxy --top-n 5 --min-conditions 4 --exit-stop 0.08  # E4(去跌停)
python scripts/backtest_limit_up_strategy.py --start 2020-01-01 --mcap-proxy --top-n 5 --min-conditions 4 --exit-stop 0.08 --adaptive-ld --ld-vol-lookback 20 --ld-vol-threshold 0.18  # 自适应跌停(不推荐)

# 数据同步
python -c "
from data.db import get_engine
from data.sync import sync_stock_daily, sync_daily_extra
e = get_engine()
sync_stock_daily(e, start_date='2026-06-12', workers=8)
sync_daily_extra(e, start_date='2026-06-12', workers=8)
"

# Web
python -m uvicorn web.main:app --host 0.0.0.0 --port 8899
# → http://localhost:8899/paper     模拟盘 (涨停+ML双栏)
# → http://localhost:8899/backtest   回测
# → http://localhost:8899/etf        ETF监控

# 数据库
pg_ctl -D /opt/homebrew/var/postgresql@18 start
pg_ctl -D /opt/homebrew/var/postgresql@18 stop
```

## 模拟盘 run_id 分配

| run_id | account_id | 策略 | 脚本 |
|--------|-----------|------|------|
| 1 | 1 | 涨停 Top-5 (4条件去跌停) | run_daily_paper_lu.py |
| 2 | 2 | 大小票 v4.0 | run_daily_paper_switch.py |
| 3 | 3 | ML 智能选股 (GBRT重排) | run_daily_paper_ml.py |

**注意**: run_id=1/2/3 各自独立，不可混用。黑名单已停用（去跌停策略表现更优）。

## 涨停策略 E4（当前最优）

### 4条件筛选（全部通过）

| # | 条件 | 参数 | 说明 |
|---|------|------|------|
| 1 | 市值 | 30–500 亿 | 中小盘，宽区间 |
| 2 | 股价 | 5–63 元 | 过滤仙股和高价股 |
| 3 | 均线 | MA5 > MA10 | 短线多头排列 |
| 4 | 涨停次数 | 近20日 >1 次 | 日收益 ≥10%，捕获动量 |

**去跌停**: 删除「近10日无跌停」条件。回测验证: Sharpe 6.36 vs 含跌停 5.28，5/7年更优。

### 关键研究发现

| 实验 | 结论 |
|------|------|
| 跌停过滤 (含 vs 去) | **去跌停更好**。仅在高波动年(2020/2024 vol>18%)含跌停更好，多数年份去跌停大幅跑赢 |
| 自适应跌停 (波动率阈值) | **不可行**。波动率是滞后指标，总是在崩盘时切到保守模式、反弹时切回激进，whipsaw 导致回撤+10pp |
| Gap 开盘过滤 | **损害收益**。A股动量股常低开高走，跳过低开票错过盘中反弹 |
| 逃顶方案 (移动止盈/MA5/持有天数) | **全部降低收益**。策略盈利靠少数大赢家充分奔跑，中途截断=自断财路。10%移动止盈是最温和的妥协(Sharpe仅降0.3) |
| 黑名单 | **已停用**。去跌停 + 8%止损已足够排雷 |

## Web 模拟盘页面

```
http://localhost:8899/paper
┌───────────────────────┐  ┌───────────────────────┐
│   🔥 涨停 Top-5       │  │   🤖 ML 智能选股       │
│   run=1 account=1     │  │   run=3 account=3     │
│   [汇总] [持仓]        │  │   [汇总] [持仓]        │
│   [权益曲线] [日盈亏]  │  │   [权益曲线] [日盈亏]  │
│   [待执行信号] [行业]  │  │   [待执行信号] [行业]  │
└───────────────────────┘  └───────────────────────┘
```

## 回测统一参数

| 参数 | 值 | 说明 |
|---|---|---|
| UNIVERSE_SIZE | 1000 | v1.8 优化 |
| INITIAL_CASH | 1,000,000 | 100万本金 |
| COMMISSION | 0.00009 | 万0.9 佣金（买卖双向） |
| STAMP_DUTY | 0.0005 | 万5 印花税（卖出单向） |
| SLIPPAGE | 0.001 | 0.1% 滑点 |
| STOP_LOSS_PCT | 0.08 | 个股止损-8% |
| MAX_DD_LIMIT | 0.25 | 组合最大回撤-25%清仓 |

选股管线：4条件筛选 → 按涨停次数排序 → Top-5选股 → 等权分配
风控管线：个股止损-8% → 组合回撤-20%减仓 → 组合回撤-25%清仓
执行规则：T+1 开盘价成交，无次日数据则跳过执行（严格A股T+1制度）

## CSV 信号导出

每日模拟盘运行后自动导出到 `data/paper_signals/signals_YYYY-MM-DD.csv`：
列: 数据日期, 排名, 股票代码, 股票名称, 评分, 收盘价, 近交易日涨跌幅(%)

## 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| PostgreSQL 连不上 | PG 没启动 | `pg_ctl -D /opt/homebrew/var/postgresql@18 start` |
| 待执行信号为空 | 脚本跑了多次 | 删掉当日信号重跑 |
| 当天买卖同一只 | T+0 fallback bug (已修复) | 现在无次日数据会跳过执行 |
| 两个策略持仓一样 | run_id 冲突 | 确保 run_id 各不同 (1/2/3) |
| 市值数据不足1年 | AKShare API限制 | 长区间回测用 `--mcap-proxy` |
| 模拟盘无盈利 | 两周太短 + 大盘下跌 | 策略靠长时间运行中抓翻倍票盈利 |
