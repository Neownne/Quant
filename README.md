# Quant — A 股涨停策略量化交易系统

> 最后更新：2026-06-14 (v5.0 回测引擎重构 + 市值数据固化 + 日终校验)  
> GitHub：[Neownne/Quant](https://github.com/Neownne/Quant)

本项目聚焦 **A 股涨停动量策略**：T 日收盘 4 条件选股，排名顺延买入，涨跌停封板延至 T+1，8% 止损，回测引擎含 10 项日终校验。其他策略（ML 选股、大小票切换、小市值 alpha、RL 等）已移入 `archive/`。

---

## 核心策略：涨停 E4（去跌停·4条件+8%止损）

### 4条件筛选

| # | 条件 | 参数 | 说明 |
|---|------|------|------|
| 1 | 市值 | 30–500 亿 | 中小盘，宽区间 |
| 2 | 股价 | 5–63 元 | 过滤仙股和高价股 |
| 3 | 均线 | MA5 > MA10 | 短线多头排列 |
| 4 | 涨停次数 | 近20日 >1 次 | 日收益 ≥9%，捕获动量 |

### 执行模型

```
T日收盘: 4条件筛选 → 按涨停次数排序 → 排名顺延（涨停跳过，下位补上）
         ├─ 主路径 T日收盘价成交
         └─ 封板顺延到 T+1 开盘
风控: 个股-8%止损 + 跌停封死顺延
交易成本: 佣金万0.9(双向) + 印花税万5(卖出单向) + 滑点0.1%
```

回测使用 `coc=True`（Cheat-On-Close），T 日收盘价执行，模拟 14:50 实时行情 → 收盘下单的未来场景。涨跌停判断按板块区分（主板 ±10%，科创板/创业板 ±20%）。

### 关键研究发现

| 实验 | 结论 |
|------|------|
| 跌停过滤 | **去跌停更好** |
| 自适应跌停 | **不可行**（whipsaw） |
| 开盘Gap过滤 | **损害收益** |
| 逃顶方案 | **全部降低收益** |
| 黑名单 | **已停用** |

---

## 快速启动

```bash
source .venv/bin/activate
pg_ctl -D /opt/homebrew/var/postgresql@18 start

# 一键回测管线（信号→回测→CSV→Web）
python scripts/run_backtest_pipeline.py --start 2025-01-01 --top-n 5

# 单独生成信号 / 跑回测
python scripts/gen_signals.py --start 2020-01-01 --end 2026-06-12 --top-n 5
python scripts/bt_backtest.py --start 2025-01-01 --end 2026-06-12 --top-n 5 \
    --signals data/signals/bt_signals.csv --exec-close

# 每日模拟盘
python scripts/run_daily_paper_lu.py --date 2026-06-12 --no-sync

# Web
python -m uvicorn web.main:app --host 0.0.0.0 --port 8899
# → http://localhost:8899/backtest  涨停回测
# → http://localhost:8899/paper     模拟盘
# → http://localhost:8899/etf       ETF监控
```

---

## 数据覆盖

| 数据表 | 起始 | 行数 | 说明 |
|--------|------|------|------|
| stock_daily | 2015-01-05 | 2779 天 | A 股日线，日均 3922 只 |
| stock_daily_extra | 2015-01-05 | 1155 万 | 市值（含隐含股本估算） |
| stock_mcap_proxy | — | 5505 只 | 隐含股本，一次性固化 |

---

## 目录结构

```
quant/
├── README.md / CLAUDE.md
├── archive/                        # 冷宫：ML/大小票/小市值/RL等非涨停策略
├── config/settings.py              # 配置中心（阈值/成本/涨跌停板别感知）
├── data/
│   ├── db.py                       # 数据库 DDL + upsert
│   ├── sync.py                     # 数据同步编排
│   ├── fetcher.py                  # AKShare 数据获取
│   ├── loader.py                   # 参数化 SQL 数据加载（含隐含股本回退）
│   └── quality.py                  # 数据质量校验
├── strategies/limit_up/
│   ├── base.py                     # 4条件筛选
│   ├── execution.py                # T日收盘主路径 + T+1顺延
│   └── pnl.py                      # 日净值/回撤
├── scripts/
│   ├── run_backtest_pipeline.py    # 一键回测管线
│   ├── bt_backtest.py              # backtrader 回测引擎（10项日终校验）
│   ├── gen_signals.py              # 信号预生成
│   ├── run_daily_paper_lu.py       # 每日涨停模拟盘
│   ├── run_daily_paper_auto.py     # 每日自动化
│   └── check_data_integrity.py     # 数据完整性检查
├── web/                            # FastAPI + HTMX + ECharts
└── tests/                          # 单元测试
```

---

## 回测引擎

### 10 项日终校验

| # | 校验项 |
|---|--------|
| 1 | 现金 ≥ 0 |
| 2 | 总资产自洽（手动 vs broker < 1%） |
| 3 | 买入必须在当日信号列表中 |
| 4 | 买入不能是涨停板 |
| 5 | 卖出不能是跌停板 |
| 6 | T+0 禁止（同日不能又买又卖同一只） |
| 7 | 持仓跌破止损线未卖出 → 断言失败 |
| 8 | 空位 + 有可买候选 → 告警 |
| 9 | 持仓不在信号也不在卖出 → 告警（跌停除外） |
| 10 | 卖出必须有原因（止损/调仓） |

### 交割单格式

```
日期,操作,股票代码,股票名称,入场价,当前价/出场价,盈亏%,股数,入场日期,总资产
2025-01-02,买入,002137,实益达,8.68,,,23000,,1000000.0
2025-01-02,持仓,002137,实益达,8.68,8.68,0.0,23000,,1000000.0
2025-01-03,卖出(止损),002137,实益达,8.65,7.82,-9.6,23000,,1031950.11
```

每日按 卖出 → 买入 → 持仓 排序，卖出标注原因（止损/调仓）。

---

## 变更记录

### v5.0 — 回测引擎重构 + 市值固化 (2026-06-14)

- 回测引擎 `bt_backtest.py` 重构：T 日收盘 coc 执行 + 排名顺延 + 日终 10 项校验
- 市值数据固化：`stock_mcap_proxy` 表 + `stock_daily_extra` 追加 947 万行估算数据，覆盖 2015 至今
- 涨跌停判断板别感知（主板 10%，科创/创业 20%，四舍五入精确比价）
- 新增 `run_backtest_pipeline.py` 一键回测管线
- 交割单固定格式：含股数、卖出原因、每日持仓快照
- 回测结果自动写入 `backtest_results` 表，Web 直接展示
- 修复前导零丢失（深市代码 000xxx 被截断）、coc 现金递减、止损回款遗漏等 20+ bug

### v4.2 — 项目精简 (2026-06-13)

- 创建 `archive/` 冷宫，归档非涨停策略代码
- 统一使用 `strategies/limit_up/` 公共模块

（更早变更见 git log。）
