# Quant — A 股量化交易系统

> 最后更新：2026-06-15 (v6.0 小市值反转 + 管线修复 + 策略实验室)  
> GitHub：[Neownne/Quant](https://github.com/Neownne/Quant)

---

## 当前策略

| 策略 | 脚本 | 角色 | 2025年 | 2020-2026 |
|------|------|------|--------|-----------|
| **小市值反转** | `bt_small_cap.py` | **主力** | **+87.2%** | +2.5% |
| 涨停 Top-N | `run_backtest_pipeline.py` | 卫星 | -61.2% | -99.9% |

方向：小市值反转打底仓，涨停精选做增强。

---

## 快速启动

```bash
source .venv/bin/activate
pg_ctl -D /opt/homebrew/var/postgresql@18 start

# 小市值反转（主力策略）
python scripts/bt_small_cap.py --start 2020-01-01 --top-n 10

# 涨停策略回测
python scripts/run_backtest_pipeline.py --start 2025-01-01 --top-n 5 --label E4

# 策略实验室（批量搜索+回测+评分）
python scripts/run_lab_forever.py --start 2020-01-01 --parallel 2

# Web
python -m uvicorn web.main:app --host 0.0.0.0 --port 8899
# → http://localhost:8899/backtest
```

---

## 小市值反转策略

**逻辑**：全 A 股市值最小的 100 只 → 近 20 日跌最多的 10 只 → 等权买入 → 每 5 天调仓

**风控**：个股 -8% 止损 | 组合 -25% 回撤熔断 | 排除 ST/科创板/次新股

| 年 | 2020 | 2021 | 2022 | 2023 | 2025 |
|----|------|------|------|------|------|
| 收益 | -14.3% | **+59.5%** | -24.5% | -1.0% | **+87.2%** |

---

## 涨停策略

4 条件筛选 → 按涨停次数排序 → T 日收盘买入 → 排名顺延（涨停跳过）→ -8% 止损

回测引擎：backtrader + coc=True + `set_shortcash(True)`，10 项日终校验  
交割单：`总资产 = 现金 + 持仓市值` 自洽，含入场日期+当前现金

---

## 目录结构

```
quant/
├── README.md / CLAUDE.md
├── config/settings.py            # 配置中心
├── data/
│   ├── db.py / loader.py         # 数据库 + 参数化 SQL
│   ├── sync.py / fetcher.py      # 数据同步
├── strategies/limit_up/          # 涨停策略逻辑
├── scripts/
│   ├── bt_small_cap.py           # 小市值反转
│   ├── bt_backtest.py            # 涨停回测引擎
│   ├── run_backtest_pipeline.py  # 涨停一键管线
│   ├── gen_signals.py            # 涨停信号生成
│   ├── run_lab_forever.py        # 策略实验室持续循环
│   ├── run_lab.py                # 实验室 CLI
│   ├── run_daily_paper_lu.py     # 每日模拟盘
│   └── check_data_integrity.py   # 数据校验
├── lab/                          # 策略实验室
│   ├── variant.py / runner.py / judge.py
│   ├── grid.py / searcher.py
│   ├── ml_runner.py / sector_runner.py
│   └── variants/                 # 32个变体 JSON
├── web/                          # FastAPI + HTMX + ECharts
├── archive/                      # 冷宫：ML/大小票/RL
└── tests/
```

---

## 回测统一参数

| 参数 | 值 |
|------|-----|
| 本金 | 1,000,000 |
| 佣金 | 万 0.9（双向） |
| 印花税 | 万 5（卖出单向） |
| 滑点 | 0.1% |
| 个股止损 | -8% |

---

## 变更记录

### v6.0 — 小市值反转 + 管线修复 (2026-06-15)
- 新增 `bt_small_cap.py`：小市值反转策略，2025年 +87.2%
- 涨停管线交割单修复：`_running_total/_running_cash` 统一口径，同日卖出→买入→持仓完全连贯
- `set_shortcash(True)` 解决 coc 下买单静默拒绝 → 持仓消失 bug
- 管线文件名含日期区间+策略标签：`trades_top{N}_{start}_{end}_{label}.csv`
- 策略实验室 32 变体 + 6 维评分 + 三档判定

### v5.0 — 回测引擎重构 + 市值固化 (2026-06-14)
- 回测引擎 `bt_backtest.py` 重构：T 日收盘 coc 执行 + 排名顺延 + 日终 10 项校验
- 市值数据固化：`stock_mcap_proxy` 表 + 947 万行估算数据
- 涨跌停判断板别感知（主板 10%，科创/创业 20%）
- 新增 `run_backtest_pipeline.py` 一键回测管线
