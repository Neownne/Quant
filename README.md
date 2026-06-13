# Quant — A 股涨停策略量化交易系统

> 最后更新：2026-06-13 (v4.2 聚焦涨停策略 + ETF监控，非涨停策略已归档)  
> GitHub：[Neownne/Quant](https://github.com/Neownne/Quant)

本项目聚焦 **A 股涨停动量策略（E4）**：T 日收盘按 4 条件选股，T+1 开盘调仓，配合个股止损与风控，输出信号、回测与模拟盘。其他策略（ML选股、大小票切换、小市值 alpha、RL 等）已移入 `archive/` 冷宫，可按需恢复。

---

## 核心策略：涨停 E4（去跌停·4条件+8%止损）

### 4条件筛选（全部通过）

| # | 条件 | 参数 | 说明 |
|---|------|------|------|
| 1 | 市值 | 30–500 亿 | 中小盘，宽区间 |
| 2 | 股价 | 5–63 元 | 过滤仙股和高价股 |
| 3 | 均线 | MA5 > MA10 | 短线多头排列 |
| 4 | 涨停次数 | 近20日 >1 次 | 日收益 ≥9%，捕获动量 |

**去跌停**：删除「近10日无跌停」条件。回测验证 Sharpe 更优。

### 执行规则

```
T日收盘: 4条件筛选 → 按涨停次数排序 → Top-5 等权分配
T+1开盘: 买入(新入选) / 卖出(落选) / 持有(仍在前5)
风控: 个股-8%止损 / 组合-20%减仓 / 组合-25%清仓
交易成本: 佣金万0.9(双向) + 印花税万5(卖出单向) + 滑点0.1%
```

**收盘价执行说明**：回测脚本 `bt_backtest.py` 默认使用收盘价执行，这是为后续接入 14:30–14:50 实时行情信号后，用收盘价直接成交做准备。当前模拟盘仍严格按 T+1 开盘价执行。

### 关键研究发现

| 实验 | 结论 |
|------|------|
| 跌停过滤 (含 vs 去) | **去跌停更好**。仅高波动年(vol>18%)含跌停更优，多数年份去跌停大幅跑赢 |
| 自适应跌停 (波动率阈值) | **不可行**。波动率滞后，总是崩盘时切保守、反弹时切激进(whipsaw) |
| 开盘Gap过滤 | **损害收益**。A股动量股常低开高走，跳过低开票错过盘中反弹 |
| 逃顶方案 (移动止盈/MA5/持有天数) | **全部降低收益**。策略靠少数大赢家充分奔跑盈利，中途截断=自断财路 |
| 黑名单 | **已停用**。去跌停 + 8%止损已足够排雷 |

---

## 架构概览

```
┌─────────────────────────────────────────────────────────┐
│           Web 监控层 (FastAPI + HTMX + ECharts)           │
│  行情看板 │ 涨停回测 │ 模拟盘 │ ETF监控 │ 数据状态        │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────────┐
│                  PostgreSQL (数据 + 模拟盘)                 │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────────┐
│              后台研究引擎 (脚本/定时任务)                    │
│  data/        数据同步、质量校验、加载工具                    │
│  strategies/limit_up/   涨停选股 + 执行 + 净值               │
│  scripts/     回测、信号生成、每日模拟盘、自动任务            │
└─────────────────────────────────────────────────────────┘
```

---

## 目录结构（精简后）

```
quant/
├── README.md / CLAUDE.md
├── archive/                     # 冷宫：ML/大小票/小市值/RL等非涨停策略
├── config/                      # 配置中心
├── data/                        # 数据层
│   ├── db.py, sync.py, fetcher.py, quality.py
│   └── loader.py                # 涨停策略通用数据加载
├── factors/                     # 因子库（保留）
├── strategies/limit_up/         # 涨停策略核心
│   ├── base.py                  # 4条件筛选
│   ├── execution.py             # T+1交易执行
│   └── pnl.py                   # 日净值/回撤
├── scripts/                     # 入口脚本
│   ├── bt_backtest.py           # backtrader涨停回测
│   ├── gen_signals.py           # 涨停信号预生成
│   ├── run_daily_paper_lu.py    # 涨停模拟盘
│   ├── run_daily_paper_auto.py  # 每日自动：同步+涨停模拟盘+ETF
│   └── run_etf_monitor.py       # ETF监控
├── web/                         # Web界面
└── tests/                       # 测试
```

---

## 快速启动

```bash
cd /Users/chenwan/Documents/quant && source .venv/bin/activate

# 数据库
pg_ctl -D /opt/homebrew/var/postgresql@18 start

# Web 界面
python -m uvicorn web.main:app --host 0.0.0.0 --port 8899
# → http://localhost:8899/paper     模拟盘
# → http://localhost:8899/backtest  涨停回测
# → http://localhost:8899/etf       ETF监控

# 生成涨停信号
python scripts/gen_signals.py --start 2025-01-01 --end 2026-06-12 --top-n 5

# 涨停策略回测（收盘价执行）
python scripts/bt_backtest.py --start 2025-01-01 --end 2026-06-12 --top-n 5 \
    --signals data/signals/bt_signals.csv --exec-close

# 每日模拟盘
cd /Users/chenwan/Documents/quant
python scripts/run_daily_paper_lu.py --date 2026-06-12 --no-sync

# 全自动（数据同步 + 涨停模拟盘 + ETF监控）
python scripts/run_daily_paper_auto.py
```

---

## 模拟盘 run_id 分配

| run_id | account_id | 策略 | 脚本 |
|--------|-----------|------|------|
| 1 | 1 | 涨停 Top-5 (E4去跌停) | run_daily_paper_lu.py |

每日运行后自动导出信号 CSV 到 `data/paper_signals/signals_YYYY-MM-DD.csv`。

---

## 冷宫（archive/）

非涨停策略代码已统一归档至 `archive/`，包括：

- ML选股回测与模拟盘（`run_ml_backtest.py`、`run_daily_paper_ml.py`）
- 大小票切换策略（`run_daily_paper_switch.py`、`small_cap/`）
- 小市值 alpha 回测
- RL 相关测试与模块
- 部分 ML/组合优化模块（`archive/models/`、`archive/portfolio/`）

详见 [archive/README.md](archive/README.md)。未来如需恢复，直接从 `archive/` 移回原路径并更新相关 `__init__.py` 即可。

---

## 变更记录

### v4.2 — 项目精简与涨停策略聚焦 (2026-06-13)

- 创建 `archive/` 冷宫，归档非涨停策略代码
- 新增 `data/loader.py` 与 `strategies/limit_up/` 公共模块
- 重构 `run_daily_paper_lu.py` / `gen_signals.py`，统一使用公共模块
- 修复模拟盘现金更新未扣交易成本 bug
- 统一涨跌停阈值为 9%
- 移除黑名单逻辑
- `bt_backtest.py` 默认本金改为 100 万
- Web 隐藏非涨停入口，保留涨停回测、模拟盘、ETF监控

### v4.1 — E4策略深化 + 模拟盘修复 (2026-06-12)

- 去跌停确认为默认
- 收盘价执行策略定位（为实时数据接入做准备）
- 模拟盘 T+1 执行修复

### v4.0 — E4 策略突破 (2026-06-10 ~ 2026-06-11)

- 去掉跌停条件 + 市值放宽到 30-500 亿 + 股价上限 63 元
- 回测 Sharpe 6.36 / 年化 354.6% / MaxDD -32.2%

（更早变更见 git log。）
