# Quant — A 股量化交易系统

> 最后更新：2026-06-12 (v4.1 去跌停策略 + 模拟盘修复 + 多项回测研究)  
> GitHub：[Neownne/Quant](https://github.com/Neownne/Quant)

---

## 核心工作流

```
因子计算 → 模型训练 → 历史回测(防过拟合) → 模拟盘验证 → 归因反馈闭环
```

---

## 回测结果（2020-2026，6.5年）

> 统一参数：100 万本金、佣金万0.9、印花税万5（卖出单向）、滑点0.1%。
> 严格 Walk-Forward 验证。

| 策略 | Sharpe | 年化 | MaxDD | 调仓 |
|------|:---:|:---:|:---:|:---:|
| **涨停策略 E4（去跌停·4条件+8%止损）** | **6.36** | **354.6%** | **-32.2%** | 日频 |
| 涨停策略 E3（含跌停·5条件+8%止损） | 5.28 | 289.8% | -31.1% | 日频 |
| 大小票 v4.0（涨停+ML） | 1.39 | 54.7% | -46.8% | 混合 |
| 小市值 alpha v2.0 | 0.93 | 25.3% | -25.5% | 周度 |
| 舞 v1.85 | 0.72 | 15.0% | -26.5% | 自适应 |

---

## 涨停策略 E4（当前最优）

### 4条件筛选（全部通过）

| # | 条件 | 参数 | 说明 |
|---|------|------|------|
| 1 | 市值 | 30–500 亿 | 中小盘，宽区间 |
| 2 | 股价 | 5–63 元 | 过滤仙股和高价股 |
| 3 | 均线 | MA5 > MA10 | 短线多头排列 |
| 4 | 涨停次数 | 近20日 >1 次 | 日收益 ≥10%，捕获动量 |

**关键设计:** 删除了「近10日无跌停」条件。A股跌停多为情绪错杀，过滤掉会错过最强反弹。

### 执行规则

```
T日收盘: 4条件筛选 → 按涨停次数排序 → Top-5 等权分配
T+1开盘: 买入(新入选) / 卖出(落选) / 持有(仍在前5)
风控: 个股-8%止损 / 组合-20%减仓 / 组合-25%清仓
```

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
│  行情看板 │ 回测对比 │ 模拟盘(涨停+ML双栏) │ ETF监控      │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────────┐
│                  PostgreSQL (31张表)                      │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────────┐
│              后台研究引擎 (脚本/定时任务)                    │
│  数据同步 → 质量校验 → 因子计算 → 训练 → 回测              │
│                                                          │
│  涨停策略 E4: 4条件规则筛选, 中小盘动量, 日频, Top-5       │
│  大小票 v4.0: 涨停+ML 动态切换, CSI1000趋势自适应          │
│  ML智能选股: E4候选池 + GBRT预测重排                      │
└─────────────────────────────────────────────────────────┘
```

---

## 模拟盘 run_id 分配

| run_id | account_id | 策略 | 脚本 |
|--------|-----------|------|------|
| 1 | 1 | 涨停 Top-5 (E4去跌停) | run_daily_paper_lu.py |
| 2 | 2 | 大小票 v4.0 | run_daily_paper_switch.py |
| 3 | 3 | ML 智能选股 (GBRT) | run_daily_paper_ml.py |

每日运行后自动导出信号CSV到 `data/paper_signals/signals_YYYY-MM-DD.csv`。

---

## 目录结构

```
quant/
├── README.md
├── CLAUDE.md
├── .env / .gitignore / requirements.txt
│
├── config/
│   ├── settings.py              # 集中配置
│   └── blacklist.json           # 黑名单(已停用)
│
├── data/                        # 数据层
│   ├── db.py                    # DDL + 连接池 + upsert
│   ├── fetcher.py               # 数据获取 (AKShare/腾讯/新浪)
│   ├── sync.py                  # 批量同步 (多进程增量)
│   ├── quality.py               # 数据质量校验
│   └── paper_signals/           # 每日信号CSV导出
│
├── factors/                     # 因子层 (83个因子)
│   ├── engine.py                # FactorEngine
│   ├── alpha101.py / alpha191_*.py / custom.py
│   ├── fundamental.py / intraday_minute.py
│   └── screening.py             # IC门禁 + 正交筛选
│
├── models/                      # ML预测层
│   ├── dataset.py / trainer.py
│   ├── regime.py                # 5状态市场检测
│   └── tuning.py                # 阈值搜索 + Optuna调优
│
├── portfolio/                   # 组合优化层
│   ├── selector.py              # 选股 (Top-N/NDrop/过滤)
│   ├── allocator.py / risk.py
│   └── paper_engine.py          # 模拟盘引擎
│
├── small_cap/                   # 小市值alpha + 大小票切换
│   ├── backtest.py
│   ├── switch_backtest.py       # v1.0 (mom_20反转)
│   └── switch_backtest_v2.py    # v4.0 (涨停替代)
│
├── scripts/                     # 工具脚本
│   ├── run_daily_paper_auto.py          # 自动模拟盘(全策略)
│   ├── run_daily_paper_lu.py            # 涨停Top-5模拟盘
│   ├── run_daily_paper_ml.py            # ML智能选股模拟盘
│   ├── run_daily_paper_switch.py        # 大小票v4.0模拟盘
│   ├── backtest_limit_up_strategy.py    # 涨停策略回测
│   ├── scan_limit_up_strategy.py        # 涨停策略每日筛选
│   ├── run_ml_backtest.py               # 舞ML回测
│   ├── run_small_cap_backtest.py        # 小市值alpha回测
│   └── run_etf_monitor.py               # ETF三因子监控
│
├── web/                         # Web界面 (FastAPI + HTMX)
│   ├── main.py
│   ├── routes/ (api/dashboard/backtest/paper/data_status/factors/etf)
│   ├── static/ / templates/
│
├── tests/                       # 测试
└── docs/                        # 文档 & 研究
```

---

## 快速启动

```bash
cd /Users/chenwan/Documents/quant && source .venv/bin/activate

# 数据库
pg_ctl -D /opt/homebrew/var/postgresql@18 start

# Web 界面
python -m uvicorn web.main:app --host 0.0.0.0 --port 8899
# → http://localhost:8899/paper     模拟盘 (涨停+ML双栏)
# → http://localhost:8899/backtest   回测

# 涨停策略回测
python scripts/backtest_limit_up_strategy.py --start 2020-01-01 --mcap-proxy --top-n 5 --min-conditions 4 --exit-stop 0.08

# 每日模拟盘
python scripts/run_daily_paper_lu.py --date 2026-06-11 --no-sync    # 涨停
python scripts/run_daily_paper_ml.py --date 2026-06-11 --no-sync    # ML
```

---

## 变更记录

### v4.1 — E4策略深化 + 模拟盘修复 (2026-06-12)

- **去跌停确认为默认**: Sharpe 6.36 vs 含跌停 5.28，5/7年更优
- **回测研究**: 自适应跌停不可行(whipsaw)，Gap过滤损害收益，逃顶方案全部降低收益
- **模拟盘修复**: T+0 bug修复(严格执行T+1)、run_id冲突解决(ML=3)、日期回退bug修复
- **Web优化**: 信号排序(rank→score DESC)、黑名单股票灰显、总收益率改用初始本金计算
- **CSV导出**: 每日信号自动保存到 data/paper_signals/
- **黑名单停用**: 去跌停 + 8%止损已足够排雷

### v4.0 — E4 策略突破 (2026-06-10 ~ 2026-06-11)

- 去掉跌停条件 + 市值放宽到 30-500 亿 + 股价上限 63 元
- E4 vs E3: Sharpe 6.93 vs 3.75(+85%), MaxDD -21.0% vs -35.0%
- ML智能选股模拟盘: E4候选池 + GBRT启发式重排
- Web双栏对比

### v3.1 — 模拟盘自动化 (2026-06-09)

- 自动模拟盘脚本、5个bug修复、ETF监控扩展至1445只

### v3.0 — 涨停策略全面优化 (2026-06-07 ~ 2026-06-08)

- 独立回测+筛选脚本、修复隔夜缺口偏差、16方向批量优化、E3(8%止损)胜出
- Web回测页重构、模拟盘双策略并排

### v2.0 — 小市值alpha + 大小票平滑分配 (2026-06-04 ~ 2026-06-06)

- 11个原创因子、7处未来函数修复、弱势期大票反转发现、大小票v1.0

### v1.x — 管线搭建 (2026-05-22 ~ 2026-06-05)

- 数据模块、Alpha101/191因子库、ML管线(XGBoost/LGBM+Walk-Forward)
- 组合优化、模拟盘引擎、批量回测、过拟合检测
- 市场状态自适应、分钟因子、行业中性化、动态反馈闭环
