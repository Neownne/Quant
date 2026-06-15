# 项目规范

## 铁律

### 回测

| # | 规则 | 原因 |
|---|------|------|
| 1 | **涨停回测用管线** `python scripts/run_backtest_pipeline.py --start X --top-n N --label NAME` | 管线自动处理信号深度、mcap-proxy、exec-close。手动拼参数必出错 |
| 2 | **小市值回测用** `python scripts/bt_small_cap.py --start X --top-n N` | 独立脚本，不依赖 backtrader，向量化回测更快 |
| 3 | **不改 bt_backtest.py 交易逻辑** | 已验证通过。`_running_total/_running_cash` 统一口径 + `set_shortcash(True)` 是正确的 |
| 4 | **新增策略写独立脚本** | 不做插件式架构。每个策略一个 py 文件，自包含回测逻辑 |

### 实验

| # | 规则 | 原因 |
|---|------|------|
| 5 | **实验室用** `python scripts/run_lab.py run --all --start 2020-01-01` | 批量跑变体，输出到统一 DB |
| 6 | **新变体只写 JSON** | `lab/variants/` 下编辑，不改任何 py 文件 |

### 行为

| # | 八荣八耻 | 具体体现 |
|---|---------|---------|
| 7 | 认真查询，不猜测接口 | 改代码前先 grep/read 确认函数签名 |
| 8 | 复用现有，不新造接口 | 能用管线就别写新脚本；能用 `load_daily_data()` 就别写新 SQL |
| 9 | 主动验证，不跳过测试 | 改完跑回测，检查交割单自洽（总资产=现金+市值，现金≥0） |
| 10 | 遵循规范，不破坏架构 | 策略脚本放 `scripts/`，公共逻辑放 `strategies/`，实验放 `lab/` |
| 11 | 承认无知，不假装理解 | 不确定的参数查 `config/settings.py`，不确定的结果问用户 |
| 12 | 谨慎重构，不盲目修改 | 改 bt_backtest.py 前确认是真正的 bug，不是表面现象 |

---

## 当前维护的策略

| 策略 | 脚本 | 状态 | 2020-2026 表现 |
|------|------|------|---------------|
| 涨停 Top-N | `run_backtest_pipeline.py` → `bt_backtest.py` | 维护中 | -99.9%，Sharpe -1.06 |
| 小市值反转 | `bt_small_cap.py` | **研究中** | +2.5%（2025年+87%，其他年波动大） |

未来方向：小市值反转打底 + 涨停精选卫星（待回测验证）。

---

## 快速参考

```bash
# ── 涨停策略 ──
python scripts/run_backtest_pipeline.py --start 2020-01-01 --top-n 5 --label E4_baseline
# → trades_top5_20200101_20260614_E4_baseline.csv

# ── 小市值反转 ──
python scripts/bt_small_cap.py --start 2020-01-01 --top-n 10
# → trades_sc_10_20200101_20260614.csv

# ── 实验室 ──
python scripts/run_lab.py run --all --start 2020-01-01    # 批量跑变体
python scripts/run_lab.py report --start 2020-01-01       # 排名报告
python scripts/run_lab_forever.py --start 2020-01-01 --parallel 2  # 持续循环

# ── 数据/模拟 ──
python scripts/run_daily_paper_auto.py                    # 每日自动化
python scripts/check_data_integrity.py                    # 数据检查

# ── Web ──
python -m uvicorn web.main:app --host 0.0.0.0 --port 8899
# → http://localhost:8899/backtest

# ── 数据库 ──
pg_ctl -D /opt/homebrew/var/postgresql@18 start
```

## 数据覆盖

| 数据表 | 起始 | 行数 | 说明 |
|--------|------|------|------|
| stock_daily | 2015-01-05 | ~2779天 | 日均 3922 只 |
| stock_daily_extra | 2015-01-05 | ~1155万 | 市值（含隐含股本估算） |
| stock_mcap_proxy | — | 5505只 | 隐含股本 |

## 回测统一参数

| 参数 | 值 |
|------|-----|
| INITIAL_CASH | 1,000,000 |
| COMMISSION | 0.00009（万0.9） |
| STAMP_DUTY | 0.0005（万5，仅卖出） |
| SLIPPAGE | 0.001（0.1%） |
| STOP_LOSS_PCT | 0.08 |

## 涨停回测管线

```bash
python scripts/run_backtest_pipeline.py --start 2020-01-01 --top-n 5 --label E4_baseline
```

文件名格式: `trades_top{N}_{start}_{end}_{label}.csv`  
管线自动: 信号深度 `max(N*4,20)` + `--mcap-proxy` + `--exec-close`  
预热: 回测开始前 90 天数据给 backtrader 均线预热  
交割单: 卖出→买入→持仓三行统一口径，`总资产=现金+持仓市值` 自洽

## 小市值反转策略

```bash
python scripts/bt_small_cap.py --start 2020-01-01 --top-n 10
```

逻辑: 市值最小100只 → 近20日跌最多 → 买10只 → 周频调仓  
风控: 个股-8%止损 + 组合-25%熔断  
不依赖 backtrader，向量化回测

分年表现:
| 2020 | 2021 | 2022 | 2023 | 2025 |
|------|------|------|------|------|
| -14.3% | **+59.5%** | -24.5% | -1.0% | **+87.2%** |

## 实验室 (lab/)

32 个变体覆盖: 筛选参数(8) + 涨停阈值(2) + 评分增强(4) + 过滤(2) + 执行机制(10) + 组合(3)

```
lab/variant.py         策略变体数据结构
lab/runner.py          批量编排（复用管线 subprocess）
lab/judge.py           6维复合评分 + 三档判定
lab/grid.py            网格变体生成
lab/searcher.py        多轮搜索模板
lab/ml_runner.py       88因子→XGBoost→三窗口
lab/sector_runner.py   行业动量轮动
```

结果写入 `backtest_results`，Web UI 直接查看。

## 代码规范

- SQL 必须参数化: `code = ANY(:codes)`
- 涨停公共逻辑放 `strategies/limit_up/`
- 新增功能优先复用 `data/loader.py`

## 已知问题（不要修）

| 问题 | 原因 |
|------|------|
| 小市值反转交割单 111 天总资产≠现金+市值 | sell 条目未统一到手工跟踪口径（待修，非紧急） |
| gen_signals `--lu-lookback` vs `--limit-up-lookback` 不一致 | 已在 `lab/variant.py` key_map 映射 |
