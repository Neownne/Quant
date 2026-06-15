# 项目规范

## 铁律

### 回测

| # | 规则 | 原因 |
|---|------|------|
| 1 | **涨停回测用管线** `python scripts/run_backtest_pipeline.py --start X --top-n N --label NAME` | 管线自动处理信号深度、mcap-proxy、exec-close |
| 2 | **小市值回测用** `python scripts/bt_small_cap.py --start X --top-n N` | 独立脚本，向量化回测 |
| 3 | **不改 bt_backtest.py 交易逻辑** | 已验证通过 |
| 4 | **新增策略写独立脚本** | 每个策略一个 py 文件，自包含回测逻辑 |

### 数据与因子

| # | 规则 | 原因 |
|---|------|------|
| 5 | **新增因子放 `factors/limit_up.py`** | 123因子统一注册在 `factors/__init__.py` |
| 6 | **因子验证用** `python scripts/validate_factors.py --start YYYY-MM-DD` | 向量化IC计算，支持滚动窗口 |
| 7 | **信号级特征用** `python scripts/featurize_signals.py` | 5分钟计算68维特征，比全市场算因子快10x |

### 交割单

| # | 规则 | 原因 |
|---|------|------|
| 8 | **文件名含日期+标签**：`trades_{策略}_{topn}_{start}_{end}_{label}.csv` | 永不覆盖，可追溯 |
| 9 | **无未来函数**：所有因子只用 ≤T 日数据 | shift(1)/rolling()/pct_change() 安全；shift(-1) 禁用 |
| 10 | **日终自洽**：总资产=现金+持仓市值，现金≥0 | 每笔交易记录当前现金字段 |

### 实验

| # | 规则 | 原因 |
|---|------|------|
| 11 | **实验室用** `python scripts/run_lab.py run --all --start 2020-01-01` | 批量跑变体 |
| 12 | **新变体只写 JSON** | `lab/variants/` 下编辑 |

### 行为

| # | 八荣八耻 | 具体体现 |
|---|---------|---------|
| 13 | 认真查询，不猜测接口 | 改代码前先 grep/read 确认函数签名 |
| 14 | 复用现有，不新造接口 | 能用 `load_daily_data()` 就别写新 SQL |
| 15 | 主动验证，不跳过测试 | 改完跑回测，检查交割单自洽 |
| 16 | 遵循规范，不破坏架构 | 策略脚本放 `scripts/`，公共逻辑放 `strategies/` |
| 17 | 承认无知，不假装理解 | 不确定的参数查 `config/settings.py` |
| 18 | 谨慎重构，不盲目修改 | 改 bt_backtest.py 前确认是真正的 bug |

---

## 当前策略矩阵

| 策略 | 脚本 | 状态 | 定位 |
|------|------|------|------|
| 小市值反转 | `bt_small_cap.py` | **主力** | 底仓，2025年+87% |
| 妖股规则 | `bt_yaogu.py` | 卫星 | 6规则评分≥6，大涨率33.7% |
| 涨停 Top-N | `run_backtest_pipeline.py` | 维护中 | 旧管线，-99.9% |
| 武器库 | `run_arsenal.py` | **日常** | 行业热力图+信号扫描 |

## 妖股规则速查

```
6规则评分（在涨停日评估）:
  +3  一字板 (lu_is_yiziban > 0)
  +2  低振幅 < 8% (lu_amplitude < 0.08)
  +1  缩量板 (lu_vol_intensity < 1.5)
  +1  非量能极值 (lu_volume_climax < 0.8)
  +1  连板 ≥ 2 (lu_streak >= 2)
  +1  缩量整理 ≥ 1天 (low_vol_streak >= 1)

score ≥ 6 → 等待首次非涨停日 → T+N日收盘买入
持有至: MA20趋势破坏 或 自适应止损(max(8%, 波动率×2))
```

## 因子库速查

```python
from factors import ALL_FACTORS  # 123个因子

# 涨停专用 29个
from factors.limit_up import LIMIT_UP_FACTORS

# 板块/全市场聚合
from factors.limit_up import compute_sector_lu_factors, compute_market_lu_extra

# IC计算
from factors.monitor import compute_rank_ic, compute_ic_series, compute_ic_summary

# 因子筛选
from factors.screening import filter_factors_by_ic, select_orthogonal_factors
```

## ML 探索结论

经过分类→回归→自适应→板块轮动全链路实验，核心结论：

1. **纯日线 OHLCV 技术因子对 A 股中期收益预测力有限**，IC 天花板 ≈ -0.10
2. **妖股规则是唯一有效产出**：不是预测收益率，而是从涨停股中筛选高质量形态
3. **ML 不能替代 alpha 来源**：XGBoost分类PR-AUC=0.19、回归R²=0.04，均无法产生持续正收益
4. **板块级信号比个股级更可靠**：155只涨停的板块 vs 单只涨停股，前者更有信息量

## 快速参考

```bash
# ── 每日 ──
python scripts/run_daily_paper_auto.py                    # 数据同步+模拟盘
python scripts/run_arsenal.py                             # 武器库面板

# ── 小市值反转（主力）──
python scripts/bt_small_cap.py --start 2020-01-01 --top-n 10

# ── 妖股规则 ──
python scripts/bt_yaogu.py --start 2020-01-01 --top-n 5 --label test

# ── 涨停管线 ──
python scripts/run_backtest_pipeline.py --start 2020-01-01 --top-n 5 --label E4_baseline

# ── 因子验证 ──
python scripts/validate_factors.py --start 2025-01-01

# ── 信号特征 ──
python scripts/featurize_signals.py --signals data/signals/bt_signals_full.csv

# ── ML训练（实验性）──
python scripts/train_signal_quality.py --start 2020-01-01

# ── Web ──
python -m uvicorn web.main:app --host 0.0.0.0 --port 8899
# → http://localhost:8899/backtest

# ── 数据库 ──
pg_ctl -D /opt/homebrew/var/postgresql@18 start
```

## 数据覆盖

| 数据表 | 起始 | 说明 |
|--------|------|------|
| stock_daily | 2015-01-05 | OHLCV + 换手率 |
| stock_daily_extra | 2015-01-05 | 市值（含隐含股本估算） |
| stock_mcap_proxy | — | 隐含股本 |
| index_daily | — | 主要指数日线 |
| concept_board | — | 概念板块映射 |
| concept_stock | — | 概念板块成分股 |

## 回测统一参数

| 参数 | 值 |
|------|-----|
| INITIAL_CASH | 1,000,000 |
| COMMISSION | 0.00009（万0.9） |
| STAMP_DUTY | 0.0005（万5，仅卖出） |
| SLIPPAGE | 0.001（0.1%） |
| STOP_LOSS_PCT | 0.08（策略可用自适应止损） |

## 已知问题（不要修）

| 问题 | 原因 |
|------|------|
| 小市值反转交割单 111 天总资产≠现金+市值 | sell 条目未统一到手工跟踪口径 |
| gen_signals `--lu-lookback` vs `--limit-up-lookback` 不一致 | 已在 `lab/variant.py` key_map 映射 |
| 妖股规则在非投机市中不出信号 | 市场风格依赖，非 bug |
