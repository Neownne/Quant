# Quant — A 股量化交易系统

> 最后更新：2026-06-16 (v7.0 妖股规则 + 因子库 + 武器库)  
> GitHub：[Neownne/Quant](https://github.com/Neownne/Quant)

---

## 当前策略

| 策略 | 脚本 | 状态 | 说明 |
|------|------|------|------|
| 小市值反转 | `bt_small_cap.py` | **主力** | +2.5%（2025年+87%） |
| 涨停 Top-N | `run_backtest_pipeline.py` | 维护中 | -99.9%，Sharpe -1.06 |
| 妖股规则 | `bt_yaogu.py` | 研究中 | 6规则评分，大涨率33.7% vs 基线12.1% |
| 武器库面板 | `run_arsenal.py` | **日常** | 每日行业热力图+信号扫描 |

方向：小市值反转打底 + 妖股规则增强 + 板块轮动择时（人工决策）。

---

## 快速启动

```bash
source .venv/bin/activate
pg_ctl -D /opt/homebrew/var/postgresql@18 start

# 每日数据同步 + 模拟盘
python scripts/run_daily_paper_auto.py

# 武器库面板（今日信号）
python scripts/run_arsenal.py

# 小市值反转
python scripts/bt_small_cap.py --start 2020-01-01 --top-n 10

# 涨停策略回测（管线）
python scripts/run_backtest_pipeline.py --start 2025-01-01 --top-n 5 --label E4

# 妖股规则回测
python scripts/bt_yaogu.py --start 2020-01-01 --top-n 5 --label test

# 因子 IC 验证
python scripts/validate_factors.py --start 2025-01-01

# Web
python -m uvicorn web.main:app --host 0.0.0.0 --port 8899
```

---

## 妖股规则策略

**核心发现**：6 条规则从涨停股中筛选高质量信号。一字板 + 低振幅 + 缩量板的组合，大涨（10日>20%）概率 33.7%，是基线（12.1%）的 2.8 倍。

### 6 条评分规则

| 权重 | 规则 | 大涨率 |
|------|------|--------|
| +3 | 一字板（lu_is_yiziban） | 33.7% |
| +2 | 低振幅 < 8%（lu_amplitude） | 23.1% |
| +1 | 缩量板（lu_vol_intensity < 1.5） | 20.1% |
| +1 | 非量能极值（lu_volume_climax < 0.8） | 19.4% |
| +1 | 连板 ≥ 2（lu_streak） | 19.2% |
| +1 | 缩量整理 ≥ 1天（low_vol_streak） | 18.8% |

signal：T日涨停评分 ≥ 6 → 等待首次非涨停日 → T+N日收盘买入 → 持有至 MA20/止损退出。

### 与旧涨停策略的关系

妖股规则 100% 被旧策略 Top-20 覆盖，但旧策略中只有 7% 是妖股高分。妖股规则是旧策略的精华提取器：

| 信号类型 | 10日均值收益 | >20%概率 |
|----------|------------|----------|
| 旧策略 Top-5 | -0.66% | 15.1% |
| 妖股 ≥ 6 | **+15.39%** | **31.6%** |
| 两者重叠 | **+26.43%** | **43.3%** |

### 局限性

只在小市值投机风格市场中有效（2020年妖股横行）。市场风格切换到基本面驱动时（2025年制造业牛市），规则信号几乎不出。**妖股规则应作为市场投机热度指标**，而非独立策略。

---

## 因子库（123因子）

在原有 94 因子上新增 29 个涨停专用因子（`factors/limit_up.py`）：

| 类别 | 新增 | 核心因子 |
|------|------|---------|
| 涨停模式 | 11 | lu_streak, lu_count_5d/20d/60d, lu_first_board, lu_freq_accel |
| 封板质量 | 8 | lu_seal_quality, lu_vol_intensity, lu_amplitude, lu_body_ratio |
| 首板蓄力 | 3 | pre_lu_vol_trend, pre_lu_ret_5d |
| 板型分类 | 3 | lu_is_yiziban, lu_is_strong_board, lu_board_strength |
| 资金流代理 | 6 | mfi_14, cmf_20, force_index, money_flow_pressure |
| 波动率结构 | 5 | vol_ratio_5_20, vol_asymmetry, vol_regime |
| 时序形态 | 5 | ma_convergence, volume_breakout, low_vol_streak |

板块共振 + 截面排名因子：`compute_sector_lu_factors()`, `compute_market_lu_extra()`

因子 IC 快速验证：`scripts/validate_factors.py`

---

## 策略武器库

`scripts/run_arsenal.py` — 每日运行，输出：
- 行业热力图（涨幅+涨停家数+上涨比）
- 妖股信号扫描（今日有无高分信号）
- 板块轮动建议（涨停潮行业识别）

```bash
python scripts/run_arsenal.py           # 今日信号
python scripts/run_arsenal.py --recent 20  # 近期策略表现
```

信号保存到 `data/arsenal/signals_YYYYMMDD.json`，不覆盖。

---

## ML 探索结论

经过分类→回归→规则提炼的完整迭代：

| 尝试 | 方法 | 结论 |
|------|------|------|
| XGBoost分类 | 预测>20%概率 | PR-AUC=0.19（基线0.15），区分度太弱 |
| XGBoost回归 | 预测实际收益率 | R²=0.04，Test IC=0.065 |
| 自适应趋势 | 滚动IC选因子 | -4.6%平盘，因子IC天花板-0.10 |
| 板块资金流 | 行业动量+宽度 | 信号无效，无法预测板块轮动 |

**核心结论**：纯日线 OHLCV 衍生技术因子对 A 股中期收益的预测力天花板很低。ML 不能替代 alpha 来源。

---

## 目录结构

```
quant/
├── README.md / CLAUDE.md
├── config/settings.py
├── data/
│   ├── db.py / loader.py / sync.py
│   ├── arsenal/            # 武器库信号输出
│   ├── factor_ic/           # 因子IC分析结果
│   ├── models/              # 训练好的ML模型
│   └── signals/             # 信号CSV
├── factors/
│   ├── __init__.py          # ALL_FACTORS (123个)
│   ├── limit_up.py          # 涨停专用因子 (29个)
│   ├── engine.py            # 因子计算引擎
│   ├── monitor.py           # IC监控
│   ├── screening.py         # 因子筛选
│   ├── market_breadth.py    # 全市场宽度
│   └── sector_breadth.py    # 板块宽度
├── strategies/limit_up/
├── scripts/
│   ├── bt_small_cap.py      # 小市值反转（主力）
│   ├── bt_backtest.py       # 涨停回测引擎
│   ├── run_backtest_pipeline.py
│   ├── gen_signals.py
│   ├── bt_yaogu.py          # 妖股规则回测 [NEW]
│   ├── bt_ml_signals.py     # ML信号回测 [NEW]
│   ├── bt_hybrid.py         # 混合策略回测 [NEW]
│   ├── bt_trend_adaptive.py # 自适应趋势 [NEW]
│   ├── featurize_signals.py # 信号级特征工程 [NEW]
│   ├── validate_factors.py  # 因子IC验证 [NEW]
│   ├── train_signal_quality.py # ML训练 [NEW]
│   ├── gen_signals_ml.py    # ML信号生成 [NEW]
│   ├── run_arsenal.py       # 武器库面板 [NEW]
│   ├── run_lab_forever.py
│   ├── run_lab.py
│   ├── run_daily_paper_auto.py
│   └── check_data_integrity.py
├── lab/
├── web/
├── archive/
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
| 个股止损 | -8%（自适应：max(8%, 波动率×2)） |

---

## 变更记录

### v7.0 — 妖股规则 + 因子库 + 武器库 (2026-06-16)
- **妖股规则**：6规则评分系统，大涨率 33.7%（基线 12.1%）
- **因子库**：94→123因子，新增 29 个涨停专用因子（`factors/limit_up.py`）
- **信号级特征工程**：`featurize_signals.py`，5分钟计算68维特征
- **因子IC验证**：`validate_factors.py`，滚动IC分析
- **策略武器库**：`run_arsenal.py`，每日行业热力图+信号扫描
- **ML探索**：分类/回归/自适应/板块轮动全链路实验，结论记录
- 新增铁律：交割单文件名含日期+标签，永不覆盖

### v6.0 — 小市值反转 + 管线修复 (2026-06-15)
- 新增 `bt_small_cap.py`：小市值反转策略，2025年 +87.2%
- 涨停管线交割单修复：`_running_total/_running_cash` 统一口径
- 策略实验室 32 变体 + 6 维评分 + 三档判定

### v5.0 — 回测引擎重构 + 市值固化 (2026-06-14)
- 回测引擎重构：coc + set_shortcash(True) + 日终10项校验
- 市值数据固化：stock_mcap_proxy 表
- 板别感知涨跌停判断
