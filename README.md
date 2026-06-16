# Quant — A 股量化交易系统

> 最后更新：2026-06-16 (v7.1 三池信号系统 + 每日邮件)  
> GitHub：[Neownne/Quant](https://github.com/Neownne/Quant)

---

## 当前策略

| 策略 | 脚本 | 状态 | 说明 |
|------|------|------|------|
| 小市值反转 | `bt_small_cap.py` | **主力** | 2025年+87% |
| 三池信号系统 | `run_daily_signals.py` | **日常** | 涨停+妖股+牛股，每日邮件推送 |
| 牛股筛选器 | `screen_bull.py` | **日常** | 缩量筑底小票，同花顺可导入 |
| 涨停 Top-N | `run_backtest_pipeline.py` | 维护中 | -99.9%，Sharpe -1.06 |
| 妖股规则 | `bt_yaogu.py` | 卫星 | 6规则评分≥6，大涨率33.7% |

---

## 快速启动

```bash
source .venv/bin/activate
pg_ctl -D /opt/homebrew/var/postgresql@18 start

# 每日数据同步 + 模拟盘
python scripts/run_daily_paper_auto.py

# 三池信号 + 邮件推送（日常用这个）
python scripts/run_daily_signals.py --exclude-gem-star --send-email

# 牛股筛选器单独运行
python scripts/screen_bull.py --exclude-gem-star --ths

# 小市值反转回测
python scripts/bt_small_cap.py --start 2020-01-01 --top-n 10

# 妖股规则回测
python scripts/bt_yaogu.py --start 2020-01-01 --top-n 5 --label test

# 涨停策略回测
python scripts/run_backtest_pipeline.py --start 2025-01-01 --top-n 5 --label E4

# Web
python -m uvicorn web.main:app --host 0.0.0.0 --port 8899
```

---

## 三池信号系统

每天早盘前运行，三个独立池子并行扫描全市场，输出信号报告 + 邮件推送 + 同花顺导入文件。

### 涨停池（4条件）

| 条件 | 参数 |
|------|------|
| 市值 | 30-500亿 |
| 股价 | 5-63元 |
| 均线 | MA5 > MA10 |
| 涨停次数 | 近20日 > 1次 |

输出当日涨停且满足筛选的股票，按涨停强度排序。

### 妖股池（6规则评分）

在涨停股中评估封板质量，评分 ≥3 入选：

| 权重 | 规则 |
|------|------|
| +3 | 一字板 |
| +2 | 低振幅 < 8% |
| +1 | 缩量板 (< 1.5x均量) |
| +1 | 非量能极值 |
| +1 | 连板 ≥ 2 |
| +1 | 缩量整理 ≥ 1天 |

### 牛股池（5条件缩量筑底）

找被市场遗忘的缩量筑底小票：

| 条件 | 参数 |
|------|------|
| 市值 | 5-50亿 |
| 趋势 | 收盘 < MA40 |
| 量能 | 成交量 < 40日均量（缩量） |
| 波动 | 20日波动率 < 3%（筑底） |
| 涨停 | 60日内涨停 < 2次（非妖股） |

综合评分 0-100，前100只入选。

### 邮件配置

`.env` 文件中配置：
```
SMTP_HOST=smtp.qq.com
SMTP_PORT=465
SMTP_USER=你的QQ号@qq.com
SMTP_PASS=QQ邮箱授权码
EMAIL_TO=收件人1@xxx.com,收件人2@xxx.com
```

### 同花顺导入

运行后生成 `data/arsenal/ths_import_YYYYMMDD.txt`，同花顺 → 自选股 → 导入 → 选择文件即可。

---

## 妖股规则策略

### 核心发现

6 条规则从涨停股中筛选高质量信号。一字板 + 低振幅 + 缩量板的组合，大涨（10日>20%）概率 33.7%，是基线（12.1%）的 2.8 倍。

### 与旧涨停策略的关系

妖股规则是旧策略的精华提取器——旧策略 Top-20 中只有 7% 是妖股高分，但重叠部分均值收益 +26.4%。

### 局限性

只在投机风格市场有效（2020年）。当市场切换到基本面驱动时（2025年），规则几乎不出信号。应作为市场投机热度指标使用。

---

## 牛股筛选器

### 发现过程

分析 1232 只涨幅 >300% 的牛股，发现启动时共同特征：
- 100% 市值 < 50亿
- 99% 低于 MA60
- 71% 前 60 日跌超 20%
- 65% 量能萎缩

短期（3-6月翻倍）画像：已跌透、正在筑底、无人关注。

### 前瞻验证

2020-2026 年月度测试，90 天前瞻：
- 均值 +36.8%，中位 +24.9%
- >20% 概率 59%，>30% 概率 42%

### 定位

海选筛子——缩小候选池从 4000+ 到 ~100 只，最终选股需人工判断板块催化。

---

## 因子库（123因子）

在原有 94 因子上新增 29 个涨停专用因子（`factors/limit_up.py`）：

| 类别 | 新增 | 核心因子 |
|------|------|---------|
| 涨停模式 | 11 | lu_streak, lu_count_5d/20d/60d, lu_first_board, lu_freq_accel |
| 封板质量 | 8 | lu_seal_quality, lu_vol_intensity, lu_amplitude, lu_body_ratio |
| 板型分类 | 3 | lu_is_yiziban, lu_is_strong_board, lu_board_strength |
| 资金流代理 | 6 | mfi_14, cmf_20, force_index, money_flow_pressure |
| 波动率结构 | 5 | vol_ratio_5_20, vol_asymmetry, vol_regime |

---

## ML 探索结论

经过分类→回归→规则提炼的完整迭代，核心结论：

1. **纯日线 OHLCV 技术因子对 A 股中期收益预测力有限**，IC 天花板 ≈ -0.10
2. **妖股规则是唯一有效产出**：不是预测收益率，而是从涨停股中筛选高质量形态
3. **ML 不能替代 alpha 来源**：XGBoost 分类 PR-AUC=0.19、回归 R²=0.04
4. **板块级信号比个股级更可靠**：155 只涨停的板块 vs 单只涨停股

---

## 目录结构

```
quant/
├── README.md / CLAUDE.md
├── .env                          # 数据库+邮箱配置
├── config/settings.py
├── data/
│   ├── db.py / loader.py / sync.py
│   ├── arsenal/                  # 每日信号+报告+同花顺导入
│   ├── factor_ic/                # 因子IC分析
│   ├── models/                   # ML模型
│   └── signals/                  # 信号CSV
├── factors/
│   ├── __init__.py               # ALL_FACTORS (123个)
│   ├── limit_up.py               # 涨停专用因子 (29个)
│   ├── engine.py / monitor.py / screening.py
│   └── market_breadth.py / sector_breadth.py
├── scripts/
│   ├── bt_small_cap.py           # 小市值反转（主力）
│   ├── run_daily_signals.py      # 三池信号系统（日常）
│   ├── screen_bull.py            # 牛股筛选器
│   ├── bt_yaogu.py               # 妖股规则回测
│   ├── bt_backtest.py            # 涨停回测引擎
│   ├── run_backtest_pipeline.py  # 涨停管线
│   ├── gen_signals.py / gen_signals_ml.py
│   ├── featurize_signals.py      # 信号特征工程
│   ├── validate_factors.py       # 因子IC验证
│   ├── bt_ml_signals.py / bt_hybrid.py / bt_trend_adaptive.py
│   ├── train_signal_quality.py
│   ├── run_arsenal.py            # 武器库面板
│   ├── run_lab.py / run_lab_forever.py
│   ├── run_daily_paper_auto.py
│   └── check_data_integrity.py
├── lab/ / web/ / archive/ / tests/
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

### v7.1 — 三池信号系统 + 每日邮件 (2026-06-16)
- **三池信号系统**：`run_daily_signals.py` 涨停+妖股+牛股并行扫描
- **牛股筛选器**：`screen_bull.py` 缩量筑底5条件，综合评分0-100
- **邮件推送**：QQ邮箱 SMTP，每日自动发送信号报告
- **同花顺导入**：自动生成自选股导入文件
- **排除创业/科创**：`--exclude-gem-star` 过滤 300/301/688
- 涨停池切换为项目标准 4 条件筛选

### v7.0 — 妖股规则 + 因子库 + 武器库 (2026-06-16)
- 妖股规则：6规则评分系统，大涨率 33.7%
- 因子库：94→123因子，29 个涨停专用因子
- 武器库：`run_arsenal.py` 每日行业热力图
- ML探索：分类/回归/自适应/板块轮动全链路实验

### v6.0 — 小市值反转 + 管线修复 (2026-06-15)
- 小市值反转策略 + 管线交割单修复

### v5.0 — 回测引擎重构 + 市值固化 (2026-06-14)
- 回测引擎：coc + set_shortcash(True) + 日终10项校验
