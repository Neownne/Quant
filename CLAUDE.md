# 项目规范

## 铁律

### 回测

| # | 规则 | 原因 |
|---|------|------|
| 1 | **涨停回测用管线** `python scripts/run_backtest_pipeline.py --start X --top-n N --label NAME` | 管线自动处理信号深度、mcap-proxy、exec-close |
| 2 | **小市值回测用** `python scripts/bt_small_cap.py --start X --top-n N` | 独立脚本，向量化回测 |
| 3 | **不改 bt_backtest.py 交易逻辑** | 已验证通过 |
| 4 | **新增策略写独立脚本** | 每个策略一个 py 文件，自包含回测逻辑 |

### 数据源

| # | 数据源 | 用途 | 状态 |
|---|--------|------|------|
| — | **akshare** | 日线/市值/指数同步 | 主力 |
| — | **腾讯行情 API** | 午盘实时扫描 | `scan_intraday.py` |
| — | **tushare (120积分)** | 行业/板块补齐 | `data/sync_tushare.py`，仅 stock_basic 1次/小时 |
| — | **AmazingData (银河证券)** | 行业/实时/财务全品类 | 7月接入，见 `docs/AmazingData开发手册.pdf` |

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
| 三池信号 | `run_daily_signals.py` | **日常** | 涨停+妖股+牛股+ETF监测，一键邮件推送 |
| 牛股筛选 | `screen_bull.py` | **日常** | 缩量筑底，同花顺可导入 |
| 妖股规则 | `bt_yaogu.py` | 卫星 | 6规则评分≥6，大涨率33.7% |
| 板块轮动 | `bt_sector_rotation.py` | **规划中** | 两层架构(宏观大类+微观细分)，7月实现 |
| 涨停 Top-N | `run_backtest_pipeline.py` | 维护中 | 旧管线，-99.9% |
| 武器库 | `run_arsenal.py` | 辅助 | 行业热力图+信号扫描 |

## 三池信号速查

```
每日运行: python scripts/run_daily_signals.py --send-email
  → 自动判断: <11:30等午盘 / 11:30-15:00午盘扫描 / >15:00日终扫描
  → 日终模式包含: 数据同步 + 三池信号 + ETF三因子监测 + 邮件推送

午盘扫描: python scripts/scan_intraday.py
  → 腾讯实时行情，9s拉5000只，涨停/跌停/板块热度

涨停池: 4条件(市值30-500亿 + 股价5-100 + MA5>MA10 + 20日涨停2-4次) → 全量展示
妖股池: 6规则(一字板+3 + 低振幅+2 + 缩量板+1 + 非量能极值+1 + 连板≥2+1 + 缩量整理+1) → ≥3入选
牛股池: 5条件(市值5-50亿 + <MA40 + 缩量 + 波动<3% + 60日无涨停) → 评分0-100
三池交集: 涨停∩妖股重点展示详情（评分+强度+连板）
ETF监测: 三因子(量能50%+方向20%+份额30%) → 50只活跃ETF，高确信/中等/正常分级
```

## 板块轮动设计

> 详细方案: `.claude/plans/1-github-2-3-buzzing-eagle.md` | 目标: 7月初 Windows + AmazingData 实现

```
两层架构:

宏 观 层 — 6大类(科技/周期/消费/金融/医药/制造)，周度打分
  打分 = 残差动量(40%) + 广度(30%) + 资金流(30%)
  拥挤度(换手率>90%分位) → 大类排除 → 选 Top 2-3

微 观 层 — 申万二级 + 概念板块，三轴打分
  趋势(40%) + 反转(35%) + 安全(25%)
  买入: 总分>60% 且 恐贪<40
  卖出: 恐贪>70 或 拥挤>90%
  一票否决: 情绪>85%分位
```

## v7.5 关键规则变更

| # | 规则 | 说明 |
|---|------|------|
| — | **数据同步全市场，策略执行仅主板** | SQL: `code !~ '^(300\|301\|688\|[48])'` |
| — | **涨停价 = round(prev_close × 1.9899, 4)** | 统一公式，移除板别感知乘数 |

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

## 涨停复盘分析（hotpoint OCR）

| # | 规则 | 原因 |
|---|------|------|
| — | **OCR 用 Surya VLM** `python scripts/bulk_ocr.py` | 102 张图，100 天数据，名称准确率 99% vs Tesseract 70% |
| — | **分析用** `python scripts/analyze_hotpoint.py` | 集中冒出关键词 + 个股再启动 + 潜在龙头 |
| — | **数据在** `data/arsenal/hotpoint/master.csv` | 101 天 × 6800+ 条记录 |
| — | **Surya 需 HF 镜像** `HF_ENDPOINT=https://hf-mirror.com` | 模型 ~2GB，首次下载后缓存 |

## 标签烙印型再启动策略（实验阶段）

> 发现：圣阳股份/剑桥科技等 OCR 关键词跨期稳定的票，首板后间隔 20+ 天再启动胜率高
> 状态：OCR 数据仅 101 天，回测样本不足。需更长历史或 concept_board 替代方案

## 自进化因子挖掘

```bash
python scripts/evolve_factors.py --rounds 3        # 3 轮进化
python scripts/evolve_factors.py --status           # DB 统计
python scripts/evolve_factors.py --top 10           # 最佳因子
```

| # | 规则 | 原因 |
|---|------|------|
| — | **因子 DB** 在 `data/factor_db.json` | 追踪每轮生成/验证/淘汰 |
| — | **规则文件** 在 `data/factor_rules.md` | 自动从 DB 提取统计规律 |
| — | **WQ alpha 参考** `/tmp/wq-alpha-research` | WorldQuant BRAIN 技能包（需账号） |

## 数据目录结构

```
data/arsenal/
  hotpoint/       ← OCR缓存 + 报告 + 图表
  pools/          ← 三池JSON (bull/yaogu/limit_up)
  reports/        ← 每日文本报告
  ths_import/     ← 同花顺导入
  daban/          ← 打板分析
```

## ML 探索结论

1. **纯日线 OHLCV 技术因子对 A 股中期收益预测力有限**，IC 天花板 ≈ -0.10
2. **妖股规则是唯一有效产出**：不是预测收益率，而是从涨停股中筛选高质量形态
3. **ML 不能替代 alpha 来源**：XGBoost分类PR-AUC=0.19、回归R²=0.04
4. **板块级信号比个股级更可靠**：155只涨停的板块 vs 单只涨停股，前者更有信息量
5. **OCR 另类数据有 alpha 潜力**：标签烙印（跨期关键词稳定度）是 WQ 没有的信号源

## 快速参考

```bash
# ── 每日 ──
python scripts/run_daily_signals.py --send-email        # 数据同步 + 三池信号 + ETF监测 + 邮件（一键）
python scripts/scan_intraday.py                         # 午盘快速扫描（腾讯实时行情）

# ── 牛股筛选 ──
python scripts/screen_bull.py --exclude-gem-star --ths   # 输出+同花顺导入

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

# ── 武器库 ──
python scripts/run_arsenal.py

# ── 涨停复盘分析 ──
python scripts/bulk_ocr.py                               # 批量 OCR hotpoint/ 图片
python scripts/analyze_hotpoint.py                       # 分析报告（集中冒出+再启动）

# ── 自进化因子挖掘 ──
python scripts/evolve_factors.py --rounds 1              # 一轮进化
python scripts/evolve_factors.py --status                # 因子DB统计
python scripts/evolve_factors.py --top 10                # 最佳因子

# ── Web ──
python -m uvicorn web.main:app --host 0.0.0.0 --port 8899

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
