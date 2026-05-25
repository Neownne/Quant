# 多因子 ML 量化选股策略 — 设计文档

> 日期：2026-05-25  
> 目标：实现实盘稳定盈利（普通行情年化 >50%，行情差 20-30%，行情好 >100%）  
> 本金：100 万 RMB | 最大回撤：25%

---

## 一、架构总览

```
原始数据 (stock_daily / daily_extra / shareholder / financial / minute)
    │
    ▼
┌─────────────────────────────┐
│  factors/  因子计算层         │
│  ├── alpha101.py  101因子   │
│  ├── alpha191.py  191因子   │
│  ├── custom.py    自定义因子 │
│  └── engine.py    计算引擎   │
└──────────┬──────────────────┘
           │ 因子值矩阵 (N只股票 × M个因子)
           ▼
┌─────────────────────────────┐
│  models/  ML预测层           │
│  ├── dataset.py   样本构造   │
│  ├── trainer.py   训练/验证  │
│  ├── predictor.py 每日预测   │
│  └── monitor.py   IC监控     │
└──────────┬──────────────────┘
           │ 每只股票预测收益排序
           ▼
┌─────────────────────────────┐
│  portfolio/  组合优化层      │
│  ├── selector.py  选股      │
│  ├── allocator.py 仓位分配  │
│  └── risk.py      风控      │
└──────────┬──────────────────┘
           │ 最终持仓 & 调仓信号
           ▼
┌─────────────────────────────┐
│  现有：backtest_runner → 回测验证 → 模拟盘 → 实盘 │
└─────────────────────────────┘
```

每层独立可测，接口明确，允许替换任意组件。

---

## 二、数据补齐（阶段一）

### 2.1 当前数据缺口

| 维度 | 现状 | 缺口 |
|---|---|---|
| 量价 | stock_daily 1,057 万行，全量覆盖 | OK |
| 估值 | stock_daily_extra 31.6 万行，仅 874 只 | 待补齐至 4,000+ |
| 股东 | stock_shareholder 42.6 万行，5,477 只 | OK |
| 财务 | 无 | P0 致命缺口，因子依赖 |
| 行业 | 无 | P1，截面中性化必需 |
| 分钟 | stock_minute 仅录制片段 | P2，暂不依赖 |

### 2.2 新增任务

| 任务 | 数据源 | 产出 | 验证 |
|---|---|---|---|
| 财务三表 fetcher + DDL | AKShare 同花顺接口 | stock_financial 表 | 5,000+ 股票有数据 |
| 财务数据全量同步 | 同上 | 利润/资产/现金流 | >100 万行 |
| 补齐估值指标 | 百度财经（已跑） | stock_daily_extra | 覆盖 >4,000 只 |
| 行业分类表 | AKShare 申万行业 | stock_industry | 每只股票有行业标签 |

### 2.3 新增表结构

```sql
-- 财务数据（同花顺财务摘要）
CREATE TABLE stock_financial (
    code            VARCHAR(10),
    report_date     DATE,           -- 报告期
    revenue         DOUBLE PRECISION, -- 营业收入（亿元）
    net_profit      DOUBLE PRECISION, -- 归母净利润（亿元）
    gross_margin    DOUBLE PRECISION, -- 毛利率 %
    net_margin      DOUBLE PRECISION, -- 净利率 %
    roe             DOUBLE PRECISION, -- ROE %
    total_assets    DOUBLE PRECISION, -- 总资产（亿元）
    total_liability DOUBLE PRECISION, -- 总负债（亿元）
    bps             DOUBLE PRECISION, -- 每股净资产
    eps             DOUBLE PRECISION, -- 每股收益
    cash_flow       DOUBLE PRECISION, -- 经营活动现金流
    PRIMARY KEY (code, report_date)
);

-- 行业分类
CREATE TABLE stock_industry (
    code            VARCHAR(10) PRIMARY KEY,
    industry_sw1    VARCHAR(50),     -- 申万一级行业
    industry_sw2    VARCHAR(50),     -- 申万二级行业
    market          VARCHAR(10)      -- 主板/创业板/科创板/北交所
);
```

---

## 三、因子库（阶段二）

### 3.1 因子计算引擎

`factors/engine.py`：

- 输入：股票代码列表、日期范围
- 输出：`pd.DataFrame`，行索引 (code, trade_date)，列 = 因子名，值 = 因子值
- 支持增量计算：新增交易日只需追加计算最新一天
- 使用模块级缓存避免重复查询数据库

### 3.2 因子体系

**Alpha101 核心因子（30+ 个）**：
优先实现均值回归型、量价背离型、动量型因子。按 Kakushadze & Tulchinsky (2015) 论文定义逐行翻译为 Python/Numpy。

**Alpha191 A股适配因子**：
国泰君安经典，补充 Alpha101 中缺失的 A 股特色因子。

**自定义因子（5-10 个）**：

| 因子 | 逻辑 |
|---|---|
| 散户参与度 | 股东户数变化率，散户增多 → 利空 |
| 市值因子 | 流通市值对数（A 股小市值溢价显著） |
| 换手率动量 | 5日换手率变化方向 |
| 估值分位 | PB 在历史 3 年分位数 |
| 利润加速度 | 净利润增速的二阶导 |

### 3.3 截面中性化

对每个因子做市值 + 行业去偏：
```
因子_中性化 = 原始因子 - β1 × 市值 - β2 × 行业哑变量
```

### 3.4 IC 监控管线

`factors/monitor.py`：

- 日度 RankIC：因子值与 T+1 收益率的截面秩相关
- ICIR：IC 均值 / IC 标准差
- IC 衰减曲线：因子值对未来 1/2/3/5/10 天的预测力

---

## 四、ML 预测层（阶段三）

### 4.1 样本构造

`models/dataset.py`：

- 横截面样本：每个交易日，每只股票一行
- 特征 X：M 个因子值（已中性化）
- 标签 y：T+1 日收益率方向（涨/跌）或连续收益率
- Walk-forward 切分：滚动 3 年训练 / 1 年验证，不做随机切分

### 4.2 模型

XGBoost 基线 → LightGBM 对比 → 选优。

分类模式（预测涨跌方向）为主，回归模式（预测收益率）作为辅助排序。

### 4.3 每日预测流程

盘后：
1. 计算当日所有股票因子值
2. 加载最新训练的模型
3. 预测每只股票的次日涨跌概率
4. 按概率降序排列，取前 N 只作为候选池
5. 传给组合优化层

### 4.4 验证标准

- RankIC 均值 > 0.05
- Walk-forward 回测年化 > 30%
- 最大回撤 < 25%
- 跑赢等权持有基准

---

## 五、组合优化层（阶段四）

### 5.1 选股

- 候选池：模型打分 top 50
- 剔除：ST/停牌/涨跌停/次新股（上市 < 60 天）
- 最终持仓：10-30 只

### 5.2 仓位分配

- 等权 + 波动率倒数加权
- 单只上限 10%
- 单行业上限 30%

### 5.3 风控

| 规则 | 参数 |
|---|---|
| 个股止损 | -8% 或 -1.5x ATR(20) |
| 组合止损 | 回撤触及 20% → 减仓至 50% |
| 最大回撤 | 回撤触及 25% → 清仓暂停 |
| 调仓频率 | 日频，盘后计算次日开盘执行 |

### 5.4 模拟盘验证

在 paper_trade 模块中跑 1-2 个月，对比模拟盘收益与回测收益，确认无过拟合。

---

## 六、项目文件结构

新增模块：

```
quant/
├── factors/                    # 新增
│   ├── __init__.py
│   ├── engine.py               # 因子计算引擎
│   ├── alpha101.py             # Alpha101 因子
│   ├── alpha191.py             # Alpha191 因子
│   ├── custom.py               # 自定义因子
│   └── monitor.py              # IC 监控
├── models/                     # 新增
│   ├── __init__.py
│   ├── dataset.py              # 样本构造
│   ├── trainer.py              # 训练/验证
│   ├── predictor.py            # 每日预测
│   └── xgb_model.py            # XGBoost 模型封装
├── portfolio/                  # 新增
│   ├── __init__.py
│   ├── selector.py             # 选股
│   ├── allocator.py            # 仓位分配
│   └── risk.py                 # 风控规则
├── data/
│   ├── db.py                   # 新增 2 张表 DDL
│   ├── fetcher.py              # 新增财务 + 行业 fetcher
│   └── sync.py                 # 新增财务 + 行业 sync
├── app/
│   └── pages/                  # 新增因子/ML 页面（后续）
└── ...
```

---

## 七、关键设计决策

1. **日频调仓**：T+1 制度下日频是最高频次
2. **横截面选股**：不做单只择时，在全市场范围内排序选 top N
3. **Walk-forward 验证**：滚动窗口，模拟实盘信息到达时点
4. **因子上限 50**：从 30 个核心因子起步，加新因子需证明正交性
5. **树模型优先**：XGBoost/LightGBM 在表格数据上比深度学习更稳
6. **RL 因子发现留待阶段四之后**：需要更大的工程投入和更扎实的因子基础

---

## 八、Superpowers 工作流

所有开发遵循以下流程：

- `brainstorming` → 澄清需求
- `writing-plans` → 制定实现计划
- `executing-plans` → 分步执行
- `test-driven-development` → 先写测试再写代码
- `verification-before-completion` → 验证后再声称完成
- `requesting-code-review` → 阶段性提交审查
- 完成后更新 PROJECT.md 并同步到 GitHub
