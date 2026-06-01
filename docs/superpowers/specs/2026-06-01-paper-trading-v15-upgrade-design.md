# 模拟盘 v1.5 升级 + 项目整理 — 设计文档

## Context

舞 v1.5 回测结果优于 v1.4（CAGR 53.2% vs 46.1%），需要将 v1.5 加入每日模拟盘，与 v1.4 并行运行并对比展示。同时项目文件随时间积累变得散乱，一并整理。

## Goals

1. Web 模拟盘增加 v1.5 面板，与 v1.4 并排对比
2. 每日脚本统一运行 v1.4 + v1.5
3. 增加权益基准叠加、每日盈亏柱状图、持仓行业分布图
4. 清理废弃文件和 __pycache__

## Design

### 1. 策略配置中心

新文件 `config/paper_strategies.py`：

```python
PAPER_STRATEGIES = [
    {
        "name": "舞", "version": "v1.4",
        "account_id": 15, "run_id": 2,
        "factor_preset": "all",    # 全因子
        "universe_size": 500,
        "forward_days": 5, "top_n": 15,
    },
    {
        "name": "舞", "version": "v1.5",
        "account_id": 17, "run_id": 4,     # 新账号
        "factor_preset": "all",             # 全因子（IC筛选自动决定）
        "universe_size": 500,
        "forward_days": 5, "top_n": 15,
    },
]
```

### 2. 每日脚本重构

`scripts/run_daily_paper.py`：

- 数据加载阶段：一次加载 share，多次复用
- 遍历 `PAPER_STRATEGIES`：
  1. 因子计算 → IC筛选 → 训练 → 预测（独立）
  2. 信号写入 `paper_signals`（各自 run_id）
  3. 执行 T-1 未执行信号 → `paper_positions` + `paper_orders`
  4. 风控检查
  5. 写入 `paper_daily_pnl`
- 每个策略记录独立的 `account_id`/`run_id`

### 3. 数据库

```sql
-- 新增 v1.5 模拟账号
INSERT INTO paper_account (id, name, initial_capital, cash)
VALUES (17, '舞v1.5', 1000000, 1000000);

-- 新增 v1.5 运行记录
INSERT INTO paper_runs (id, strategy_id, version_id, start_date, initial_capital, status)
VALUES (4, 15, 17, CURRENT_DATE, 1000000, 'running');
```

### 4. Web 布局

`paper.html` 改为三区：

```
┌────────────────────────────────────────────┐
│  策略选择器  +  初始资金  +  创建模拟盘      │
├──────────────────┬─────────────────────────┤
│   舞 v1.4        │   舞 v1.5               │
│   (account=15)   │   (account=17)          │
│   run_id=2       │   run_id=4              │
│                  │                         │
│  [权益曲线+基准]  │  [权益曲线+基准]         │
│  [每日盈亏柱状图] │  [每日盈亏柱状图]        │
│  [持仓行业饼图]   │  [持仓行业饼图]          │
│  [持仓表]        │  [持仓表]               │
│  [已平仓]        │  [已平仓]               │
├──────────────────┴─────────────────────────┤
│   小市值 v1.0 (account=16, run_id=3)        │
│   [权益曲线] [持仓表] [已平仓]               │
└────────────────────────────────────────────┘
```

### 5. 图表增强

每个策略面板新增 3 类图表（复用 ECharts + app.js 的 `data-chart` 机制）：

| 图表 | 类型 | 数据源 | API |
|------|------|--------|-----|
| 权益曲线 + 基准叠加 | 双线图 | paper_daily_pnl + index_daily | 现有 `/api/paper-run/{id}` 扩展 |
| 每日盈亏 | 柱状图(红绿) | paper_daily_pnl.daily_return | 同上 |
| 持仓行业分布 | 饼图 | paper_positions JOIN stock_industry | 新增 `/api/paper-sector/{account_id}` |

所有图表通过服务器端渲染 `data-chart` JSON 属性，由 `app.js` 的 `initChart()` 初始化。

### 6. API 变更

| 端点 | 改动 | 说明 |
|------|------|------|
| `/api/paper-run/{run_id}` | 扩展 | 返回中增加基准数据、每日盈亏数据 |
| `/api/paper-sector/{account_id}` | 🆕 | 返回持仓行业分布 JSON |

### 7. 文件变更清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `config/paper_strategies.py` | 🆕 | 策略配置 |
| `scripts/run_daily_paper.py` | ✏️ 重写 | 多策略循环 |
| `web/templates/paper.html` | ✏️ | 三区布局 |
| `web/routes/api.py` | ✏️ | 新图表API |
| `web/static/app.js` | ✏️ | 柱状图/饼图支持 |
| `strategies/` | ❌ 删除 | 废弃静态策略 |
| `output/` | 📁 整理 | 分子目录 |
| `VERSION.md` | ✏️ | v1.6 规划 |
| `CLAUDE.md` | ✏️ | 项目结构 + 版本 |

### 8. 验证

- `python scripts/run_daily_paper.py` — 双策略运行，无错误，各自写入独立 paper_* 记录
- `python -m uvicorn web.main:app` — Web 页面三区正确加载
- 权益曲线显示双线（策略净值 + 上证基准）
- 每日盈亏柱状图红绿正确
- 持仓行业饼图展示正确
- 小市值面板正常显示
