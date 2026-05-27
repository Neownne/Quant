# 量化平台架构重构设计

> 日期：2026-05-27  
> 状态：已确认，待实施

---

## 目标

重构项目架构，Web 端定位为只读监控+可视化面板，策略研究/训练/调参全部在后台完成，结果持久化后由 Web 端读取展示。

## 核心原则

- **Web 只读，后台读写**：策略代码不进入 Web 进程
- **防过拟合**：回测阶段强制多项检查，不通过不可发布
- **可追溯**：因子权重行级存储，支持时间旅行查询
- **数据质量优先**：每日同步后强制校验，不通过则阻断下游

---

## 一、整体架构

```
┌─────────────────────────────────────────────────┐
│                   Web 监控层                      │
│          FastAPI + HTMX + Alpine.js              │
│  行情看板 │ 回测对比 │ 模拟盘 │ 数据状态 │ 因子监控  │
└──────────────────────┬──────────────────────────┘
                       │ 只读查询
┌──────────────────────┴──────────────────────────┐
│                 PostgreSQL                       │
│  行情表 │ 基本面表 │ 回测结果表 │ 模拟盘表 │ 因子IC表 │
│  新增：策略配置/版本/权重历史/血缘/可用性/健康/      │
│        指令队列/信号因子/归因/数据质量             │
└──────────────────────┬──────────────────────────┘
                       │ 读写
┌──────────────────────┴──────────────────────────┐
│              后台研究引擎（脚本/定时任务）           │
│                                                  │
│  数据同步 → 质量校验 → 因子计算 → 模型训练 → 回测   │
│                         (带防过拟合检查)            │
│                                      │           │
│                              结果写入DB           │
│                                                  │
│  模拟盘引擎 → 持仓跟踪 → 归因分析 → 自动调参+熔断   │
└─────────────────────────────────────────────────┘
```

---

## 二、Web 层设计

**技术栈：** FastAPI + Jinja2 模板 + HTMX（局部刷新）+ Alpine.js（轻量交互）+ ECharts CDN

**5 个页面：**

| 页面 | 路由 | 核心内容 |
|---|---|---|
| 行情看板 | `/` | 自选股列表分组 + 实时报价表 + K线图（可叠加技术指标） |
| 回测对比 | `/backtest` | 多策略权益曲线叠加图 + 指标排名表 + 参数详情 + 过拟合标记颜色 |
| 模拟盘监控 | `/paper` | 日期选择 + 账户净值曲线 + 持仓 + 信号 + 委托/成交/平仓历史 |
| 数据状态 | `/data` | 各表覆盖度统计 + 同步日志 + 质量校验结果 + 手动触发同步 |
| 因子监控 | `/factors` | IC序列/衰减曲线/覆盖率 + ML模型表现 + 单因子时序分析 |

**HTMX 典型交互：**
- 行情看板选股 → `hx-get="/api/kline/{code}"` → 局部刷新K线
- 回测对比筛选 → `hx-get="/api/backtest-compare?..."` → 刷新图表+排名
- 数据页同步 → `hx-post="/api/sync/trigger"` → 返回日志流

---

## 三、数据层设计

### 3.1 新增/修改的数据库表

| 表名 | 用途 |
|---|---|
| `strategy_configs` | 策略基本信息（id, name, type, description, created_at） |
| `strategy_versions` | 模型架构级版本（算法类型、特征列表版本），不存权重JSON |
| `factor_weights_history` | **行级存储**：strategy_id, factor_name, weight, effective_date, source(auto/manual), reason |
| `factor_lineage` | 因子血缘：source_fields, computation_formula_hash, last_validated_at, upstream_factors |
| `factor_availability` | 每个因子每交易日 data_ready_at, latency_ms，防未来函数 |
| `backtest_results` | 新增 quality 字段（valid/suspect/invalid）+ metrics_json + equity_curve_json |
| `paper_runs` | 模拟盘运行：start_date, end_date, status, initial_capital |
| `paper_signals` | 每日选股信号，关联 paper_runs |
| `signal_factors` | **行级存储**：signal_id, factor_name, value，替代 JSON blob |
| `paper_positions` | 持仓明细：entry/exit 时间价格，pnl |
| `signal_attribution` | 归因分析：每个信号的 P&L + factor_contrib |
| `weight_adjustments` | 调权记录：confidence_level, source(auto/manual) |
| `strategy_health` | 每日健康评分：IC/回撤/regime_tag/status/action_required |
| `strategy_commands` | 人工干预指令队列（替代直接改DB） |
| `data_quality_log` | 每日数据校验结果 |

### 3.2 日常数据流

```
15:30  数据同步(sync.py)
         ↓
      质量校验关卡（覆盖率/空值率/异常跳变/涨跌停冻结）
         ↓ 不通过 → 阻断 + 告警
         ↓ 通过
16:00  因子计算 → factor_values + factor_lineage + factor_availability
         ↓
17:00  模拟盘选股(如有运行中的paper_run) → signal_factors(行级)
```

---

## 四、策略研究闭环

### 4.1 研究流程

```
1. 因子计算(scripts/factor_compute.py, 定时任务)
       │
2. 模型训练(models/trainer.py) → new strategy_version
       │
3. 历史回测(scripts/run_ml_backtest.py)
       │  防过拟合检查（见4.2）
       │  通过 → backtest_results(quality=valid)
       │  不通过 → quality=suspect，标记具体原因
       │
4. 模拟盘验证
   │  选择版本 + 起始日期 → paper_run
   │  至少跑1个月 + 夏普/回撤偏差<30% → 标记 production-ready
       │
5. 归因分析 + 自动调参（见4.3）+ 人工指令（见4.4）
```

### 4.2 回测防过拟合机制

所有回测结果写入前必须通过以下检查：

| 机制 | 说明 | 建议阈值 |
|---|---|---|
| Walk-forward 切分 | 训练/验证/测试按时间先后，禁止 shuffle | 如 36个月训练→6个月验证→6个月测试 |
| 多时段稳健性 | 测试集至少覆盖 2 种 market regime | 牛/熊/震荡中至少2种 |
| 参数敏感性 | 最优参数 ±10%，结果波动 >20% → overfit | 网格搜索时自动计算 |
| 最少交易次数 | 测试期 <30 笔 → 统计不可靠 | 日频约半年最低 |
| 样本外一致性 | 验证集夏普 / 训练集夏普 < 0.3 → 严重过拟合 | 写入 metrics_json |
| 调整后夏普 | adjusted_sharpe = sharpe * sqrt(N_trades/N_params) | VC维度惩罚 |
| 模拟盘验证期 | 新版本模拟盘 ≥1月 + 指标偏差 <30% | 达标方可标记 production-ready |

### 4.3 自动调参逻辑（含防过拟合）

```
触发条件（需同时满足）：
  1. 某因子 IC 衰减连续 >= 3 期
  2. p-value < 0.05

动作：
  → 降权 20%
  → 写入 weight_adjustments（source=auto, confidence_level=0.95/0.99）

不满足则 log_warning("IC衰减但不显著，暂不调整")

权重计算：
  运行时通过 effective_date 回溯 factor_weights_history
  → 先取 manual 基准权重 → 叠加 auto 调整 → 归一化
```

### 4.4 人工干预

Web 端或命令行写入 `strategy_commands`，后台异步执行：

| command_type | 说明 |
|---|---|
| `adjust_weight` | 手动修改某因子权重 |
| `pause` | 暂停模拟盘 |
| `resume` | 恢复模拟盘 |
| `rollback` | 回滚到某历史版本 |
| `retrain` | 用最新数据重新训练 |

所有指令记录执行结果，支持回滚。

### 4.5 策略熔断

`strategy_health` 每日更新：

| status | 判定 | action |
|---|---|---|
| normal | 7日 IC > 0, 回撤 < 10% | 无 |
| warning | IC 转负 或 回撤 > 15% | 通知、人工关注 |
| critical | IC 连续负 或 回撤 > 25% | 自动切备用策略（等权组合/空仓） |

---

## 五、同步自动化

数据同步使用系统级定时任务，不需要手动运行：

- **macOS launchd**：plist 配置 `~/Library/LaunchAgents/com.quant.sync.plist`，每日 15:30 触发 `python -m data.sync`
- **备选：crontab**：`30 15 * * 1-5 cd /path/to/quant && python -m data.sync`
- Web 端数据状态页仍保留"手动触发同步"按钮（调 launchd `launchctl start`）
- 当前 `app/main.py` 中的 `schedule` 线程在切换 FastAPI 后删除

---

## 六、现有项目清理

删除以下内容：

| 删除 | 原因 |
|---|---|
| `app/pages/5_📝_Strategy_Editor.py` | Web 不写策略 |
| `app/utils/backtest_runner.py` | 回测在后台脚本跑 |
| `app/utils/ml_backtest.py` | 同上 |
| `app/utils/ml_config_manager.py` | 同上 |
| `app/utils/stock_pools.py` | 股票池逻辑移到后台脚本 |
| `app/pages/6_📦_Stock_Pools.py` | 同上 |
| `app/pages/7_🔴_Recorder.py` | 录制逻辑移到后台 |
| `app/pages/8_📊_ML_Monitor.py` | 合并到因子监控页 |
| `app/main.py` 的 schedule 线程 | 改用 launchd |

保留并改造：
- `app/utils/data_loader.py` → 改造为 FastAPI 路由的数据查询层，去掉 backtrader 相关
- `app/utils/account_manager.py` → 保留，模拟盘账户管理仍需要

策略/因子/模型/组合层代码不动（非 Web 部分）。现有 Streamlit 全部删除后重建为 FastAPI。

---

## 七、排除项

- 暂不做研究/生产环境物理隔离（Part 3 延后）
- 不重写现有脚本，只加质量关卡 + 结果入库层 + 防过拟合检查
- 分钟数据暂用新浪接口，后续可切换付费 API

---

## 八、实施后文档更新

实施完成后需更新：
- `README.md`：架构图、目录结构、工作流同步最新设计
- `docs/project-learning-guide.md`：删除已废弃模块的说明，更新系统架构章节

---

## 九、前端技术选择

- **框架：** FastAPI + Jinja2 + HTMX + Alpine.js + ECharts CDN
- **不引入：** npm/Node.js 构建工具链、前端 SPA 框架
- **图表：** K线图、权益曲线等用 ECharts（CDN 直引），简单指标用内联 SVG/CSS 或 Alpine.js 数据绑定
