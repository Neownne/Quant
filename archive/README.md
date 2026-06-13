# 冷宫（archive/）

本目录存放已从主项目移出的非涨停策略代码。它们不再参与日常维护，但保留完整历史，便于未来按需恢复或参考。

## 归档原因

项目决定聚焦 **涨停策略 + ETF监控**，暂停维护以下策略线：

- ML 选股（舞、GBRT重排）
- 大小票动态切换
- 小市值 alpha
- 强化学习（RL）相关实验
- 复杂的组合优化/多因子引擎

## 目录说明

```
archive/
├── data/                       # 归档的模型与数据文件
│   ├── gbrt_model.pkl
│   └── limit_up_e3_2026_daily.csv
├── hotpoint/                   # 热点图 PNG 序列
├── models/                     # ML/RL 模型模块
│   ├── dataset.py
│   ├── dual_period.py
│   ├── predictor.py
│   ├── regime.py
│   ├── sector_dataset.py
│   ├── sector_model.py
│   ├── trainer.py
│   └── tuning.py
├── pictures/                   # 图片资源（已清空或仅 .DS_Store）
├── portfolio/                  # 组合优化模块
│   ├── allocator.py
│   ├── paper_engine.py
│   ├── risk.py
│   ├── sector_filter.py
│   ├── selector.py
│   └── small_cap_engine.py
├── scripts/                    # 非涨停策略脚本
│   ├── run_daily_paper_ml.py
│   ├── run_daily_paper_switch.py
│   ├── run_ml_backtest.py
│   └── run_small_cap_backtest.py
├── small_cap/                  # 小市值/大小票策略
│   ├── backtest.py
│   ├── switch_backtest.py
│   └── switch_backtest_v2.py
└── tests/                      # 非涨停相关测试
    ├── test_attribution.py
    ├── test_models.py
    ├── test_overfit_check.py
    ├── test_portfolio.py
    ├── test_regime.py
    ├── test_rl_dynamic.py
    ├── test_rl_env.py
    ├── test_rl_models.py
    ├── test_rl_predictor.py
    ├── test_rl_trainer.py
    ├── test_sector_breadth.py
    ├── test_sector_dataset.py
    ├── test_sector_filter.py
    └── test_sector_model.py
```

## 恢复方法

1. 将需要的文件/目录从 `archive/` 移回项目根目录的原始位置。
2. 恢复 `models/__init__.py` 和 `portfolio/__init__.py` 中的导出。
3. 恢复 `web/routes/api.py` 的 `/strategy-summary` 多策略卡片（如需要）。
4. 恢复 `web/templates/base.html`、`paper.html`、`backtest.html` 中的非涨停入口（如需要）。
5. 恢复 `scripts/run_daily_paper_auto.py` 中对相应脚本的调用。
6. 运行 `python -m compileall .` 和 `pytest tests/` 验证无导入错误。

## 依赖关系备忘

- `portfolio/paper_engine.py` 依赖 `models.regime`、`portfolio.selector/allocator/risk`
- `scripts/run_ml_backtest.py` 依赖 `models.*`、`portfolio.*`、`factors.*`
- `scripts/run_daily_paper_switch.py` 仅依赖 `data.db` 和 `config.settings`
- `scripts/run_daily_paper_ml.py` 依赖 `data.db`、`data.sync`、`config.settings` 和 `data/gbrt_model.pkl`

## 注意事项

- 冷宫代码不再与主项目同步更新，移回后可能需要适配当前数据表结构。
- 主项目已统一涨跌停阈值为 9%，冷宫中的旧脚本仍可能使用 9.9%/9.8%/9.7% 阈值，恢复时请注意校准。
- 主项目已移除黑名单逻辑，恢复旧脚本时如需黑名单功能需自行维护。
