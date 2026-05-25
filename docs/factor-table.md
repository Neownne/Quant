# A 股多因子选股 — 因子全景表

> 日期：2026-05-25  
> 合计：现有 37 个 + 候选 28 个 = 65 个因子  
> 新增标准：逐因子验证 IC 显著性 + 与已有因子相关性 < 0.7（正交性门禁）

---

## 一、现有因子（37 个）

### 1.1 反转型（3）

| 因子 | 公式 | 数据 | 预期 IC | 状态 |
|---|---|---|---|---|
| rev_5 | -(C_t - C_{t-5}) / C_{t-5} | close | 正（A股短期反转显著） | ✅ 已有 |
| rev_10 | -(C_t - C_{t-10}) / C_{t-10} | close | 正 | ✅ 已有 |
| rev_20 | -(C_t - C_{t-20}) / C_{t-20} | close | 正 | ✅ 已有 |

### 1.2 动量型（3）

| 因子 | 公式 | 数据 | 预期 IC | 状态 |
|---|---|---|---|---|
| mom_20 | (C_t - C_{t-20}) / C_{t-20} | close | 负（A股中期反转） | ✅ 已有 |
| mom_60 | (C_t - C_{t-60}) / C_{t-60} | close | 正（长期动量） | ✅ 已有 |
| ema_ratio_5_20 | EMA(5) / EMA(20) - 1 | close | 正 | ✅ 已有 |

### 1.3 波动率型（2）

| 因子 | 公式 | 数据 | 预期 IC | 状态 |
|---|---|---|---|---|
| vol_20 | std(ret, 20) × √252 | close | 负（低波异象） | ✅ 已有 |
| atr_14 | ATR(14) / Close | high/low/close | 负 | ✅ 已有 |

### 1.4 量价型（3）

| 因子 | 公式 | 数据 | 预期 IC | 状态 |
|---|---|---|---|---|
| vol_ratio_5_20 | MA(vol,5) / MA(vol,20) | volume | 正 | ✅ 已有 |
| vpt | (C×V - MA(C×V,20)) / MA(C×V,20) | close/volume | 正 | ✅ 已有 |
| vwap_ratio | Close / VWAP(20) - 1 | high/low/close/volume | 正 | ✅ 已有 |

### 1.5 趋势强度型（5）

| 因子 | 公式 | 数据 | 预期 IC | 状态 |
|---|---|---|---|---|
| macd_dif | EMA(12) - EMA(26) | close | 正 | ✅ 已有 |
| macd_signal | EMA(macd_dif, 9) | close | 正 | ✅ 已有 |
| macd_hist | (DIF - DEA) / Close | close | 正 | ✅ 已有 |
| rsi_14 | RSI(14) Wilder | close | 负（超买反转） | ✅ 已有 |
| rsi_7 | RSI(7) Wilder | close | 负 | ✅ 已有 |

### 1.6 通道型（2）

| 因子 | 公式 | 数据 | 预期 IC | 状态 |
|---|---|---|---|---|
| bb_position | (C - mid) / (4×std) | close | 正（突破） | ✅ 已有 |
| bb_width | (4×std) / SMA(20) | close | 负 | ✅ 已有 |

### 1.7 流动性型（3）

| 因子 | 公式 | 数据 | 预期 IC | 状态 |
|---|---|---|---|---|
| turnover_5 | MA(turnover, 5) | turnover | 负（高换手利空） | ✅ 已有 |
| illiquidity | MA(|ret| / (V×C), 20) | close/volume | 正（非流动溢价） | ✅ 已有 |
| amount_ratio | MA(amount,5) / MA(amount,20) | amount | 正 | ✅ 已有 |

### 1.8 高阶统计型（9）

| 因子 | 公式 | 数据 | 预期 IC | 状态 |
|---|---|---|---|---|
| skewness_20 | skew(ret, 20) | close | 负（负偏反转） | ✅ 已有 |
| kurtosis_20 | kurt(ret, 20) | close | 负 | ✅ 已有 |
| high_low_ratio | MA((H-L)/C, 20) | high/low/close | 负 | ✅ 已有 |
| max_dd_20 | (C - High_20) / High_20 | close | 正（反转） | ✅ 已有 |
| corr_c_v | corr(C, V, 10) | close/volume | 正 | ✅ 已有 |
| co_ratio | (C-O) / (H-L) | open/high/low/close | 正 | ✅ 已有 |
| up_day_ratio | MA(ret>0, 20) | close | 正 | ✅ 已有 |
| price_position | (C-L_20) / (H_20-L_20) | high/low/close | 正 | ✅ 已有 |
| vol_swing | |V/MA(V,20)-1| × sign(ret) | close/volume | 正 | ✅ 已有 |

### 1.9 A 股自定义型（7）

| 因子 | 公式 | 数据 | 预期 IC | 状态 |
|---|---|---|---|---|
| log_mcap | -log(流通市值) | market_cap（extra） | 正（小盘溢价） | ⚠️ 需 extra_data |
| turnover_mom | Δturnover(5) / std(turnover,20) | turnover | 负 | ✅ 已有 |
| pb_pct | -PB在500日中分位数 | pb（extra） | 正（低估值） | ⚠️ 需 extra_data |
| sh_change | -Δ股东户数(60日) | shareholder_count（extra） | 正（散户减少） | ⚠️ 需 extra_data |
| vol_conv | -|V/MA(V,5) - V/MA(V,20)| | volume | 正 | ✅ 已有 |
| intra_vol | EMA((H-L)/O, 5) | open/high/low | 负 | ✅ 已有 |
| gap_ratio | (O_t - C_{t-1}) / C_{t-1} | open/close | 正（缺口延续） | ✅ 已有 |

---

## 二、Alpha191 A 股候选因子（28 个）

> 来源：国泰君安 Alpha191 论文 + A 股实证文献  
> 入选标准：1) A 股回测有效 2) 与已有因子相关性低 3) 数据可得

### 2.1 换手率类（5）

| 因子 | 公式 | 数据 | 预期 IC | 优先级 |
|---|---|---|---|---|
| turnover_skew | skew(turnover, 20) | turnover | 正（换手右偏=资金进场） | ⭐⭐⭐ |
| turnover_cv | std(turnover,20) / mean(turnover,20) | turnover | 负（换手稳定>剧烈） | ⭐⭐⭐ |
| turnover_ma_dev | turnover / MA(turnover,60) - 1 | turnover | 正（异常放量=关注） | ⭐⭐ |
| turnover_ret_corr | corr(turnover, ret, 20) | turnover/close | 正（量价配合） | ⭐⭐⭐ |
| free_turnover_ratio | turnover / float_share_ratio | turnover/float_share（extra） | 负 | ⭐ |

### 2.2 隔夜/开盘效应类（4）

| 因子 | 公式 | 数据 | 预期 IC | 优先级 |
|---|---|---|---|---|
| overnight_ret | (O_t - C_{t-1}) / C_{t-1} | open/close | 正（隔夜涨延续） | ⭐⭐⭐ |
| overnight_ret_std | std(overnight_ret, 10) | open/close | 负 | ⭐⭐ |
| open_auction_jump | (O_t - MA(O,5)) / MA(O,5) | open | 正 | ⭐⭐ |
| gap_ma_dev | gap_ratio - MA(gap_ratio, 20) | open/close | 正 | ⭐⭐ |

### 2.3 资金流向类（6）

| 因子 | 公式 | 数据 | 预期 IC | 优先级 |
|---|---|---|---|---|
| money_flow | Σ((C-L)-(H-C))×V / ΣV, 10日 | high/low/close/volume | 正 | ⭐⭐⭐ |
| obv_roc | (OBV_t - OBV_{t-20}) / OBV_{t-20} | close/volume | 正 | ⭐⭐ |
| force_index | EMA(ΔC × V, 2) | close/volume | 正 | ⭐⭐ |
| cwt | C × V × turnover 的5日变化率 | close/volume/turnover | 正 | ⭐ |
| volume_climax | (V_t - max(V_{t-20..t-1})) / max(V_{t-20..t-1}) | volume | 负（天量见顶） | ⭐⭐⭐ |
| vwap_momentum | VWAP(5) / VWAP(20) - 1 | high/low/close/volume | 正 | ⭐⭐ |

### 2.4 日内形态类（4）

| 因子 | 公式 | 数据 | 预期 IC | 优先级 |
|---|---|---|---|---|
| upper_shadow | (H - max(O,C)) / (H-L) | open/high/low/close | 负（上影利空） | ⭐⭐⭐ |
| lower_shadow | (min(O,C) - L) / (H-L) | open/high/low/close | 正（下影利多） | ⭐⭐⭐ |
| body_ratio | |C-O| / (H-L) | open/high/low/close | 正（实体大=趋势强） | ⭐⭐ |
| intra_day_rev | (C-O) / (H-L) — 盘中反转度 | open/high/low/close | 正 | ⭐⭐ |

### 2.5 波动率高阶类（5）

| 因子 | 公式 | 数据 | 预期 IC | 优先级 |
|---|---|---|---|---|
| vol_of_vol | std(std(ret,5), 20) | close | 负（波动率不稳定） | ⭐⭐⭐ |
| down_vol_ratio | std(ret_neg, 20) / std(ret, 20) | close | 负（下行波动占比高利空） | ⭐⭐⭐ |
| tail_risk | ret_5pct_quantile(60日) | close | 负 | ⭐⭐ |
| beta_20 | cov(ret, ret_mkt, 20) / var(ret_mkt, 20) | close + index | 负（低beta溢价A股显著） | ⭐⭐⭐ |
| ret_asymmetry | (mean(ret_pos) - |mean(ret_neg)|) / std(ret) | close | 正 | ⭐⭐ |

### 2.6 流动性高阶类（4）

| 因子 | 公式 | 数据 | 预期 IC | 优先级 |
|---|---|---|---|---|
| amihud_5 | MA(|ret| / amount, 5) × 10^10 | close/amount | 正（非流动溢价） | ⭐⭐ |
| dollar_volume | MA(amount, 20) 的对数 | amount | 负（大盘股弱于小盘） | ⭐⭐⭐ |
| turnover_breakout | (turnover - min_turnover_60) / (max_turnover_60 - min_turnover_60) | turnover | 正 | ⭐⭐ |
| bid_ask_proxy | MA((H-L)/V, 20) — 买卖价差代理 | high/low/volume | 正（流动性补偿） | ⭐ |

---

## 三、新增因子实施顺序

按方案 A 优先级，边实现边验证：

```
Phase 3a: 换手率类（5个）→ IC 验证 + 相关性门禁
Phase 3b: 日内形态类（4个）→ IC 验证 + 相关性门禁
Phase 3c: 资金流向类（6个）→ IC 验证 + 相关性门禁
Phase 3d: 波动率高阶类（5个，含 beta）→ IC 验证 + 相关性门禁
Phase 3e: 隔夜效应类（4个）→ IC 验证 + 相关性门禁
Phase 3f: 流动性高阶类（4个）→ IC 验证 + 相关性门禁
```

每阶段输出：通过门禁的因子数 / 候选数，新增后全量 E2E 准确率变化。

### 门禁标准

- **IC 门禁**：|RankIC| 均值 > 0.02 且 t 统计量 > 2.0
- **正交性门禁**：与已有因子最大 |correlation| < 0.7
- **边际贡献**：加入后 E2E 准确率不下降（允许 0.2% 误差）

---

## 四、实施节奏

| 步骤 | 内容 | 预期产出 |
|---|---|---|
| 1 | 实现换手率 5 因子 + 门禁 | `factors/alpha191_turnover.py` |
| 2 | 实现日内形态 4 因子 + 门禁 | `factors/alpha191_intraday.py` |
| 3 | 加载 extra_data（估值+股东） | 激活 log_mcap/pb_pct/sh_change |
| 4 | 市场状态识别模块 | `models/regime.py` |
| 5 | 因子正交性筛选 | `factors/screening.py` |
| 6 | XGBoost + LightGBM 集成 | 更新 `models/trainer.py` |
| 7 | Optuna 阈值+超参联合调优 | `models/tuning.py` |
