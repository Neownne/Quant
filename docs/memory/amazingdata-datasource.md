---
name: amazingdata-datasource
description: 中国银河证券星耀数智 AmazingData — 替代 tushare 的行业/板块/行情数据源
metadata: 
  node_type: memory
  type: reference
  originSessionId: 27010fa6-19f6-4418-9a57-b2ce03efc1d3
---

## 概述

**AmazingData** 是中国银河证券星耀数智量化平台的金融数据 SDK（Python），提供行情、基础数据、财务、行业指数、可转债等全品类数据。

- SDK: `AmazingData` + `tgw` (wheel 安装)
- 支持 Python 3.8-3.13, Linux/Windows
- 需账号密码 + IP/Port 登录
- PDF 手册: `~/Desktop/AmazingData开发手册.pdf` (148页, V1.0.24, 2025-12-16)

## 登录方式

```python
import AmazingData as ad
ad.login(username='...', password='...', host='ip', port=port)
```

## 本项目会用到的关键接口

### 1. 行业数据（替代 tushare，补齐 stock_industry 表）

| 接口 | 说明 |
|------|------|
| `InfoData.get_industry_base_info()` | 行业指数基本信息：INDEX_CODE, LEVEL_TYPE(1/2/3级), LEVEL1_NAME, LEVEL2_NAME, LEVEL3_NAME |
| `InfoData.get_industry_constituent(code_list)` | 行业指数成分股：INDEX_CODE, CON_CODE, INDATE, OUTDATE |

→ 可完全替代 tushare 的 `pro.industry()` + 申万分类

### 2. 股票基础信息（补齐 stock_basic 表）

| 接口 | 说明 |
|------|------|
| `BaseData.get_code_list(security_type)` | 每日最新代码表（早9点前更新） |
| `InfoData.get_stock_basic(code_list)` | 证券基础信息：MARKET_CODE, SECURITY_NAME, LISTPLATE_NAME, LISTDATE, IS_LISTED |
| `InfoData.get_history_stock_status(code_list)` | 历史证券状态：PRE_CLOSE, HIGH_LIMITED, LOW_LIMITED, IS_ST, IS_SUSP, IS_WD, IS_XR |

### 3. 实时行情（替代腾讯 API，用于午盘扫描）

| 接口 | 说明 |
|------|------|
| `MarketData.get_snapshot(code_list)` | Level-1 实时快照（含涨跌幅、量价、委比等），比腾讯 API 更全更快 |
| `MarketData.get_kline(code_list, period)` | 实时 K 线数据 |

### 4. 历史行情（可替代 akshare 日线同步）

| 接口 | 说明 |
|------|------|
| `MarketData.get_history_kline(code_list, start, end)` | 历史 K 线，2013年至今 |
| `InfoData.get_backward_factor(code_list)` | 后复权因子 |

### 5. 指数成分股

| 接口 | 说明 |
|------|------|
| `InfoData.get_index_constituent(code_list)` | 交易所指数成分股，约600+指数 |
| `InfoData.get_index_weight(code_list)` | 指数权重（仅支持上证50/沪深300/中证500/中证800/中证1000） |

### 6. 其他有价值的数据

- 财务数据：资产负债表/现金流量表/利润表/业绩快报/业绩预告
- 股东股本：十大股东/股东户数/股本结构/股权冻结/限售解禁
- 融资融券：成交汇总/交易明细
- 交易异动：龙虎榜/大宗交易
- 可转债全套数据
- 金融算子：数学/统计/时序函数

## vs tushare 对比

| 维度 | tushare (120积分) | AmazingData |
|------|-------------------|-------------|
| 行业分类 | ❌ 无权限 | ✅ L1/L2/L3 行业指数 |
| 基础信息 | stock_basic 1次/小时 | ✅ 每日代码表+证券基础 |
| 实时行情 | ❌ | ✅ Level-1 快照 |
| 历史K线 | ✅ 50次/分钟 | ✅ |
| 财务数据 | ❌ | ✅ 全套 |
| 股东股本 | ❌ | ✅ |
| 安装方式 | pip | wheel (银河内网/网盘) |
| 联网要求 | 公网 | 需银河 VPN/内网 |

## 部署注意

用户提到 7 月份在虚拟机上接入。需要：
1. 银河证券 VPN 或内网环境
2. 从银河网盘下载 wheel 文件
3. 安装 `tgw` + `AmazingData` wheel
4. 获取账号密码
5. 在 [[industry-sector-data-gap]] 的基础上，用 AmazingData 完全填补行业/板块数据缺口

## 相关记忆

- [[industry-sector-data-gap]] — 当前行业/板块数据缺口（47.6% 无行业、SW L2 全空）
- [[check-claude-md-before-coding]] — 写代码前看铁律
