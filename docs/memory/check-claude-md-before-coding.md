---
name: check-claude-md-before-coding
description: Always read and follow CLAUDE.md rules before writing any code
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 27010fa6-19f6-4418-9a57-b2ce03efc1d3
---

每次写代码前必须先 Review [CLAUDE.md](mdc:CLAUDE.md) 中的铁律和规范：涨停阈值用板别感知 `_get_limit()`、回测用指定脚本、新增因子放 `factors/limit_up.py`、不猜测接口先 grep/read 确认、复用现有函数不新造轮子。

**Why:** 用户明确要求，避免重蹈之前 7 处硬编码涨停阈值不一致的问题。

**How to apply:** 每次改代码前，先 Read CLAUDE.md 确认相关规则，特别是：
- [[threshold-consistency]] 涨停阈值统一用 `_get_limit()`
- 不改 `bt_backtest.py` 交易逻辑
- 新增策略写独立脚本
- 复用 `load_daily_data()` 等现有函数
