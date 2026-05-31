# 双策略并行 — 舞 + 小市 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有舞策略旁新增小市值事件驱动策略，Web左右分栏展示，两条策略并行跑模拟盘。

**Architecture:** 新增3个文件(回测/模拟盘/引擎)，修改2个Web文件(API+模板)，DB新增1条策略+1个账户+1个run。小市引擎复用TradingConfig参数。

**Tech Stack:** Python/Pandas/SQLAlchemy + FastAPI/HTMX/ECharts

---

### Task 1: SmallCapEngine — 事件驱动引擎

**Files:**
- Create: `portfolio/small_cap_engine.py`

- [ ] **Step 1: 创建引擎类骨架**

```python
"""SmallCapEngine：小市值事件驱动引擎（低开反转 + 线性止盈）。

每日流程:
1. 构建候选池（最小市值N只）
2. 扫描入场信号（低开反转7条件）
3. 检查持仓出场（止损/移动止盈/破开盘价）
4. 生成买卖单→写入DB
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy import text
from datetime import date
from data.db import get_engine
from config.settings import TradingConfig


class SmallCapEngine:

    def __init__(self, account_id: int, run_id: int,
                 stock_num: int = 10, universe_size: int = 300):
        self.account_id = account_id
        self.run_id = run_id
        self.stock_num = stock_num
        self.universe_size = universe_size
        self.commission = TradingConfig.COMMISSION
        self.stamp_duty = TradingConfig.STAMP_DUTY
        self.slippage = 0.0015

    def seasonal_skip(self, dt) -> bool:
        m, d = dt.month, dt.day
        return m in (1, 4) or (m == 12 and d >= 20) or (m == 3 and d >= 20)

    def build_universe(self, engine, ref_date: str) -> list[str]:
        sql = f"""
            SELECT code FROM stock_basic
            WHERE is_st = FALSE
            AND list_date <= '{ref_date}'::date - INTERVAL '375 days'
            AND code NOT LIKE '688%' AND code NOT LIKE '300%'
            AND code NOT LIKE '4%' AND code NOT LIKE '8%'
            ORDER BY code LIMIT {self.universe_size}
        """
        return pd.read_sql(text(sql), engine)["code"].tolist()

    def run_daily(self, trade_date, prev_date, ohlcv_data,
                  positions: dict, cash: float,
                  peak_value: float) -> dict:
        """执行单日周期。返回更新后的 positions, cash, peak_value, orders"""
        pass
```

- [ ] **Step 2: 实现入场信号扫描**

在 `run_daily` 中添加入场逻辑：

```python
def _scan_entries(self, today_data, prev_data, positions, lookback_data,
                  industry_map, cash):
    """扫描低开反转入场信号。返回 buy_orders"""
    buy_orders = []
    if len(positions) >= self.stock_num:
        return buy_orders

    today_codes = set(today_data["code"].unique())
    prev_codes = set(prev_data["code"].unique())
    common = today_codes & prev_codes

    candidates = []
    for code in common:
        if code in positions:
            continue
        t_row = today_data[today_data["code"] == code].iloc[0]
        p_row = prev_data[prev_data["code"] == code].iloc[0]
        today_open = float(t_row["open"])
        prev_close = float(p_row["close"])
        prev_low = float(p_row["low"])
        prev_high = float(p_row["high"])

        # 条件1: 低开 > 0.75%
        gap = (today_open - prev_close) / prev_close
        if not (today_open < prev_low and gap < -0.0075):
            continue

        # 条件2: 近4日振幅 < 10%
        if code not in lookback_data:
            continue
        lb = lookback_data[code]
        if len(lb) < 2:
            continue
        lb_high = max(r["close"] for r in lb)
        lb_low = min(r["close"] for r in lb)
        if (lb_high - lb_low) / lb_low > 0.10:
            continue

        # 条件3: 昨收 > 5日均线
        if len(lb) >= 5:
            ma5 = np.mean([r["close"] for r in lb[-5:]])
        else:
            ma5 = np.mean([r["close"] for r in lb])
        if prev_close <= ma5:
            continue

        # 条件4: 昨日非涨停
        if prev_high > 0 and prev_close >= prev_high * 0.995:
            continue

        # 条件5: 行业分散
        ind = industry_map.get(code, "未知")
        existing_inds = {industry_map.get(c, "未知") for c in positions}
        if ind in existing_inds:
            continue

        candidates.append((code, gap))

    candidates.sort(key=lambda x: x[1])
    slots = self.stock_num - len(positions)
    for code, _ in candidates[:slots]:
        row = today_data[today_data["code"] == code].iloc[0]
        price = float(row["close"])
        per_stock = cash / max(slots, 1)
        shares = int(per_stock / price / 100) * 100
        cost = shares * price * (1 + self.commission + self.slippage)
        if shares >= 100 and cost <= cash:
            cash -= cost
            buy_orders.append({"code": code, "price": price, "shares": shares, "gap": gap})
    return buy_orders, cash
```

- [ ] **Step 3: 实现出场检查**

```python
def _check_exits(self, today_data, positions, today_open_map):
    """检查持仓出场条件。返回 to_sell"""
    to_sell = []
    today_codes = set(today_data["code"].unique())
    for code, pos in list(positions.items()):
        if code not in today_codes:
            continue
        row = today_data[today_data["code"] == code].iloc[0]
        close_p = float(row["close"])
        high_p = float(row["high"])
        open_p = today_open_map.get(code, pos["cost"])

        # 更新当日最高
        pos["today_high"] = max(pos.get("today_high", open_p), high_p)

        # 1. 跌停跳过
        limit_down = open_p * (0.80 if code.startswith('3') else 0.90) * 0.995
        if close_p <= limit_down:
            continue

        profit_pct = (close_p - pos["cost"]) / pos["cost"]

        # 2. 硬止损 -3%
        if profit_pct <= -0.03:
            to_sell.append((code, "hard_stop", close_p))
            continue

        # 3. 破开盘价
        if close_p < open_p:
            to_sell.append((code, "below_open", close_p))
            continue

        # 4. 线性移动止盈
        day_gain = (close_p - open_p) / open_p
        if day_gain > 0.01:
            today_high = pos.get("today_high", close_p)
            max_dd = min(0.02, 0.005 + (day_gain - 0.01) * (0.015 / 0.09))
            max_dd = max(0.003, max_dd)
            if close_p < today_high * (1 - max_dd):
                to_sell.append((code, "trailing_stop", close_p))
                continue
    return to_sell
```

- [ ] **Step 4: 实现完整的 run_daily 方法**

```python
def run_daily(self, trade_date, prev_date, today_data, prev_data,
              positions: dict, cash: float, peak_value: float,
              lookback_data: dict, industry_map: dict) -> dict:
    """返回 {positions, cash, peak_value, orders, total_value}"""
    if today_data is None or today_data.empty:
        return {"positions": positions, "cash": cash, "peak_value": peak_value,
                "orders": [], "total_value": cash}

    dt = pd.Timestamp(trade_date)

    # 季节空仓
    if self.seasonal_skip(dt):
        for code in list(positions.keys()):
            row = today_data[today_data["code"] == code]
            if not row.empty:
                p = float(row.iloc[0]["close"])
            else:
                p = positions[code]["cost"]
            cash += positions[code]["shares"] * p * (1 - self.stamp_duty - self.commission - self.slippage)
            del positions[code]
        total_value = cash
        peak_value = max(peak_value, total_value)
        return {"positions": {}, "cash": cash, "peak_value": peak_value,
                "orders": [], "total_value": total_value}

    today_open_map = dict(zip(today_data["code"], today_data["open"].astype(float)))

    # 出场
    sells = self._check_exits(today_data, positions, today_open_map)
    sell_orders = []
    for code, reason, price in sells:
        pos = positions.pop(code, None)
        if pos:
            revenue = pos["shares"] * price * (1 - self.stamp_duty - self.commission - self.slippage)
            cash += revenue
            sell_orders.append({"code": code, "direction": "SELL",
                                "price": price, "volume": pos["shares"], "reason": reason})

    # 入场
    buys, cash = self._scan_entries(today_data, prev_data, positions,
                                     lookback_data, industry_map, cash)
    buy_orders = []
    for b in buys:
        positions[b["code"]] = {"shares": b["shares"], "cost": b["price"],
                                 "today_high": b["price"], "today_open": today_open_map.get(b["code"], b["price"])}
        buy_orders.append({"code": b["code"], "direction": "BUY",
                           "price": b["price"], "volume": b["shares"], "reason": "signal"})

    # 净值计算
    position_value = 0.0
    for code, pos in positions.items():
        if code in today_data["code"].values:
            p = float(today_data[today_data["code"] == code].iloc[0]["close"])
        else:
            p = pos["cost"]
        position_value += pos["shares"] * p
    total_value = cash + position_value
    peak_value = max(peak_value, total_value)

    all_orders = sell_orders + buy_orders
    return {"positions": positions, "cash": cash, "peak_value": peak_value,
            "orders": all_orders, "total_value": total_value,
            "n_buys": len(buy_orders), "n_sells": len(sell_orders)}
```

- [ ] **Step 5: 验证编译**

```bash
python3 -c "import ast; ast.parse(open('portfolio/small_cap_engine.py').read()); print('OK')"
```

- [ ] **Step 6: Commit**

```bash
git add portfolio/small_cap_engine.py
git commit -m "feat: SmallCapEngine — 小市值事件驱动引擎"
```

---

### Task 2: 小市值回测脚本

**Files:**
- Create: `scripts/run_small_cap_backtest.py`

- [ ] **Step 1: 创建回测主逻辑**

```python
#!/usr/bin/env python
"""小市值事件驱动回测"""
import sys, os, argparse, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy import text
from data.db import get_engine
from config.settings import TradingConfig
from portfolio.small_cap_engine import SmallCapEngine

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="20200101")
    parser.add_argument("--end", default="20260528")
    parser.add_argument("--stock-num", type=int, default=10)
    parser.add_argument("--universe-size", type=int, default=300)
    args = parser.parse_args()

    engine = get_engine()
    eg = SmallCapEngine(account_id=0, run_id=0,
                        stock_num=args.stock_num, universe_size=args.universe_size)

    # 加载交易日历
    dates = pd.read_sql(text(f"""
        SELECT DISTINCT trade_date FROM stock_daily
        WHERE trade_date BETWEEN '{args.start}' AND '{args.end}'
        ORDER BY trade_date
    """), engine)
    dates["trade_date"] = pd.to_datetime(dates["trade_date"])
    trade_dates = sorted(dates["trade_date"].tolist())

    # 加载全量价格数据
    universe = eg.build_universe(engine, args.start)
    all_codes = set(universe)
    cl = ",".join([f"'{c}'" for c in all_codes])
    price_data = pd.read_sql(text(f"""
        SELECT code, trade_date, open, high, low, close, volume, amount
        FROM stock_daily WHERE code IN ({cl})
        AND trade_date BETWEEN '{args.start}' AND '{args.end}'
        ORDER BY code, trade_date
    """), engine)
    price_data["trade_date"] = pd.to_datetime(price_data["trade_date"])
    price_by_date = {dt: group.set_index("code") for dt, group in price_data.groupby("trade_date")}

    # 行业映射
    ind_df = pd.read_sql(text(f"SELECT code, industry_sw1 FROM stock_industry WHERE code IN ({cl})"), engine)
    industry_map = dict(zip(ind_df["code"], ind_df["industry_sw1"]))

    # 状态
    cash = TradingConfig.INITIAL_CASH
    positions = {}
    peak_value = cash
    equity = {}
    last_universe_month = -1

    # 回测循环
    for i, dt in enumerate(trade_dates):
        dt_str = dt.strftime("%Y-%m-%d")
        if dt.month != last_universe_month:
            universe = eg.build_universe(engine, dt_str)
            all_codes = set(universe)
            last_universe_month = dt.month

        today = price_by_date.get(dt)
        prev = price_by_date.get(trade_dates[i-1]) if i > 0 else None

        # 构建lookback_data
        lookback = {}
        for code in all_codes:
            lb = []
            for j in range(max(0,i-15), i):
                ld = trade_dates[j]
                ld_data = price_by_date.get(ld)
                if ld_data is not None and code in ld_data.index:
                    lb.append({"close": float(ld_data.loc[code, "close"]),
                               "low": float(ld_data.loc[code, "low"]),
                               "high": float(ld_data.loc[code, "high"])})
            if lb:
                lookback[code] = lb

        result = eg.run_daily(dt, trade_dates[i-1] if i>0 else dt,
                              today, prev, positions, cash, peak_value,
                              lookback, industry_map)
        positions = result["positions"]
        cash = result["cash"]
        peak_value = result["peak_value"]
        equity[dt_str] = result["total_value"]

    # 计算指标
    eq_vals = list(equity.values())
    total_return = eq_vals[-1]/eq_vals[0]-1 if len(eq_vals)>=2 else 0
    n_years = len(eq_vals)/252
    cagr = (1+total_return)**(1/max(n_years,0.2))-1
    peak = eq_vals[0]
    max_dd=0.0
    for v in eq_vals:
        if v>peak: peak=v
        max_dd=max(max_dd,(peak-v)/peak)

    print(f"总收益:{total_return:.2%} 年化:{cagr:.2%} maxDD:{max_dd:.2%}")

    # 写入DB（略，同舞策略格式）
    engine.dispose()

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 运行验证**

```bash
python3 scripts/run_small_cap_backtest.py --start 20220101 --end 20260528 --stock-num 10
```

- [ ] **Step 3: Commit**

```bash
git add scripts/run_small_cap_backtest.py
git commit -m "feat: 小市值事件驱动回测脚本"
```

---

### Task 3: DB种子数据 + Web双策略展示

**Files:**
- Modify: `scripts/init_paper_trading.py` (新增小市账户)
- Modify: `web/routes/api.py` (回测列表+模拟盘支持双策略)
- Modify: `web/templates/paper.html` (左右分栏)
- Modify: `web/templates/backtest.html` (双策略标记)

- [ ] **Step 1: 初始化小市种子数据**

```bash
python3 -c "
from data.db import get_engine; from sqlalchemy import text
e=get_engine()
with e.begin() as c:
    # 策略配置
    c.execute(text(\"INSERT INTO strategy_configs (name,type,description) VALUES ('小市','static','小市值事件驱动低开反转') ON CONFLICT (name) DO NOTHING\"))
    sid = c.execute(text(\"SELECT id FROM strategy_configs WHERE name='小市'\")).fetchone()[0]
    # 版本
    sv = c.execute(text(\"SELECT id FROM strategy_versions WHERE strategy_id=:sid\"),{'sid':sid}).fetchone()
    if not sv:
        c.execute(text(\"INSERT INTO strategy_versions (strategy_id,version,algorithm_type,feature_list_version) VALUES (:sid,'v1.0','event_driven','small_cap_v1')\"),{'sid':sid})
    # 账户
    acc = c.execute(text(\"SELECT id FROM paper_account WHERE name='小市-日频模拟盘'\")).fetchone()
    if not acc:
        c.execute(text(\"INSERT INTO paper_account (name,initial_capital,cash) VALUES ('小市-日频模拟盘',1000000,1000000)\"))
    aid = c.execute(text(\"SELECT id FROM paper_account WHERE name='小市-日频模拟盘'\")).fetchone()[0]
    # Run
    run = c.execute(text(\"SELECT id FROM paper_runs WHERE strategy_id=:sid AND status='running'\"),{'sid':sid}).fetchone()
    if not run:
        vid = c.execute(text(\"SELECT id FROM strategy_versions WHERE strategy_id=:sid\"),{'sid':sid}).fetchone()[0]
        c.execute(text(\"INSERT INTO paper_runs (strategy_id,version_id,start_date,initial_capital,status) VALUES (:sid,:vid,CURRENT_DATE,1000000,'running')\"),{'sid':sid,'vid':vid})
    rid = c.execute(text(\"SELECT id FROM paper_runs WHERE strategy_id=:sid AND status='running'\"),{'sid':sid}).fetchone()[0]
    print(f'小市: account_id={aid}, run_id={rid}')
e.dispose()
"
```

- [ ] **Step 2: Web回测列表 — 显示双策略**

修改 `web/routes/api.py` 中 `/backtest` 路由，表格增加策略名列：

```python
# 已有代码已含 strategy_name 列，无需改动
# 确认回测列表HTML中显示 name + version + CAGR
```

- [ ] **Step 3: Web模拟盘 — 左右分栏**

修改 `web/templates/paper.html`，主内容区改为双栏：

```html
<div style="display:grid; grid-template-columns: 1fr 1fr; gap: 20px;">
    <div id="paper-left">
        <h3>舞 (ML因子选股)</h3>
        <div hx-get="/api/paper-run/2?account_id=15" hx-trigger="load"></div>
    </div>
    <div id="paper-right">
        <h3>小市 (事件驱动)</h3>
        <div hx-get="/api/paper-run/3?account_id=16" hx-trigger="load"></div>
    </div>
</div>
```

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "feat: Web双策略展示 — 左右分栏+小市种子数据"
```

---

### Task 4: 小市值模拟盘每日驱动

**Files:**
- Create: `scripts/run_small_cap_paper.py`

- [ ] **Step 1: 创建小市每日脚本**

```python
#!/usr/bin/env python
"""小市值模拟盘每日驱动"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from loguru import logger
from sqlalchemy import text
from datetime import date, timedelta
from data.db import get_engine
from config.settings import TradingConfig
from portfolio.small_cap_engine import SmallCapEngine

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--account-id", type=int, default=16)
    parser.add_argument("--run-id", type=int, default=3)
    parser.add_argument("--stock-num", type=int, default=10)
    parser.add_argument("--universe-size", type=int, default=300)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    engine = get_engine()
    eg = SmallCapEngine(account_id=args.account_id, run_id=args.run_id,
                        stock_num=args.stock_num, universe_size=args.universe_size)

    # 加载最新两个交易日数据
    latest = pd.read_sql(text("SELECT MAX(trade_date) FROM stock_daily"), engine).iloc[0,0]
    prev_dt = pd.read_sql(text(f"SELECT MAX(trade_date) FROM stock_daily WHERE trade_date < '{latest}'"), engine).iloc[0,0]
    logger.info(f"数据日: {prev_dt} → {latest}")

    # 加载持仓和现金
    with engine.connect() as conn:
        row = conn.execute(text("SELECT cash FROM paper_account WHERE id=:aid"),{"aid":args.account_id}).fetchone()
        cash = float(row[0]) if row else TradingConfig.INITIAL_CASH
        pos_rows = conn.execute(text("""
            SELECT stock_code, SUM(quantity), SUM(entry_price*quantity)/SUM(quantity)
            FROM paper_positions WHERE run_id=:rid AND exit_date IS NULL
            GROUP BY stock_code
        """),{"rid":args.run_id}).fetchall()
        positions = {}
        for r in pos_rows:
            if int(r[1])>0:
                positions[str(r[0])]={"shares":int(r[1]),"cost":float(r[2]),"today_high":float(r[2])}

    # 加载价格数据
    universe = eg.build_universe(engine, str(latest))
    cl = ",".join([f"'{c}'" for c in list(universe)+list(positions.keys())])
    today_data = pd.read_sql(text(f"""
        SELECT code, open, high, low, close FROM stock_daily
        WHERE code IN ({cl}) AND trade_date='{latest}'
    """), engine)
    prev_data = pd.read_sql(text(f"""
        SELECT code, open, high, low, close FROM stock_daily
        WHERE code IN ({cl}) AND trade_date='{prev_dt}'
    """), engine)

    # 行业映射
    ind_df = pd.read_sql(text(f"SELECT code, industry_sw1 FROM stock_industry WHERE code IN ({cl})"), engine)
    industry_map = dict(zip(ind_df["code"], ind_df["industry_sw1"]))

    # lookback data
    lookback = {}
    for code in cl.split(","):
        code = code.strip("'")
        hist = pd.read_sql(text(f"""
            SELECT close, low, high FROM stock_daily
            WHERE code='{code}' AND trade_date<'{latest}' ORDER BY trade_date DESC LIMIT 15
        """), engine)
        if not hist.empty:
            lookback[code] = hist.to_dict("records")

    peak_row = conn.execute(text("SELECT COALESCE(MAX(total_value),0) FROM paper_daily_pnl WHERE account_id=:aid"),{"aid":args.account_id}).fetchone()
    peak = float(peak_row[0]) if peak_row else cash

    result = eg.run_daily(latest, prev_dt, today_data, prev_data,
                          positions, cash, peak, lookback, industry_map)

    if args.dry_run:
        logger.info(f"[DRY RUN] 买入{result['n_buys']} 卖出{result['n_sells']} 总资产{result['total_value']:,.0f}")
    else:
        # 写入DB（同舞策略模式：paper_orders/paper_positions/paper_daily_pnl）
        with engine.begin() as conn:
            for o in result["orders"]:
                conn.execute(text("""
                    INSERT INTO paper_orders (account_id,code,direction,price,volume,amount,status,note)
                    VALUES (:aid,:code,:dir,:price,:vol,:amt,'filled',:note)
                """),{"aid":args.account_id,"code":o["code"],"dir":o["direction"],
                      "price":o["price"],"vol":o["volume"],"amt":o["price"]*o["volume"],"note":o.get("reason","")})
            # 更新持仓（简化：清空重建）
            conn.execute(text("DELETE FROM paper_positions WHERE run_id=:rid"),{"rid":args.run_id})
            for code, pos in result["positions"].items():
                conn.execute(text("""
                    INSERT INTO paper_positions (run_id,stock_code,entry_date,entry_price,quantity)
                    VALUES (:rid,:sc,CURRENT_DATE,:ep,:qty)
                """),{"rid":args.run_id,"sc":code,"ep":pos["cost"],"qty":pos["shares"]})
            # 现金
            conn.execute(text("UPDATE paper_account SET cash=:c WHERE id=:aid"),{"c":result["cash"],"aid":args.account_id})
            # 净值
            conn.execute(text("""
                INSERT INTO paper_daily_pnl (account_id,trade_date,cash,position_value,total_value)
                VALUES (:aid,CURRENT_DATE,:cash,:pv,:tv)
                ON CONFLICT (account_id,trade_date) DO UPDATE SET total_value=:tv2
            """),{"aid":args.account_id,"cash":result["cash"],"pv":result["total_value"]-result["cash"],
                  "tv":result["total_value"],"tv2":result["total_value"]})
        logger.info(f"执行完成: 买入{result['n_buys']} 卖出{result['n_sells']} 总资产{result['total_value']:,.0f}")

    engine.dispose()

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/run_small_cap_paper.py
git commit -m "feat: 小市值模拟盘每日驱动脚本"
```

---

### Verification

- [ ] 回测对比页显示舞(46.1%) + 小市(待测)
- [ ] 模拟盘页左右各显示一套持仓/信号/权益曲线
- [ ] 小市回测CAGR > 0
- [ ] 两条策略各自独立运行不冲突
