"""震荡网格策略：限价单阶梯式高抛低吸。

源自 Loughborough 项目的 ShockStrategy，适配到 backtrader 日线回测框架。
核心逻辑：价格每下跌 buy_step 挂一单限价买入，成交后在成本价 + sell_step 挂限价卖出。
"""
import backtrader as bt


class GridShockStrategy(bt.Strategy):
    params = (
        ("buy_step", 0.02),        # 买入阶梯（元）
        ("sell_step", 0.02),       # 卖出阶梯（元）
        ("size", 500),             # 每次交易股数
        ("max_positions", 5),      # 最大同时持仓批次数
        ("ma_period", 30),         # 均线周期
        ("ma_discount", 0.03),     # 低于均线超 3% 时加倍买入步长
    )

    def __init__(self):
        self.ma30 = bt.ind.SMA(period=self.p.ma_period)
        # ref -> (limit_price, order_obj)
        self._buy_orders: dict[int, tuple] = {}
        # ref -> (limit_price, cost_basis, order_obj)
        self._sell_orders: dict[int, tuple] = {}
        # 已成交未卖出的成本价列表
        self._cost_bases: list[float] = []

    # -- helpers --
    def _active_buy_count(self):
        return len(self._buy_orders) + len(self._cost_bases)

    def _min_buy_price(self):
        bp = [p for p, _ in self._buy_orders.values()]
        return min(bp + self._cost_bases) if (bp or self._cost_bases) else float("inf")

    def _max_buy_price(self):
        bp = [p for p, _ in self._buy_orders.values()]
        return max(bp) if bp else 0.0

    def _min_cost(self):
        return min(self._cost_bases) if self._cost_bases else float("inf")

    def _covered_costs(self):
        return {cb for _, cb, _ in self._sell_orders.values()}

    # -- order notifications --
    def notify_order(self, order):
        ref = order.ref
        if order.status == order.Completed:
            if order.isbuy():
                fill_price = order.executed.price
                self._cost_bases.append(fill_price)
                self._buy_orders.pop(ref, None)
            else:
                if ref in self._sell_orders:
                    _, cb, _ = self._sell_orders.pop(ref)
                    if cb in self._cost_bases:
                        self._cost_bases.remove(cb)

        elif order.status in (order.Canceled, order.Margin, order.Rejected):
            if ref in self._sell_orders:
                _, cb, _ = self._sell_orders.pop(ref)
                if cb not in self._cost_bases:
                    self._cost_bases.append(cb)
            self._buy_orders.pop(ref, None)

    # -- main loop --
    def next(self):
        price = self.data.close[0]
        ma_val = self.ma30[0]

        # ---- 撤单 ----
        for ref, (bp, bo) in list(self._buy_orders.items()):
            if price - bp > self.p.buy_step * 3:
                self.cancel(bo)

        for ref, (sp, cb, so) in list(self._sell_orders.items()):
            if sp - price > self.p.sell_step * 2.5:
                self.cancel(so)

        # ---- 对未配置卖单的成本价挂限价卖单 ----
        for cb in list(self._cost_bases):
            if cb in self._covered_costs():
                continue
            sell_price = round(cb + self.p.sell_step, 2)
            so = self.sell(exectype=bt.Order.Limit, price=sell_price, size=self.p.size)
            self._sell_orders[so.ref] = (sell_price, cb, so)

        # ---- 买入系数（MA30 偏离调整步长） ----
        buy_coef = 1.0
        if ma_val > 0:
            dev = (price - ma_val) / price if price > 0 else 0
            if dev > self.p.ma_discount:
                return  # 价格远高于均线，暂停买入
            elif dev < -self.p.ma_discount:
                buy_coef = 2.0  # 价格远低于均线，加大买入步长
        buy_step = self.p.buy_step * buy_coef

        if self._active_buy_count() >= self.p.max_positions:
            return

        min_buy = self._min_buy_price()
        min_cost = self._min_cost()
        max_buy = self._max_buy_price()

        # 初始买入
        if not self._cost_bases and not self._buy_orders:
            buy_price = round(price - buy_step, 2)
            bo = self.buy(exectype=bt.Order.Limit, price=buy_price, size=self.p.size)
            self._buy_orders[bo.ref] = (buy_price, bo)
            return

        # 策略1：现价低于最低买入价和最低成本价，继续向下挂买单
        if min_cost < 900 and price < min_cost and price < min_buy:
            base = min(min_buy, min_cost)
            if price < base - buy_step:
                buy_price = round(price, 2)
            else:
                buy_price = round(base - buy_step, 2)
            bo = self.buy(exectype=bt.Order.Limit, price=buy_price, size=self.p.size)
            self._buy_orders[bo.ref] = (buy_price, bo)
            return

        # 策略2：最高买单与最低成本价间有空洞，在空洞中挂买单
        if max_buy > 0 and min_cost < 900 and (min_cost - max_buy) > buy_step * 1.8:
            if price > max_buy:
                buy_price = round(max_buy + buy_step, 2)
                bo = self.buy(exectype=bt.Order.Limit, price=buy_price, size=self.p.size)
                self._buy_orders[bo.ref] = (buy_price, bo)
            elif price < min_cost:
                buy_price = round(min_cost - buy_step, 2)
                bo = self.buy(exectype=bt.Order.Limit, price=buy_price, size=self.p.size)
                self._buy_orders[bo.ref] = (buy_price, bo)
