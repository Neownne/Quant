import backtrader as bt


class RSIStrategy(bt.Strategy):
    """RSI 超买超卖策略：RSI < oversold 买入，RSI > overbought 卖出。"""

    params = (
        ("period", 14),
        ("oversold", 30),
        ("overbought", 70),
    )

    def __init__(self):
        self.rsi = bt.ind.RSI(period=self.params.period)

    def next(self):
        if not self.position:
            if self.rsi < self.params.oversold:
                self.buy()
        elif self.rsi > self.params.overbought:
            self.close()
