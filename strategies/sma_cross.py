import backtrader as bt


class SMACross(bt.Strategy):
    """双均线交叉策略：短期均线上穿长期均线买入，下穿卖出。"""

    params = (
        ("fast", 5),
        ("slow", 20),
    )

    def __init__(self):
        self.fast_ma = bt.ind.SMA(period=self.params.fast)
        self.slow_ma = bt.ind.SMA(period=self.params.slow)
        self.crossover = bt.ind.CrossOver(self.fast_ma, self.slow_ma)

    def next(self):
        if not self.position:
            if self.crossover > 0:
                self.buy()
        elif self.crossover < 0:
            self.close()
