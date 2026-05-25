"""
实盘数据录制。
用法：
    python -m data.recorder                    # 录制分钟K线（默认模式）
    python -m data.recorder --mode tick        # 收盘后抓取逐笔数据
    python -m data.recorder --watchlist 000001,600519,300750  # 自选股

也支持编程调用：创建 RecorderSession 后在后台线程运行。
"""
import argparse
import queue
import threading
import time
from datetime import date, datetime
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError as FutureTimeoutError

import pandas as pd
from loguru import logger

from config.settings import DataConfig
from data.db import init_db, upsert_df, get_engine
from data.fetcher import fetch_minute_data, fetch_tick_data

DEFAULT_WATCHLIST = [
    "000001", "000002", "000858", "002415",
    "300750", "600036", "600519", "601318",
]


# ---------- 时间判断 ----------

def is_trading_time() -> bool:
    now = datetime.now()
    morning = now.replace(hour=9, minute=30, second=0, microsecond=0)
    morning_end = now.replace(hour=11, minute=30, second=0, microsecond=0)
    afternoon = now.replace(hour=13, minute=0, second=0, microsecond=0)
    afternoon_end = now.replace(hour=15, minute=0, second=0, microsecond=0)
    return (morning <= now <= morning_end) or (afternoon <= now <= afternoon_end)


def is_trading_day() -> bool:
    return date.today().weekday() < 5


def trading_status() -> str:
    """返回当前交易状态描述。"""
    if not is_trading_day():
        return "非交易日（周末/节假日）"
    now = datetime.now()
    if now.hour < 9 or (now.hour == 9 and now.minute < 30):
        return "盘前（距开盘还有 {} 分钟）".format(
            (datetime.now().replace(hour=9, minute=30, second=0) - now).seconds // 60
        )
    if is_trading_time():
        return "交易中"
    if now.hour < 13:
        return "午休"
    if now.hour >= 15:
        return "已收盘"
    return "未知"


# ---------- 录制会话 ----------

class RecorderSession:
    """可编程控制的录制会话，在后台线程中运行。"""

    def __init__(self, watchlist: list[str], mode: str = "minute", period: str = "1"):
        self.watchlist = watchlist
        self.mode = mode          # "minute" | "tick"
        self.period = period      # "1"|"5"|"15"|"30"|"60"
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.log_queue: queue.Queue = queue.Queue()  # 线程安全的日志
        self._engine = None

    def _emit(self, level: str, msg: str):
        """向日志队列写入一条消息。"""
        try:
            self.log_queue.put_nowait({
                "time": datetime.now().strftime("%H:%M:%S"),
                "level": level,
                "msg": msg,
            })
        except queue.Full:
            pass

    def _run_minute(self):
        """分钟K线录制循环（在后台线程中运行）。"""
        self._emit("info", f"分钟K线录制启动，{len(self.watchlist)} 只标的，周期 {self.period}min")
        engine = get_engine()

        while not self._stop_event.is_set():
            if not is_trading_day():
                self._emit("info", "非交易日，录制停止")
                break

            if not is_trading_time():
                self._emit("info", "非交易时段，10 分钟后重试...")
                self._stop_event.wait(600)
                continue

            round_start = time.time()
            n_total = 0

            with ProcessPoolExecutor(max_workers=4) as executor:
                futures = {
                    executor.submit(fetch_minute_data, code, self.period): code
                    for code in self.watchlist
                }
                for future in as_completed(futures):
                    code = futures[future]
                    try:
                        df = future.result(timeout=30)
                    except FutureTimeoutError:
                        self._emit("error", f"{code} 分钟数据超时")
                        continue
                    except Exception as e:
                        self._emit("error", f"{code} 失败: {e}")
                        continue

                    if df.empty:
                        continue
                    today = pd.Timestamp.now().normalize()
                    df = df[df["trade_time"] >= today]
                    if not df.empty:
                        n = upsert_df(df, "stock_minute", engine)
                        n_total += n

            elapsed = time.time() - round_start
            self._emit("success", f"本轮写入 {n_total} 条，耗时 {elapsed:.1f}s，{len(self.watchlist)} 只标的")

            self._stop_event.wait(60)

        engine.dispose()
        self._emit("info", "分钟K线录制已停止")

    def _run_tick(self):
        """收盘后抓取逐笔数据。"""
        self._emit("info", f"逐笔数据抓取启动，{len(self.watchlist)} 只标的")
        engine = get_engine()
        trade_date = date.today()

        for code in self.watchlist:
            if self._stop_event.is_set():
                break
            try:
                df = fetch_tick_data(code, trade_date)
                if not df.empty:
                    n = upsert_df(df, "stock_tick", engine)
                    self._emit("success", f"{code} +{n} 笔逐笔")
                else:
                    self._emit("info", f"{code} 无逐笔数据")
            except Exception as e:
                self._emit("error", f"{code} 逐笔失败: {e}")
            time.sleep(0.5)

        engine.dispose()
        self._emit("info", "逐笔数据抓取完成")

    def start(self):
        """启动录制（后台线程）。"""
        if self._thread and self._thread.is_alive():
            self._emit("warning", "录制已在运行中")
            return

        self._stop_event.clear()
        target = self._run_minute if self.mode == "minute" else self._run_tick
        self._thread = threading.Thread(target=target, daemon=True)
        self._thread.start()

    def stop(self):
        """停止录制。"""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)
        self._emit("info", "录制已停止")

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def get_logs(self) -> list[dict]:
        """取出日志队列中所有待消费的消息。"""
        msgs = []
        while True:
            try:
                msgs.append(self.log_queue.get_nowait())
            except queue.Empty:
                break
        return msgs


# ============================================================
#  CLI 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="实盘数据录制工具")
    parser.add_argument("--mode", choices=["minute", "tick"], default="minute")
    parser.add_argument("--watchlist", default="", help="自选股代码，逗号分隔")
    parser.add_argument("--period", default="1", choices=["1", "5", "15", "30", "60"])
    args = parser.parse_args()

    watchlist = (
        [c.strip() for c in args.watchlist.split(",") if c.strip()]
        if args.watchlist
        else DEFAULT_WATCHLIST
    )

    logger.info("初始化数据库表结构 ...")
    init_db()

    session = RecorderSession(watchlist, mode=args.mode, period=args.period)
    logger.info("按 Ctrl+C 停止录制")

    # CLI 模式：同步阻塞运行
    session.start()
    try:
        while session.is_running:
            # 消费并打印日志
            for entry in session.get_logs():
                level = entry["level"]
                msg = entry["msg"]
                if level == "error":
                    logger.error(msg)
                elif level == "warning":
                    logger.warning(msg)
                elif level == "success":
                    logger.success(msg)
                else:
                    logger.info(msg)
            time.sleep(0.5)
    except KeyboardInterrupt:
        session.stop()


if __name__ == "__main__":
    main()
