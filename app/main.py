import os
import subprocess
import sys
import threading
import time

import schedule
import streamlit as st

# 确保项目根目录在 Python 路径中，页面文件才能 import data/strategies/app 等模块
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

st.set_page_config(page_title="Quant 量化交易系统", page_icon="📊", layout="wide")

# ---- 每日同步线程 ----
def _run_sync():
    """在子进程中运行股票日线同步（ETF/基金同步暂时禁用）"""
    for mode in ["stock-daily"]:  # etf-daily, fund-nav disabled for now
        subprocess.run(
            [sys.executable, "-m", "data.sync", "--mode", mode],
            cwd=_PROJECT_ROOT,
        )


def _scheduler_loop():
    schedule.every().day.at("18:00").do(_run_sync)
    while True:
        schedule.run_pending()
        time.sleep(60)


if "scheduler_started" not in st.session_state:
    t = threading.Thread(target=_scheduler_loop, daemon=True)
    t.start()
    st.session_state.scheduler_started = True

# ---- 页面导航 ----
pages = [
    st.Page("pages/1_📈_Watchlist.py", title="实时报价", icon="📈"),
    st.Page("pages/2_📊_Charts.py", title="K线图", icon="📊"),
    st.Page("pages/3_🧪_Backtest.py", title="策略回测", icon="🧪"),
    st.Page("pages/4_📋_Paper_Trade.py", title="模拟盘", icon="📋"),
    st.Page("pages/5_📝_Strategy_Editor.py", title="策略编辑器", icon="📝"),
    st.Page("pages/6_📦_Stock_Pools.py", title="股票池", icon="📦"),
    st.Page("pages/7_🔴_Recorder.py", title="数据录制", icon="🔴"),
    st.Page("pages/8_📊_ML_Monitor.py", title="ML策略监控", icon="📊"),
]

pg = st.navigation(pages)
pg.run()
