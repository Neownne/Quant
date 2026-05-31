import sys
import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from web.routes import dashboard, backtest, paper, data_status, factors, api, etf_monitor

app = FastAPI(title="Quant Monitor")

app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

app.include_router(dashboard.router)
app.include_router(backtest.router)
app.include_router(paper.router)
app.include_router(data_status.router)
app.include_router(factors.router)
app.include_router(api.router)
app.include_router(etf_monitor.router)
