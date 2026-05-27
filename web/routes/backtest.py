from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from web.templates_loader import templates

router = APIRouter(tags=["backtest"])


@router.get("/backtest", response_class=HTMLResponse)
async def backtest_page(request: Request):
    return templates.TemplateResponse(request, "backtest.html", {"active_page": "backtest"})
