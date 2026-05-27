from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from web.templates_loader import templates

router = APIRouter(tags=["dashboard"])


@router.get("/", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    return templates.TemplateResponse(request, "dashboard.html", {"active_page": "dashboard"})
