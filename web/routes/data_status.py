from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from web.templates_loader import templates

router = APIRouter(tags=["data_status"])


@router.get("/data", response_class=HTMLResponse)
async def data_status_page(request: Request):
    return templates.TemplateResponse(request, "data.html", {"active_page": "data"})
