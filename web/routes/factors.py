from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from web.templates_loader import templates

router = APIRouter(tags=["factors"])


@router.get("/factors", response_class=HTMLResponse)
async def factors_page(request: Request):
    return templates.TemplateResponse(request, "factors.html", {"active_page": "factors"})
