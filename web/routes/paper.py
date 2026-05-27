from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from web.templates_loader import templates

router = APIRouter(tags=["paper"])


@router.get("/paper", response_class=HTMLResponse)
async def paper_page(request: Request):
    return templates.TemplateResponse(request, "paper.html", {"active_page": "paper"})
