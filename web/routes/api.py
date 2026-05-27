from fastapi import APIRouter

router = APIRouter(prefix="/api", tags=["api"])


@router.get("/ping")
async def ping():
    return {"status": "ok"}
