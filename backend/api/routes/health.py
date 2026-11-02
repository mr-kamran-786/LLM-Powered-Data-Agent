"""Health check: confirms the server is up and shows the active config."""

from fastapi import APIRouter

from backend.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict:
    settings = get_settings()
    return {
        "status": "ok",
        "llm_model": settings.llm_model,
        "upload_dir_exists": settings.upload_dir.exists(),
    }
