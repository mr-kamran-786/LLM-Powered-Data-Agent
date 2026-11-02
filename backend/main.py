"""FastAPI entry point. Run with: uvicorn backend.main:app --reload"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import plotly
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.agent import prompts
from backend.api.routes import chat, export, health, sessions
from backend.config import PROJECT_ROOT, get_settings
from backend.sandbox.manager import pool

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
FRONTEND_DIR = PROJECT_ROOT / "frontend" / "web"
PLOTLY_JS = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    settings = get_settings()                        # fail fast if .env is broken
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    print(f"Data Agent starting. Model: {settings.llm_model}")
    yield
    # --- Shutdown ---
    pool.shutdown()                                  # stop every sandbox worker
    print("Data Agent shutting down.")


app = FastAPI(
    title="Data Agent API",
    description="LLM-powered data analysis agent",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow a separate dev frontend (Vite/Streamlit) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:8501"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(sessions.router)
app.include_router(chat.router)
app.include_router(export.router)


@app.get("/api/config", tags=["meta"])
def config() -> dict:
    s = get_settings()
    return {
        "default_model": s.llm_model,
        "models": s.available_models,
        "critic_enabled": s.critic_enabled,
        "max_upload_mb": s.max_upload_mb,
        "max_retries": s.max_retries,
        "prompt_versions": prompts.versions(),
    }


@app.get("/vendor/plotly.min.js", include_in_schema=False)
def plotly_js() -> FileResponse:
    """Plotly ships its JS bundle inside the Python package: the UI works offline."""
    return FileResponse(PLOTLY_JS, media_type="application/javascript")


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
