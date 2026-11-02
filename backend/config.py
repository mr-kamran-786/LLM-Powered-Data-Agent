"""Central configuration. Every setting lives here, loaded from .env."""

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# data-agent/ folder (this file is data-agent/backend/config.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",  # ignore unrelated variables in .env
    )

    # --- LLM ---
    gemini_api_key: SecretStr = SecretStr("")  # hidden when printed or logged
    llm_model: str = "gemini/gemini-3.1-flash-lite"
    llm_temperature: float | None = None  # None = model's default (Gemini 3 needs 1.0)
    llm_max_tokens: int = 4096
    # Models offered in the UI model picker (F13). Comma-separated LiteLLM model ids.
    llm_available_models: str = (
        "gemini/gemini-3.1-flash-lite,gemini/gemini-3.5-flash-lite,gemini/gemini-3-flash-preview,"
        "gemini/gemini-3.5-flash,gemini/gemini-3.1-pro-preview,ollama/qwen2.5-coder:7b"
    )
    # Tried in order when the chosen model is overloaded (503), rate-limited (429) or retired (404).
    llm_fallback_models: str = "gemini/gemini-3.5-flash-lite,gemini/gemini-3-flash-preview,gemini/gemini-3.5-flash"
    ollama_base_url: str = "http://localhost:11434"
    # Client-side limit per model; the Gemini free tier allows 15 requests/minute. 0 = unlimited.
    llm_requests_per_minute: int = 14

    # --- Agent limits (from the plan) ---
    max_agent_steps: int = 10
    max_retries: int = 3
    code_timeout_seconds: int = 60
    critic_enabled: bool = True          # LLM critique in the Verifier (F14 multi-agent mode)
    history_turns_verbatim: int = 4      # older turns are folded into a rolling summary (F8)

    # --- Cost control (non-functional requirement "Cost") ---
    session_token_budget: int = 2_000_000  # input + output tokens per session

    # --- Sandbox ---
    sandbox_memory_mb: int = 4096        # enforced on Linux/Docker via RLIMIT_AS
    tool_output_chars: int = 4000        # tool results are truncated before going back to the LLM

    # --- Data ---
    max_upload_mb: int = 500           # requirement F1
    upload_dir: Path = PROJECT_ROOT / "uploads"
    db_path: Path = PROJECT_ROOT / "uploads" / "sessions.sqlite3"
    prompts_dir: Path = PROJECT_ROOT / "prompts"
    trace_dir: Path = PROJECT_ROOT / "uploads" / "traces"

    @property
    def fallback_models(self) -> list[str]:
        return [m.strip() for m in self.llm_fallback_models.split(",") if m.strip()]

    @property
    def available_models(self) -> list[str]:
        models = [m.strip() for m in self.llm_available_models.split(",") if m.strip()]
        if self.llm_model not in models:
            models.insert(0, self.llm_model)
        return models


@lru_cache
def get_settings() -> Settings:
    """Load settings once and reuse them everywhere."""
    return Settings()
