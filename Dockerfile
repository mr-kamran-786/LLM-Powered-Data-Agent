# Data Agent: API + web UI in one image.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY backend ./backend
COPY frontend ./frontend
COPY prompts ./prompts
COPY eval ./eval

# Non-root user; uploads live on a volume.
RUN useradd --create-home --uid 1000 agent && mkdir -p /app/uploads && chown -R agent /app/uploads
USER agent

EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
