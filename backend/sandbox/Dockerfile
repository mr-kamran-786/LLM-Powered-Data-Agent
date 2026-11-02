# Sandbox image for SANDBOX_BACKEND=docker: one container per session, started by
# backend/sandbox/manager.py with --network none, CPU/memory limits and a read-only data mount.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 MPLCONFIGDIR=/tmp/mpl
RUN pip install pandas pyarrow numpy scipy scikit-learn statsmodels plotly matplotlib duckdb tabulate \
    && useradd --uid 1000 --create-home sandbox
COPY worker.py /sandbox/worker.py
USER sandbox
WORKDIR /work
