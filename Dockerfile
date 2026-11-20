FROM ghcr.io/astral-sh/uv:0.10.10 AS uv
FROM python:3.12-slim AS runtime

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
COPY --from=uv /uv /bin/uv
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-dev --no-managed-python \
    && groupadd -g 10001 plantguard && useradd -u 10001 -g 10001 plantguard
COPY backend/app ./backend/app
COPY ml ./ml
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=20s --timeout=5s CMD ["python","-c","import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)"]
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--limit-concurrency", "32"]
