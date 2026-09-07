# syntax=docker/dockerfile:1

# ---------- builder: install all dependencies into a self-contained venv ----------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# CPU-only torch wheel is ~1GB smaller than the default CUDA build. Install it
# first so `pip install -r requirements.txt` sees torch already satisfied and
# never pulls the CUDA variant from PyPI.
RUN pip install "torch>=2.2,<3" --index-url https://download.pytorch.org/whl/cpu

# Dependency layer: only rebuilt when requirements.txt changes.
COPY requirements.txt ./
RUN pip install -r requirements.txt

# App layer: package metadata needs pyproject.toml and README.md present.
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-deps .

# ---------- test: full dependency set + pytest (used by CI) ----------
FROM builder AS test

RUN pip install "pytest>=8" "ruff>=0.6"
COPY tests/ ./tests/
COPY examples/ ./examples/

CMD ["pytest"]

# ---------- runtime: slim final image ----------
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/hf_cache \
    PATH="/opt/venv/bin:$PATH"

COPY --from=builder /opt/venv /opt/venv

# Non-root user with writable homes for artifacts, mounted data, and the HF cache.
RUN useradd --create-home --shell /bin/bash finroute

WORKDIR /app
COPY --chown=finroute:finroute configs/ ./configs/
COPY --chown=finroute:finroute examples/ ./examples/
RUN mkdir -p /app/artifacts /app/data /app/hf_cache \
    && chown -R finroute:finroute /app

USER finroute

VOLUME ["/app/artifacts", "/app/hf_cache"]

# Entrypoint is the FinRoute CLI; pick a subcommand per run:
#   docker run --rm finroute build-index --pdf-dir data/pdfs ...
ENTRYPOINT ["finroute"]
CMD ["--help"]
