# syntax=docker/dockerfile:1.7
#
# This image packages the tabula-ocr API service only. It is a thin FastAPI
# process that calls out to a separately deployed vLLM server on the MI300X
# (see deploy/provision_mi300x.sh and docker-compose.yml's `vlm` service) —
# the OCR model itself does not ship in this image, which keeps rebuilds of
# the API fast and keeps the accelerator-facing serving stack independently
# upgradable.

# ---- builder: resolve and build wheels in isolation from the runtime -----
FROM python:3.12-slim-bookworm AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

# Only the dependency manifest is copied first so this layer is cached across
# source-only changes, which is most commits.
COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir .

# ---- runtime: minimal, non-root, no build toolchain -----------------------
FROM python:3.12-slim-bookworm AS runtime

LABEL org.opencontainers.image.title="tabula-ocr" \
      org.opencontainers.image.description="Schema-locked document extraction with evidence and abstention." \
      org.opencontainers.image.source="https://github.com/example/tabula-ocr" \
      org.opencontainers.image.licenses="MIT"

RUN groupadd --system --gid 1000 tabula \
    && useradd --system --uid 1000 --gid tabula --create-home tabula

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TABULA_ENVIRONMENT=prod \
    TABULA_LOG_FORMAT=json \
    TABULA_SCHEMA_DIR=/app/schemas

COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY --chown=tabula:tabula schemas ./schemas

USER tabula
EXPOSE 8080

# Liveness only — readiness (which touches the GPU-backed model) is a
# Kubernetes readinessProbe against /readyz, not something the container
# healthcheck should gate its own restart on.
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8080/healthz', timeout=2)" || exit 1

ENTRYPOINT ["uvicorn", "tabula_ocr.api:create_app", "--factory", \
            "--host", "0.0.0.0", "--port", "8080"]
CMD ["--workers", "2"]
