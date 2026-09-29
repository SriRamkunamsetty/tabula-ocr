"""HTTP surface.

Three deliberate choices here:

* **Liveness and readiness are different questions.** ``/healthz`` answers "is
  this process alive" and never touches the GPU; ``/readyz`` answers "can this
  process serve traffic" and checks the upstream model and the schema registry.
  Wiring a Kubernetes liveness probe to a dependency check is how a slow model
  turns into a restart loop.
* **Errors are mapped once.** Every handler raises a :class:`TabulaError`
  subclass; a single exception handler turns it into a stable JSON body with a
  machine-readable ``code``, so clients branch on codes rather than on prose.
* **The response always carries evidence.** There is no endpoint that returns a
  bare value, because a bare value cannot be audited.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from tabula_ocr import __version__
from tabula_ocr.config import Settings, get_settings
from tabula_ocr.errors import TabulaError
from tabula_ocr.models import DocumentResult, ExtractionRequest
from tabula_ocr.observability import (
    INFLIGHT,
    REQUEST_LATENCY,
    REQUESTS,
    bind_request_id,
    configure_logging,
    get_logger,
    new_request_id,
)
from tabula_ocr.pipeline import OCRService
from tabula_ocr.prompts.registry import DEFAULT_VERSION, list_versions
from tabula_ocr.schema_registry import SchemaRegistry
from tabula_ocr.vlm.client import OpenAICompatibleVLM, VLMClient

__all__ = ["create_app", "router"]

_log = get_logger(__name__)
router = APIRouter()


class HealthResponse(BaseModel):
    """Liveness payload."""

    status: str
    service: str
    version: str


class ReadyResponse(BaseModel):
    """Readiness payload, including which dependencies were checked."""

    ready: bool
    checks: dict[str, str]


class SchemaListResponse(BaseModel):
    """Discovery payload for callers choosing an extraction schema."""

    schemas: list[str]
    prompt_versions: list[dict[str, str]]
    default_prompt_version: str


def create_app(
    settings: Settings | None = None,
    service: OCRService | None = None,
) -> FastAPI:
    """Build the ASGI application.

    Args:
        settings: Configuration override; the process singleton is used when
            omitted.
        service: Pre-built pipeline, which is how tests inject a scripted model
            without any monkeypatching.
    """
    resolved = settings or get_settings()
    configure_logging(resolved.log_level, resolved.log_format)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        registry = SchemaRegistry(resolved.schema_dir)
        if service is not None:
            app.state.service = service
        else:
            registry.load_all()  # fail fast on a malformed schema
            client: VLMClient = OpenAICompatibleVLM(resolved)
            app.state.service = OCRService(resolved, client, registry)
        app.state.settings = resolved
        app.state.started_at = time.time()
        _log.info(
            "service_started",
            version=__version__,
            environment=resolved.environment,
            model=resolved.vlm_model,
            passes=resolved.passes,
        )
        try:
            yield
        finally:
            await app.state.service._vlm.aclose()
            _log.info("service_stopped")

    app = FastAPI(
        title="TABULA OCR",
        version=__version__,
        summary="Schema-locked document extraction with evidence and abstention.",
        lifespan=lifespan,
    )
    app.include_router(router)

    @app.middleware("http")
    async def correlate(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("x-request-id") or new_request_id()
        bind_request_id(request_id)
        INFLIGHT.inc()
        try:
            response: Response = await call_next(request)
        finally:
            INFLIGHT.dec()
        response.headers["x-request-id"] = request_id
        return response

    @app.exception_handler(TabulaError)
    async def handle_tabula_error(_request: Request, exc: TabulaError) -> JSONResponse:
        _log.warning("request_failed", code=exc.code, message=exc.message)
        return JSONResponse(status_code=exc.http_status, content={"error": exc.to_dict()})

    return app


@router.get("/healthz", response_model=HealthResponse, tags=["ops"])
async def healthz() -> HealthResponse:
    """Liveness. Deliberately checks nothing external."""
    return HealthResponse(status="ok", service="tabula-ocr", version=__version__)


@router.get("/readyz", response_model=ReadyResponse, tags=["ops"])
async def readyz(request: Request) -> ReadyResponse:
    """Readiness. Verifies the schema registry and the upstream model."""
    checks: dict[str, str] = {}
    service: OCRService = request.app.state.service

    try:
        names = service.registry.names()
        checks["schemas"] = f"ok ({len(names)} loaded)"
    except Exception as exc:
        checks["schemas"] = f"error: {exc}"

    try:
        await service._vlm.complete(
            system="ping",
            user="ping",
            images_b64=[],
            temperature=0.0,
            max_tokens=1,
            operation="readiness",
        )
        checks["vlm"] = "ok"
    except Exception as exc:
        checks["vlm"] = f"error: {type(exc).__name__}"

    return ReadyResponse(ready=all(v.startswith("ok") for v in checks.values()), checks=checks)


@router.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    """Prometheus exposition."""
    body: bytes = generate_latest()
    return Response(content=body, media_type=CONTENT_TYPE_LATEST)


@router.get("/v1/schemas", response_model=SchemaListResponse, tags=["extraction"])
async def schemas(request: Request) -> SchemaListResponse:
    """List the extraction schemas and prompt versions available."""
    service: OCRService = request.app.state.service
    return SchemaListResponse(
        schemas=service.registry.names(),
        prompt_versions=[{"version": v, "summary": s} for v, s in list_versions()],
        default_prompt_version=DEFAULT_VERSION,
    )


@router.post("/v1/extract", response_model=DocumentResult, tags=["extraction"])
async def extract(request: Request, payload: ExtractionRequest) -> DocumentResult:
    """Extract a schema's fields from a document, with evidence per field."""
    service: OCRService = request.app.state.service
    started = time.perf_counter()
    try:
        result = await service.extract(payload)
    except TabulaError as exc:
        REQUESTS.labels(schema=payload.schema_name, outcome=exc.code).inc()
        raise
    REQUESTS.labels(schema=payload.schema_name, outcome="success").inc()
    REQUEST_LATENCY.labels(schema=payload.schema_name).observe(time.perf_counter() - started)
    return result
