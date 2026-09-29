"""Structured logging, request correlation and Prometheus metrics.

Logs are JSON by default so they land in a log pipeline without a regex; a
``console`` renderer is available for local work. Every log line inside a
request carries the same ``request_id``, bound once by the API middleware via a
context variable, so a single failing document can be traced across the layout,
recognition and consensus stages without threading an argument through every
call.
"""

from __future__ import annotations

import logging
import sys
import uuid
from collections.abc import Mapping, MutableMapping
from contextvars import ContextVar
from typing import Any

import structlog
from prometheus_client import Counter, Gauge, Histogram

__all__ = [
    "FIELDS_ABSTAINED",
    "FIELDS_EXTRACTED",
    "INFLIGHT",
    "REQUESTS",
    "REQUEST_LATENCY",
    "UNGROUNDED_VALUES",
    "VLM_FAILURES",
    "VLM_LATENCY",
    "bind_request_id",
    "configure_logging",
    "current_request_id",
    "get_logger",
    "new_request_id",
]

_request_id: ContextVar[str] = ContextVar("request_id", default="-")

# --- Metrics -------------------------------------------------------------
# Label cardinality is kept deliberately low: schema names and outcomes are
# bounded sets, document ids are never used as labels.
REQUESTS = Counter(
    "tabula_requests_total", "Extraction requests handled.", ["schema", "outcome"]
)
REQUEST_LATENCY = Histogram(
    "tabula_request_latency_seconds",
    "End-to-end extraction latency.",
    ["schema"],
    buckets=(0.25, 0.5, 1, 2, 4, 8, 16, 32, 64),
)
VLM_LATENCY = Histogram(
    "tabula_vlm_latency_seconds",
    "Latency of a single vision-model call.",
    ["operation"],
    buckets=(0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32),
)
VLM_FAILURES = Counter("tabula_vlm_failures_total", "Upstream model failures.", ["reason"])
FIELDS_EXTRACTED = Counter(
    "tabula_fields_extracted_total", "Fields returned with a value.", ["schema"]
)
FIELDS_ABSTAINED = Counter(
    "tabula_fields_abstained_total", "Fields the service declined.", ["schema", "reason"]
)
UNGROUNDED_VALUES = Counter(
    "tabula_ungrounded_values_total",
    "Candidate values rejected because they were absent from the page text.",
    ["schema"],
)
INFLIGHT = Gauge("tabula_inflight_requests", "Extraction requests in flight.")


def configure_logging(level: str = "INFO", fmt: str = "json") -> None:
    """Configure structlog and the stdlib root logger once, at startup."""
    renderer: Any = (
        structlog.processors.JSONRenderer()
        if fmt == "json"
        else structlog.dev.ConsoleRenderer(colors=False)
    )
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=getattr(logging, level))
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            _inject_request_id,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level)),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def _inject_request_id(
    _logger: Any, _name: str, event: MutableMapping[str, Any]
) -> Mapping[str, Any]:
    event.setdefault("request_id", _request_id.get())
    return event


def get_logger(name: str) -> Any:
    """Return a bound logger for ``name``."""
    return structlog.get_logger(name)


def new_request_id() -> str:
    """Generate a short, collision-resistant request id."""
    return uuid.uuid4().hex[:16]


def bind_request_id(request_id: str) -> None:
    """Bind ``request_id`` for the current async context."""
    _request_id.set(request_id)


def current_request_id() -> str:
    """Return the request id bound to the current context."""
    return _request_id.get()
