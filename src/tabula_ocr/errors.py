"""Typed errors.

Every failure mode the service can produce has a class here with a stable
``code``. The API layer maps ``code`` to an HTTP status once, so adding a new
error never means touching the request handlers.
"""

from __future__ import annotations


class TabulaError(Exception):
    """Base class for all service errors."""

    code = "internal_error"
    http_status = 500

    def __init__(self, message: str, *, detail: str | None = None) -> None:
        """Create the error with a human message and optional machine detail."""
        super().__init__(message)
        self.message = message
        self.detail = detail

    def to_dict(self) -> dict[str, str | None]:
        """Render the error as a stable JSON body."""
        return {"code": self.code, "message": self.message, "detail": self.detail}


class ConfigurationError(TabulaError):
    """The deployment is misconfigured; retrying will not help."""

    code = "configuration_error"
    http_status = 500


class SchemaNotFoundError(TabulaError):
    """The caller asked for an extraction schema that is not registered."""

    code = "schema_not_found"
    http_status = 404


class InvalidDocumentError(TabulaError):
    """The uploaded payload could not be decoded into page images."""

    code = "invalid_document"
    http_status = 422


class VLMTimeoutError(TabulaError):
    """The vision model did not answer inside the configured budget."""

    code = "vlm_timeout"
    http_status = 504


class VLMProtocolError(TabulaError):
    """The vision model answered, but not in the contracted shape."""

    code = "vlm_protocol_error"
    http_status = 502


class CircuitOpenError(TabulaError):
    """The upstream model is failing; requests are being shed deliberately."""

    code = "upstream_unavailable"
    http_status = 503
