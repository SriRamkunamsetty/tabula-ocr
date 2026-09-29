"""Domain models for the TABULA OCR service.

Every value that leaves this service is wrapped in a :class:`FieldResult`, which
carries not only the extracted value but the evidence supporting it: which pixel
region it came from, the verbatim source quote, how many independent decoding
passes agreed on it, and whether the service is confident enough to stand behind
it. Callers are expected to branch on :attr:`FieldResult.status` rather than
assuming a value is present.

The models are intentionally strict (``extra="forbid"``) so that a drifting
model response fails loudly at the boundary instead of silently propagating.
"""

from __future__ import annotations

import enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "BoundingBox",
    "DocumentResult",
    "ExtractionRequest",
    "FieldResult",
    "FieldStatus",
    "PageResult",
    "PassObservation",
    "Provenance",
    "UsageStats",
]


class _Strict(BaseModel):
    """Base model with production-safe defaults."""

    model_config = ConfigDict(extra="forbid", frozen=False, str_strip_whitespace=True)


class BoundingBox(_Strict):
    """Axis-aligned region of a page, in normalised ``[0, 1]`` coordinates.

    Normalised coordinates keep provenance valid regardless of the DPI a page was
    rasterised at, which matters because the layout pass and the recognition pass
    may run at different resolutions.
    """

    x0: float = Field(ge=0.0, le=1.0)
    y0: float = Field(ge=0.0, le=1.0)
    x1: float = Field(ge=0.0, le=1.0)
    y1: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _ordered(self) -> BoundingBox:
        """Reject boxes with non-positive extent."""
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError("bounding box must have positive width and height")
        return self

    @property
    def area(self) -> float:
        """Fraction of the page covered by this box."""
        return (self.x1 - self.x0) * (self.y1 - self.y0)

    def to_pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        """Return the box as integer pixel coordinates for an image of this size."""
        return (
            round(self.x0 * width),
            round(self.y0 * height),
            round(self.x1 * width),
            round(self.y1 * height),
        )

    def iou(self, other: BoundingBox) -> float:
        """Intersection-over-union with ``other``; 0.0 when the boxes are disjoint."""
        ix0, iy0 = max(self.x0, other.x0), max(self.y0, other.y0)
        ix1, iy1 = min(self.x1, other.x1), min(self.y1, other.y1)
        if ix1 <= ix0 or iy1 <= iy0:
            return 0.0
        intersection = (ix1 - ix0) * (iy1 - iy0)
        union = self.area + other.area - intersection
        return intersection / union if union > 0 else 0.0


class FieldStatus(enum.StrEnum):
    """Outcome of extracting a single field.

    ``ABSTAINED`` is a first-class success: the service determined it could not
    support a value with evidence and declined to guess. Downstream systems
    should route abstentions to human review, never treat them as errors.
    """

    EXTRACTED = "extracted"
    ABSTAINED = "abstained"
    NOT_PRESENT = "not_present"


class Provenance(_Strict):
    """Where a value came from, and how well it is supported by the page."""

    page: int = Field(ge=1, description="1-based page number.")
    bbox: BoundingBox | None = Field(
        default=None, description="Pixel region the model attributed the value to."
    )
    quote: str | None = Field(
        default=None, description="Verbatim text the model claims to have read."
    )
    grounded: bool = Field(
        default=False,
        description="True when the value was located in the page's reference text.",
    )
    grounding_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Best fuzzy match against page text."
    )


class PassObservation(_Strict):
    """One decoding pass's opinion about one field.

    Consensus is computed over a list of these, so keeping the pass identity and
    the raw (pre-normalisation) value makes every confidence score auditable.
    """

    pass_id: str
    raw_value: Any
    normalized_value: str | None = None
    bbox: BoundingBox | None = None
    quote: str | None = None


class FieldResult(_Strict):
    """A single extracted field together with its supporting evidence."""

    name: str
    value: Any = None
    status: FieldStatus = FieldStatus.EXTRACTED
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    agreement: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Fraction of passes that agreed."
    )
    provenance: Provenance | None = None
    reason: str | None = Field(
        default=None, description="Why the service abstained, when it did."
    )
    observations: list[PassObservation] = Field(default_factory=list)

    @model_validator(mode="after")
    def _abstention_has_no_value(self) -> FieldResult:
        """Guarantee that a non-extracted field never carries a value."""
        if self.status is not FieldStatus.EXTRACTED and self.value is not None:
            raise ValueError("non-extracted fields must not carry a value")
        return self

    @property
    def is_actionable(self) -> bool:
        """True when a caller may consume the value without human review."""
        return self.status is FieldStatus.EXTRACTED


class UsageStats(_Strict):
    """Token and timing accounting for one request."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    passes: int = 0
    latency_ms: float = 0.0
    model: str = "unknown"

    @property
    def total_tokens(self) -> int:
        """Prompt plus completion tokens."""
        return self.prompt_tokens + self.completion_tokens

    def cost_usd(self, gpu_hourly_rate: float, throughput_tokens_per_s: float) -> float:
        """Amortised GPU cost of this request.

        Self-hosted serving has no per-token price list, so cost is derived from
        the hourly rate of the accelerator and the measured throughput of the
        deployment. See ``benchmarks/bench_throughput.py`` for how the
        throughput figure is produced.
        """
        if throughput_tokens_per_s <= 0:
            return 0.0
        gpu_seconds = self.total_tokens / throughput_tokens_per_s
        return (gpu_hourly_rate / 3600.0) * gpu_seconds


class PageResult(_Strict):
    """Full-page transcription plus the regions the layout pass identified."""

    page: int = Field(ge=1)
    text: str = ""
    width: int = Field(default=0, ge=0)
    height: int = Field(default=0, ge=0)
    regions: list[BoundingBox] = Field(default_factory=list)


class DocumentResult(_Strict):
    """The response envelope returned by the service."""

    document_id: str
    schema_name: str
    fields: list[FieldResult] = Field(default_factory=list)
    pages: list[PageResult] = Field(default_factory=list)
    usage: UsageStats = Field(default_factory=UsageStats)
    warnings: list[str] = Field(default_factory=list)

    def field(self, name: str) -> FieldResult | None:
        """Return the named field, or ``None`` when the schema did not define it."""
        return next((f for f in self.fields if f.name == name), None)

    @property
    def abstention_rate(self) -> float:
        """Fraction of schema fields the service declined to answer."""
        if not self.fields:
            return 0.0
        declined = sum(1 for f in self.fields if f.status is FieldStatus.ABSTAINED)
        return declined / len(self.fields)

    @property
    def mean_confidence(self) -> float:
        """Mean confidence across fields the service did answer."""
        answered = [f.confidence for f in self.fields if f.is_actionable]
        return sum(answered) / len(answered) if answered else 0.0


class ExtractionRequest(_Strict):
    """Input contract for ``POST /v1/extract``."""

    document_id: str = Field(min_length=1, max_length=128)
    images_b64: list[str] = Field(
        min_length=1, max_length=50, description="Base64-encoded page images."
    )
    schema_name: str = Field(min_length=1, max_length=64)
    reference_text: str | None = Field(
        default=None,
        description=(
            "Optional trusted text layer (e.g. from a born-digital PDF). When "
            "supplied it is used for grounding instead of a transcription pass."
        ),
    )
    passes: int | None = Field(default=None, ge=1, le=5)
    prompt_version: str | None = None

    @field_validator("images_b64")
    @classmethod
    def _non_empty_images(cls, value: list[str]) -> list[str]:
        """Reject blank page payloads at the boundary."""
        if any(not item.strip() for item in value):
            raise ValueError("image payloads must not be empty")
        return value
