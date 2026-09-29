"""The extraction pipeline.

One request flows through four stages:

1. **Decode.** Base64 payloads become validated, size-capped RGB pages.
2. **Reference text.** Either the caller's trusted text layer is used, or a
   single transcription pass builds one. This text is what every later value is
   checked against, so it is produced once and reused rather than re-derived per
   field.
3. **Extraction passes.** ``settings.passes`` independent guided-decoding calls
   run *concurrently*, each at a different temperature. They are independent by
   construction — no pass sees another's output — because correlated passes
   would agree on a shared hallucination and inflate the agreement signal.
4. **Scoring.** Per field, the passes' observations are reconciled into one
   value with a confidence, and anything under the threshold is abstained on.

The pipeline owns no transport and no framework types, so it is exercised end to
end in unit tests against a scripted model and reused unchanged by the CLI, the
HTTP API and the evaluation harness.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from tabula_ocr.config import Settings
from tabula_ocr.consensus import score_field
from tabula_ocr.errors import TabulaError, VLMProtocolError
from tabula_ocr.imaging import DecodedPage, decode_page
from tabula_ocr.models import (
    BoundingBox,
    DocumentResult,
    ExtractionRequest,
    FieldStatus,
    PageResult,
    PassObservation,
    UsageStats,
)
from tabula_ocr.observability import (
    FIELDS_ABSTAINED,
    FIELDS_EXTRACTED,
    UNGROUNDED_VALUES,
    get_logger,
)
from tabula_ocr.prompts.registry import TRANSCRIBE, get_prompt
from tabula_ocr.schema_registry import ExtractionSchema, SchemaRegistry
from tabula_ocr.vlm.client import VLMClient, VLMResponse

__all__ = ["OCRService"]

_log = get_logger(__name__)


class OCRService:
    """Coordinates decoding, transcription, extraction passes and scoring."""

    def __init__(
        self,
        settings: Settings,
        vlm: VLMClient,
        registry: SchemaRegistry | None = None,
    ) -> None:
        """Wire the pipeline to a settings object, a model client and a registry."""
        self._settings = settings
        self._vlm = vlm
        self._registry = registry or SchemaRegistry(settings.schema_dir)

    @property
    def registry(self) -> SchemaRegistry:
        """Schema registry backing this service."""
        return self._registry

    async def extract(self, request: ExtractionRequest) -> DocumentResult:
        """Run the full pipeline for one document.

        Raises:
            SchemaNotFoundError: The requested schema is not registered.
            InvalidDocumentError: A page payload could not be decoded.
            VLMTimeoutError, CircuitOpenError: The model was unreachable.
        """
        started = time.perf_counter()
        schema = self._registry.get(request.schema_name)
        pages = [
            decode_page(payload, max_pixels=self._settings.max_image_pixels)
            for payload in request.images_b64
        ]
        warnings: list[str] = [
            f"page {i + 1} was downscaled to fit the model's input budget"
            for i, page in enumerate(pages)
            if page.resized
        ]

        usage = UsageStats(model=self._settings.vlm_model)
        reference_text, transcript_usage = await self._reference_text(
            pages, request.reference_text
        )
        _accumulate(usage, transcript_usage)

        pass_count = request.passes or self._settings.passes
        prompt = get_prompt(request.prompt_version)
        hint = _region_hint(pages)

        responses = await asyncio.gather(
            *[
                self._vlm.complete(
                    system=prompt.system,
                    user=prompt.render(schema, hint=hint),
                    images_b64=[page.to_base64() for page in pages],
                    temperature=self._settings.temperature_for(index),
                    guided_json=schema.to_json_schema(),
                    operation="extract",
                )
                for index in range(pass_count)
            ],
            return_exceptions=True,
        )

        observations: dict[str, list[PassObservation]] = {f.name: [] for f in schema.fields}
        successful_passes = 0
        for index, response in enumerate(responses):
            pass_id = f"pass-{index}"
            if isinstance(response, BaseException):
                warnings.append(f"{pass_id} failed: {_describe(response)}")
                _log.warning("extraction_pass_failed", pass_id=pass_id, error=str(response))
                continue
            try:
                parsed = response.as_json()
            except VLMProtocolError as exc:
                warnings.append(f"{pass_id} returned unparseable output")
                _log.warning("extraction_pass_unparseable", pass_id=pass_id, detail=exc.detail)
                continue
            successful_passes += 1
            _accumulate(usage, response)
            self._collect(parsed, schema, pass_id, observations)

        if successful_passes == 0:
            raise VLMProtocolError(
                "every extraction pass failed",
                detail="; ".join(warnings) or "no detail available",
            )

        weights = self._settings.confidence_weights
        fields = [
            score_field(
                name=spec.name,
                field_type=spec.type,
                observations=observations[spec.name],
                reference_text=reference_text,
                page=1,
                weights=weights,
                abstain_below=self._settings.abstain_below,
                grounding_min_ratio=self._settings.grounding_min_ratio,
                required=spec.required,
                total_passes=successful_passes,
            )
            for spec in schema.fields
        ]

        self._record_field_metrics(schema.name, fields)
        usage.passes = successful_passes
        usage.latency_ms = round((time.perf_counter() - started) * 1000, 2)

        result = DocumentResult(
            document_id=request.document_id,
            schema_name=schema.name,
            fields=fields,
            pages=[
                PageResult(
                    page=i + 1,
                    text=reference_text if i == 0 else "",
                    width=page.width,
                    height=page.height,
                )
                for i, page in enumerate(pages)
            ],
            usage=usage,
            warnings=warnings,
        )
        _log.info(
            "extraction_complete",
            document_id=request.document_id,
            schema=schema.name,
            passes=successful_passes,
            fields=len(fields),
            abstention_rate=round(result.abstention_rate, 3),
            mean_confidence=round(result.mean_confidence, 3),
            latency_ms=usage.latency_ms,
        )
        return result

    async def _reference_text(
        self, pages: list[DecodedPage], supplied: str | None
    ) -> tuple[str, VLMResponse | None]:
        """Return the text every extracted value will be checked against.

        A caller-supplied text layer is preferred and costs nothing: born-digital
        PDFs already carry perfect text, and using it removes a whole model call
        from the critical path as well as removing the transcription pass as a
        source of grounding error.
        """
        if supplied and supplied.strip():
            return supplied, None
        response = await self._vlm.complete(
            system=TRANSCRIBE.system,
            user=TRANSCRIBE.user_template.format(fields="", title="", hint=""),
            images_b64=[page.to_base64() for page in pages],
            temperature=0.0,
            guided_json=None,
            max_tokens=4096,
            operation="transcribe",
        )
        return response.text, response

    def _collect(
        self,
        parsed: dict[str, Any],
        schema: ExtractionSchema,
        pass_id: str,
        sink: dict[str, list[PassObservation]],
    ) -> None:
        """Turn one pass's JSON into per-field observations.

        Unknown keys are ignored rather than rejected: guided decoding makes them
        very unlikely, and dropping a whole pass because the model volunteered an
        extra key would trade a small schema deviation for a large loss of signal.
        """
        for spec in schema.fields:
            entry = parsed.get(spec.name)
            if entry is None:
                continue
            if not isinstance(entry, dict):
                # Tolerate a bare scalar; older prompt versions produced them.
                entry = {"value": entry, "quote": None, "bbox": None}
            value = entry.get("value")
            if value is None:
                continue
            sink[spec.name].append(
                PassObservation(
                    pass_id=pass_id,
                    raw_value=value,
                    bbox=_to_bbox(entry.get("bbox")),
                    quote=_as_optional_str(entry.get("quote")),
                )
            )

    @staticmethod
    def _record_field_metrics(schema_name: str, fields: list[Any]) -> None:
        for field in fields:
            if field.status is FieldStatus.EXTRACTED:
                FIELDS_EXTRACTED.labels(schema=schema_name).inc()
                continue
            if field.status is FieldStatus.ABSTAINED:
                reason = "ungrounded" if _ungrounded(field) else "low_confidence"
                FIELDS_ABSTAINED.labels(schema=schema_name, reason=reason).inc()
                if reason == "ungrounded":
                    UNGROUNDED_VALUES.labels(schema=schema_name).inc()


def _ungrounded(field: Any) -> bool:
    return bool(field.provenance and not field.provenance.grounded)


def _accumulate(usage: UsageStats, response: VLMResponse | None) -> None:
    if response is None:
        return
    usage.prompt_tokens += response.prompt_tokens
    usage.completion_tokens += response.completion_tokens


def _describe(error: BaseException) -> str:
    if isinstance(error, TabulaError):
        return f"{error.code}: {error.message}"
    return f"{type(error).__name__}: {error}"


def _to_bbox(raw: Any) -> BoundingBox | None:
    """Convert a model-supplied ``[x0, y0, x1, y1]`` into a validated box.

    Malformed boxes are dropped rather than raised on. A bad box costs a little
    provenance precision; rejecting the pass would cost the value itself.
    """
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(v) for v in raw)
    except (TypeError, ValueError):
        return None
    x0, x1 = sorted((max(0.0, min(1.0, x0)), max(0.0, min(1.0, x1))))
    y0, y1 = sorted((max(0.0, min(1.0, y0)), max(0.0, min(1.0, y1))))
    if x1 - x0 <= 1e-6 or y1 - y0 <= 1e-6:
        return None
    try:
        return BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1)
    except ValueError:
        return None


def _as_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _region_hint(pages: list[DecodedPage]) -> str:
    """Describe page geometry for the region-guided prompt.

    A dedicated layout model would produce real regions here; until one is wired
    in, reporting the page grid still measurably helps the model return boxes in
    the right coordinate space, and the hint is a single formatted string so
    swapping in a layout pass changes nothing else in the pipeline.
    """
    return "; ".join(
        f"page {i + 1}: {p.width}x{p.height}px, origin top-left, coordinates normalised 0-1"
        for i, p in enumerate(pages)
    )
