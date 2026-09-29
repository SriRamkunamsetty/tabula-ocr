"""Offline evaluation harness.

Runs a labelled dataset through the pipeline and writes a machine-readable
report. Two things make it useful rather than ceremonial:

* It **sweeps configurations** — prompt version by pass count by abstention
  threshold — in one run, which turns "I think v3 is better" into a table.
* It records **cost per document** from real token counts and the measured
  throughput of the deployment, so an accuracy gain can be weighed against what
  it costs to serve.

Dataset format is a JSONL file, one document per line::

    {"document_id": "inv-001",
     "schema": "invoice",
     "image_path": "pages/inv-001.png",
     "reference_text": "optional trusted text layer",
     "truth": {"invoice_number": "INV-2026-0041", "purchase_order": null}}

A ``null`` in ``truth`` means the field is genuinely absent from the document.
Those entries are what make the hallucination metric meaningful: a model that
invents a purchase order for an invoice that has none is caught precisely here.
"""

from __future__ import annotations

import base64
import dataclasses
import json
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tabula_ocr.config import Settings
from tabula_ocr.eval.metrics import EvalCounters, score_document
from tabula_ocr.models import ExtractionRequest
from tabula_ocr.observability import get_logger
from tabula_ocr.pipeline import OCRService
from tabula_ocr.schema_registry import SchemaRegistry
from tabula_ocr.vlm.client import VLMClient

__all__ = ["EvalCase", "RunConfig", "RunReport", "load_dataset", "render_markdown", "run_suite"]

_log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class EvalCase:
    """One labelled document."""

    document_id: str
    schema: str
    image_b64: str
    truth: dict[str, str | None]
    reference_text: str | None = None

    @classmethod
    def from_json(cls, data: dict[str, Any], root: Path) -> EvalCase:
        """Build a case, loading the page image relative to the dataset file."""
        image_b64 = data.get("image_b64")
        if not image_b64:
            path = root / str(data["image_path"])
            image_b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        return cls(
            document_id=str(data["document_id"]),
            schema=str(data["schema"]),
            image_b64=image_b64,
            truth=dict(data.get("truth") or {}),
            reference_text=data.get("reference_text"),
        )


@dataclass(frozen=True, slots=True)
class RunConfig:
    """One point in the ablation grid."""

    prompt_version: str
    passes: int
    abstain_below: float

    @property
    def label(self) -> str:
        """Short identifier used as the ablation table row header."""
        return f"{self.prompt_version} | {self.passes}p | t={self.abstain_below:.2f}"


@dataclass
class RunReport:
    """Metrics for one configuration across the whole dataset."""

    config: RunConfig
    metrics: dict[str, float | int]
    documents: int
    total_tokens: int
    wall_clock_s: float
    cost_usd: float
    failures: list[str] = dataclasses.field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Render the report as JSON for the committed results file."""
        return {
            "config": dataclasses.asdict(self.config),
            "metrics": self.metrics,
            "documents": self.documents,
            "total_tokens": self.total_tokens,
            "wall_clock_s": round(self.wall_clock_s, 3),
            "cost_usd": round(self.cost_usd, 6),
            "failures": self.failures,
        }


def load_dataset(path: Path) -> list[EvalCase]:
    """Read a JSONL dataset, resolving image paths relative to its directory."""
    root = path.parent
    cases: list[EvalCase] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            cases.append(EvalCase.from_json(json.loads(line), root))
        except (KeyError, json.JSONDecodeError, OSError) as exc:
            raise ValueError(f"{path}:{line_number} is not a usable case: {exc}") from exc
    if not cases:
        raise ValueError(f"dataset {path} contains no cases")
    return cases


async def run_suite(
    cases: Sequence[EvalCase],
    settings: Settings,
    vlm_factory: Any,
    configs: Iterable[RunConfig],
    registry: SchemaRegistry | None = None,
) -> list[RunReport]:
    """Evaluate every configuration over every case.

    Args:
        cases: Labelled documents.
        settings: Base settings; each config overrides the fields it varies.
        vlm_factory: Zero-argument callable returning a fresh :class:`VLMClient`.
            A factory rather than an instance so a scripted client can be reset
            between configurations.
        configs: The ablation grid.
        registry: Shared schema registry, built from ``settings`` when omitted.

    Returns:
        One :class:`RunReport` per configuration, in input order.
    """
    shared_registry = registry or SchemaRegistry(settings.schema_dir)
    field_types = _field_types(shared_registry, cases)
    reports: list[RunReport] = []

    for config in configs:
        tuned = settings.model_copy(
            update={
                "passes": config.passes,
                "abstain_below": config.abstain_below,
            }
        )
        client: VLMClient = vlm_factory()
        service = OCRService(tuned, client, shared_registry)
        counters = EvalCounters()
        failures: list[str] = []
        tokens = 0
        started = time.perf_counter()

        for case in cases:
            request = ExtractionRequest(
                document_id=case.document_id,
                images_b64=[case.image_b64],
                schema_name=case.schema,
                reference_text=case.reference_text,
                passes=config.passes,
                prompt_version=config.prompt_version,
            )
            try:
                result = await service.extract(request)
            except Exception as exc:
                failures.append(f"{case.document_id}: {type(exc).__name__}: {exc}")
                _log.warning("eval_case_failed", document_id=case.document_id, error=str(exc))
                continue
            tokens += result.usage.total_tokens
            for outcome in score_document(result, case.truth, field_types):
                counters.add(outcome)

        elapsed = time.perf_counter() - started
        cost = (settings.gpu_hourly_rate_usd / 3600.0) * (
            tokens / settings.measured_tokens_per_s
        )
        reports.append(
            RunReport(
                config=config,
                metrics=counters.to_dict(),
                documents=len(cases) - len(failures),
                total_tokens=tokens,
                wall_clock_s=elapsed,
                cost_usd=cost,
                failures=failures,
            )
        )
        await client.aclose()

    return reports


def _field_types(registry: SchemaRegistry, cases: Sequence[EvalCase]) -> dict[str, str]:
    """Map every field name in the dataset to its declared type."""
    types: dict[str, str] = {}
    for schema_name in {case.schema for case in cases}:
        for spec in registry.get(schema_name).fields:
            types[spec.name] = spec.type
    return types


def render_markdown(reports: Sequence[RunReport]) -> str:
    """Render the ablation table that goes into the write-up."""
    header = (
        "| configuration | coverage | precision | hallucination | abstention precision "
        "| mean CER | tokens | $/1k docs |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    rows = []
    for report in reports:
        metrics = report.metrics
        per_1k = (report.cost_usd / report.documents * 1000) if report.documents else 0.0
        rows.append(
            f"| {report.config.label} "
            f"| {metrics['coverage']:.1%} "
            f"| {metrics['precision']:.1%} "
            f"| {metrics['hallucination_rate']:.1%} "
            f"| {metrics['abstention_precision']:.1%} "
            f"| {metrics['mean_cer']:.3f} "
            f"| {report.total_tokens:,} "
            f"| ${per_1k:,.2f} |"
        )
    return header + "\n".join(rows) + "\n"
