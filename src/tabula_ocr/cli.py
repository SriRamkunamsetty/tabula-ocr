"""Command-line interface.

``tabula`` is the operational entry point: extract a single document, run the
evaluation suite, list schemas, or run a GPU-free demo. Every command shares the
same pipeline object the HTTP service uses, so behaviour observed from the shell
is the behaviour that ships.
"""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from typing import Any

import typer

from tabula_ocr import __version__
from tabula_ocr.config import Settings, get_settings
from tabula_ocr.eval.harness import RunConfig, load_dataset, render_markdown, run_suite
from tabula_ocr.models import ExtractionRequest
from tabula_ocr.observability import configure_logging
from tabula_ocr.pipeline import OCRService
from tabula_ocr.prompts.registry import DEFAULT_VERSION, list_versions
from tabula_ocr.schema_registry import SchemaRegistry
from tabula_ocr.vlm.client import OpenAICompatibleVLM
from tabula_ocr.vlm.fake import PassScript, ScriptedVLM

app = typer.Typer(
    add_completion=False,
    help="TABULA OCR - schema-locked document extraction with evidence.",
    no_args_is_help=True,
)


def _settings(overrides: dict[str, Any] | None = None) -> Settings:
    settings = get_settings()
    return settings.model_copy(update=overrides) if overrides else settings


@app.command()
def version() -> None:
    """Print the service version."""
    typer.echo(f"tabula-ocr {__version__}")


@app.command("schemas")
def list_schemas() -> None:
    """List extraction schemas and prompt versions."""
    settings = _settings()
    registry = SchemaRegistry(settings.schema_dir)
    typer.echo("Extraction schemas:")
    for name in registry.names():
        schema = registry.get(name)
        typer.echo(f"  {name:<12} {len(schema.fields)} fields - {schema.title}")
    typer.echo(f"\nPrompt versions (default: {DEFAULT_VERSION}):")
    for version_id, summary in list_versions():
        typer.echo(f"  {version_id:<18} {summary}")


@app.command()
def extract(
    image: Path = typer.Option(..., exists=True, readable=True, help="Page image."),
    schema: str = typer.Option("invoice", help="Extraction schema name."),
    passes: int = typer.Option(None, min=1, max=5, help="Override pass count."),
    prompt_version: str = typer.Option(None, help="Override prompt version."),
    reference_text: Path = typer.Option(
        None, exists=True, help="Trusted text layer, skips the transcription pass."
    ),
    output: Path = typer.Option(None, help="Write the JSON result here."),
) -> None:
    """Extract one document against a live vision model."""
    settings = _settings()
    configure_logging(settings.log_level, "console")
    request = ExtractionRequest(
        document_id=image.stem,
        images_b64=[base64.b64encode(image.read_bytes()).decode("ascii")],
        schema_name=schema,
        passes=passes,
        prompt_version=prompt_version,
        reference_text=reference_text.read_text(encoding="utf-8") if reference_text else None,
    )
    client = OpenAICompatibleVLM(settings)
    service = OCRService(settings, client)
    result = asyncio.run(_run_and_close(service, client, request))
    payload = result.model_dump(mode="json", exclude={"pages"})
    rendered = json.dumps(payload, indent=2)
    if output:
        output.write_text(rendered, encoding="utf-8")
        typer.echo(f"wrote {output}")
    else:
        typer.echo(rendered)
    _print_summary(result)


@app.command()
def evaluate(
    dataset: Path = typer.Option(..., exists=True, help="JSONL evaluation set."),
    report: Path = typer.Option(Path("reports/eval.json"), help="Where to write JSON."),
    thresholds: str = typer.Option("0.50,0.62,0.75", help="Abstention thresholds."),
    prompts: str = typer.Option(DEFAULT_VERSION, help="Comma-separated prompt versions."),
    passes: str = typer.Option("1,2", help="Comma-separated pass counts."),
) -> None:
    """Run the ablation grid over a labelled dataset."""
    settings = _settings()
    configure_logging(settings.log_level, "console")
    cases = load_dataset(dataset)
    grid = [
        RunConfig(prompt_version=p, passes=int(n), abstain_below=float(t))
        for p in prompts.split(",")
        for n in passes.split(",")
        for t in thresholds.split(",")
    ]
    reports = asyncio.run(
        run_suite(cases, settings, lambda: OpenAICompatibleVLM(settings), grid)
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps([r.to_dict() for r in reports], indent=2), encoding="utf-8")
    typer.echo(render_markdown(reports))
    typer.echo(f"wrote {report}")


@app.command()
def demo() -> None:
    """Run the pipeline end to end with no GPU, using a scripted model.

    Two passes are scripted: one reads the page correctly, the other invents a
    purchase order that never appears in the text. The demo shows the invented
    value being caught by grounding and abstained on, which is the behaviour the
    whole design exists to produce.
    """
    settings = _settings({"passes": 2, "environment": "dev"})
    configure_logging("INFO", "console")
    page_text = (
        "ACME INSTRUMENTS PVT LTD\n"
        "Invoice INV-2026-0041   Date: 14/03/2026\n"
        "Phone: +91 80 4123 9900\n"
        "Grand Total: Rs. 1,24,500.00\n"
    )
    scripted = ScriptedVLM(
        scripts=[
            PassScript(
                values={
                    "invoice_number": "INV-2026-0041",
                    "invoice_date": "14/03/2026",
                    "supplier_name": "ACME INSTRUMENTS PVT LTD",
                    "supplier_phone": "+91 80 4123 9900",
                    "total_amount": "1,24,500.00",
                    "currency": "INR",
                    "purchase_order": None,
                }
            ),
            PassScript(
                values={
                    "invoice_number": "INV-2026-0041",
                    "invoice_date": "14/03/2026",
                    "supplier_name": "ACME INSTRUMENTS PVT LTD",
                    "supplier_phone": "+91 80 4123 9900",
                    "total_amount": "1,24,500.00",
                    "currency": "INR",
                    "purchase_order": "PO-88213",
                }
            ),
        ],
        transcription=page_text,
    )
    service = OCRService(settings, scripted, SchemaRegistry(settings.schema_dir))
    request = ExtractionRequest(
        document_id="demo-invoice",
        images_b64=[_one_pixel_png()],
        schema_name="invoice",
        reference_text=page_text,
    )
    result = asyncio.run(service.extract(request))
    _print_summary(result)
    for field in result.fields:
        marker = "OK " if field.is_actionable else "HOLD"
        value = field.value if field.is_actionable else f"({field.reason})"
        typer.echo(f"  [{marker}] {field.name:<16} conf={field.confidence:.2f}  {value}")


async def _run_and_close(service: OCRService, client: Any, request: ExtractionRequest) -> Any:
    try:
        return await service.extract(request)
    finally:
        await client.aclose()


def _print_summary(result: Any) -> None:
    settings = get_settings()
    cost = result.usage.cost_usd(settings.gpu_hourly_rate_usd, settings.measured_tokens_per_s)
    typer.echo(
        f"\n{result.document_id}: {len(result.fields)} fields, "
        f"{result.abstention_rate:.0%} abstained, "
        f"mean confidence {result.mean_confidence:.2f}, "
        f"{result.usage.total_tokens} tokens, "
        f"{result.usage.latency_ms:.0f} ms, ${cost:.5f}"
    )


def _one_pixel_png() -> str:
    """Smallest valid PNG, so the demo needs no sample assets on disk."""
    return (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmM"
        "IQAAAABJRU5ErkJggg=="
    )


if __name__ == "__main__":  # pragma: no cover
    app()
