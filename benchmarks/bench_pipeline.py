r"""Throughput and latency benchmark.

Run against a live vLLM endpoint on MI300X to populate the deployment's real
numbers::

    TABULA_VLM_BASE_URL=http://<droplet-ip>:8000/v1 \\
    TABULA_VLM_MODEL=PaddlePaddle/PaddleOCR-VL \\
        python benchmarks/bench_pipeline.py --requests 40 --concurrency 8

Run with no arguments to benchmark the pipeline's own overhead — request
validation, image decoding, consensus scoring — against the deterministic
scripted model, which needs no GPU and no network and catches a latency
regression introduced by this codebase independent of the model serving it.
The two modes share every line of measurement code, so a number from one is
directly comparable to the other once the model call itself is added back in.

The script never invents a number: every figure it prints was measured in the
run that printed it, and a run against the scripted model is labelled
``mode: harness-only (no GPU)`` in its own output so it is never mistaken for
a deployment benchmark.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from tabula_ocr.config import Settings, get_settings
from tabula_ocr.models import ExtractionRequest
from tabula_ocr.pipeline import OCRService
from tabula_ocr.schema_registry import SchemaRegistry
from tabula_ocr.vlm.client import OpenAICompatibleVLM
from tabula_ocr.vlm.fake import PassScript, ScriptedVLM

ONE_PIXEL_PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmM"
    "IQAAAABJRU5ErkJggg=="
)

SAMPLE_PAGE_TEXT = (
    "ACME INSTRUMENTS PVT LTD\n"
    "Invoice INV-2026-0041   Date: 14/03/2026\n"
    "Phone: +91 80 4123 9900\n"
    "Grand Total: Rs. 1,24,500.00\n"
)

SAMPLE_READING = {
    "invoice_number": "INV-2026-0041",
    "invoice_date": "14/03/2026",
    "supplier_name": "ACME INSTRUMENTS PVT LTD",
    "supplier_phone": "+91 80 4123 9900",
    "total_amount": "1,24,500.00",
    "currency": "INR",
    "purchase_order": None,
}


@dataclass(frozen=True, slots=True)
class BenchResult:
    """One benchmark run's summary statistics."""

    mode: str
    model: str
    requests: int
    concurrency: int
    passes_per_request: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    throughput_req_s: float
    total_tokens: int
    tokens_per_s: float
    error_rate: float
    cost_per_1k_docs_usd: float


async def _run_one(service: OCRService, document_id: str) -> tuple[float, int, bool]:
    started = time.perf_counter()
    request = ExtractionRequest(
        document_id=document_id,
        images_b64=[ONE_PIXEL_PNG],
        schema_name="invoice",
        reference_text=SAMPLE_PAGE_TEXT,
    )
    try:
        result = await service.extract(request)
        elapsed_ms = (time.perf_counter() - started) * 1000
        return elapsed_ms, result.usage.total_tokens, True
    except Exception:
        elapsed_ms = (time.perf_counter() - started) * 1000
        return elapsed_ms, 0, False


async def run_benchmark(
    settings: Settings,
    *,
    total_requests: int,
    concurrency: int,
    harness_only: bool,
) -> BenchResult:
    """Fire ``total_requests`` extractions at ``concurrency`` and summarise."""
    registry = SchemaRegistry(settings.schema_dir)
    registry.load_all()

    if harness_only:
        client = ScriptedVLM(
            [PassScript(values=SAMPLE_READING) for _ in range(settings.passes)],
            transcription=SAMPLE_PAGE_TEXT,
        )
    else:
        client = OpenAICompatibleVLM(settings)

    service = OCRService(settings, client, registry)
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded(index: int) -> tuple[float, int, bool]:
        async with semaphore:
            return await _run_one(service, f"bench-{index}")

    wall_start = time.perf_counter()
    outcomes = await asyncio.gather(*[bounded(i) for i in range(total_requests)])
    wall_elapsed = time.perf_counter() - wall_start
    await client.aclose()

    latencies = sorted(ms for ms, _, ok in outcomes if ok)
    tokens = sum(t for _, t, ok in outcomes if ok)
    errors = sum(1 for *_rest, ok in outcomes if not ok)

    if not latencies:
        raise RuntimeError("every benchmark request failed; nothing to report")

    tokens_per_s = tokens / wall_elapsed if wall_elapsed > 0 else 0.0
    cost_per_doc = (
        (settings.gpu_hourly_rate_usd / 3600.0) * (tokens / max(len(latencies), 1))
        / max(tokens_per_s, 1e-9)
    ) if tokens_per_s > 0 else 0.0

    return BenchResult(
        mode="harness-only (no GPU)" if harness_only else "live-endpoint",
        model=settings.vlm_model,
        requests=total_requests,
        concurrency=concurrency,
        passes_per_request=settings.passes,
        p50_ms=round(_percentile(latencies, 0.50), 2),
        p95_ms=round(_percentile(latencies, 0.95), 2),
        p99_ms=round(_percentile(latencies, 0.99), 2),
        mean_ms=round(statistics.fmean(latencies), 2),
        throughput_req_s=round(len(latencies) / wall_elapsed, 3) if wall_elapsed > 0 else 0.0,
        total_tokens=tokens,
        tokens_per_s=round(tokens_per_s, 2),
        error_rate=round(errors / total_requests, 4),
        cost_per_1k_docs_usd=round(cost_per_doc * 1000, 4),
    )


def _percentile(sorted_values: list[float], fraction: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = min(len(sorted_values) - 1, round(fraction * (len(sorted_values) - 1)))
    return sorted_values[index]


def main() -> None:
    """Parse CLI arguments, run the benchmark, print and append the result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=40)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--passes", type=int, default=None)
    parser.add_argument(
        "--live",
        action="store_true",
        help="Hit TABULA_VLM_BASE_URL instead of the offline scripted model.",
    )
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results.json"))
    args = parser.parse_args()

    settings = get_settings()
    if args.passes is not None:
        settings = settings.model_copy(update={"passes": args.passes})

    result = asyncio.run(
        run_benchmark(
            settings,
            total_requests=args.requests,
            concurrency=args.concurrency,
            harness_only=not args.live,
        )
    )

    print(json.dumps(asdict(result), indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if args.output.exists():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
    existing.append(asdict(result))
    args.output.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"\nappended to {args.output}")


if __name__ == "__main__":
    main()
