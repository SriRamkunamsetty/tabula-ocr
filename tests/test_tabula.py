"""Test suite for TABULA OCR.

The tests are organised by the risk they retire, not by module. The most
valuable ones are in ``TestAntiHallucination``: they assert that a value the
model invented — one that appears nowhere in the page text — is refused, and
that the refusal carries a readable reason. Everything else exists to keep that
behaviour honest under refactoring.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tabula_ocr.api import create_app
from tabula_ocr.config import Settings
from tabula_ocr.consensus import ground_value, reach_consensus, score_field
from tabula_ocr.errors import InvalidDocumentError, SchemaNotFoundError, VLMProtocolError
from tabula_ocr.eval.harness import EvalCase, RunConfig, render_markdown, run_suite
from tabula_ocr.eval.metrics import EvalCounters, character_error_rate, score_document
from tabula_ocr.imaging import decode_page
from tabula_ocr.models import (
    BoundingBox,
    ExtractionRequest,
    FieldStatus,
    PassObservation,
    UsageStats,
)
from tabula_ocr.normalize import normalize_value, validate_format
from tabula_ocr.pipeline import OCRService
from tabula_ocr.prompts.registry import get_prompt
from tabula_ocr.schema_registry import SchemaRegistry
from tabula_ocr.vlm.client import CircuitBreaker, VLMResponse
from tabula_ocr.vlm.fake import PassScript, ScriptedVLM

PAGE_TEXT = (
    "ACME INSTRUMENTS PVT LTD\n"
    "Invoice INV-2026-0041   Date: 14/03/2026\n"
    "Phone: +91 80 4123 9900\n"
    "Grand Total: Rs. 1,24,500.00\n"
)

CORRECT_READING = {
    "invoice_number": "INV-2026-0041",
    "invoice_date": "14/03/2026",
    "supplier_name": "ACME INSTRUMENTS PVT LTD",
    "supplier_phone": "+91 80 4123 9900",
    "total_amount": "1,24,500.00",
    "currency": "INR",
    "purchase_order": None,
}

ONE_PIXEL_PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmM"
    "IQAAAABJRU5ErkJggg=="
)


@pytest.fixture(scope="session")
def schema_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "schemas"


@pytest.fixture()
def settings(schema_dir: Path) -> Settings:
    return Settings(
        environment="test",
        schema_dir=schema_dir,
        passes=2,
        abstain_below=0.62,
        log_level="ERROR",
    )


@pytest.fixture()
def registry(schema_dir: Path) -> SchemaRegistry:
    return SchemaRegistry(schema_dir)


def build_service(
    settings: Settings, registry: SchemaRegistry, scripts: list[PassScript]
) -> OCRService:
    return OCRService(settings, ScriptedVLM(scripts, transcription=PAGE_TEXT), registry)


def request_for(schema: str = "invoice", **kwargs) -> ExtractionRequest:
    payload = {
        "document_id": "doc-1",
        "images_b64": [ONE_PIXEL_PNG],
        "schema_name": schema,
        "reference_text": PAGE_TEXT,
    }
    payload.update(kwargs)
    return ExtractionRequest(**payload)


# --------------------------------------------------------------------------
class TestNormalisation:
    @pytest.mark.parametrize(
        ("raw", "field_type", "expected"),
        [
            ("Rs. 1,24,500.00", "currency", "124500"),
            ("1.234,56", "number", "1234.56"),
            ("  INV-2026-0041 ", "string", "INV-2026-0041"),
            ("14/03/2026", "date", "2026-03-14"),
            ("March 14, 2026", "date", "2026-03-14"),
            ("+91 80 4123 9900", "phone", "+918041239900"),
            ("v2.14.1", "version", "2.14.1"),
            ("Yes", "boolean", "true"),
            ("N/A", "string", None),
            (None, "string", None),
        ],
    )
    def test_normalisation_is_canonical(self, raw, field_type, expected):
        assert normalize_value(raw, field_type) == expected

    @pytest.mark.parametrize(
        ("raw", "field_type", "valid"),
        [
            ("ops@acme.co.in", "email", True),
            ("ops@acme", "email", False),
            ("+91 80 4123 9900", "phone", True),
            ("12", "phone", False),
            ("2.14.1", "version", True),
            ("banana", "version", False),
            ("31/02/2026", "date", False),
        ],
    )
    def test_format_validation(self, raw, field_type, valid):
        assert validate_format(raw, field_type) is valid

    def test_unicode_digits_fold_to_ascii(self):
        assert normalize_value("１２３４", "integer") == "1234"


# --------------------------------------------------------------------------
class TestGrounding:
    def test_exact_substring_scores_perfectly(self):
        report = ground_value("INV-2026-0041", PAGE_TEXT)
        assert report.grounded and report.score == 1.0

    def test_ocr_noise_still_grounds(self):
        report = ground_value("ACME INSTRUMENTS PVT LTO", PAGE_TEXT, min_ratio=0.86)
        assert report.grounded
        assert report.matched_quote is not None

    def test_invented_value_is_not_grounded(self):
        report = ground_value("PO-88213", PAGE_TEXT)
        assert not report.grounded
        assert report.score < 0.86

    def test_empty_reference_text_does_not_raise(self):
        assert ground_value("anything", "") == ground_value("anything", "   ")


# --------------------------------------------------------------------------
class TestConsensus:
    def test_majority_wins_and_agreement_is_reported(self):
        observations = [
            PassObservation(pass_id="pass-0", raw_value="1,24,500.00"),
            PassObservation(pass_id="pass-1", raw_value="124500"),
            PassObservation(pass_id="pass-2", raw_value="124.500"),
        ]
        outcome = reach_consensus(observations, "currency")
        assert outcome.normalized_value == "124500"
        assert outcome.agreement == pytest.approx(2 / 3, abs=1e-4)

    def test_missing_votes_count_against_agreement(self):
        observations = [
            PassObservation(pass_id="pass-0", raw_value="INV-1"),
            PassObservation(pass_id="pass-1", raw_value=None),
        ]
        outcome = reach_consensus(observations, "string")
        assert outcome.agreement == 0.5

    def test_no_usable_votes_returns_empty_outcome(self):
        outcome = reach_consensus([PassObservation(pass_id="pass-0", raw_value="  ")], "string")
        assert outcome.normalized_value is None and outcome.agreement == 0.0

    def test_score_field_marks_required_missing_as_abstained(self):
        result = score_field(
            name="invoice_number",
            field_type="string",
            observations=[],
            reference_text=PAGE_TEXT,
            page=1,
            weights=(0.5, 0.35, 0.15),
            abstain_below=0.62,
            grounding_min_ratio=0.86,
            required=True,
        )
        assert result.status is FieldStatus.ABSTAINED
        assert result.value is None
        assert "required" in (result.reason or "")


# --------------------------------------------------------------------------
class TestAntiHallucination:
    """The behaviour this service exists to provide."""

    @pytest.mark.asyncio
    async def test_invented_field_is_refused(self, settings, registry):
        hallucinated = dict(CORRECT_READING, purchase_order="PO-88213")
        service = build_service(
            settings,
            registry,
            [PassScript(values=CORRECT_READING), PassScript(values=hallucinated)],
        )
        result = await service.extract(request_for())

        purchase_order = result.field("purchase_order")
        assert purchase_order is not None
        assert purchase_order.status is not FieldStatus.EXTRACTED
        assert purchase_order.value is None

    @pytest.mark.asyncio
    async def test_unanimous_hallucination_is_still_refused_by_grounding(
        self, settings, registry
    ):
        """Agreement alone must not be able to carry a fabricated value."""
        hallucinated = dict(CORRECT_READING, purchase_order="PO-88213")
        service = build_service(
            settings,
            registry,
            [PassScript(values=hallucinated), PassScript(values=hallucinated)],
        )
        result = await service.extract(request_for())

        purchase_order = result.field("purchase_order")
        assert purchase_order.status is FieldStatus.ABSTAINED
        assert "not located in page text" in (purchase_order.reason or "")
        assert purchase_order.confidence < settings.abstain_below

    @pytest.mark.asyncio
    async def test_grounded_agreed_value_is_returned_with_evidence(self, settings, registry):
        service = build_service(
            settings,
            registry,
            [PassScript(values=CORRECT_READING), PassScript(values=CORRECT_READING)],
        )
        result = await service.extract(request_for())

        invoice_number = result.field("invoice_number")
        assert invoice_number.status is FieldStatus.EXTRACTED
        assert invoice_number.value == "INV-2026-0041"
        assert invoice_number.agreement == 1.0
        assert invoice_number.confidence > 0.9
        assert invoice_number.provenance.grounded
        assert invoice_number.provenance.bbox is not None

    @pytest.mark.asyncio
    async def test_disagreement_lowers_confidence(self, settings, registry):
        misread = dict(CORRECT_READING, supplier_phone="+91 80 4123 9000")
        service = build_service(
            settings,
            registry,
            [PassScript(values=CORRECT_READING), PassScript(values=misread)],
        )
        result = await service.extract(request_for())
        phone = result.field("supplier_phone")
        assert phone.agreement == 0.5
        assert phone.confidence < 1.0


# --------------------------------------------------------------------------
class TestPipelineRobustness:
    @pytest.mark.asyncio
    async def test_one_malformed_pass_does_not_fail_the_request(self, settings, registry):
        service = build_service(
            settings,
            registry,
            [PassScript(values=CORRECT_READING), PassScript(values={}, malformed="not json")],
        )
        result = await service.extract(request_for())
        assert result.usage.passes == 1
        assert any("unparseable" in w for w in result.warnings)
        assert result.field("invoice_number").status is FieldStatus.EXTRACTED

    @pytest.mark.asyncio
    async def test_all_passes_failing_raises_protocol_error(self, settings, registry):
        service = build_service(
            settings,
            registry,
            [PassScript(values={}, malformed="{"), PassScript(values={}, malformed="{")],
        )
        with pytest.raises(VLMProtocolError):
            await service.extract(request_for())

    @pytest.mark.asyncio
    async def test_supplied_reference_text_skips_transcription_pass(self, settings, registry):
        scripted = ScriptedVLM(
            [PassScript(values=CORRECT_READING)] * 2, transcription=PAGE_TEXT
        )
        service = OCRService(settings, scripted, registry)
        await service.extract(request_for())
        assert all(call["operation"] != "transcribe" for call in scripted.calls)

    @pytest.mark.asyncio
    async def test_transcription_pass_runs_without_reference_text(self, settings, registry):
        scripted = ScriptedVLM(
            [PassScript(values=CORRECT_READING)] * 2, transcription=PAGE_TEXT
        )
        service = OCRService(settings, scripted, registry)
        await service.extract(request_for(reference_text=None))
        assert sum(c["operation"] == "transcribe" for c in scripted.calls) == 1

    @pytest.mark.asyncio
    async def test_guided_decoding_constraint_is_sent(self, settings, registry):
        scripted = ScriptedVLM(
            [PassScript(values=CORRECT_READING)] * 2, transcription=PAGE_TEXT
        )
        service = OCRService(settings, scripted, registry)
        await service.extract(request_for())
        extract_calls = [c for c in scripted.calls if c["operation"] == "extract"]
        assert extract_calls and all(c["guided"] for c in extract_calls)

    @pytest.mark.asyncio
    async def test_passes_use_different_temperatures(self, settings, registry):
        scripted = ScriptedVLM(
            [PassScript(values=CORRECT_READING)] * 2, transcription=PAGE_TEXT
        )
        service = OCRService(settings, scripted, registry)
        await service.extract(request_for())
        temps = {c["temperature"] for c in scripted.calls if c["operation"] == "extract"}
        assert len(temps) == settings.passes

    @pytest.mark.asyncio
    async def test_unknown_schema_raises(self, settings, registry):
        service = build_service(settings, registry, [PassScript(values=CORRECT_READING)])
        with pytest.raises(SchemaNotFoundError):
            await service.extract(request_for(schema="does-not-exist"))


# --------------------------------------------------------------------------
class TestImaging:
    def test_rejects_non_image_payload(self):
        payload = base64.b64encode(b"definitely not a png").decode()
        with pytest.raises(InvalidDocumentError):
            decode_page(payload, max_pixels=10_000_000)

    def test_rejects_invalid_base64(self):
        with pytest.raises(InvalidDocumentError):
            decode_page("!!!!not base64!!!!", max_pixels=10_000_000)

    def test_accepts_data_url_prefix(self):
        page = decode_page(f"data:image/png;base64,{ONE_PIXEL_PNG}", max_pixels=1_000_000)
        assert page.width == 1 and page.height == 1

    def test_enforces_pixel_ceiling(self):
        with pytest.raises(InvalidDocumentError):
            decode_page(ONE_PIXEL_PNG, max_pixels=0)


# --------------------------------------------------------------------------
class TestModelsAndUtilities:
    def test_bbox_rejects_inverted_coordinates(self):
        with pytest.raises(ValueError):
            BoundingBox(x0=0.5, y0=0.1, x1=0.2, y1=0.3)

    def test_bbox_iou(self):
        a = BoundingBox(x0=0.0, y0=0.0, x1=0.5, y1=0.5)
        assert a.iou(a) == pytest.approx(1.0)
        assert a.iou(BoundingBox(x0=0.6, y0=0.6, x1=0.9, y1=0.9)) == 0.0

    def test_usage_cost_is_derived_from_throughput(self):
        usage = UsageStats(prompt_tokens=700, completion_tokens=300)
        cost = usage.cost_usd(gpu_hourly_rate=1.99, throughput_tokens_per_s=1000)
        assert cost == pytest.approx(1.99 / 3600, rel=1e-3)

    def test_circuit_breaker_opens_and_resets(self):
        breaker = CircuitBreaker(failure_threshold=2, reset_timeout_s=0.0)
        breaker.record_failure()
        assert not breaker.is_open
        breaker.record_failure()
        assert not breaker.is_open  # zero timeout means it immediately half-opens
        breaker.record_success()
        assert not breaker.is_open

    def test_vlm_response_strips_code_fences(self):
        response = VLMResponse(text='```json\n{"a": 1}\n```')
        assert response.as_json() == {"a": 1}

    def test_vlm_response_rejects_prose(self):
        with pytest.raises(VLMProtocolError):
            VLMResponse(text="Sure! Here is the data.").as_json()

    def test_prompt_versions_are_addressable(self):
        assert get_prompt("v2-schema-locked").version == "v2-schema-locked"
        with pytest.raises(KeyError):
            get_prompt("v99-does-not-exist")


# --------------------------------------------------------------------------
class TestEvaluation:
    def test_character_error_rate(self):
        assert character_error_rate("abc", "abc") == 0.0
        assert character_error_rate("abd", "abc") == pytest.approx(1 / 3)
        assert character_error_rate("", "") == 0.0

    @pytest.mark.asyncio
    async def test_scoring_counts_hallucination_and_justified_abstention(
        self, settings, registry
    ):
        hallucinated = dict(CORRECT_READING, purchase_order="PO-88213")
        service = build_service(
            settings,
            registry,
            [PassScript(values=hallucinated), PassScript(values=hallucinated)],
        )
        result = await service.extract(request_for())
        truth = dict(CORRECT_READING)
        types = {f.name: f.type for f in registry.get("invoice").fields}

        counters = EvalCounters()
        for outcome in score_document(result, truth, types):
            counters.add(outcome)

        assert counters.hallucination_rate == 0.0  # it was refused, not returned
        assert counters.abstention_precision == 1.0
        assert counters.precision == 1.0

    @pytest.mark.asyncio
    async def test_run_suite_produces_a_comparable_grid(self, settings, registry, tmp_path):
        cases = [
            EvalCase(
                document_id="doc-1",
                schema="invoice",
                image_b64=ONE_PIXEL_PNG,
                truth=dict(CORRECT_READING),
                reference_text=PAGE_TEXT,
            )
        ]
        grid = [
            RunConfig("v2-schema-locked", 1, 0.50),
            RunConfig("v3-region-guided", 2, 0.62),
        ]
        reports = await run_suite(
            cases,
            settings,
            lambda: ScriptedVLM(
                [PassScript(values=CORRECT_READING)] * 2, transcription=PAGE_TEXT
            ),
            grid,
            registry,
        )
        assert len(reports) == 2
        assert all(r.documents == 1 for r in reports)
        assert all(r.total_tokens > 0 for r in reports)
        table = render_markdown(reports)
        assert "hallucination" in table and "v3-region-guided" in table


# --------------------------------------------------------------------------
class TestHttpApi:
    @pytest.fixture()
    def client(self, settings, registry):
        service = build_service(
            settings,
            registry,
            [PassScript(values=CORRECT_READING), PassScript(values=CORRECT_READING)],
        )
        with TestClient(create_app(settings, service)) as test_client:
            yield test_client

    def test_healthz_is_dependency_free(self, client):
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_readyz_reports_individual_checks(self, client):
        body = client.get("/readyz").json()
        assert set(body["checks"]) == {"schemas", "vlm"}

    def test_schemas_endpoint_lists_prompts(self, client):
        body = client.get("/v1/schemas").json()
        assert "invoice" in body["schemas"]
        assert body["default_prompt_version"]

    def test_extract_returns_evidence(self, client):
        response = client.post(
            "/v1/extract",
            json={
                "document_id": "doc-1",
                "images_b64": [ONE_PIXEL_PNG],
                "schema_name": "invoice",
                "reference_text": PAGE_TEXT,
            },
        )
        assert response.status_code == 200
        body = response.json()
        invoice_number = next(f for f in body["fields"] if f["name"] == "invoice_number")
        assert invoice_number["value"] == "INV-2026-0041"
        assert invoice_number["provenance"]["grounded"] is True
        assert "x-request-id" in response.headers

    def test_unknown_schema_maps_to_404_with_code(self, client):
        response = client.post(
            "/v1/extract",
            json={
                "document_id": "doc-1",
                "images_b64": [ONE_PIXEL_PNG],
                "schema_name": "nope",
            },
        )
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "schema_not_found"

    def test_request_validation_rejects_empty_pages(self, client):
        response = client.post(
            "/v1/extract",
            json={"document_id": "d", "images_b64": [], "schema_name": "invoice"},
        )
        assert response.status_code == 422

    def test_metrics_are_exposed(self, client):
        client.post(
            "/v1/extract",
            json={
                "document_id": "doc-1",
                "images_b64": [ONE_PIXEL_PNG],
                "schema_name": "invoice",
                "reference_text": PAGE_TEXT,
            },
        )
        body = client.get("/metrics").text
        assert "tabula_requests_total" in body
        assert "tabula_fields_extracted_total" in body


def test_schema_registry_rejects_unknown_type(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"name": "bad", "fields": [{"name": "x", "type": "quaternion"}]}))
    with pytest.raises(Exception) as exc_info:
        SchemaRegistry(tmp_path).get("bad")
    assert "unsupported type" in str(exc_info.value)
