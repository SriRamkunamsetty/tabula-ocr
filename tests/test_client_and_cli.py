"""Tests for the upstream client and the command-line interface.

The client tests use ``httpx.MockTransport`` rather than a live endpoint, so the
retry, backoff, circuit-breaker and payload-shape behaviour is asserted exactly
— including that a 4xx is *not* retried, which is the difference between a
failed request and a burnt GPU-hour budget.
"""

from __future__ import annotations

import json

import httpx
import pytest
from typer.testing import CliRunner

from tabula_ocr.cli import app
from tabula_ocr.config import Settings
from tabula_ocr.errors import CircuitOpenError, VLMProtocolError, VLMTimeoutError
from tabula_ocr.vlm.client import OpenAICompatibleVLM

runner = CliRunner()


def completion(text: str, *, prompt_tokens: int = 10, completion_tokens: int = 5) -> dict:
    return {
        "model": "test-vlm",
        "choices": [{"message": {"content": text}}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }


def build_client(handler, **overrides) -> OpenAICompatibleVLM:
    settings = Settings(
        environment="test",
        vlm_max_retries=overrides.pop("vlm_max_retries", 2),
        breaker_failure_threshold=overrides.pop("breaker_failure_threshold", 5),
        breaker_reset_timeout_s=overrides.pop("breaker_reset_timeout_s", 60.0),
        log_level="ERROR",
        **overrides,
    )
    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(transport=transport, base_url=settings.vlm_base_url)
    return OpenAICompatibleVLM(settings, client=http)


async def call(client: OpenAICompatibleVLM, **kwargs):
    defaults = {
        "system": "s",
        "user": "u",
        "images_b64": ["Zm9v"],
        "temperature": 0.0,
        "operation": "extract",
    }
    defaults.update(kwargs)
    return await client.complete(**defaults)


class TestPayloadShape:
    @pytest.mark.asyncio
    async def test_guided_json_is_sent_both_ways(self):
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content))
            return httpx.Response(200, json=completion('{"ok": 1}'))

        client = build_client(handler)
        await call(client, guided_json={"type": "object"})
        assert captured["guided_json"] == {"type": "object"}
        assert captured["extra_body"]["guided_json"] == {"type": "object"}
        await client.aclose()

    @pytest.mark.asyncio
    async def test_images_become_data_urls(self):
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content))
            return httpx.Response(200, json=completion("{}"))

        client = build_client(handler)
        await call(client)
        parts = captured["messages"][1]["content"]
        assert parts[0]["type"] == "text"
        assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")
        await client.aclose()

    @pytest.mark.asyncio
    async def test_usage_is_propagated(self):
        client = build_client(
            lambda r: httpx.Response(
                200, json=completion("{}", prompt_tokens=700, completion_tokens=120)
            )
        )
        response = await call(client)
        assert response.prompt_tokens == 700
        assert response.completion_tokens == 120
        assert response.model == "test-vlm"
        await client.aclose()


class TestFailureHandling:
    @pytest.mark.asyncio
    async def test_server_errors_are_retried_then_surface_as_timeout(self):
        attempts = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["n"] += 1
            return httpx.Response(503, text="overloaded")

        client = build_client(handler, vlm_max_retries=2)
        with pytest.raises(VLMTimeoutError):
            await call(client)
        assert attempts["n"] == 3  # initial attempt plus two retries
        await client.aclose()

    @pytest.mark.asyncio
    async def test_client_errors_are_not_retried(self):
        attempts = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["n"] += 1
            return httpx.Response(400, text="bad request")

        client = build_client(handler, vlm_max_retries=3)
        with pytest.raises(VLMProtocolError):
            await call(client)
        assert attempts["n"] == 1
        await client.aclose()

    @pytest.mark.asyncio
    async def test_transport_error_recovers_on_retry(self):
        state = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            state["n"] += 1
            if state["n"] == 1:
                raise httpx.ConnectError("connection refused", request=request)
            return httpx.Response(200, json=completion('{"ok": true}'))

        client = build_client(handler, vlm_max_retries=2)
        response = await call(client)
        assert response.as_json() == {"ok": True}
        await client.aclose()

    @pytest.mark.asyncio
    async def test_breaker_opens_and_sheds_load(self):
        client = build_client(
            lambda r: httpx.Response(500, text="boom"),
            vlm_max_retries=0,
            breaker_failure_threshold=2,
            breaker_reset_timeout_s=60.0,
        )
        for _ in range(2):
            with pytest.raises(VLMTimeoutError):
                await call(client)
        with pytest.raises(CircuitOpenError):
            await call(client)
        await client.aclose()

    @pytest.mark.asyncio
    async def test_missing_completion_is_a_protocol_error(self):
        client = build_client(lambda r: httpx.Response(200, json={"choices": []}))
        with pytest.raises(VLMProtocolError):
            await call(client)
        await client.aclose()


class TestCli:
    def test_version(self):
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "tabula-ocr" in result.stdout

    def test_schemas_lists_registered_documents(self, monkeypatch):
        monkeypatch.setenv("TABULA_SCHEMA_DIR", "schemas")
        result = runner.invoke(app, ["schemas"])
        assert result.exit_code == 0
        assert "invoice" in result.stdout
        assert "rulebook" in result.stdout

    def test_demo_runs_without_a_gpu_and_refuses_the_invented_field(self, monkeypatch):
        monkeypatch.setenv("TABULA_SCHEMA_DIR", "schemas")
        result = runner.invoke(app, ["demo"])
        assert result.exit_code == 0, result.stdout
        assert "invoice_number" in result.stdout
        # The scripted second pass invents a purchase order; it must be held.
        purchase_order_line = next(
            line for line in result.stdout.splitlines() if "purchase_order" in line
        )
        assert "HOLD" in purchase_order_line
