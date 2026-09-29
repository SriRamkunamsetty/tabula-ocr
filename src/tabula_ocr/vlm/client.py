"""Client for the vision model served by vLLM on ROCm.

The serving engine speaks the OpenAI chat-completions protocol, so this client
is deliberately thin. What it adds is the operational behaviour a production
caller needs and the raw protocol does not provide:

* a bounded concurrency semaphore, so a burst of documents cannot queue more
  work on the accelerator than it can serve inside the request timeout;
* retries with exponential backoff and jitter, on transport errors and 5xx only
  — a 4xx is a bug in our request and retrying it just burns budget;
* a circuit breaker, so when the model is down the service sheds load in
  microseconds instead of holding every connection open for the full timeout;
* ``guided_json`` pass-through, which is what makes the response structurally
  valid by construction rather than by parsing luck.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from tabula_ocr.config import Settings
from tabula_ocr.errors import CircuitOpenError, VLMProtocolError, VLMTimeoutError
from tabula_ocr.observability import VLM_FAILURES, VLM_LATENCY, get_logger

__all__ = ["CircuitBreaker", "OpenAICompatibleVLM", "VLMClient", "VLMResponse"]

_log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class VLMResponse:
    """Normalised view of one model completion."""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = "unknown"
    latency_ms: float = 0.0

    def as_json(self) -> dict[str, Any]:
        """Parse the completion as a JSON object.

        Guided decoding should make this infallible, but a serving engine
        without the constraint enabled will happily return prose. Fenced code
        blocks are stripped before parsing because that is the one deviation
        seen often enough in practice to be worth handling rather than failing.
        """
        text = self.text.strip()
        if text.startswith("```"):
            text = text.split("```")[1] if "```" in text[3:] else text[3:]
            text = text.removeprefix("json").strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise VLMProtocolError(
                "model response was not valid JSON",
                detail=f"{exc}; first 200 chars: {text[:200]!r}",
            ) from exc
        if not isinstance(parsed, dict):
            raise VLMProtocolError(
                "model response was valid JSON but not an object",
                detail=f"got {type(parsed).__name__}",
            )
        return parsed


class VLMClient(Protocol):
    """Interface the pipeline depends on.

    The pipeline is written against this protocol rather than a concrete client
    so that tests, benchmarks and the offline evaluation harness can substitute
    a deterministic implementation without patching module internals.
    """

    async def complete(
        self,
        *,
        system: str,
        user: str,
        images_b64: list[str],
        temperature: float,
        guided_json: dict[str, Any] | None = None,
        max_tokens: int = 2048,
        operation: str = "extract",
    ) -> VLMResponse:
        """Run one multimodal completion."""
        ...

    async def aclose(self) -> None:
        """Release transport resources."""
        ...


@dataclass
class CircuitBreaker:
    """Minimal three-state breaker: closed, open, half-open.

    Half-open is implicit: once the reset timeout elapses a single request is
    allowed through, and its outcome either closes the breaker or re-opens it
    for another interval.
    """

    failure_threshold: int
    reset_timeout_s: float
    _failures: int = field(default=0, init=False)
    _opened_at: float | None = field(default=None, init=False)

    @property
    def is_open(self) -> bool:
        """True while the breaker is shedding load."""
        if self._opened_at is None:
            return False
        if time.monotonic() - self._opened_at >= self.reset_timeout_s:
            self._opened_at = None
            self._failures = self.failure_threshold - 1
            return False
        return True

    def record_success(self) -> None:
        """Reset the failure count and close the breaker."""
        self._failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        """Count a failure and open the breaker once the threshold is reached."""
        self._failures += 1
        if self._failures >= self.failure_threshold:
            self._opened_at = time.monotonic()


class OpenAICompatibleVLM:
    """Production client for a vLLM/SGLang endpoint."""

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        """Build a client; an injected transport is used by tests and benchmarks."""
        self._settings = settings
        self._client = client or httpx.AsyncClient(
            base_url=settings.vlm_base_url,
            timeout=httpx.Timeout(
                settings.vlm_timeout_s, connect=settings.vlm_connect_timeout_s
            ),
            headers={"Authorization": f"Bearer {settings.vlm_api_key}"},
            limits=httpx.Limits(max_connections=settings.vlm_max_concurrency * 2),
        )
        self._semaphore = asyncio.Semaphore(settings.vlm_max_concurrency)
        self._breaker = CircuitBreaker(
            settings.breaker_failure_threshold, settings.breaker_reset_timeout_s
        )

    async def complete(
        self,
        *,
        system: str,
        user: str,
        images_b64: list[str],
        temperature: float,
        guided_json: dict[str, Any] | None = None,
        max_tokens: int = 2048,
        operation: str = "extract",
    ) -> VLMResponse:
        """Run one multimodal completion, retrying transient failures."""
        if self._breaker.is_open:
            VLM_FAILURES.labels(reason="circuit_open").inc()
            raise CircuitOpenError(
                "vision model circuit breaker is open",
                detail="upstream has failed repeatedly; shedding load",
            )

        payload = self._build_payload(
            system, user, images_b64, temperature, guided_json, max_tokens
        )
        last_error: Exception | None = None

        for attempt in range(self._settings.vlm_max_retries + 1):
            started = time.perf_counter()
            try:
                async with self._semaphore:
                    response = await self._client.post("/chat/completions", json=payload)
                if response.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        f"upstream {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                if response.status_code >= 400:
                    VLM_FAILURES.labels(reason=f"http_{response.status_code}").inc()
                    raise VLMProtocolError(
                        f"vision model rejected the request ({response.status_code})",
                        detail=response.text[:500],
                    )
                elapsed_ms = (time.perf_counter() - started) * 1000
                VLM_LATENCY.labels(operation=operation).observe(elapsed_ms / 1000)
                self._breaker.record_success()
                return self._parse(response.json(), elapsed_ms)

            except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError) as exc:
                last_error = exc
                self._breaker.record_failure()
                reason = type(exc).__name__
                VLM_FAILURES.labels(reason=reason).inc()
                if attempt == self._settings.vlm_max_retries:
                    break
                backoff = self._backoff_seconds(attempt)
                _log.warning(
                    "vlm_call_retry",
                    attempt=attempt + 1,
                    max_attempts=self._settings.vlm_max_retries + 1,
                    backoff_s=round(backoff, 3),
                    reason=reason,
                    operation=operation,
                )
                await asyncio.sleep(backoff)

        raise VLMTimeoutError(
            "vision model did not return a usable response",
            detail=f"{type(last_error).__name__}: {last_error}",
        )

    def _build_payload(
        self,
        system: str,
        user: str,
        images_b64: list[str],
        temperature: float,
        guided_json: dict[str, Any] | None,
        max_tokens: int,
    ) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{"type": "text", "text": user}]
        for image in images_b64:
            url = image if image.startswith("data:") else f"data:image/png;base64,{image}"
            content.append({"type": "image_url", "image_url": {"url": url}})

        payload: dict[str, Any] = {
            "model": self._settings.vlm_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if guided_json is not None:
            # vLLM accepts the constraint at the top level; the nested form under
            # extra_body is sent as well so the same payload works against
            # gateways that forward only OpenAI-canonical keys.
            payload["guided_json"] = guided_json
            payload["extra_body"] = {"guided_json": guided_json}
        return payload

    @staticmethod
    def _parse(body: dict[str, Any], elapsed_ms: float) -> VLMResponse:
        try:
            text = body["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise VLMProtocolError(
                "vision model response was missing a completion",
                detail=str(body)[:500],
            ) from exc
        usage = body.get("usage") or {}
        return VLMResponse(
            text=text,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            model=str(body.get("model", "unknown")),
            latency_ms=round(elapsed_ms, 2),
        )

    @staticmethod
    def _backoff_seconds(attempt: int) -> float:
        """Exponential backoff with full jitter, capped at eight seconds."""
        return random.uniform(0, min(8.0, 0.5 * (2**attempt)))

    async def aclose(self) -> None:
        """Close the underlying HTTP transport."""
        await self._client.aclose()
