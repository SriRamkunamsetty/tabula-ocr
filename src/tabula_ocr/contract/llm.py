"""A small OpenAI-compatible vision client for contract mode.

One request asks the model for a JSON object ``{"kind": ..., "lines": [...]}``: what sort of
object the picture shows and the text lines it reads, top to bottom, character for character.
It degrades to ``None`` instead of raising: a slow, down or confused model must yield an
*empty best guess*, never a crash and never a missing output file.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import time
from typing import Any, Protocol

import httpx

from tabula_ocr.contract.config import ContractSettings

__all__ = ["OpenAIVision", "VisionModel", "extract_json_object", "parse_reading"]

_log = logging.getLogger(__name__)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": ["us_plate", "cn_plate", "sign", "other"]},
        "lines": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["kind", "lines"],
    "additionalProperties": False,
}

PROMPT = (
    "You are an exact OCR engine. Read the text in this image and reply with ONE JSON object "
    '{"kind": ..., "lines": [...]} and nothing else.\n'
    "kind is one of: us_plate (a United States licence plate), cn_plate (a Chinese licence "
    "plate), sign (a road or traffic sign, plaque or notice), other.\n"
    "lines are the text lines in reading order, top to bottom, exactly as printed: keep every "
    "character, do not add labels, units or explanation, do not correct or guess spelling. "
    "For a licence plate list every line printed on it, including a state name or slogan if "
    "one is printed. For a Chinese plate the first character is a province character and the "
    "second is a letter: keep both. Letters O and I never appear on Chinese plates; "
    "digits 0 and 1 do. Use uppercase for Latin letters."
)


class VisionModel(Protocol):
    """What the runner needs from a model."""

    async def read(self, png: bytes, *, timeout_s: float) -> dict[str, Any] | None:
        """Return ``{"kind": str, "lines": [str]}`` for the image, or ``None``."""
        ...

    async def wait_ready(self, budget_s: float) -> bool:
        """Block until the model server answers, or ``budget_s`` runs out."""
        ...

    async def aclose(self) -> None:
        """Release network resources."""
        ...


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Pull the first JSON object out of model text (tolerates fences and chatter)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    fenced = _FENCE.search(text)
    if fenced:
        text = fenced.group(1).strip()
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char == "{":
            try:
                value, _ = decoder.raw_decode(text[start:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
    return None


def parse_reading(payload: dict[str, Any] | None) -> tuple[str, list[str]] | None:
    """Validate a reply into ``(kind, lines)``; tolerate ``text`` or a string ``lines``."""
    if not payload:
        return None
    raw = payload.get("lines", payload.get("text", []))
    if isinstance(raw, str):
        raw = raw.splitlines()
    if not isinstance(raw, list):
        return None
    lines = [str(item) for item in raw if str(item).strip()]
    return str(payload.get("kind", "other")), lines


class OpenAIVision:
    """Talks to a vLLM endpoint over the OpenAI chat-completions API."""

    def __init__(
        self, settings: ContractSettings, client: httpx.AsyncClient | None = None
    ) -> None:
        """Create a client; tests inject a transport-backed ``httpx.AsyncClient``."""
        self._client = client or httpx.AsyncClient(
            base_url=settings.llm_base_url,
            headers={"Authorization": f"Bearer {settings.llm_api_key}"},
        )
        self._model = settings.llm_model or None
        self._schema_supported = True

    async def aclose(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    async def _resolve_model(self, timeout_s: float) -> str | None:
        if self._model:
            return self._model
        try:
            response = await self._client.get("/models", timeout=timeout_s)
            response.raise_for_status()
            data = response.json().get("data") or []
            self._model = data[0]["id"] if data else None
        except (httpx.HTTPError, ValueError, KeyError, IndexError):
            self._model = None
        return self._model

    async def wait_ready(self, budget_s: float) -> bool:
        """Poll ``/models`` until it answers with a model, up to ``budget_s`` seconds."""
        deadline = time.monotonic() + max(0.0, budget_s)
        while True:
            if await self._resolve_model(timeout_s=3.0):
                return True
            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(1.0)

    async def _complete(self, payload: dict[str, Any], timeout_s: float) -> str | None:
        try:
            response = await self._client.post(
                "/chat/completions", json=payload, timeout=timeout_s
            )
        except httpx.HTTPError as exc:
            _log.warning("llm transport error: %s", type(exc).__name__)
            return None
        if response.status_code == 400 and "response_format" in payload:
            self._schema_supported = False  # structured output rejected: retry as plain text
            plain = {k: v for k, v in payload.items() if k != "response_format"}
            return await self._complete(plain, timeout_s)
        if response.status_code >= 400:
            _log.warning("llm http error: %s", response.status_code)
            return None
        try:
            content = response.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError, TypeError):
            return None
        return str(content) if content is not None else None

    async def read(self, png: bytes, *, timeout_s: float) -> dict[str, Any] | None:
        """Ask the vision model what this image says."""
        model = await self._resolve_model(min(timeout_s, 5.0))
        if not model:
            return None
        data_uri = f"data:image/png;base64,{base64.b64encode(png).decode('ascii')}"
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                }
            ],
            "temperature": 0.0,
            "max_tokens": 200,
        }
        if self._schema_supported:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "reading", "schema": SCHEMA, "strict": True},
            }
        text = await self._complete(payload, timeout_s)
        return extract_json_object(text) if text else None
