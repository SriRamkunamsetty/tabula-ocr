"""A deterministic, GPU-free stand-in for the vision model.

This is not a mock in the throwaway sense. It is a scripted model used by three
real callers: the unit tests, ``make demo`` (so the service can be exercised on
a laptop with no accelerator attached), and the evaluation harness when it runs
in ``--offline`` mode to check the harness itself.

It can be told to disagree between passes, to hallucinate values that are absent
from the page, and to emit malformed JSON, which is how the abstention and
grounding logic gets tested against the failure modes that matter rather than
only against the happy path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from tabula_ocr.vlm.client import VLMResponse

__all__ = ["PassScript", "ScriptedVLM"]


@dataclass
class PassScript:
    """What the fake model should return for one decoding pass.

    Attributes:
        values: Field name to printed value. ``None`` means the model reports
            the field as absent.
        quotes: Optional per-field verbatim line; defaults to the value itself.
        boxes: Optional per-field normalised ``[x0, y0, x1, y1]``.
        malformed: When set, the pass returns this raw string instead of JSON,
            which exercises the protocol-error path.
    """

    values: dict[str, str | None]
    quotes: dict[str, str] = field(default_factory=dict)
    boxes: dict[str, list[float]] = field(default_factory=dict)
    malformed: str | None = None


class ScriptedVLM:
    """Replays a fixed list of :class:`PassScript` objects, in order."""

    def __init__(
        self,
        scripts: list[PassScript],
        *,
        transcription: str = "",
        model: str = "scripted-vlm",
        latency_ms: float = 12.0,
    ) -> None:
        """Create a scripted model from an ordered list of pass scripts."""
        if not scripts:
            raise ValueError("ScriptedVLM requires at least one PassScript")
        self._scripts = scripts
        self._transcription = transcription
        self._model = model
        self._latency_ms = latency_ms
        self.calls: list[dict[str, Any]] = []
        self._extract_index = 0

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
        """Return the next scripted response for ``operation``."""
        self.calls.append(
            {
                "operation": operation,
                "temperature": temperature,
                "images": len(images_b64),
                "guided": guided_json is not None,
                "system": system,
                "user": user,
            }
        )

        if operation == "transcribe":
            return self._response(self._transcription, prompt_tokens=420)

        script = self._scripts[min(self._extract_index, len(self._scripts) - 1)]
        self._extract_index += 1
        if script.malformed is not None:
            return self._response(script.malformed)

        payload: dict[str, Any] = {}
        for name, value in script.values.items():
            payload[name] = {
                "value": value,
                "quote": script.quotes.get(name, value),
                "bbox": script.boxes.get(name, [0.1, 0.1, 0.4, 0.15] if value else None),
            }
        return self._response(json.dumps(payload), prompt_tokens=610)

    def _response(self, text: str, *, prompt_tokens: int = 100) -> VLMResponse:
        return VLMResponse(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=max(1, len(text) // 4),
            model=self._model,
            latency_ms=self._latency_ms,
        )

    async def aclose(self) -> None:
        """No-op; present so the fake satisfies the client protocol."""
        return None
