"""Extraction schemas and the JSON Schema handed to the serving engine.

The service never asks a model to "return JSON" and hope. Each extraction
schema is compiled into a JSON Schema that is passed to vLLM as a guided
decoding constraint, so the token sampler is restricted to sequences that parse.
Structural validity stops being a prompt-engineering problem and becomes a
property of the decoder.

Schemas are plain JSON files on disk so an operator can add a document type
without a deployment, and are validated on load so a malformed one fails at
startup rather than on the first request.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from tabula_ocr.errors import ConfigurationError, SchemaNotFoundError
from tabula_ocr.normalize import SUPPORTED_TYPES

__all__ = ["ExtractionSchema", "FieldSpec", "SchemaRegistry"]


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """One field the model is asked to find."""

    name: str
    type: str
    description: str
    required: bool = False
    examples: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FieldSpec:
        """Build a field spec from its JSON form, validating the type."""
        try:
            name = str(data["name"]).strip()
            field_type = str(data["type"]).strip()
        except KeyError as exc:  # pragma: no cover - guarded by loader tests
            raise ConfigurationError(f"field is missing key {exc}") from exc
        if not name:
            raise ConfigurationError("field name must not be empty")
        if field_type not in SUPPORTED_TYPES:
            raise ConfigurationError(
                f"field '{name}' has unsupported type '{field_type}'; "
                f"supported types are {sorted(SUPPORTED_TYPES)}"
            )
        return cls(
            name=name,
            type=field_type,
            description=str(data.get("description", "")).strip(),
            required=bool(data.get("required", False)),
            examples=tuple(str(e) for e in data.get("examples", ())),
        )


@dataclass(frozen=True, slots=True)
class ExtractionSchema:
    """A named set of fields, compiled for guided decoding."""

    name: str
    title: str
    fields: tuple[FieldSpec, ...]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExtractionSchema:
        """Build a schema from its JSON form, rejecting duplicate fields."""
        name = str(data.get("name", "")).strip()
        if not name:
            raise ConfigurationError("schema is missing a 'name'")
        raw_fields = data.get("fields") or []
        if not raw_fields:
            raise ConfigurationError(f"schema '{name}' defines no fields")
        fields = tuple(FieldSpec.from_dict(f) for f in raw_fields)
        seen: set[str] = set()
        for field in fields:
            if field.name in seen:
                raise ConfigurationError(f"schema '{name}' repeats field '{field.name}'")
            seen.add(field.name)
        return cls(name=name, title=str(data.get("title", name)), fields=fields)

    def field(self, name: str) -> FieldSpec | None:
        """Return the named field spec, if the schema defines one."""
        return next((f for f in self.fields if f.name == name), None)

    def to_json_schema(self) -> dict[str, Any]:
        """Compile to the JSON Schema passed to vLLM as ``guided_json``.

        Every field is represented as an object carrying the value plus its
        evidence, and every one of those keys is marked required. Making the
        evidence structurally mandatory is what forces the model to commit to a
        quote and a box for each value instead of returning a bare string that
        cannot be verified.
        """
        properties: dict[str, Any] = {}
        for field in self.fields:
            properties[field.name] = {
                "type": "object",
                "description": field.description or f"The document's {field.name}.",
                "properties": {
                    "value": {
                        "type": ["string", "null"],
                        "description": "Exact text as printed, or null if absent.",
                    },
                    "quote": {
                        "type": ["string", "null"],
                        "description": "Verbatim surrounding line copied from the page.",
                    },
                    "bbox": {
                        "type": ["array", "null"],
                        "items": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "minItems": 4,
                        "maxItems": 4,
                        "description": "Normalised [x0, y0, x1, y1] of the value.",
                    },
                },
                "required": ["value", "quote", "bbox"],
                "additionalProperties": False,
            }
        return {
            "type": "object",
            "title": self.title,
            "properties": properties,
            "required": [f.name for f in self.fields],
            "additionalProperties": False,
        }

    def describe_fields(self) -> str:
        """Render the field list for inclusion in the prompt."""
        lines = []
        for field in self.fields:
            hint = f" e.g. {', '.join(field.examples)}" if field.examples else ""
            flag = " (required)" if field.required else ""
            lines.append(f"- {field.name} [{field.type}]{flag}: {field.description}{hint}")
        return "\n".join(lines)


class SchemaRegistry:
    """Loads and caches extraction schemas from a directory of JSON files."""

    def __init__(self, directory: Path) -> None:
        """Point the registry at a directory of schema JSON files."""
        self._directory = Path(directory)
        self._cache: dict[str, ExtractionSchema] = {}

    @property
    def directory(self) -> Path:
        """Directory the registry reads from."""
        return self._directory

    def load_all(self) -> dict[str, ExtractionSchema]:
        """Eagerly load every schema; call once at startup to fail fast."""
        if not self._directory.is_dir():
            raise ConfigurationError(f"schema directory not found: {self._directory}")
        for path in sorted(self._directory.glob("*.json")):
            schema = self._load_file(path)
            self._cache[schema.name] = schema
        if not self._cache:
            raise ConfigurationError(f"no schemas found in {self._directory}")
        return dict(self._cache)

    def get(self, name: str) -> ExtractionSchema:
        """Return a schema by name, raising :class:`SchemaNotFoundError`."""
        if name in self._cache:
            return self._cache[name]
        path = self._directory / f"{name}.json"
        if not path.is_file():
            raise SchemaNotFoundError(
                f"unknown extraction schema '{name}'",
                detail=f"available: {', '.join(self.names()) or 'none'}",
            )
        schema = self._load_file(path)
        self._cache[schema.name] = schema
        return schema

    def names(self) -> list[str]:
        """Names of all schemas visible to the registry."""
        on_disk = (
            {p.stem for p in self._directory.glob("*.json")}
            if self._directory.is_dir()
            else set()
        )
        return sorted(on_disk | set(self._cache))

    def _load_file(self, path: Path) -> ExtractionSchema:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ConfigurationError(
                f"schema file {path.name} is not valid JSON", detail=str(exc)
            ) from exc
        data.setdefault("name", path.stem)
        return ExtractionSchema.from_dict(data)


@lru_cache(maxsize=8)
def _cached_registry(directory: str) -> SchemaRegistry:
    return SchemaRegistry(Path(directory))
