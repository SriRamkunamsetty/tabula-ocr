"""Versioned prompts.

The brief for this mini-challenge is explicitly about *prompting* a multimodal
model, so prompts are treated as versioned, testable assets rather than string
literals buried in the call site. Each version is immutable once released; an
improvement is a new version, so an evaluation run recorded against ``v2`` stays
meaningful after ``v3`` ships.

``docs/prompt_ablation.md`` records what each version bought on the evaluation
set. The numbers there are produced by ``make eval``, not by hand.
"""

from __future__ import annotations

from dataclasses import dataclass

from tabula_ocr.schema_registry import ExtractionSchema

__all__ = ["DEFAULT_VERSION", "PROMPTS", "PromptVersion", "get_prompt", "list_versions"]


@dataclass(frozen=True, slots=True)
class PromptVersion:
    """One prompting strategy, identified by a stable version string."""

    version: str
    summary: str
    system: str
    user_template: str

    def render(self, schema: ExtractionSchema, hint: str = "") -> str:
        """Fill the template for a concrete extraction schema."""
        return self.user_template.format(
            fields=schema.describe_fields(),
            title=schema.title,
            hint=hint or "None.",
        )


_ABSTAIN_RULE = (
    "If a field is not physically printed on this page, set its value to null. "
    "Never infer, compute, translate or complete a value from context. A null is "
    "a correct answer; a plausible guess is a defect."
)

V1_BASELINE = PromptVersion(
    version="v1-baseline",
    summary="Plain instruction. Control condition for the ablation table.",
    system="You are an OCR assistant. Read documents and return the requested fields.",
    user_template=("Read this {title} and return the following fields as JSON.\n\n{fields}\n"),
)

V2_SCHEMA_LOCKED = PromptVersion(
    version="v2-schema-locked",
    summary=(
        "Adds the evidence contract: every field must carry a verbatim quote and "
        "a bounding box, and absence must be reported as null."
    ),
    system=(
        "You are a document transcription engine. You transcribe exactly what is "
        "printed, character for character. You do not interpret, summarise or "
        "correct the document. Your output is consumed by an automated system "
        "that verifies every value against the page image."
    ),
    user_template=(
        "Extract the following fields from this {title}.\n\n"
        "{fields}\n\n"
        "Rules:\n"
        "1. Copy each value exactly as printed, preserving punctuation and case.\n"
        f"2. {_ABSTAIN_RULE}\n"
        "3. For every field, also return `quote`: the full printed line the value "
        "appears in, copied verbatim.\n"
        "4. For every field, also return `bbox`: [x0, y0, x1, y1] normalised to "
        "0-1 of page width and height, tightly around the value.\n\n"
        "Page notes: {hint}"
    ),
)

V3_REGION_GUIDED = PromptVersion(
    version="v3-region-guided",
    summary=(
        "v2 plus the layout pass's region inventory, so the model attends to "
        "candidate regions instead of re-reading the whole page per field."
    ),
    system=V2_SCHEMA_LOCKED.system,
    user_template=(
        "Extract the following fields from this {title}.\n\n"
        "{fields}\n\n"
        "Rules:\n"
        "1. Copy each value exactly as printed, preserving punctuation and case.\n"
        f"2. {_ABSTAIN_RULE}\n"
        "3. Return `quote` (the verbatim printed line) and `bbox` "
        "([x0, y0, x1, y1] normalised to 0-1) for every field.\n"
        "4. Work region by region. Do not merge values that sit in different "
        "regions of the page.\n\n"
        "Detected regions on this page: {hint}"
    ),
)

TRANSCRIBE = PromptVersion(
    version="transcribe-v1",
    summary="Full-page transcription used to build the grounding reference text.",
    system=(
        "You are a document transcription engine. Transcribe the page exactly, "
        "preserving reading order and line breaks. Do not summarise or omit."
    ),
    user_template=(
        "Transcribe every character visible on this page in natural reading "
        "order. Preserve line breaks. Render tables as pipe-delimited rows. "
        "Output plain text only, with no commentary.\n\n{fields}{title}{hint}"
    ),
)

PROMPTS: dict[str, PromptVersion] = {
    p.version: p for p in (V1_BASELINE, V2_SCHEMA_LOCKED, V3_REGION_GUIDED, TRANSCRIBE)
}

DEFAULT_VERSION = V3_REGION_GUIDED.version


def get_prompt(version: str | None = None) -> PromptVersion:
    """Return a prompt version, falling back to the current default.

    An unknown version raises rather than silently degrading, so a typo in a
    benchmark configuration cannot quietly invalidate a whole evaluation run.
    """
    key = version or DEFAULT_VERSION
    if key not in PROMPTS:
        raise KeyError(
            f"unknown prompt version '{key}'; available: {', '.join(sorted(PROMPTS))}"
        )
    return PROMPTS[key]


def list_versions() -> list[tuple[str, str]]:
    """Return ``(version, summary)`` for every registered prompt."""
    return [(p.version, p.summary) for p in PROMPTS.values()]
