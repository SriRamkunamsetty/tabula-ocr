"""Consensus, grounding and confidence — the anti-hallucination core.

A vision-language model asked for a field will almost always return something.
The problem in production is not that it fails, it is that it fails *quietly*.
This module exists to make quiet failure loud, using three independent signals
that a fabricated value cannot satisfy simultaneously:

1. **Agreement.** The same page is decoded several times under different
   sampling conditions. A value genuinely present on the page is stable across
   passes; an invented one drifts.
2. **Grounding.** The value must be locatable in the page's reference text.
   Fuzzy matching is used rather than exact equality because the reference text
   itself comes from OCR and carries its own character errors, but a value that
   was never on the page scores near zero however generous the matcher is.
3. **Format validity.** A date that does not parse or a phone number of the
   wrong length is evidence of misreading, independent of the other two signals.

The three signals are **not** peers. Grounding is a necessary condition: a value
that cannot be located on the page is never returned, however many passes agreed
on it and however well-formed it looks. This is deliberate and was chosen after
the weighted-sum design failed its own test — with agreement at 0.5 and format
validity at 0.15 of the weight, two passes that shared a hallucination summed to
0.65 and cleared a 0.62 threshold with zero grounding. Two correlated model
samples are not independent evidence about the world, so no amount of agreement
between them may substitute for the value being on the page.

Agreement and format validity therefore act as supporting evidence: they set the
confidence *within* the grounded set, which is what downstream triage sorts on.
Weights and the abstention threshold live in :class:`Settings` so they can be
tuned against a labelled set rather than argued about.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from rapidfuzz import fuzz

from tabula_ocr.models import (
    BoundingBox,
    FieldResult,
    FieldStatus,
    PassObservation,
    Provenance,
)
from tabula_ocr.normalize import canonical_text, normalize_value, validate_format

__all__ = [
    "ConsensusOutcome",
    "GroundingReport",
    "ground_value",
    "reach_consensus",
    "score_field",
]


@dataclass(frozen=True, slots=True)
class GroundingReport:
    """Result of checking a candidate value against a page's reference text."""

    grounded: bool
    score: float
    matched_quote: str | None


def ground_value(
    value: str,
    reference_text: str,
    *,
    min_ratio: float = 0.86,
    window_slack: int = 12,
) -> GroundingReport:
    """Check whether ``value`` actually appears in ``reference_text``.

    A sliding window the length of the candidate (plus slack) is scanned across
    the reference text and the best partial-ratio match is kept. Exact
    containment short-circuits to a perfect score, which is the common case and
    keeps the hot path cheap.

    Args:
        value: Candidate value as read by the model.
        reference_text: Trusted text for the page — either the PDF text layer or
            a full-page transcription pass.
        min_ratio: Similarity at or above which the value counts as grounded.
        window_slack: Extra characters allowed around the candidate length, to
            absorb OCR insertions such as stray spaces.

    Returns:
        A :class:`GroundingReport`. An empty reference text yields a not-grounded
        report rather than an exception, because a blank page is a legitimate
        input and should abstain, not 500.
    """
    candidate = value.strip()
    if not candidate or not reference_text.strip():
        return GroundingReport(False, 0.0, None)

    haystack_canonical = canonical_text(reference_text)
    needle_canonical = canonical_text(candidate)
    if needle_canonical and needle_canonical in haystack_canonical:
        return GroundingReport(True, 1.0, candidate)

    # partial_ratio finds the best-matching substring of the haystack, which is
    # exactly the question being asked, and is O(n*m) with a fast C kernel.
    score = fuzz.partial_ratio(candidate.casefold(), reference_text.casefold()) / 100.0
    matched: str | None = None
    if score >= min_ratio:
        matched = _best_window(candidate, reference_text, window_slack)
    return GroundingReport(score >= min_ratio, round(score, 4), matched)


def _best_window(value: str, reference_text: str, slack: int) -> str | None:
    """Return the substring of ``reference_text`` most similar to ``value``."""
    width = len(value) + slack
    if width >= len(reference_text):
        return reference_text.strip() or None
    best_score, best_text = 0.0, None
    step = max(1, width // 4)
    for start in range(0, len(reference_text) - width + 1, step):
        window = reference_text[start : start + width]
        score = fuzz.ratio(value.casefold(), window.casefold())
        if score > best_score:
            best_score, best_text = score, window
    return best_text.strip() if best_text else None


@dataclass(frozen=True, slots=True)
class ConsensusOutcome:
    """Winning value for a field and how strongly the passes supported it."""

    normalized_value: str | None
    display_value: str | None
    agreement: float
    supporting: tuple[PassObservation, ...]


def reach_consensus(
    observations: list[PassObservation],
    field_type: str,
    total_passes: int | None = None,
) -> ConsensusOutcome:
    """Pick the value the decoding passes agree on.

    Votes are grouped by canonicalised normalised value, so ``"1,240.00"`` and
    ``"1240"`` land in the same bucket.

    Agreement is the winning bucket's share of ``total_passes`` — every pass that
    ran, not just the ones that produced a value. A pass that reported the field
    as absent is a vote against it being reliably present, and counting only the
    passes that spoke would score a field read by one pass out of three as
    unanimous, which is precisely backwards on the fields most likely to be
    hallucinated.

    Args:
        observations: Values produced for this field, at most one per pass.
        field_type: Declared type, used to normalise before grouping.
        total_passes: Number of passes that completed. Defaults to the number of
            observations, which is only correct when every pass answered.

    Returns:
        The winning value and its agreement share. Ties are broken by the
        earliest pass index, keeping the function deterministic.
    """
    denominator = max(total_passes or len(observations), len(observations), 1)
    if not observations:
        return ConsensusOutcome(None, None, 0.0, ())

    buckets: dict[str, list[PassObservation]] = defaultdict(list)
    normalised_by_pass: dict[str, str] = {}
    for observation in observations:
        normalized = normalize_value(observation.raw_value, field_type)
        if normalized is None:
            continue
        normalised_by_pass[observation.pass_id] = normalized
        buckets[canonical_text(normalized) or normalized].append(observation)

    if not buckets:
        return ConsensusOutcome(None, None, 0.0, ())

    key = max(buckets, key=lambda k: (len(buckets[k]), -_first_index(buckets[k])))
    winners = buckets[key]
    agreement = len(winners) / denominator
    normalized = normalised_by_pass[winners[0].pass_id]
    display = str(winners[0].raw_value).strip()
    return ConsensusOutcome(normalized, display, round(agreement, 4), tuple(winners))


def _first_index(observations: list[PassObservation]) -> int:
    """Lowest numeric suffix among pass ids, used only as a tie-breaker."""
    indices = []
    for observation in observations:
        _, _, tail = observation.pass_id.rpartition("-")
        indices.append(int(tail) if tail.isdigit() else 0)
    return min(indices) if indices else 0


def score_field(
    *,
    name: str,
    field_type: str,
    observations: list[PassObservation],
    reference_text: str,
    page: int,
    weights: tuple[float, float, float],
    abstain_below: float,
    grounding_min_ratio: float,
    required: bool = False,
    total_passes: int | None = None,
) -> FieldResult:
    """Combine consensus, grounding and format validity into a final result.

    Grounding is applied as a veto and the other two signals as a weighted
    score. A value that is not locatable in the reference text is abstained on
    outright; among values that *are* locatable, agreement and format validity
    decide whether confidence clears the threshold. When no reference text is
    available at all the veto is skipped — there is nothing to check against —
    and the field is scored on the remaining signals, with the missing evidence
    recorded in the result.

    Returns:
        A :class:`FieldResult` whose ``status`` is ``EXTRACTED`` only when the
        combined confidence clears ``abstain_below``. Fields that no pass read at
        all come back as ``NOT_PRESENT`` when optional and ``ABSTAINED`` when
        the schema marks them required, since a missing required field is a
        finding rather than an absence.
    """
    outcome = reach_consensus(observations, field_type, total_passes)
    w_agreement, w_grounding, w_format = weights

    if outcome.normalized_value is None:
        status = FieldStatus.ABSTAINED if required else FieldStatus.NOT_PRESENT
        reason = (
            "required field not read by any decoding pass"
            if required
            else "field absent from document"
        )
        return FieldResult(
            name=name,
            value=None,
            status=status,
            confidence=0.0,
            agreement=0.0,
            reason=reason,
            observations=observations,
        )

    grounding = ground_value(
        outcome.display_value or outcome.normalized_value,
        reference_text,
        min_ratio=grounding_min_ratio,
    )
    format_ok = validate_format(outcome.display_value, field_type)
    confidence = (
        w_agreement * outcome.agreement
        + w_grounding * grounding.score
        + w_format * (1.0 if format_ok else 0.0)
    )
    if not grounding.grounded:
        # Ungrounded values are vetoed below, but the number still has to be
        # meaningful: a review queue sorted by confidence must not put an
        # unsupported value above a merely ambiguous one. Scaling by the
        # grounding score keeps the field readable as "how likely is this
        # correct" rather than "how much did the model like it".
        confidence *= grounding.score
    confidence = round(min(max(confidence, 0.0), 1.0), 4)

    provenance = Provenance(
        page=page,
        bbox=_merge_boxes([o.bbox for o in outcome.supporting if o.bbox]),
        quote=grounding.matched_quote or outcome.supporting[0].quote,
        grounded=grounding.grounded,
        grounding_score=grounding.score,
    )

    has_reference = bool(reference_text.strip())
    if (has_reference and not grounding.grounded) or confidence < abstain_below:
        return FieldResult(
            name=name,
            value=None,
            status=FieldStatus.ABSTAINED,
            confidence=confidence,
            agreement=outcome.agreement,
            provenance=provenance,
            reason=_abstention_reason(outcome.agreement, grounding, format_ok),
            observations=observations,
        )

    return FieldResult(
        name=name,
        value=outcome.display_value,
        status=FieldStatus.EXTRACTED,
        confidence=confidence,
        agreement=outcome.agreement,
        provenance=provenance,
        reason=None,
        observations=observations,
    )


def _abstention_reason(agreement: float, grounding: GroundingReport, format_ok: bool) -> str:
    """Human-readable explanation, ordered by which signal failed hardest."""
    causes: list[str] = []
    if not grounding.grounded:
        causes.append(f"value not located in page text (score {grounding.score:.2f})")
    if agreement < 1.0:
        causes.append(f"decoding passes disagreed (agreement {agreement:.2f})")
    if not format_ok:
        causes.append("value failed format validation")
    return "; ".join(causes) or "confidence below threshold"


def _merge_boxes(boxes: list[BoundingBox]) -> BoundingBox | None:
    """Union of the supporting passes' boxes, so the UI highlights one region."""
    if not boxes:
        return None
    return BoundingBox(
        x0=min(b.x0 for b in boxes),
        y0=min(b.y0 for b in boxes),
        x1=max(b.x1 for b in boxes),
        y1=max(b.y1 for b in boxes),
    )
