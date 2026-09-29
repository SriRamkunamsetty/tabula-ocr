"""Evaluation metrics.

Character error rate and field accuracy are table stakes; the two metrics that
actually decide whether this service is safe to put in front of a business
process are the last two:

* **Hallucination rate** — of the values the service *did* return, how many were
  wrong. A system that returns everything and is right 85% of the time is worse
  than one that returns 70% of fields and is right 99% of the time, because the
  first one silently poisons downstream data and the second one raises its hand.
* **Abstention precision** — when the service declined, was it right to decline?
  Without this number, abstention can be gamed by abstaining on everything.

Together they make the confidence threshold a tunable engineering decision with
a visible cost on both sides, rather than a magic number.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rapidfuzz.distance import Levenshtein

from tabula_ocr.models import DocumentResult, FieldStatus
from tabula_ocr.normalize import canonical_text, normalize_value

__all__ = ["EvalCounters", "FieldOutcome", "character_error_rate", "score_document"]


def character_error_rate(prediction: str, truth: str) -> float:
    """Levenshtein distance normalised by the length of the ground truth.

    Returns 0.0 when both strings are empty and 1.0 when the truth is empty but
    a prediction was made, which keeps the metric defined on absent fields
    instead of dividing by zero.
    """
    if not truth:
        return 0.0 if not prediction else 1.0
    return min(1.0, Levenshtein.distance(prediction, truth) / len(truth))


@dataclass(frozen=True, slots=True)
class FieldOutcome:
    """Per-field comparison against ground truth."""

    document_id: str
    field: str
    predicted: str | None
    expected: str | None
    status: FieldStatus
    confidence: float
    correct: bool
    cer: float

    @property
    def was_answered(self) -> bool:
        """True when the service returned a value rather than abstaining."""
        return self.status is FieldStatus.EXTRACTED

    @property
    def is_hallucination(self) -> bool:
        """Returned a value that is wrong, or invented one for an absent field."""
        return self.was_answered and not self.correct

    @property
    def is_justified_abstention(self) -> bool:
        """Declined on a field it would probably have got wrong anyway."""
        return not self.was_answered and (self.expected is None or not self.correct)


@dataclass
class EvalCounters:
    """Aggregates :class:`FieldOutcome` values into the report metrics."""

    outcomes: list[FieldOutcome] = field(default_factory=list)

    def add(self, outcome: FieldOutcome) -> None:
        """Record one field outcome."""
        self.outcomes.append(outcome)

    @property
    def total(self) -> int:
        """Number of fields scored."""
        return len(self.outcomes)

    @property
    def answered(self) -> list[FieldOutcome]:
        """Outcomes where the service returned a value."""
        return [o for o in self.outcomes if o.status is FieldStatus.EXTRACTED]

    @property
    def declined(self) -> list[FieldOutcome]:
        """Outcomes where the service abstained."""
        return [o for o in self.outcomes if o.status is not FieldStatus.EXTRACTED]

    @property
    def field_accuracy(self) -> float:
        """Correct answers over all fields, counting an abstention as incorrect."""
        if not self.total:
            return 0.0
        return sum(1 for o in self.outcomes if o.correct and o.was_answered) / self.total

    @property
    def precision(self) -> float:
        """Correct answers over answers given — the number that matters."""
        answered = self.answered
        if not answered:
            return 0.0
        return sum(1 for o in answered if o.correct) / len(answered)

    @property
    def coverage(self) -> float:
        """Share of fields the service was willing to answer."""
        return len(self.answered) / self.total if self.total else 0.0

    @property
    def hallucination_rate(self) -> float:
        """Share of *answers* that were wrong."""
        answered = self.answered
        if not answered:
            return 0.0
        return sum(1 for o in answered if o.is_hallucination) / len(answered)

    @property
    def abstention_precision(self) -> float:
        """Share of abstentions that were the right call."""
        declined = self.declined
        if not declined:
            return 1.0
        return sum(1 for o in declined if o.is_justified_abstention) / len(declined)

    @property
    def mean_cer(self) -> float:
        """Mean character error rate over answered fields."""
        answered = self.answered
        if not answered:
            return 0.0
        return sum(o.cer for o in answered) / len(answered)

    def to_dict(self) -> dict[str, float | int]:
        """Render the metrics as a JSON-safe dictionary."""
        return {
            "fields": self.total,
            "answered": len(self.answered),
            "coverage": round(self.coverage, 4),
            "precision": round(self.precision, 4),
            "hallucination_rate": round(self.hallucination_rate, 4),
            "abstention_precision": round(self.abstention_precision, 4),
            "mean_cer": round(self.mean_cer, 4),
        }


def score_document(
    result: DocumentResult,
    truth: dict[str, str | None],
    field_types: dict[str, str],
) -> list[FieldOutcome]:
    """Compare one extraction result against ground truth.

    Correctness is judged on *normalised* values, so a model that reads
    ``01/04/2026`` where the label says ``2026-04-01`` is scored correct. The
    alternative — exact string match — mostly measures formatting conventions
    rather than reading accuracy.
    """
    outcomes: list[FieldOutcome] = []
    for name, expected in truth.items():
        result_field = result.field(name)
        if result_field is None:
            continue
        field_type = field_types.get(name, "string")
        predicted_raw = result_field.value
        predicted = normalize_value(predicted_raw, field_type)
        expected_norm = normalize_value(expected, field_type)

        if result_field.status is FieldStatus.EXTRACTED:
            correct = bool(
                expected_norm is not None
                and predicted is not None
                and canonical_text(predicted) == canonical_text(expected_norm)
            )
        else:
            correct = expected_norm is None

        outcomes.append(
            FieldOutcome(
                document_id=result.document_id,
                field=name,
                predicted=str(predicted_raw) if predicted_raw is not None else None,
                expected=expected,
                status=result_field.status,
                confidence=result_field.confidence,
                correct=correct,
                cer=character_error_rate(predicted or "", expected_norm or ""),
            )
        )
    return outcomes
