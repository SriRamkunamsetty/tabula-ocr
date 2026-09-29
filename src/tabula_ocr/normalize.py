"""Value normalisation and format validation.

Two different jobs live here and are deliberately kept apart.

*Normalisation* makes two textually different readings of the same value
comparable, so that ``"Rs. 1,240.00"`` and ``"1240.0"`` count as agreement
rather than as a disputed field. Consensus operates on normalised values.

*Validation* asks whether a value is well-formed for its declared type. A phone
number that does not parse is evidence that the model misread it, so validation
feeds the confidence score rather than raising.

Both are pure functions with no I/O, which is what makes the confidence score
reproducible across runs.
"""

from __future__ import annotations

import datetime as dt
import re
import unicodedata
from collections.abc import Callable
from typing import Any

__all__ = [
    "SUPPORTED_TYPES",
    "canonical_text",
    "normalize_value",
    "validate_format",
]

SUPPORTED_TYPES = frozenset(
    {
        "string",
        "text",
        "number",
        "currency",
        "integer",
        "date",
        "phone",
        "email",
        "version",
        "boolean",
    }
)

_WS_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^0-9a-z]+")
_CURRENCY_RE = re.compile(r"[^\d.,\-]")
_NUMBER_TOKEN_RE = re.compile(r"-?\d[\d.,]*")
_PHONE_KEEP_RE = re.compile(r"[^\d+]")
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s.]+\.[^@\s]+$")
_SEMVER_RE = re.compile(r"^v?\d+(\.\d+){0,3}(-[0-9A-Za-z.\-]+)?(\+[0-9A-Za-z.\-]+)?$")

_DATE_FORMATS = (
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%d-%m-%Y",
    "%d %b %Y",
    "%d %B %Y",
    "%b %d, %Y",
    "%B %d, %Y",
    "%Y/%m/%d",
)


def canonical_text(value: str) -> str:
    """Aggressively fold a string for equality comparison.

    Unicode is NFKC-normalised first so that full-width digits, ligatures and
    the several dash characters OCR models emit collapse onto their ASCII
    equivalents before anything else runs.
    """
    folded = unicodedata.normalize("NFKC", value).casefold()
    return _NON_ALNUM_RE.sub("", folded)


def _normalize_number(value: str) -> str | None:
    # Extract the numeric token rather than deleting non-numeric characters:
    # stripping characters from "Rs. 1,24,500.00" leaves a leading separator
    # (".1,24,500.00") that no decimal-point heuristic can recover from.
    match = _NUMBER_TOKEN_RE.search(value)
    if match is None:
        return None
    cleaned = match.group(0).rstrip(".,")
    if not cleaned:
        return None
    # Treat the last separator as the decimal point when both appear, which
    # handles both 1,234.56 and 1.234,56 without guessing a locale up front.
    if "," in cleaned and "." in cleaned:
        decimal_sep = "," if cleaned.rfind(",") > cleaned.rfind(".") else "."
        thousands_sep = "." if decimal_sep == "," else ","
        cleaned = cleaned.replace(thousands_sep, "").replace(decimal_sep, ".")
    elif "," in cleaned:
        parts = cleaned.split(",")
        cleaned = cleaned.replace(",", "." if len(parts[-1]) == 2 else "")
    try:
        number = float(cleaned)
    except ValueError:
        return None
    return f"{number:.4f}".rstrip("0").rstrip(".") or "0"


def _normalize_date(value: str) -> str | None:
    text = _WS_RE.sub(" ", value.strip())
    for fmt in _DATE_FORMATS:
        try:
            return dt.datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def _normalize_phone(value: str) -> str | None:
    digits = _PHONE_KEEP_RE.sub("", value)
    if digits.startswith("+"):
        digits = "+" + digits[1:].replace("+", "")
    core = digits.lstrip("+")
    if not 7 <= len(core) <= 15:
        return None
    return digits


def _normalize_version(value: str) -> str | None:
    token = value.strip().lstrip("vV")
    return token if _SEMVER_RE.match(f"v{token}") else None


def _normalize_boolean(value: str) -> str | None:
    folded = value.strip().casefold()
    if folded in {"true", "yes", "y", "1", "checked", "x"}:
        return "true"
    if folded in {"false", "no", "n", "0", "unchecked", ""}:
        return "false"
    return None


_NORMALIZERS: dict[str, Callable[[str], str | None]] = {
    "number": _normalize_number,
    "currency": _normalize_number,
    "integer": lambda v: n.split(".")[0] if (n := _normalize_number(v)) else None,
    "date": _normalize_date,
    "phone": _normalize_phone,
    "version": _normalize_version,
    "boolean": _normalize_boolean,
    "email": lambda v: v.strip().casefold() or None,
}


def normalize_value(value: Any, field_type: str) -> str | None:
    """Return a canonical string form of ``value``, or ``None`` if unusable.

    ``None`` is a meaningful result: it means this pass produced nothing that
    can be compared, and consensus will treat it as a missing vote rather than
    as a vote for the empty string.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        value = "true" if value else "false"
    text = str(value).strip()
    if not text or text.casefold() in {"null", "none", "n/a", "na", "-", "--"}:
        return None

    normalizer = _NORMALIZERS.get(field_type)
    if normalizer is not None:
        return normalizer(text)
    return _WS_RE.sub(" ", text)


def validate_format(value: Any, field_type: str) -> bool:
    """Return True when ``value`` is well-formed for ``field_type``.

    Unknown types validate as ``True``: an unrecognised type is a schema
    authoring issue, and failing every field of it would hide the real problem
    behind a wall of low-confidence results.
    """
    normalized = normalize_value(value, field_type)
    if normalized is None:
        return False
    match field_type:
        case "email":
            return bool(_EMAIL_RE.match(normalized))
        case "phone":
            return bool(_normalize_phone(normalized))
        case "version":
            return bool(_normalize_version(normalized))
        case "date":
            return bool(_normalize_date(normalized) or _DATE_ISO_RE.match(normalized))
        case "integer":
            return normalized.lstrip("-").isdigit()
        case "number" | "currency":
            try:
                float(normalized)
            except ValueError:
                return False
            return True
        case "boolean":
            return normalized in {"true", "false"}
        case _:
            return bool(normalized)


_DATE_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
