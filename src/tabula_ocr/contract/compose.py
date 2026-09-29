"""Deterministic composition and voting: turn raw model lines into the graded ``text``.

The model is asked only to *see*: which kind of object this is and which text lines it reads.
Everything the challenge's rules decide is plain code, so it is testable and cannot drift with
a prompt: dropping the printed jurisdiction banner but never a Chinese province character,
joining lines top to bottom with single spaces, repairing characters a Chinese plate can never
contain, and picking one answer out of several independent readings.

The grader compares after normalisation (uppercase; whitespace and ``- . _ ·`` removed), so
votes are counted on the normalised form: ``7-ABC-123`` and ``7 abc 123`` are the same vote.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

__all__ = [
    "Candidate",
    "Reading",
    "Verdict",
    "canonical_kind",
    "compose",
    "grader_normalize",
    "vote",
]

_MIDDLE_DOT = chr(0x00B7)
_DASHES = chr(0x2010) + "-" + chr(0x2015)
_STRIP = re.compile(rf"[\s\-._{_MIDDLE_DOT}{_DASHES}]+")
_SPACES = re.compile(r"\s+")
_JUNK_EDGES = "\"'`" + chr(0x201C) + chr(0x201D) + chr(0x2018) + chr(0x2019) + "|[](){}<>,;:"

# The first character of a mainland plate. The second is always a letter.
PROVINCES = "京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领"
_CN_SUFFIX = "挂学警港澳"
_CN_PLATE = re.compile(rf"([{PROVINCES}])([A-Z])([A-Z0-9]{{4,6}}[{_CN_SUFFIX}]?)")

_STATES = (
    "ALABAMA|ALASKA|ARIZONA|ARKANSAS|CALIFORNIA|COLORADO|CONNECTICUT|DELAWARE|FLORIDA|GEORGIA|"
    "HAWAII|IDAHO|ILLINOIS|INDIANA|IOWA|KANSAS|KENTUCKY|LOUISIANA|MAINE|MARYLAND|"
    "MASSACHUSETTS|MICHIGAN|MINNESOTA|MISSISSIPPI|MISSOURI|MONTANA|NEBRASKA|NEVADA|OHIO|"
    "OKLAHOMA|OREGON|PENNSYLVANIA|TENNESSEE|TEXAS|UTAH|VERMONT|VIRGINIA|WASHINGTON|"
    "WISCONSIN|WYOMING|NEW HAMPSHIRE|NEW JERSEY|NEW MEXICO|NEW YORK|NORTH CAROLINA|"
    "NORTH DAKOTA|RHODE ISLAND|SOUTH CAROLINA|SOUTH DAKOTA|WEST VIRGINIA|"
    "DISTRICT OF COLUMBIA"
)
_SLOGANS = (
    "THE GOLDEN STATE|GOLDEN STATE|THE LONE STAR STATE|LONE STAR STATE|SUNSHINE STATE|"
    "THE SUNSHINE STATE|THE EMPIRE STATE|EMPIRE STATE|THE PEACH STATE|PEACH STATE|"
    "LAND OF ENCHANTMENT|LIVE FREE OR DIE|THE GREAT LAKES STATE|GREAT LAKES STATE|"
    "THE BUCKEYE STATE|THE GARDEN STATE|GARDEN STATE|THE GRAND CANYON STATE|"
    "GRAND CANYON STATE|THE EVERGREEN STATE|EVERGREEN STATE|THE SILVER STATE|SILVER STATE|"
    "THE BEEHIVE STATE|BEEHIVE STATE|THE SHOW ME STATE|THE VOLUNTEER STATE|"
    "THE KEYSTONE STATE|THE OCEAN STATE|THE FIRST STATE|THE PALMETTO STATE|"
    "THE HAWKEYE STATE|THE SUNFLOWER STATE|THE BLUEGRASS STATE|THE PELICAN STATE|"
    "THE PINE TREE STATE|THE OLD LINE STATE|THE CONSTITUTION STATE|THE GRANITE STATE|"
    "THE MOUNTAIN STATE|THE BAY STATE|THE NORTH STAR STATE|THE LAND OF LINCOLN|"
    "THE HOOSIER STATE|THE NATURAL STATE|THE MAGNOLIA STATE|THE COWBOY STATE|"
    "THE BIG SKY COUNTRY|BIG SKY COUNTRY|THE PEACE GARDEN STATE|THE SOONER STATE|"
    "THE HEART OF IT ALL|THE ALOHA STATE|ALOHA STATE|THE LAST FRONTIER|"
    "THE GEM STATE|GEM STATE|THE CENTENNIAL STATE|CENTENNIAL STATE|THE TAR HEEL STATE|"
    "FIRST IN FLIGHT|LIFE'S BETTER HERE|SEE THE REAL WYOMING|WHERE THE WEST BEGINS"
)
_BANNERS = sorted(set(_STATES.split("|")) | set(_SLOGANS.split("|")), key=len, reverse=True)
_BANNER_RE = re.compile(
    r"(?<![A-Z0-9])(?:" + "|".join(re.escape(b) for b in _BANNERS if b) + r")(?![A-Z0-9])",
    re.IGNORECASE,
)

_KINDS = {"us_plate", "cn_plate", "sign", "other"}


def grader_normalize(text: str) -> str:
    """Normalise like the official grader: uppercase, drop whitespace and ``- . _ ·``."""
    return _STRIP.sub("", text).upper()


def canonical_kind(kind: object) -> str:
    """Map whatever the model called the object onto one of our four kinds."""
    name = str(kind or "").strip().lower().replace(" ", "_").replace("-", "_")
    if name in _KINDS:
        return name
    if "plate" in name or "licen" in name:
        return (
            "cn_plate" if ("cn" in name or "china" in name or "chinese" in name) else "us_plate"
        )
    if "sign" in name or "stop" in name or "speed" in name:
        return "sign"
    return "other"


@dataclass(frozen=True)
class Reading:
    """What one view of the image produced, as the model reported it."""

    view: str
    weight: float
    kind: str
    lines: tuple[str, ...]


@dataclass(frozen=True)
class Candidate:
    """A composed answer from one view."""

    view: str
    weight: float
    text: str
    kind: str


@dataclass(frozen=True)
class Verdict:
    """The voted result."""

    text: str
    confidence: float
    kind: str
    candidates: tuple[Candidate, ...] = ()


def _clean_lines(lines: tuple[str, ...] | list[str]) -> list[str]:
    cleaned: list[str] = []
    for raw in lines:
        line = _SPACES.sub(" ", str(raw)).strip().strip(_JUNK_EDGES).strip()
        if line:
            cleaned.append(line)
    return cleaned


def _plate_score(line: str) -> float:
    """How much a line looks like a registration: 5-8 characters mixing letters and digits."""
    core = re.sub(r"[^A-Za-z0-9]", "", line)
    if not core:
        return float("-inf")
    size = len(core)
    score = 0.0
    if 5 <= size <= 8:
        score += 3.0
    elif 2 <= size <= 4:
        score += 1.0
    else:
        score -= abs(size - 8)
    has_digit = any(c.isdigit() for c in core)
    has_alpha = any(c.isalpha() for c in core)
    if has_digit and has_alpha:
        score += 2.0
    elif has_digit or has_alpha:
        score += 0.5
    return score


def find_cn_plate(text: str) -> str | None:
    """Locate a mainland registration (province + letter + 4-6 more) inside free text."""
    squeezed = grader_normalize(text)
    match = _CN_PLATE.search(squeezed)
    if match is None:
        # ``I`` and ``O`` never occur on these plates, so a model that wrote them was reading
        # ``1`` and ``0``; retry with the tail repaired before giving up.
        repaired = re.sub(
            rf"([{PROVINCES}][A-Z])([A-Z0-9]+)",
            lambda m: m.group(1) + m.group(2).replace("O", "0").replace("I", "1"),
            squeezed,
        )
        match = _CN_PLATE.search(repaired)
        if match is None:
            return None
    tail = match.group(3).replace("O", "0").replace("I", "1")
    return f"{match.group(1)}{match.group(2)}{tail}"


def _compose_us_plate(lines: list[str]) -> str:
    stripped = _clean_lines([_BANNER_RE.sub(" ", line) for line in lines])
    if not stripped:
        return " ".join(lines)
    best = max(stripped, key=_plate_score)
    return best if _plate_score(best) > float("-inf") else " ".join(stripped)


def compose(kind: str, lines: tuple[str, ...] | list[str]) -> str:
    """Compose the graded string from the lines a view read.

    Plates: the number only (banner and slogan dropped), except a Chinese plate, whose leading
    province character is part of the number. Signs and anything else: every line, top to
    bottom, joined with single spaces.
    """
    cleaned = _clean_lines(lines)
    if not cleaned:
        return ""
    cn = find_cn_plate(" ".join(cleaned))
    if cn is not None:
        return cn
    if kind == "us_plate":
        return _compose_us_plate(cleaned)
    return " ".join(cleaned)


def _plausibility(kind: str, text: str) -> float:
    return _plate_score(text) if kind in ("us_plate", "cn_plate") else float(len(text))


def _majority_kind(readings: list[Reading]) -> str:
    tally: dict[str, float] = defaultdict(float)
    for reading in readings:
        if find_cn_plate(" ".join(reading.lines)) is not None:
            tally["cn_plate"] += reading.weight
        else:
            tally[reading.kind] += reading.weight
    return max(tally.items(), key=lambda item: item[1])[0] if tally else "other"


def _charwise(candidates: list[Candidate], length: int) -> str:
    """Position-by-position weighted vote across equally long readings."""
    columns: list[dict[str, float]] = [defaultdict(float) for _ in range(length)]
    for cand in candidates:
        for position, char in enumerate(grader_normalize(cand.text)):
            columns[position][char] += cand.weight
    return "".join(max(col.items(), key=lambda kv: kv[1])[0] for col in columns)


def vote(readings: list[Reading]) -> Verdict:
    """Choose the final text from several independent readings of one image."""
    usable = [r for r in readings if any(line.strip() for line in r.lines)]
    if not usable:
        return Verdict("", 0.0, "other")
    kind = _majority_kind(usable)
    candidates = [Candidate(r.view, r.weight, compose(kind, r.lines), kind) for r in usable]
    candidates = [c for c in candidates if grader_normalize(c.text)]
    if not candidates:
        return Verdict("", 0.0, kind)

    weight: dict[str, float] = defaultdict(float)
    display: dict[str, str] = {}
    for cand in candidates:
        key = grader_normalize(cand.text)
        weight[key] += cand.weight
        display.setdefault(key, cand.text)  # views are ordered original-first
    total = sum(weight.values())
    winner = max(weight, key=lambda k: (weight[k], _plausibility(kind, display[k])))
    text = display[winner]
    support = weight[winner] / total

    # No clear consensus, yet several readings of the same length: their disagreements are
    # usually single characters that different renderings got wrong in different places.
    if support < 0.6:
        length_weight: dict[int, float] = defaultdict(float)
        for cand in candidates:
            length_weight[len(grader_normalize(cand.text))] += cand.weight
        common = max(length_weight, key=lambda n: length_weight[n])
        same = [c for c in candidates if len(grader_normalize(c.text)) == common]
        if len(same) >= 3:
            merged = _charwise(same, common)
            if merged != winner:
                text, support = merged, max(support, weight.get(merged, 0.0) / total)
    confidence = round(min(0.99, 0.35 + 0.64 * support), 2)
    return Verdict(text, confidence, kind, tuple(candidates))
