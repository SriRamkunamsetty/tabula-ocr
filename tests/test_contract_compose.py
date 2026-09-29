import pytest

from tabula_ocr.contract.compose import (
    Reading,
    canonical_kind,
    compose,
    find_cn_plate,
    grader_normalize,
    vote,
)


def read(view: str, weight: float, kind: str, *lines: str) -> Reading:
    return Reading(view, weight, kind, tuple(lines))


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("7-ABC-123", "7ABC123"), ("7 abc 123", "7ABC123"), ("A·12_3.4", "A1234")],
)
def test_grader_normalisation_matches_published_rule(raw, expected):
    assert grader_normalize(raw) == expected


@pytest.mark.parametrize(
    ("kind", "lines", "expected"),
    [
        ("us_plate", ["CALIFORNIA", "7ABC123"], "7ABC123"),
        ("us_plate", ["CALIFORNIA 7ABC123"], "7ABC123"),
        ("us_plate", ["TEXAS", "ABC 1234", "THE LONE STAR STATE"], "ABC 1234"),
        ("us_plate", ["NEW YORK", "EMPIRE STATE", "KLM-4821"], "KLM-4821"),
        ("us_plate", ["7ABC123"], "7ABC123"),
        ("sign", ["SPEED", "LIMIT", "65"], "SPEED LIMIT 65"),
        ("sign", ["STOP"], "STOP"),
        ("sign", ["  ADVISORY ", "", "SPEED 35 "], "ADVISORY SPEED 35"),
        ("other", [], ""),
    ],
)
def test_composition_rules(kind, lines, expected):
    assert compose(kind, lines) == expected


def test_chinese_plate_keeps_province_and_letter():
    assert compose("cn_plate", ["京", "A12345"]) == "京A12345"
    assert compose("cn_plate", ["京A·12345"]) == "京A12345"
    # New-energy plates carry six trailing characters.
    assert compose("cn_plate", ["粤B D12345"]) == "粤BD12345"


def test_chinese_plate_is_not_mistaken_for_us_plate():
    assert compose("us_plate", ["沪A 8K266"]) == "沪A8K266"


def test_chinese_plate_never_contains_letters_o_or_i():
    assert find_cn_plate("京AO1I34") == "京A01134"
    assert find_cn_plate("京A12O45") == "京A12045"
    assert find_cn_plate("沪A8KI66") == "沪A8K166"


def test_no_chinese_plate_in_ordinary_text():
    assert find_cn_plate("SPEED LIMIT 65") is None


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("us_plate", "us_plate"),
        ("US license plate", "us_plate"),
        ("Chinese licence plate", "cn_plate"),
        ("Speed limit sign", "sign"),
        ("poster", "other"),
        (None, "other"),
    ],
)
def test_kind_names_are_canonicalised(given, expected):
    assert canonical_kind(given) == expected


def test_vote_prefers_the_majority_reading():
    verdict = vote(
        [
            read("original", 1.5, "us_plate", "CALIFORNIA", "7ABC123"),
            read("contrast", 1.0, "us_plate", "7ABC128"),
            read("denoised", 1.0, "us_plate", "7-ABC-123"),
            read("equalized", 1.0, "us_plate", "7ABC123"),
        ]
    )
    assert grader_normalize(verdict.text) == "7ABC123"
    assert verdict.confidence > 0.8


def test_one_bad_view_cannot_win_against_the_rest():
    verdict = vote(
        [
            read("original", 1.5, "sign", "STOP"),
            read("contrast", 1.0, "sign", "STOP"),
            read("denoised", 1.0, "sign", "SIOP"),
        ]
    )
    assert verdict.text == "STOP"


def test_original_view_breaks_a_two_way_tie():
    verdict = vote(
        [
            read("original", 1.5, "us_plate", "7ABC123"),
            read("contrast", 1.5, "us_plate", "7ABC128"),
        ]
    )
    assert grader_normalize(verdict.text) == "7ABC123"


def test_charwise_vote_repairs_errors_in_different_places():
    verdict = vote(
        [
            read("original", 1.5, "us_plate", "7ABC128"),
            read("contrast", 1.0, "us_plate", "7A8C123"),
            read("denoised", 1.0, "us_plate", "7ABC123"),
            read("equalized", 1.0, "us_plate", "TABC123"),
        ]
    )
    assert grader_normalize(verdict.text) == "7ABC123"


def test_votes_are_counted_on_the_normalised_form():
    verdict = vote(
        [
            read("original", 1.5, "us_plate", "7 abc 123"),
            read("contrast", 1.0, "us_plate", "7-ABC-123"),
            read("denoised", 1.0, "us_plate", "7ABC123"),
        ]
    )
    assert verdict.confidence >= 0.95


def test_chinese_plate_wins_the_kind_vote_even_if_the_model_says_us_plate():
    verdict = vote(
        [
            read("original", 1.5, "us_plate", "京A12345"),
            read("contrast", 1.0, "cn_plate", "京", "A12345"),
        ]
    )
    assert verdict.kind == "cn_plate"
    assert verdict.text == "京A12345"


def test_empty_readings_give_an_empty_verdict():
    verdict = vote([read("original", 1.5, "sign"), read("contrast", 1.0, "sign", "  ")])
    assert (verdict.text, verdict.confidence) == ("", 0.0)
    assert vote([]).text == ""
