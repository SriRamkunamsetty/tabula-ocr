import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from PIL import Image

from tabula_ocr.contract.cli import main
from tabula_ocr.contract.config import ContractSettings
from tabula_ocr.contract.llm import OpenAIVision, extract_json_object, parse_reading
from tabula_ocr.contract.runner import output_path, read_image, run_image, write_output


class ScriptedVision:
    """Answers each view from a script; the call order matches the view order."""

    def __init__(self, replies: list[dict[str, Any] | Exception | None]) -> None:
        self.replies = replies
        self.calls = 0

    async def read(self, png: bytes, *, timeout_s: float) -> dict[str, Any] | None:
        reply = self.replies[self.calls % len(self.replies)]
        self.calls += 1
        if isinstance(reply, Exception):
            raise reply
        return reply

    async def wait_ready(self, budget_s: float) -> bool:
        return True

    async def aclose(self) -> None:
        return None


def settings(tmp_path: Path, **kwargs: Any) -> ContractSettings:
    return ContractSettings(output_dir=tmp_path / "out", **kwargs)


def make_image(tmp_path: Path, name: str = "image_01.png") -> Path:
    path = tmp_path / name
    Image.new("RGB", (300, 100), (240, 240, 240)).save(path)
    return path


def plate(*lines: str, kind: str = "us_plate") -> dict[str, Any]:
    return {"kind": kind, "lines": list(lines)}


def load(target: Path) -> dict[str, Any]:
    return json.loads(target.read_text(encoding="utf-8"))


def test_output_file_is_named_after_the_input_and_has_the_graded_shape(tmp_path):
    image = make_image(tmp_path, "image_07.png")
    model = ScriptedVision([plate("CALIFORNIA", "7ABC123")])
    run_image(image, settings(tmp_path), model, watchdog=False)
    payload = load(tmp_path / "out" / "image_07_output.json")
    assert set(payload) == {"text", "confidence"}
    assert payload["text"] == "7ABC123"
    assert 0.0 <= payload["confidence"] <= 1.0


def test_stem_handles_dotted_names_and_every_format(tmp_path):
    cfg = settings(tmp_path)
    assert output_path(cfg, Path("/app/input/image_01.tiff")).name == "image_01_output.json"
    assert output_path(cfg, Path("/app/input/a.b.jpeg")).name == "a.b_output.json"


def test_one_bad_view_is_outvoted(tmp_path):
    image = make_image(tmp_path)
    model = ScriptedVision(
        [plate("7ABC123"), plate("7ABC128"), plate("7ABC123"), plate("7ABC123")]
    )
    run_image(image, settings(tmp_path), model, watchdog=False)
    assert load(tmp_path / "out" / "image_01_output.json")["text"] == "7ABC123"


def test_failed_views_are_ignored_when_others_succeed(tmp_path):
    image = make_image(tmp_path)
    model = ScriptedVision([RuntimeError("boom"), None, plate("STOP", kind="sign")])
    run_image(image, settings(tmp_path), model, watchdog=False)
    assert load(tmp_path / "out" / "image_01_output.json")["text"] == "STOP"


def test_model_down_leaves_a_valid_placeholder(tmp_path):
    image = make_image(tmp_path)
    run_image(
        image, settings(tmp_path, model_wait_s=0.0), ScriptedVision([None]), watchdog=False
    )
    assert load(tmp_path / "out" / "image_01_output.json") == {"text": "", "confidence": 0.0}


def test_unreadable_image_still_leaves_a_valid_file(tmp_path):
    bad = tmp_path / "image_02.png"
    bad.write_bytes(b"garbage")
    run_image(bad, settings(tmp_path), ScriptedVision([plate("X")]), watchdog=False)
    assert load(tmp_path / "out" / "image_02_output.json") == {"text": "", "confidence": 0.0}


def test_missing_image_still_leaves_a_valid_file(tmp_path):
    model = ScriptedVision([plate("X")])
    run_image(tmp_path / "gone.png", settings(tmp_path), model, watchdog=False)
    assert load(tmp_path / "out" / "gone_output.json")["text"] == ""


def test_chinese_text_is_written_as_real_characters(tmp_path):
    image = make_image(tmp_path)
    model = ScriptedVision([plate("京", "A12345", kind="cn_plate")])
    run_image(image, settings(tmp_path), model, watchdog=False)
    raw = (tmp_path / "out" / "image_01_output.json").read_text(encoding="utf-8")
    assert "京A12345" in raw  # not escaped: the grader wants the characters as they appear


def test_slow_model_is_cut_off_but_progress_is_kept(tmp_path):
    class Slow(ScriptedVision):
        async def read(self, png: bytes, *, timeout_s: float) -> dict[str, Any] | None:
            self.calls += 1
            if self.calls > 1:
                await asyncio.sleep(30)
            return plate("7ABC123")

    image = make_image(tmp_path)
    cfg = settings(tmp_path, budget_s=3.0, view_timeout_s=2.0)
    run_image(image, cfg, Slow([None]), watchdog=False)
    assert load(tmp_path / "out" / "image_01_output.json")["text"] == "7ABC123"


def test_read_image_returns_a_verdict_without_touching_the_disk(tmp_path):
    image = make_image(tmp_path)
    model = ScriptedVision([plate("SPEED", "LIMIT", "65", kind="sign")])
    verdict = asyncio.run(read_image(image, settings(tmp_path), model))
    assert verdict.text == "SPEED LIMIT 65"
    assert not (tmp_path / "out").exists()


def test_write_output_is_atomic_and_leaves_no_temp_files(tmp_path):
    target = tmp_path / "o" / "x_output.json"
    write_output(target, {"text": "A", "confidence": 0.5})
    write_output(target, {"text": "B", "confidence": 0.6})
    assert [p.name for p in target.parent.iterdir()] == ["x_output.json"]
    assert load(target)["text"] == "B"


# ------------------------------------------------------------------------------- the CLI


def test_cli_ignores_unknown_flags_and_always_exits_zero(tmp_path, monkeypatch, capsys):
    image = make_image(tmp_path)
    monkeypatch.setenv("TABULA_OCR_LLM_BASE_URL", "http://127.0.0.1:9/v1")  # nothing listens
    monkeypatch.setenv("TABULA_OCR_MODEL_WAIT_S", "0")
    monkeypatch.setenv("TABULA_OCR_BUDGET_S", "4")
    argv = ["--input-image", str(image), "--output-dir", str(tmp_path / "o"), "--bogus", "1"]
    assert main(argv) == 0
    assert "ignoring unrecognized" in capsys.readouterr().err
    assert load(tmp_path / "o" / "image_01_output.json") == {"text": "", "confidence": 0.0}


def test_cli_prints_chinese_text_even_when_stdout_is_not_utf8(tmp_path, monkeypatch):
    import io
    import sys

    from tabula_ocr.contract import cli
    from tabula_ocr.contract.compose import Verdict

    monkeypatch.setattr(
        cli, "run_image", lambda *_a, **_k: Verdict("京A12345", 0.9, "cn_plate")
    )
    raw = io.BytesIO()
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(raw, encoding="cp1252"))
    assert (
        cli.main(["--input-image", str(make_image(tmp_path)), "--output-dir", str(tmp_path)])
        == 0
    )
    sys.stdout.flush()
    assert "京A12345" in raw.getvalue().decode("utf-8")


def test_cli_without_arguments_is_a_polite_no_op(capsys):
    assert main([]) == 0
    assert "usage" in capsys.readouterr().err


# ---------------------------------------------------------------------- the HTTP client


def _client(handler, **kwargs) -> OpenAIVision:
    http = httpx.AsyncClient(base_url="http://model/v1", transport=httpx.MockTransport(handler))
    return OpenAIVision(ContractSettings(**kwargs), http)


def _chat(content: str) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


def _read_once(handler) -> dict[str, Any] | None:
    async def go() -> dict[str, Any] | None:
        client = _client(handler)
        try:
            return await client.read(b"\x89PNG", timeout_s=5)
        finally:
            await client.aclose()

    return asyncio.run(go())


def test_client_sends_a_data_uri_image_and_parses_the_json_reply():
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "vlm"}]})
        seen.update(json.loads(request.content))
        return _chat('```json\n{"kind": "sign", "lines": ["STOP"]}\n```')

    assert _read_once(handler) == {"kind": "sign", "lines": ["STOP"]}
    content = seen["messages"][0]["content"]
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert seen["model"] == "vlm"
    assert seen["temperature"] == 0.0


def test_client_retries_without_structured_output_when_the_server_rejects_it():
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "vlm"}]})
        body = json.loads(request.content)
        bodies.append(body)
        if "response_format" in body:
            return httpx.Response(400, json={"error": "unsupported"})
        return _chat('{"kind": "other", "lines": ["HI"]}')

    assert _read_once(handler) == {"kind": "other", "lines": ["HI"]}
    assert "response_format" in bodies[0]
    assert "response_format" not in bodies[1]


@pytest.mark.parametrize("status", [500, 503, 404])
def test_client_returns_none_on_server_errors(status):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "vlm"}]})
        return httpx.Response(status)

    assert _read_once(handler) is None


def test_wait_ready_gives_up_after_its_budget():
    async def go() -> bool:
        client = _client(lambda request: httpx.Response(503))
        try:
            return await client.wait_ready(0.0)
        finally:
            await client.aclose()

    assert asyncio.run(go()) is False


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('noise {"kind": "sign", "lines": ["A"]} tail', {"kind": "sign", "lines": ["A"]}),
        ('<think>hmm {x}</think>{"a": 1}', {"a": 1}),
        ("no json here", None),
    ],
)
def test_json_extraction_is_tolerant(text, expected):
    assert extract_json_object(text) == expected


def test_reading_parser_tolerates_common_model_deviations():
    assert parse_reading({"kind": "sign", "lines": ["A", " ", "B"]}) == ("sign", ["A", "B"])
    assert parse_reading({"text": "7ABC123\nLINE2"}) == ("other", ["7ABC123", "LINE2"])
    assert parse_reading({"lines": 5}) is None
    assert parse_reading(None) is None
