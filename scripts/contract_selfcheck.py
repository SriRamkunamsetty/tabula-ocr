r"""Replay the grader's invocation model against this repository, then score like the grader.

    python scripts/contract_selfcheck.py                     # synthetic images + scripted model
    python scripts/contract_selfcheck.py --live              # your real model
    python scripts/contract_selfcheck.py --live --images mc2-starter-kit/images \\
        --expected mc2-starter-kit/expected.json

What it does that in-process tests cannot: every image is a separate OS process
(``app.py --input-image ...``), exactly as the harness runs them, the model is reached over real
HTTP with real base64 images, and each image's wall-clock time is checked against the 30 second
limit. Scoring follows the published rule: 20 points per image, ``text`` must match the expected
answer after normalisation (uppercase; whitespace and ``- . _ ·`` removed).

The scripted mode uses ten synthetic images, one per awkward case the grader promises (PNG,
JPEG, TIFF; huge, tiny, transparent, palette, 16-bit, EXIF-rotated, multi-frame), each with a
unique aspect ratio so the scripted server can tell them apart. It validates parsing, image
handling, voting, timing and the file contract. It does **not** measure how well a real model
reads characters: that needs ``--live`` on real hardware.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tabula_ocr.contract.compose import grader_normalize  # noqa: E402

PER_IMAGE_LIMIT_S = 30.0
POINTS = 20
SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def _canvas(size: tuple[int, int], label: str, mode: str = "RGB") -> Image.Image:
    image = Image.new("RGB", size, (235, 235, 235))
    ImageDraw.Draw(image).text((6, 6), label, fill=(20, 20, 20))
    return image.convert(mode) if mode != "RGB" else image


def build_images(folder: Path) -> list[tuple[Path, str, dict[str, Any]]]:
    """Ten images and, for each, the expected text and the scripted model's reply."""
    folder.mkdir(parents=True, exist_ok=True)
    plate = {"kind": "us_plate", "lines": ["CALIFORNIA", "7ABC123"]}
    cn = {"kind": "cn_plate", "lines": ["京", "A12345"]}
    speed = {"kind": "sign", "lines": ["SPEED", "LIMIT", "65"]}
    stop = {"kind": "sign", "lines": ["STOP"]}
    texas = {"kind": "us_plate", "lines": ["TEXAS", "KLM-4821", "THE LONE STAR STATE"]}
    items: list[tuple[str, Image.Image, dict[str, str | int], str, dict[str, Any]]] = []

    def add(
        name: str,
        image: Image.Image,
        save: dict[str, Any],
        expected: str,
        reply: dict[str, Any],
    ) -> None:
        items.append((name, image, save, expected, reply))

    add("image_01.png", _canvas((640, 210), "us"), {}, "7ABC123", plate)
    add("image_02.jpg", _canvas((520, 150), "cn"), {"quality": 90}, "京A12345", cn)
    add("image_03.tif", _canvas((400, 400), "speed"), {}, "SPEED LIMIT 65", speed)
    add("image_04.png", _canvas((6000, 2000), "huge stop"), {}, "STOP", stop)
    add("image_05.jpg", _canvas((91, 31), "tiny"), {}, "KLM4821", texas)
    add("image_06.png", Image.new("RGBA", (333, 90), (0, 0, 0, 0)), {}, "7ABC123", plate)
    add("image_07.tif", Image.new("I;16", (350, 97), 900), {}, "SPEED LIMIT 65", speed)
    add("image_08.jpg", _canvas((230, 700), "rotated"), {}, "STOP", stop)
    add("image_09.tiff", _canvas((300, 123), "frames"), {}, "京A12345", cn)
    add("image_10.png", _canvas((500, 178), "palette").convert("P"), {}, "KLM4821", texas)

    out: list[tuple[Path, str, dict[str, Any]]] = []
    for name, image, save, expected, reply in items:
        path = folder / name
        if name == "image_08.jpg":
            exif = Image.Exif()
            exif[0x0112] = 6  # stored sideways; the model must be shown it upright
            image = image.rotate(90, expand=True)
            image.save(path, exif=exif, **save)
        elif name == "image_09.tiff":
            image.save(
                path, save_all=True, append_images=[_canvas((50, 50), "other frame")], **save
            )
        else:
            image.save(path, **save)
        out.append((path, expected, reply))
    return out


def _aspect(width: int, height: int) -> float:
    return round(width / height, 2)


class ScriptedServer:
    """A tiny OpenAI-compatible vision server: replies by the aspect ratio it is shown."""

    def __init__(self, images: list[tuple[Path, str, dict[str, Any]]]) -> None:
        """Start serving on a free local port."""
        from tabula_ocr.contract.views import load_image

        table: dict[float, dict[str, Any]] = {}
        for path, _, reply in images:
            width, height = load_image(path).size
            table[_aspect(width, height)] = reply

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args: object) -> None:  # silence
                return

            def _send(self, payload: dict[str, Any]) -> None:
                body = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                self._send({"data": [{"id": "scripted-vlm"}]})

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                uri = body["messages"][0]["content"][1]["image_url"]["url"].split(",", 1)[1]
                with Image.open(io.BytesIO(base64.b64decode(uri))) as shown:
                    key = _aspect(*shown.size)
                near = min(table, key=lambda known: abs(known - key))
                reply = (
                    table[near] if abs(near - key) < 0.03 else {"kind": "other", "lines": []}
                )
                self._send({"choices": [{"message": {"content": json.dumps(reply)}}]})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()


def _load_expected(path: Path) -> dict[str, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        raw = {
            str(row.get("image") or row.get("file")): row.get("text", row.get("answer", ""))
            for row in raw
        }
    return {Path(k).stem: str(v) for k, v in raw.items()}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--live", action="store_true", help="use the model at TABULA_OCR_LLM_BASE_URL"
    )
    parser.add_argument("--images", type=Path, help="folder of PNG/JPEG/TIFF images")
    parser.add_argument("--expected", type=Path, help="JSON: {image_name: text} or a list")
    args = parser.parse_args()

    work = Path(tempfile.mkdtemp(prefix="tabula-ocr-selfcheck-"))
    env = {**os.environ, "TABULA_OCR_OUTPUT_DIR": str(work / "out")}
    server: ScriptedServer | None = None
    if args.images:
        if not args.live or not args.expected:
            print("--images needs --live and --expected", file=sys.stderr)
            return 2
        expected = _load_expected(args.expected)
        cases = [
            (p, expected.get(p.stem, ""), {})
            for p in sorted(args.images.iterdir())
            if p.suffix.lower() in SUFFIXES
        ]
    else:
        cases = build_images(work / "input")
        if not args.live:
            server = ScriptedServer(cases)
            env["TABULA_OCR_LLM_BASE_URL"] = f"http://127.0.0.1:{server.port}/v1"

    score, problems = 0, []
    print(f"{'#':>2} {'time':>6}  {'result':6} image")
    for number, (image, expected_text, _) in enumerate(cases, start=1):
        started = time.monotonic()
        done = subprocess.run(
            [sys.executable, str(REPO / "app.py"), "--input-image", str(image)],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        elapsed = time.monotonic() - started
        target = work / "out" / f"{image.stem}_output.json"
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
            assert isinstance(payload["text"], str)
        except (OSError, ValueError, KeyError, AssertionError):
            payload = None
            problems.append(f"{image.name}: missing or malformed output file")
        if elapsed > PER_IMAGE_LIMIT_S:
            problems.append(
                f"{image.name}: took {elapsed:.1f}s (limit {PER_IMAGE_LIMIT_S:.0f}s)"
            )
        if done.returncode != 0:
            problems.append(f"{image.name}: exit code {done.returncode}")
        ok = payload is not None and grader_normalize(payload["text"]) == grader_normalize(
            expected_text
        )
        score += POINTS if ok else 0
        print(f"{number:>2} {elapsed:5.1f}s  {'PASS' if ok else 'FAIL':6} {image.name}")
        if not ok and payload is not None:
            print(f"     got {payload['text']!r}  expected {expected_text!r}")

    print(f"\nscore: {score}/{POINTS * len(cases)}")
    for problem in problems:
        print(f"PROBLEM: {problem}")
    if server:
        server.close()
    return 0 if score == POINTS * len(cases) and not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
