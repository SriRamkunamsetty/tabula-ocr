"""The harness entrypoint: ``python3 /app/app.py --input-image /app/input/image_01.png``.

Unrecognised flags are ignored with a warning (a crash scores zero, a surplus flag should not).
``--input-dir`` is a convenience for local checks: it runs every image in a directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from tabula_ocr.contract.config import ContractSettings
from tabula_ocr.contract.runner import output_path, run_image

__all__ = ["main"]

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="app.py",
        description="TABULA OCR, Mini-Challenge 2 contract mode",
        allow_abbrev=False,
    )
    parser.add_argument("--input-image", metavar="PATH", help="the image to read")
    parser.add_argument(
        "--input-dir", metavar="DIR", help="local check: read every image in a directory"
    )
    parser.add_argument("--output-dir", help="override the output directory (/app/output)")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one invocation. Always exits 0: the output file, not the exit code, is the result."""
    for stream in (sys.stdout, sys.stderr):  # a Chinese plate must never crash the printout
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args, unknown = _parser().parse_known_args(argv)
    if unknown:
        print(f"app.py: ignoring unrecognized arguments: {unknown}", file=sys.stderr)
    settings = ContractSettings.from_env()
    if args.output_dir:
        settings = replace(settings, output_dir=Path(args.output_dir))

    images: list[Path] = []
    if args.input_image:
        images.append(Path(args.input_image))
    if args.input_dir:
        folder = Path(args.input_dir)
        images.extend(
            sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
            if folder.is_dir()
            else []
        )
    if not images:
        print("usage: app.py --input-image PATH", file=sys.stderr)
        return 0
    for image in images:
        verdict = run_image(image, settings)
        print(
            json.dumps(
                {
                    "output": str(output_path(settings, image)),
                    "text": verdict.text,
                    "confidence": verdict.confidence,
                },
                ensure_ascii=False,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
