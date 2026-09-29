"""Harness entrypoint: this file is ``/app/app.py`` inside the submission image."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from tabula_ocr.contract.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
