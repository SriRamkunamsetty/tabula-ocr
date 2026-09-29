"""Contract mode: the exact shape the Mini-Challenge 2 grader runs and scores.

The grader executes ``python3 /app/app.py --input-image PATH`` once per image inside an
already-running container and compares ``text`` from ``<name>_output.json`` after a fixed
normalisation. Nothing else is scored, so this subpackage is deliberately small: prepare the
image, ask one vision model for the characters several ways, compose and vote deterministically.
"""

from tabula_ocr.contract.config import ContractSettings
from tabula_ocr.contract.runner import read_image, run_image

__all__ = ["ContractSettings", "read_image", "run_image"]
