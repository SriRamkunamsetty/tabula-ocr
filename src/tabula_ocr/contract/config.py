"""Contract-mode settings, read straight from ``TABULA_OCR_*`` environment variables.

Not ``pydantic-settings``: the harness starts a new process per image and this mode should
spend its 30 seconds on the image, not on a settings framework.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

__all__ = ["ContractSettings"]


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class ContractSettings:
    """All contract-mode knobs."""

    llm_base_url: str = "http://127.0.0.1:8000/v1"
    llm_model: str = ""  # empty: use whichever model the server reports first
    llm_api_key: str = "EMPTY"
    output_dir: Path = Path("/app/output")

    budget_s: float = 24.0  # the hard limit is 30 s per image, process start included
    view_timeout_s: float = 16.0
    model_wait_s: float = 8.0  # how long one image waits for a model that is still loading
    max_side: int = 1600  # longest edge sent to the model
    min_side: int = 640  # tiny crops are upscaled to at least this long edge
    max_views: int = 4

    @classmethod
    def from_env(cls) -> ContractSettings:
        """Build settings from the environment."""
        env = os.environ
        default = cls()
        return cls(
            llm_base_url=env.get("TABULA_OCR_LLM_BASE_URL", default.llm_base_url).rstrip("/"),
            llm_model=env.get("TABULA_OCR_LLM_MODEL", default.llm_model),
            llm_api_key=env.get("TABULA_OCR_LLM_API_KEY", default.llm_api_key),
            output_dir=Path(env.get("TABULA_OCR_OUTPUT_DIR", str(default.output_dir))),
            budget_s=_float("TABULA_OCR_BUDGET_S", default.budget_s),
            view_timeout_s=_float("TABULA_OCR_VIEW_TIMEOUT_S", default.view_timeout_s),
            model_wait_s=_float("TABULA_OCR_MODEL_WAIT_S", default.model_wait_s),
            max_side=_int("TABULA_OCR_MAX_SIDE", default.max_side),
            min_side=_int("TABULA_OCR_MIN_SIDE", default.min_side),
            max_views=max(1, _int("TABULA_OCR_MAX_VIEWS", default.max_views)),
        )
