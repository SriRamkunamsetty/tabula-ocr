"""Runtime configuration.

All configuration is read from the environment exactly once, at process start,
and is immutable thereafter. Nothing in the codebase reads ``os.environ``
directly; anything that needs a knob takes a :class:`Settings` instance, which
keeps tests hermetic and makes the full configuration surface greppable in one
file.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["Settings", "get_settings"]


class Settings(BaseSettings):
    """Service configuration, sourced from ``TABULA_*`` environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="TABULA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    # --- Service identity -------------------------------------------------
    service_name: str = "tabula-ocr"
    environment: Literal["dev", "staging", "prod", "test"] = "dev"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["json", "console"] = "json"

    # --- Upstream vision model (vLLM on ROCm, OpenAI-compatible) ----------
    vlm_base_url: str = Field(
        default="http://localhost:8000/v1",
        description="OpenAI-compatible endpoint served by vLLM on the MI300X.",
    )
    vlm_model: str = Field(
        default="PaddlePaddle/PaddleOCR-VL",
        description="Model id as registered with the serving engine.",
    )
    vlm_api_key: str = Field(default="EMPTY", repr=False)
    vlm_timeout_s: float = Field(default=90.0, gt=0)
    vlm_connect_timeout_s: float = Field(default=5.0, gt=0)
    vlm_max_retries: int = Field(default=3, ge=0, le=10)
    vlm_max_concurrency: int = Field(default=8, ge=1, le=256)

    # --- Circuit breaker --------------------------------------------------
    breaker_failure_threshold: int = Field(default=5, ge=1)
    breaker_reset_timeout_s: float = Field(default=30.0, gt=0)

    # --- Extraction behaviour --------------------------------------------
    passes: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Independent decoding passes used to form a consensus.",
    )
    pass_temperatures: tuple[float, ...] = Field(default=(0.0, 0.35))
    render_dpi: int = Field(default=200, ge=72, le=600)
    max_image_pixels: int = Field(default=40_000_000, ge=1_000_000)

    # --- Confidence and abstention ---------------------------------------
    weight_agreement: float = Field(default=0.5, ge=0, le=1)
    weight_grounding: float = Field(default=0.35, ge=0, le=1)
    weight_format: float = Field(default=0.15, ge=0, le=1)
    abstain_below: float = Field(
        default=0.62,
        ge=0.0,
        le=1.0,
        description="Fields scoring below this threshold are abstained on.",
    )
    grounding_min_ratio: float = Field(default=0.86, ge=0.0, le=1.0)

    # --- Cost accounting --------------------------------------------------
    gpu_hourly_rate_usd: float = Field(
        default=1.99, ge=0, description="AMD Developer Cloud 1x MI300X list rate."
    )
    measured_tokens_per_s: float = Field(default=1400.0, gt=0)

    # --- Paths ------------------------------------------------------------
    schema_dir: Path = Field(default=Path("schemas"))

    @field_validator("vlm_base_url")
    @classmethod
    def _strip_trailing_slash(cls, value: str) -> str:
        """Normalise the base URL so path joins never double up."""
        return value.rstrip("/")

    @field_validator("pass_temperatures")
    @classmethod
    def _temperatures_in_range(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        """Reject sampling temperatures the serving engine will not accept."""
        if not value:
            raise ValueError("at least one pass temperature is required")
        if any(t < 0.0 or t > 2.0 for t in value):
            raise ValueError("temperatures must be within [0.0, 2.0]")
        return value

    def temperature_for(self, pass_index: int) -> float:
        """Temperature for a given pass, cycling if fewer are configured."""
        return self.pass_temperatures[pass_index % len(self.pass_temperatures)]

    @property
    def confidence_weights(self) -> tuple[float, float, float]:
        """Confidence weights normalised to sum to 1.0."""
        total = self.weight_agreement + self.weight_grounding + self.weight_format
        if total <= 0:
            return (1.0, 0.0, 0.0)
        return (
            self.weight_agreement / total,
            self.weight_grounding / total,
            self.weight_format / total,
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
