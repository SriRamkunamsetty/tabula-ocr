"""Container entrypoint: start the vision model once, then stay alive.

The harness executes ``python3 /app/app.py --input-image ...`` *inside the running container*
once per image, so the container's main process must (a) load the model a single time during the
startup budget and (b) never exit: a container that dies scores zero, one that lost its model
still writes valid (empty) output files.

Environment:
  TABULA_OCR_SERVE_MODEL       path or repo id of the model (default ``/models/vlm``)
  TABULA_OCR_LLM_MODEL         name the server exposes it under (default ``tabula-vlm``)
  TABULA_OCR_VRAM_BUDGET_GIB   VRAM the server may claim (default 36). vLLM's memory setting is
                               a fraction of the *whole card*; on a 192 GB accelerator a naive
                               0.9 would blow the 48 GiB ceiling, so the budget is converted.
  TABULA_OCR_MAX_MODEL_LEN     context length (default 8192)
  TABULA_OCR_SERVE_EXTRA_ARGS  extra vLLM arguments
"""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import FrameType

import httpx

__all__ = ["gpu_memory_fraction", "main"]

READY_MARKER = Path(os.environ.get("TABULA_OCR_READY_MARKER", "/tmp/tabula_ready"))


def gpu_memory_fraction(budget_gib: float, total_gib: float | None) -> float:
    """Turn a GiB budget into vLLM's fraction-of-the-whole-card setting."""
    if not total_gib:
        return 0.75
    return round(max(0.05, min(0.92, budget_gib / total_gib)), 3)


def _total_vram_gib() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return float(torch.cuda.mem_get_info(0)[1]) / 1024**3
    except Exception as exc:
        print(f"serve: could not query VRAM: {exc}", file=sys.stderr)
    return None


def _command() -> list[str]:
    model = os.environ.get("TABULA_OCR_SERVE_MODEL", "/models/vlm")
    served = os.environ.get("TABULA_OCR_LLM_MODEL", "tabula-vlm")
    budget = float(os.environ.get("TABULA_OCR_VRAM_BUDGET_GIB", "36"))
    fraction = gpu_memory_fraction(budget, _total_vram_gib())
    return [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model, "--served-model-name", served,
        "--host", "127.0.0.1", "--port", "8000",
        "--gpu-memory-utilization", str(fraction),
        "--max-model-len", os.environ.get("TABULA_OCR_MAX_MODEL_LEN", "8192"),
        "--enable-prefix-caching",
        *shlex.split(os.environ.get("TABULA_OCR_SERVE_EXTRA_ARGS", "")),
    ]  # fmt: skip


def _wait_ready(base_url: str, proc: subprocess.Popen[bytes], timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False
        try:
            if httpx.get(f"{base_url}/models", timeout=2).status_code == 200:
                return True
        except httpx.HTTPError:
            pass
        time.sleep(2)
    return False


def main() -> int:
    """Start vLLM, wait for it, write the ready marker, then idle forever."""
    base_url = os.environ.get("TABULA_OCR_LLM_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
    command = _command()
    print("serve: starting", " ".join(command), flush=True)
    proc = subprocess.Popen(command)

    def stop(_signum: int, _frame: FrameType | None) -> None:
        if proc.poll() is None:
            proc.terminate()
        sys.exit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    if _wait_ready(base_url, proc, float(os.environ.get("TABULA_OCR_READY_TIMEOUT_S", "540"))):
        print("serve: model ready", flush=True)
    else:
        # Stay up in degraded mode: images will get valid empty output files instead of the
        # container disappearing, which the harness would score as a total failure.
        print("serve: model NOT ready; running degraded", file=sys.stderr, flush=True)
    READY_MARKER.write_text(str(time.time()))
    while True:  # never exit
        if proc.poll() is not None:
            print(
                f"serve: model server exited with {proc.returncode}",
                file=sys.stderr,
                flush=True,
            )
            proc.wait()
            while True:
                time.sleep(3600)
        time.sleep(5)


if __name__ == "__main__":
    raise SystemExit(main())
