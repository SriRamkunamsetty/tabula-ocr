"""Turn any PNG, JPEG or TIFF into a few model-ready views of the same picture.

The grader promises PNG, JPEG or TIFF "of any size and aspect ratio", so this module owns every
way an image can be awkward: multi-frame TIFFs, EXIF-rotated phone JPEGs, palette/16-bit/CMYK
modes, transparency, images far larger than the model's native resolution and plates so small
the model would see a smear. It never raises for a readable image and raises one clear error
(``ImageUnreadableError``) for one that is not.

Why several views: a vision-language model can misread a character on one rendering and get it
right on another (glare, low contrast, JPEG noise). Reading the *same* picture through
different, deterministic pre-processing and voting on the result removes many one-off misreads
without a second model. Each view is a cheap Pillow transform.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from PIL import Image, ImageFilter, ImageOps

__all__ = ["ImageUnreadableError", "View", "load_image", "make_views", "to_png"]

Image.MAX_IMAGE_PIXELS = 400_000_000  # large scans are legitimate; refuse only absurd ones


class ImageUnreadableError(Exception):
    """The file is missing, empty, truncated beyond repair, or not an image at all."""


@dataclass(frozen=True)
class View:
    """One rendering of the input image."""

    name: str
    image: Image.Image
    weight: float


def _scale_to_8bit(image: Image.Image) -> Image.Image:
    """Stretch the real value range of a 16-bit or float greyscale image onto 0-255."""
    extrema = cast("tuple[float, float]", image.getextrema())
    low, high = float(extrema[0]), float(extrema[1])
    span = max(1e-9, high - low)
    return image.point(lambda v: (v - low) * 255 / span).convert("L")


def _flatten(image: Image.Image) -> Image.Image:
    """Convert any mode to RGB, compositing transparency onto white."""
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        canvas = Image.new("RGB", rgba.size, (255, 255, 255))
        canvas.paste(rgba, mask=rgba.getchannel("A"))
        return canvas
    if image.mode in ("I;16", "I;16L", "I;16B", "I", "F"):
        return _scale_to_8bit(image).convert("RGB")
    return image.convert("RGB")


def load_image(source: Path | bytes) -> Image.Image:
    """Decode the first frame, honour EXIF orientation and return an RGB image."""
    try:
        data = source if isinstance(source, bytes) else source.read_bytes()
    except OSError as exc:
        raise ImageUnreadableError(f"cannot read file: {exc}") from exc
    if not data:
        raise ImageUnreadableError("file is empty")
    try:
        with Image.open(io.BytesIO(data)) as opened:
            opened.seek(0)  # multi-frame TIFF/GIF: the first frame is the picture
            opened.load()
            oriented = ImageOps.exif_transpose(opened) or opened
            return _flatten(oriented)
    except (OSError, ValueError, SyntaxError, EOFError, Image.DecompressionBombError) as exc:
        raise ImageUnreadableError(
            f"not a decodable image: {type(exc).__name__}: {exc}"
        ) from exc


def fit(image: Image.Image, *, max_side: int, min_side: int) -> Image.Image:
    """Scale so the longest edge lies in ``[min_side, max_side]`` (never distorting aspect)."""
    longest = max(image.size)
    if longest > max_side:
        scale = max_side / longest
    elif longest < min_side:
        scale = min_side / longest
    else:
        return image
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def _gamma(image: Image.Image, gamma: float) -> Image.Image:
    table = [round(255 * ((i / 255) ** gamma)) for i in range(256)]
    return image.point(table * len(image.getbands()))


def make_views(
    image: Image.Image, *, max_side: int, min_side: int, limit: int = 4
) -> list[View]:
    """Return up to ``limit`` renderings; the untouched one comes first and weighs most."""
    base = fit(image, max_side=max_side, min_side=min_side)
    grey = ImageOps.grayscale(base)
    stretched = ImageOps.autocontrast(base, cutoff=1)
    views = [
        View("original", base, 1.5),
        View(
            "contrast",
            _gamma(stretched, 0.85).filter(ImageFilter.UnsharpMask(radius=2, percent=120)),
            1.0,
        ),
        View(
            "denoised",
            base.filter(ImageFilter.MedianFilter(3)).filter(
                ImageFilter.UnsharpMask(radius=1.5, percent=160, threshold=2)
            ),
            1.0,
        ),
        View("equalized", ImageOps.equalize(grey).convert("RGB"), 1.0),
    ]
    return views[: max(1, limit)]


def to_png(image: Image.Image) -> bytes:
    """Encode losslessly: the model must see exactly the pixels the view produced."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
