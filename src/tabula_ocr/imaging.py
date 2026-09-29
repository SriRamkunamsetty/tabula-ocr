"""Page image handling.

Documents arrive as base64 payloads from arbitrary sources, so this module is a
trust boundary. It enforces a decompressed-pixel ceiling before anything is
rendered (a 6 KB PNG can expand to gigabytes, which is a denial-of-service
vector, not a hypothetical), converts to a consistent colour space, and caps the
long edge so that a 600-dpi scan does not blow the vision tower's token budget.

Crops are produced here too, so that the region-guided prompt can send a tight
image of a candidate region instead of a whole page.
"""

from __future__ import annotations

import base64
import binascii
import io
from dataclasses import dataclass

from PIL import Image, ImageOps, UnidentifiedImageError

from tabula_ocr.errors import InvalidDocumentError
from tabula_ocr.models import BoundingBox

__all__ = ["MAX_LONG_EDGE", "DecodedPage", "crop_region", "decode_page", "encode_image"]

MAX_LONG_EDGE = 2048
"""Longest edge, in pixels, sent to the model.

Above roughly 2k the marginal accuracy on document text is small while the
visual token count — and therefore both latency and the chance of the model
losing track of the page — keeps growing.
"""


@dataclass(frozen=True, slots=True)
class DecodedPage:
    """A validated, normalised page image."""

    image: Image.Image
    width: int
    height: int
    resized: bool

    def to_base64(self, fmt: str = "PNG") -> str:
        """Re-encode the normalised image for transport to the model."""
        return encode_image(self.image, fmt=fmt)


def decode_page(payload: str, *, max_pixels: int) -> DecodedPage:
    """Decode a base64 page image and normalise it for inference.

    Args:
        payload: Base64 string, optionally carrying a ``data:`` URL prefix.
        max_pixels: Hard ceiling on decompressed pixels.

    Returns:
        A :class:`DecodedPage` in RGB, EXIF-rotated, with its long edge capped.

    Raises:
        InvalidDocumentError: The payload was not decodable base64, was not a
            recognisable image, or exceeded ``max_pixels``.
    """
    raw = payload.split(",", 1)[1] if payload.startswith("data:") else payload
    try:
        blob = base64.b64decode(raw, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise InvalidDocumentError("page payload is not valid base64") from exc
    if not blob:
        raise InvalidDocumentError("page payload decoded to zero bytes")

    try:
        with Image.open(io.BytesIO(blob)) as probe:
            pixels = probe.width * probe.height
            if pixels > max_pixels:
                raise InvalidDocumentError(
                    "page image exceeds the decompressed pixel limit",
                    detail=f"{pixels} pixels > limit {max_pixels}",
                )
            probe.load()
            image = ImageOps.exif_transpose(probe).convert("RGB")
    except UnidentifiedImageError as exc:
        raise InvalidDocumentError("page payload is not a recognisable image") from exc
    except OSError as exc:  # truncated or corrupt file
        raise InvalidDocumentError("page image could not be read", detail=str(exc)) from exc

    resized = False
    long_edge = max(image.width, image.height)
    if long_edge > MAX_LONG_EDGE:
        scale = MAX_LONG_EDGE / long_edge
        image = image.resize(
            (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
            Image.Resampling.LANCZOS,
        )
        resized = True

    return DecodedPage(image=image, width=image.width, height=image.height, resized=resized)


def encode_image(image: Image.Image, *, fmt: str = "PNG") -> str:
    """Encode a PIL image as a base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def crop_region(page: DecodedPage, bbox: BoundingBox, *, padding: float = 0.01) -> Image.Image:
    """Crop a normalised region from a page, with a little padding.

    Padding matters: a box the model produced is often a character or two tight,
    and cropping exactly on it can shear the first glyph, which then reads as a
    different character on the verification pass.
    """
    x0 = max(0.0, bbox.x0 - padding)
    y0 = max(0.0, bbox.y0 - padding)
    x1 = min(1.0, bbox.x1 + padding)
    y1 = min(1.0, bbox.y1 + padding)
    box = BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1).to_pixels(page.width, page.height)
    return page.image.crop(box)
