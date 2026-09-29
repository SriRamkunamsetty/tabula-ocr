import contextlib
import io
from pathlib import Path

import pytest
from PIL import Image

from tabula_ocr.contract.views import ImageUnreadableError, fit, load_image, make_views, to_png


def _save(tmp_path: Path, name: str, image: Image.Image, **kwargs) -> Path:
    path = tmp_path / name
    image.save(path, **kwargs)
    return path


@pytest.mark.parametrize(
    ("name", "fmt"),
    [("a.png", "PNG"), ("a.jpg", "JPEG"), ("a.tif", "TIFF"), ("a.tiff", "TIFF")],
)
def test_all_three_promised_formats_decode(tmp_path, name, fmt):
    path = _save(tmp_path, name, Image.new("RGB", (120, 40), (200, 30, 30)), format=fmt)
    image = load_image(path)
    assert image.mode == "RGB"
    assert image.size == (120, 40)


@pytest.mark.parametrize("mode", ["L", "P", "RGBA", "LA", "1", "CMYK", "I;16"])
def test_unusual_modes_become_rgb(tmp_path, mode):
    source = Image.new(mode, (64, 32))
    path = tmp_path / "m.tif"
    source.save(path, format="TIFF")
    assert load_image(path).mode == "RGB"


def test_transparent_pixels_land_on_white_not_black(tmp_path):
    rgba = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
    path = _save(tmp_path, "t.png", rgba)
    assert load_image(path).getpixel((0, 0)) == (255, 255, 255)


def test_sixteen_bit_range_is_scaled_not_clipped(tmp_path):
    grey = Image.new("I;16", (4, 4), 1000)
    grey.putpixel((0, 0), 2000)
    path = _save(tmp_path, "g.tif", grey, format="TIFF")
    loaded = load_image(path)
    assert loaded.getpixel((0, 0)) == (255, 255, 255)
    assert loaded.getpixel((1, 1)) == (0, 0, 0)


def test_only_the_first_tiff_frame_is_read(tmp_path):
    first = Image.new("RGB", (30, 10), (255, 0, 0))
    second = Image.new("RGB", (30, 10), (0, 0, 255))
    path = tmp_path / "multi.tif"
    first.save(path, format="TIFF", save_all=True, append_images=[second])
    assert load_image(path).getpixel((0, 0)) == (255, 0, 0)


def test_exif_orientation_is_applied(tmp_path):
    source = Image.new("RGB", (60, 20), (10, 10, 10))
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 degrees to display
    path = _save(tmp_path, "r.jpg", source, format="JPEG", exif=exif)
    assert load_image(path).size == (20, 60)


@pytest.mark.parametrize(
    "payload", [b"", b"not an image at all", b"\x89PNG\r\n\x1a\ntruncated"]
)
def test_unreadable_input_raises_one_clear_error(payload):
    with pytest.raises(ImageUnreadableError):
        load_image(payload)


def test_missing_file_is_unreadable(tmp_path):
    with pytest.raises(ImageUnreadableError):
        load_image(tmp_path / "nope.png")


def test_truncated_jpeg_is_unreadable_or_decodes_without_crashing(tmp_path):
    buffer = io.BytesIO()
    Image.new("RGB", (200, 200), (5, 60, 90)).save(buffer, format="JPEG")
    path = tmp_path / "cut.jpg"
    path.write_bytes(buffer.getvalue()[: len(buffer.getvalue()) // 3])
    with contextlib.suppress(ImageUnreadableError):
        assert load_image(path).mode == "RGB"


@pytest.mark.parametrize(
    ("size", "expected_longest"),
    [((4000, 1000), 1600), ((100, 30), 640), ((900, 300), 900), ((30, 900), 900)],
)
def test_fit_bounds_the_long_edge_and_keeps_aspect(size, expected_longest):
    out = fit(Image.new("RGB", size), max_side=1600, min_side=640)
    assert max(out.size) == expected_longest
    assert abs(out.width / out.height - size[0] / size[1]) < 0.05


def test_views_are_ordered_original_first_and_heaviest():
    views = make_views(
        Image.new("RGB", (300, 100), (120, 120, 120)), max_side=1600, min_side=640
    )
    assert views[0].name == "original"
    assert views[0].weight == max(v.weight for v in views)
    assert len(views) == 4
    assert all(v.image.size == views[0].image.size for v in views)
    assert all(v.image.mode == "RGB" for v in views)


def test_view_limit_is_respected_and_at_least_one_view_exists():
    image = Image.new("RGB", (300, 100))
    assert len(make_views(image, max_side=1600, min_side=640, limit=2)) == 2
    assert len(make_views(image, max_side=1600, min_side=640, limit=0)) == 1


def test_png_encoding_round_trips_losslessly():
    source = Image.new("RGB", (50, 20), (1, 2, 3))
    assert Image.open(io.BytesIO(to_png(source))).convert("RGB").getpixel((3, 3)) == (1, 2, 3)
