"""Tests for the compositing pipeline.

These exercise the real functions in ``common.imaging`` — the same code the
GitHub Actions worker calls — using synthetic images, so they need neither the
ONNX model nor a Telegram token.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image

from common import config, imaging

BLUE = (0, 0, 255)
RED = (255, 0, 0)


def _blob(image: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return buf.getvalue()


def _cutout_with_blob(size=(200, 200), blob=(80, 60), offset=None) -> Image.Image:
    """A fully transparent frame with an opaque rectangle in the middle."""
    canvas = Image.new("RGBA", size, (0, 0, 0, 0))
    x0 = offset[0] if offset else (size[0] - blob[0]) // 2
    y0 = offset[1] if offset else (size[1] - blob[1]) // 2
    box = (x0, y0, x0 + blob[0], y0 + blob[1])
    Image.Image.paste(canvas, Image.new("RGBA", blob, RED + (255,)), box)
    return canvas


# --- load_image -------------------------------------------------------------


@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "WEBP"])
def test_load_image_accepts_supported_formats(fmt):
    image = Image.new("RGB", (64, 48), BLUE)
    loaded = imaging.load_image(_blob(image, fmt))
    assert loaded.mode == "RGBA"
    assert loaded.size == (64, 48)


@pytest.mark.parametrize("fmt", ["GIF", "BMP", "TIFF"])
def test_load_image_rejects_other_formats(fmt):
    blob = _blob(Image.new("RGB", (32, 32), BLUE), fmt)
    with pytest.raises(imaging.ImageRejected):
        imaging.load_image(blob)


def test_load_image_rejects_garbage():
    with pytest.raises(imaging.ImageRejected):
        imaging.load_image(b"not an image at all")


def test_load_image_rejects_empty():
    with pytest.raises(imaging.ImageRejected):
        imaging.load_image(b"")


def test_load_image_enforces_byte_limit(monkeypatch):
    monkeypatch.setattr(config, "MAX_BYTES", 10)
    blob = _blob(Image.new("RGB", (64, 48), BLUE))
    with pytest.raises(imaging.ImageRejected, match="19 МБ|слишком большой"):
        imaging.load_image(blob)


def test_load_image_enforces_pixel_limit(monkeypatch):
    monkeypatch.setattr(config, "MAX_PIXELS", 100)
    blob = _blob(Image.new("RGB", (64, 48), BLUE))
    with pytest.raises(imaging.ImageRejected, match="мегапиксел"):
        imaging.load_image(blob)


def test_load_image_applies_exif_orientation():
    """A photo shot in portrait but stored landscape must be rotated for us."""
    exif = Image.Exif()
    exif[274] = 6  # Orientation: rotate 90 CW
    buf = io.BytesIO()
    Image.new("RGB", (100, 50), BLUE).save(buf, format="JPEG", exif=exif)
    assert imaging.load_image(buf.getvalue()).size == (50, 100)


# --- cover_square -----------------------------------------------------------


@pytest.mark.parametrize("source", [(640, 480), (480, 640), (500, 500), (2000, 300)])
def test_cover_square_produces_exact_square(source):
    out = imaging.cover_square(Image.new("RGB", source, BLUE), 256)
    assert out.size == (256, 256)
    assert out.mode == "RGB"


def test_cover_square_fills_every_pixel():
    out = imaging.cover_square(Image.new("RGB", (640, 480), BLUE), 128)
    assert out.getpixel((0, 0)) == BLUE
    assert out.getpixel((127, 127)) == BLUE


def test_cover_square_rejects_non_positive_size():
    with pytest.raises(ValueError):
        imaging.cover_square(Image.new("RGB", (10, 10), BLUE), 0)


# --- trim_transparent -------------------------------------------------------


def test_trim_transparent_crops_to_the_object():
    cutout = _cutout_with_blob(size=(200, 200), blob=(80, 60), offset=(50, 70))
    trimmed = imaging.trim_transparent(cutout)
    assert trimmed is not None
    assert trimmed.size == (80, 60)
    assert trimmed.getpixel((0, 0))[3] == 255


def test_trim_transparent_ignores_faint_noise():
    cutout = _cutout_with_blob(size=(120, 120), blob=(40, 40), offset=(40, 40))
    # A single 5/255 alpha speck far from the object must not widen the bbox.
    cutout.putpixel((2, 2), (0, 0, 0, 5))
    assert imaging.trim_transparent(cutout, threshold=20).size == (40, 40)


def test_trim_transparent_returns_none_for_empty_mask():
    assert imaging.trim_transparent(Image.new("RGBA", (50, 50), (0, 0, 0, 0))) is None


# --- fit_object -------------------------------------------------------------


def test_fit_object_never_upscales():
    small = Image.new("RGBA", (100, 50), RED + (255,))
    assert imaging.fit_object(small, 1000).size == (100, 50)


def test_fit_object_downscales_keeping_ratio():
    big = Image.new("RGBA", (1000, 500), RED + (255,))
    out = imaging.fit_object(big, 200)
    assert out.size == (200, 100)


def test_fit_object_rejects_non_positive_box():
    with pytest.raises(ValueError):
        imaging.fit_object(Image.new("RGBA", (10, 10)), 0)


# --- composite --------------------------------------------------------------


def test_composite_puts_object_on_background():
    cutout = _cutout_with_blob(size=(200, 200), blob=(80, 60))
    background = Image.new("RGB", (100, 100), BLUE)
    out = imaging.composite(cutout, background, size=256)

    assert out.size == (256, 256)
    # Corners stay pure background.
    assert out.getpixel((0, 0))[:3] == BLUE
    assert out.getpixel((255, 255))[:3] == BLUE
    # Centre is the product: an 80x60 blob centred at ((256-80)//2, (256-60)//2).
    assert out.getpixel((128, 128))[:3] == RED


def test_composite_output_is_opaque():
    cutout = _cutout_with_blob()
    out = imaging.composite(cutout, Image.new("RGB", (64, 64), BLUE), size=128)
    assert out.getchannel("A").getextrema() == (255, 255)


def test_composite_scales_large_object_to_fill():
    # A 1600x1200 object into a 512 canvas at fill=0.8 must end up <= 410 px wide.
    cutout = _cutout_with_blob(size=(1600, 1200), blob=(1600, 1200), offset=(0, 0))
    out = imaging.composite(cutout, Image.new("RGB", (64, 64), BLUE), size=512, fill=0.8)
    assert out.size == (512, 512)
    # The widest red run through the middle row cannot exceed 410 px.
    row = [out.getpixel((x, 256))[:3] for x in range(512)]
    assert sum(1 for px in row if px == RED) <= 410


def test_composite_raises_when_nothing_was_cut_out():
    empty = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    with pytest.raises(imaging.ImageRejected):
        imaging.composite(empty, Image.new("RGB", (64, 64), BLUE), size=128)


@pytest.mark.parametrize("fill", [0, -0.5, 1.5])
def test_composite_rejects_bad_fill(fill):
    with pytest.raises(ValueError):
        imaging.composite(
            _cutout_with_blob(), Image.new("RGB", (64, 64), BLUE), size=128, fill=fill
        )


# --- encoders ---------------------------------------------------------------


def test_encode_jpeg_flattens_alpha_onto_white():
    half = Image.new("RGBA", (40, 40), (0, 0, 0, 0))
    blob = imaging.encode_jpeg(half)
    assert blob[:3] == b"\xff\xd8\xff"
    decoded = Image.open(io.BytesIO(blob))
    assert decoded.mode == "RGB"
    # JPEG is lossy, so allow a small deviation from pure white.
    r, g, b = decoded.getpixel((20, 20))
    assert min(r, g, b) > 240


def test_encode_png_keeps_alpha():
    decoded = Image.open(io.BytesIO(imaging.encode_png(_cutout_with_blob((40, 40), (20, 20)))))
    assert decoded.mode == "RGBA"


def test_make_preview_caps_dimensions():
    blob = imaging.make_preview(Image.new("RGB", (4000, 3000), BLUE), 640)
    preview = Image.open(io.BytesIO(blob))
    assert preview.format == "JPEG"
    assert max(preview.size) <= 640


def test_make_preview_accepts_raw_bytes():
    blob = imaging.make_preview(_blob(Image.new("RGB", (900, 900), BLUE), "JPEG"), 300)
    assert max(Image.open(io.BytesIO(blob)).size) <= 300


def test_normalize_background_downscales():
    blob = imaging.normalize_background(_blob(Image.new("RGB", (5000, 4000), BLUE)), max_side=800)
    out = Image.open(io.BytesIO(blob))
    assert out.format == "PNG"
    assert max(out.size) <= 800
