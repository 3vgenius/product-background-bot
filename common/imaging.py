"""Pure image helpers.

This module deliberately has **no** dependency on rembg or Telegram, so the
whole compositing pipeline can be unit-tested without downloading a 200 MB+
ONNX model or holding a bot token.
"""

from __future__ import annotations

import io

from PIL import Image, ImageOps, UnidentifiedImageError

from . import config

Image.MAX_IMAGE_PIXELS = config.MAX_PIXELS


class ImageRejected(ValueError):
    """Raised for input the bot refuses to process. The message is user-facing."""


def load_image(raw: bytes, max_pixels: int | None = None) -> Image.Image:
    """Validate raw bytes and return an RGBA image, EXIF orientation applied.

    ``max_pixels`` overrides the global cap for callers that decode in a
    memory-constrained process (see ``normalize_background``).
    """
    limit = max_pixels or config.MAX_PIXELS
    if not raw:
        raise ImageRejected("Пустой файл.")
    if len(raw) > config.MAX_BYTES:
        raise ImageRejected("Файл слишком большой. Максимум — 19 МБ.")
    try:
        image = Image.open(io.BytesIO(raw))
        fmt = image.format
        width, height = image.size
        if fmt not in config.ALLOWED_FORMATS:
            raise ImageRejected(
                f"Формат {fmt or 'неизвестен'} не поддерживается. Нужны JPEG, PNG или WebP."
            )
        # Checked from the header, before decode: rejecting an oversized image
        # must not cost the memory needed to materialise it.
        if width * height > limit:
            raise ImageRejected(
                f"Изображение слишком большое: максимум {limit // 1_000_000} мегапикселей."
            )
        image = ImageOps.exif_transpose(image)
        image.load()
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as exc:
        raise ImageRejected(
            "Не удалось прочитать изображение. Отправьте JPEG, PNG или WebP."
        ) from exc
    return image.convert("RGBA")


def cover_square(image: Image.Image, size: int) -> Image.Image:
    """Scale+crop an image so it exactly fills a ``size`` x ``size`` square."""
    if size <= 0:
        raise ValueError("size must be positive")
    fitted = ImageOps.fit(
        image.convert("RGB"),
        (size, size),
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.5),
    )
    return fitted.convert("RGB")


def trim_transparent(image: Image.Image, threshold: int = 20) -> Image.Image | None:
    """Crop to the alpha bounding box, ignoring faint segmentation noise.

    Returns ``None`` when the segmentation mask is empty (nothing was found).
    """
    rgba = image.convert("RGBA")
    alpha = rgba.getchannel("A")
    mask = alpha.point(lambda p: 255 if p > threshold else 0)
    bbox = mask.getbbox()
    if bbox is None:
        return None
    return rgba.crop(bbox)


def fit_object(image: Image.Image, box: int) -> Image.Image:
    """Scale an image down (never up) so it fits inside ``box`` x ``box``."""
    if box <= 0:
        raise ValueError("box must be positive")
    result = image.convert("RGBA").copy()
    result.thumbnail((box, box), Image.Resampling.LANCZOS)
    return result


def composite(
    cutout: Image.Image,
    background: Image.Image,
    size: int = config.DEFAULT_SIZE,
    fill: float = config.DEFAULT_FILL,
) -> Image.Image:
    """Place ``cutout`` centred on ``background``, trimmed and scaled to ``fill``.

    ``cutout`` may carry a full-frame alpha channel; it is trimmed to the object
    first so that the framing depends on the product, not on the original shot.
    """
    if not 0 < fill <= 1:
        raise ValueError("fill must be in (0, 1]")
    trimmed = trim_transparent(cutout)
    if trimmed is None:
        raise ImageRejected(
            "Не удалось найти товар на фотографии. Попробуйте другой снимок."
        )
    obj = fit_object(trimmed, int(round(size * fill)))
    canvas = cover_square(background, size).convert("RGBA")
    x = (size - obj.width) // 2
    y = (size - obj.height) // 2
    canvas.alpha_composite(obj, (x, y))
    return canvas


def encode_jpeg(image: Image.Image, quality: int = config.JPEG_QUALITY) -> bytes:
    """Encode as JPEG. Alpha is flattened onto white, as marketplaces expect."""
    rgba = image.convert("RGBA")
    flat = Image.new("RGB", rgba.size, (255, 255, 255))
    flat.paste(rgba, mask=rgba.getchannel("A"))
    out = io.BytesIO()
    flat.save(out, format="JPEG", quality=quality, subsampling=0)
    return out.getvalue()


def encode_png(image: Image.Image) -> bytes:
    out = io.BytesIO()
    image.convert("RGBA").save(out, format="PNG", optimize=True)
    return out.getvalue()


def make_preview(raw_or_image: bytes | Image.Image, size: int = 800) -> bytes:
    """Small JPEG used for the /backgrounds gallery."""
    image = raw_or_image if isinstance(raw_or_image, Image.Image) else load_image(raw_or_image)
    preview = image.convert("RGB").copy()
    preview.thumbnail((size, size), Image.Resampling.LANCZOS)
    out = io.BytesIO()
    preview.save(out, format="JPEG", quality=85)
    return out.getvalue()


def normalize_background(raw: bytes, max_side: int = 3000) -> bytes:
    """Re-encode an uploaded background so we never store an oversized original.

    Uses the tighter background pixel cap: this runs inside the Render process,
    where the whole image is decoded before being thumbnailed down.
    """
    image = load_image(raw, max_pixels=config.MAX_BACKGROUND_PIXELS)
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    # JPEG, not PNG: backgrounds are photographs, and a 3000 px PNG weighs
    # 10-15 MB - slow to base64 through the Contents API on a 0.1-CPU dyno and
    # heavy to pull back for every gallery preview. Transparency is flattened
    # to white by encode_jpeg; a background has no use for alpha anyway.
    return encode_jpeg(image, quality=config.BACKGROUND_JPEG_QUALITY)
