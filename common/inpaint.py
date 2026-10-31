"""Remove the objects that already sit on a background picture.

Pipeline (runs on the GitHub Actions runner, never on Render):

1. ``object_mask``   - rembg in mask-only mode marks the salient objects.
2. ``grow_mask``     - the mask is dilated a little so shadows / halos go too.
3. ``inpaint``       - the masked area is filled from its surroundings.
   * ``lama``   : LaMa (ONNX, 512x512, downloaded once and cached) - good
                  on textures, walls, tables, fabrics.
   * ``opencv`` : ``cv2.inpaint`` (Telea) - always available, fine on flat and
                  gradient backgrounds, blurry on textures. Used when LaMa is
                  not available or fails, so the feature degrades instead of
                  breaking.

``clean_background`` ties them together and refuses clearly wrong inputs:
nothing to remove, or an "object" that covers most of the frame (that is not a
background with a vase on it, that is a product photo).

Everything heavy is imported lazily so the Render process and the unit tests
never pay for numpy/onnxruntime/cv2 unless they use this module.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from PIL import Image, ImageFilter

from . import config

log = logging.getLogger(__name__)


class NothingToClean(ValueError):
    """The background has no object worth removing (mask too small)."""


class RefusedToClean(ValueError):
    """The 'object' is most of the picture; cleaning would destroy it."""


# --- 1. mask ------------------------------------------------------------------


def object_mask(image: Image.Image, model: str) -> Image.Image:
    """Binary-ish ``L`` mask of the salient objects, same size as ``image``."""
    from rembg import new_session, remove

    started = time.monotonic()
    session = new_session(model)
    result = remove(image.convert("RGB"), session=session, only_mask=True)
    log.info("object mask in %.1fs", time.monotonic() - started)
    if not isinstance(result, Image.Image):
        import io

        result = Image.open(io.BytesIO(result))
    mask = result.convert("L")
    if mask.size != image.size:
        mask = mask.resize(image.size, Image.Resampling.BILINEAR)
    return mask


def grow_mask(mask: Image.Image, radius: int) -> Image.Image:
    """Dilate and binarise. Dilation takes shadows and soft edges along; a
    tight mask leaves a visible ghost of the removed object."""
    binary = mask.point(lambda v: 255 if v >= 128 else 0)
    if radius <= 0:
        return binary
    size = 2 * radius + 1
    return binary.filter(ImageFilter.MaxFilter(size))


def mask_coverage(mask: Image.Image) -> float:
    """Fraction of pixels that are masked, 0..1."""
    histogram = mask.histogram()
    total = sum(histogram)
    if not total:
        return 0.0
    return sum(histogram[128:]) / total


# --- 2. inpainting backends ---------------------------------------------------

LAMA_SIZE = 512


def lama_model_path() -> Path:
    home = os.getenv("LAMA_HOME") or os.path.join(
        os.getenv("REMBG_HOME") or os.path.join(os.path.expanduser("~"), ".rembg"), "models", "lama"
    )
    return Path(home) / "lama_fp32.onnx"


def _download(url: str, dest: Path) -> None:
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    log.info("downloading %s", url)
    with requests.get(url, stream=True, timeout=600) as response:
        response.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in response.iter_content(1 << 20):
                fh.write(chunk)
    tmp.replace(dest)


def lama_session():
    """Load (downloading once) the LaMa ONNX model. Raises on any failure;
    the caller decides whether to fall back."""
    import onnxruntime as ort

    path = lama_model_path()
    if not path.exists():
        _download(config.LAMA_ONNX_URL, path)
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def inpaint_lama(image: Image.Image, mask: Image.Image, session=None) -> Image.Image:
    """Inpaint with LaMa at 512x512, then paste the filled area back at full
    resolution. Only masked pixels change; everything else is the original."""
    import numpy as np

    session = session or lama_session()
    rgb = image.convert("RGB")
    small = rgb.resize((LAMA_SIZE, LAMA_SIZE), Image.Resampling.LANCZOS)
    small_mask = mask.resize((LAMA_SIZE, LAMA_SIZE), Image.Resampling.NEAREST)

    img_arr = np.asarray(small, dtype=np.float32) / 255.0  # HWC 0..1
    img_arr = np.transpose(img_arr, (2, 0, 1))[None]  # 1x3xHxW
    mask_arr = (np.asarray(small_mask, dtype=np.float32) >= 128).astype(np.float32)[None, None]

    inputs = session.get_inputs()
    if len(inputs) < 2:
        raise RuntimeError(f"unexpected LaMa inputs: {[i.name for i in inputs]}")
    # Order by shape, not by name: the image input has 3 channels, the mask 1.
    feed = {}
    for meta in inputs:
        shape = list(meta.shape)
        channels = shape[1] if len(shape) == 4 else None
        feed[meta.name] = mask_arr if channels == 1 else img_arr
    out = session.run(None, feed)[0]  # 1x3xHxW
    out = np.asarray(out, dtype=np.float32)[0]
    if out.max() <= 1.5:  # some exports emit 0..1, others 0..255
        out = out * 255.0
    out = np.clip(out, 0, 255).astype(np.uint8)
    filled_small = Image.fromarray(np.transpose(out, (1, 2, 0)), "RGB")

    filled = filled_small.resize(rgb.size, Image.Resampling.LANCZOS)
    soft_mask = mask.filter(ImageFilter.GaussianBlur(2))
    return Image.composite(filled, rgb, soft_mask)


def inpaint_opencv(image: Image.Image, mask: Image.Image, radius: int = 7) -> Image.Image:
    """Telea inpainting. Good on flat/gradient areas, smeary on textures."""
    import cv2
    import numpy as np

    rgb = np.asarray(image.convert("RGB"))
    m = (np.asarray(mask.convert("L")) >= 128).astype(np.uint8) * 255
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    filled = cv2.inpaint(bgr, m, radius, cv2.INPAINT_TELEA)
    return Image.fromarray(cv2.cvtColor(filled, cv2.COLOR_BGR2RGB), "RGB")


def inpaint_shiftmap(image: Image.Image, mask: Image.Image) -> Image.Image:
    """Patch-based (exemplar) inpainting from opencv-contrib: fills the hole by
    copying real patches of the same picture, so textures (grass, fabric,
    wood, concrete) stay sharp instead of turning into a smear. This is what
    "content-aware fill" does. Slower than Telea (~15 s per 2 MP with a
    half-frame hole) but it is the only CPU method here that does not leave
    a visible blot on a large hole."""
    import cv2
    import numpy as np

    if not hasattr(cv2, "xphoto"):
        raise RuntimeError("opencv-contrib (cv2.xphoto) is not installed")
    rgb = np.asarray(image.convert("RGB"))
    hole = np.asarray(mask.convert("L")) >= 128
    # xphoto's convention is inverted: non-zero = keep, zero = fill.
    keep = (~hole).astype(np.uint8) * 255
    bgr = np.ascontiguousarray(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    filled = np.zeros_like(bgr)
    cv2.xphoto.inpaint(bgr, keep, filled, cv2.xphoto.INPAINT_SHIFTMAP)
    out = cv2.cvtColor(filled, cv2.COLOR_BGR2RGB)
    out[~hole] = rgb[~hole]  # keep untouched pixels bit-identical
    return Image.fromarray(out, "RGB")


# Backend chain. "auto" tries the exemplar method first: it is proven here on
# large holes over texture, needs no download and no network. LaMa is opt-in
# (INPAINT_BACKEND=lama) until it has been seen working on the runner. Telea
# is the last resort so that the user always gets *something* plus an honest
# label in the caption.
_CHAIN = {
    "auto": ("shiftmap", "opencv"),
    "shiftmap": ("shiftmap",),
    "lama": ("lama",),
    "opencv": ("opencv",),
    "lama-auto": ("lama", "shiftmap", "opencv"),
}
_ENGINES = {"lama": inpaint_lama, "shiftmap": inpaint_shiftmap, "opencv": inpaint_opencv}


def inpaint(image: Image.Image, mask: Image.Image, backend: str = "auto") -> tuple[Image.Image, str]:
    """Return (filled image, backend actually used)."""
    chain = _CHAIN.get((backend or "auto").lower())
    if chain is None:
        raise ValueError(f"unknown INPAINT_BACKEND {backend!r}; use one of {sorted(_CHAIN)}")
    last_error: Exception | None = None
    for name in chain:
        try:
            started = time.monotonic()
            result = _ENGINES[name](image, mask)
            log.info("%s inpaint in %.1fs", name, time.monotonic() - started)
            return result, name
        except Exception as exc:  # noqa: BLE001 - degrade, do not break
            last_error = exc
            if name == chain[-1]:
                raise
            log.warning("%s unavailable (%s), falling back to %s", name, exc, chain[chain.index(name) + 1])
    raise RuntimeError(str(last_error))  # pragma: no cover - loop always returns or raises


# --- 3. the whole thing ------------------------------------------------------


def clean_background(
    image: Image.Image,
    model: str,
    backend: str = "auto",
    *,
    min_coverage: float = 0.004,
    max_coverage: float = 0.60,
    grow_radius: int | None = None,
) -> tuple[Image.Image, float, str]:
    """Remove the salient objects from ``image``.

    Returns (cleaned image, fraction of the frame that was filled, backend).
    Raises NothingToClean / RefusedToClean with user-facing Russian messages.
    """
    rgb = image.convert("RGB")
    raw_mask = object_mask(rgb, model)
    raw_coverage = mask_coverage(raw_mask)
    if raw_coverage < min_coverage:
        raise NothingToClean(
            "На фоне не нашлось объектов, которые стоило бы убрать — оставляю как есть."
        )
    if raw_coverage > max_coverage:
        raise RefusedToClean(
            f"Объект занимает {raw_coverage:.0%} кадра — это не фон с предметом, а сам предмет. "
            "Очистку не делаю, чтобы не испортить картинку."
        )
    if grow_radius is None:
        # ~1.2% of the long side: 36 px on a 3000 px picture, 4 px on a 300 px one.
        grow_radius = max(3, int(max(rgb.size) * 0.012))
    mask = grow_mask(raw_mask, grow_radius)
    cleaned, used = inpaint(rgb, mask, backend)
    return cleaned, mask_coverage(mask), used
