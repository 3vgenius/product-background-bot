"""Object removal from backgrounds.

OpenCV runs for real. rembg and LaMa are stubbed: the mask comes from a fake
``rembg`` module, the LaMa path is exercised with a fake onnxruntime session
so that the pre/post-processing (layout, scaling, blend-back) is verified
without the 200 MB model.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest
from PIL import Image, ImageDraw

from common import inpaint

W, H = 600, 400


def _gradient(w=W, h=H):
    img = Image.new("RGB", (w, h))
    img.putdata([(int(60 + 120 * x / w), int(70 + 100 * y / h), 110) for y in range(h) for x in range(w)])
    return img


def _with_object(box=(380, 120, 520, 320)):
    img = _gradient()
    ImageDraw.Draw(img).rectangle(box, fill=(220, 30, 30))
    return img


def _box_mask(box=(380, 120, 520, 320), size=(W, H)):
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rectangle(box, fill=255)
    return mask


@pytest.fixture()
def fake_rembg_mask(monkeypatch):
    """rembg stub whose only_mask output is the red box (or whatever is set)."""
    state = {"mask": _box_mask(), "calls": []}

    def new_session(model):
        state["calls"].append(("session", model))
        return model

    def remove(image, session=None, only_mask=False, **kw):
        state["calls"].append(("remove", session, only_mask))
        assert only_mask is True
        return state["mask"]

    module = types.ModuleType("rembg")
    module.new_session = new_session
    module.remove = remove
    monkeypatch.setitem(sys.modules, "rembg", module)
    return state


# --- mask helpers ------------------------------------------------------------


def test_grow_mask_dilates_and_binarises():
    mask = Image.new("L", (50, 50), 0)
    ImageDraw.Draw(mask).rectangle((20, 20, 29, 29), fill=200)  # soft value, not 255
    grown = inpaint.grow_mask(mask, 3)
    assert grown.getpixel((25, 25)) == 255
    assert grown.getpixel((17, 25)) == 255  # 3 px outside the box
    assert grown.getpixel((15, 25)) == 0
    assert set(np.unique(np.asarray(grown))) <= {0, 255}


def test_grow_mask_zero_radius_only_binarises():
    mask = Image.new("L", (10, 10), 100)
    assert np.asarray(inpaint.grow_mask(mask, 0)).max() == 0
    mask = Image.new("L", (10, 10), 200)
    assert np.asarray(inpaint.grow_mask(mask, 0)).min() == 255


def test_mask_coverage():
    assert inpaint.mask_coverage(Image.new("L", (10, 10), 0)) == 0.0
    assert inpaint.mask_coverage(Image.new("L", (10, 10), 255)) == 1.0
    assert abs(inpaint.mask_coverage(_box_mask()) - (141 * 201) / (W * H)) < 1e-6


# --- OpenCV backend (real) ----------------------------------------------------


def test_opencv_inpaint_fills_a_box_on_a_gradient():
    clean = _gradient()
    filled = inpaint.inpaint_opencv(_with_object(), inpaint.grow_mask(_box_mask(), 6))
    a = np.asarray(filled).astype(int)
    b = np.asarray(clean).astype(int)
    inside = np.abs(a - b)[120:320, 380:520].mean()
    outside = np.abs(a - b)[:, :300].mean()
    assert inside < 12, inside  # visually indistinguishable on a smooth gradient
    assert outside == 0  # untouched pixels are bit-identical
    assert filled.size == clean.size and filled.mode == "RGB"


def _grass(w=W, h=H, seed=1):
    """Multi-scale noise texture: what Telea smears and ShiftMap must keep."""
    import cv2

    rng = np.random.default_rng(seed)

    def layer(scale):
        small = rng.random((h // scale + 1, w // scale + 1)).astype(np.float32)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)

    tex = 0.5 * layer(2) + 0.3 * layer(6) + 0.2 * layer(24)
    tex = (tex - tex.min()) / (tex.max() - tex.min())
    rgb = np.stack([30 + 60 * tex, 90 + 110 * tex, 25 + 40 * tex], -1).astype(np.uint8)
    return Image.fromarray(rgb, "RGB")


def _detail(image, mask):
    """Mean |Laplacian| inside vs outside the mask: a smear scores low inside."""
    import cv2

    grey = np.asarray(image.convert("L")).astype(np.float32)
    lap = np.abs(cv2.Laplacian(grey, cv2.CV_32F))
    m = np.asarray(mask) >= 128
    return lap[m].mean(), lap[~m].mean()


def test_shiftmap_keeps_texture_inside_a_large_hole():
    grass = _grass()
    mask = _box_mask((150, 60, 450, 340))  # 35 % of the frame
    damaged = grass.copy()
    damaged.paste((40, 40, 50), mask.getbbox())
    filled = inpaint.inpaint_shiftmap(damaged, mask)
    inside, outside = _detail(filled, mask)
    assert inside > 0.7 * outside, (inside, outside)  # texture, not a blot
    t_inside, _ = _detail(inpaint.inpaint_opencv(damaged, mask), mask)
    assert t_inside < 0.5 * outside  # and this is exactly why Telea is last
    assert filled.getpixel((20, 20)) == grass.getpixel((20, 20))  # untouched outside
    assert filled.size == grass.size and filled.mode == "RGB"


def test_auto_chain_is_shiftmap_then_opencv(monkeypatch):
    order = []
    monkeypatch.setattr(inpaint, "_ENGINES", {
        "shiftmap": lambda i, m: (order.append("shiftmap"), (_ for _ in ()).throw(RuntimeError("x")))[1],
        "opencv": lambda i, m: (order.append("opencv"), i)[1],
        "lama": lambda i, m: (order.append("lama"), i)[1],
    })
    _, used = inpaint.inpaint(_with_object(), _box_mask(), backend="auto")
    assert used == "opencv" and order == ["shiftmap", "opencv"]


def test_auto_never_touches_lama(monkeypatch):
    called = []
    monkeypatch.setattr(inpaint, "_ENGINES", {**inpaint._ENGINES, "lama": lambda i, m: called.append(1)})
    _, used = inpaint.inpaint(_with_object(), _box_mask(), backend="auto")
    assert used == "shiftmap" and called == []


def test_explicit_backend_does_not_fall_back(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no model")

    monkeypatch.setattr(inpaint, "_ENGINES", {**inpaint._ENGINES, "lama": boom})
    with pytest.raises(RuntimeError):
        inpaint.inpaint(_with_object(), _box_mask(), backend="lama")
    _, used = inpaint.inpaint(_with_object(), _box_mask(), backend="opencv")
    assert used == "opencv"


def test_lama_auto_chain_falls_through_to_shiftmap(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no model")

    monkeypatch.setattr(inpaint, "_ENGINES", {**inpaint._ENGINES, "lama": boom})
    _, used = inpaint.inpaint(_with_object(), _box_mask(), backend="lama-auto")
    assert used == "shiftmap"


def test_unknown_backend_is_an_error():
    with pytest.raises(ValueError, match="INPAINT_BACKEND"):
        inpaint.inpaint(_with_object(), _box_mask(), backend="magic")


# --- LaMa pre/post-processing with a fake session ----------------------------


class _Meta:
    def __init__(self, name, shape):
        self.name, self.shape = name, shape


class FakeLama:
    """Returns a flat grey 0..1 image: easy to recognise after blend-back."""

    def __init__(self, image_first=True, scale01=True):
        self.image_first = image_first
        self.scale01 = scale01
        self.feeds = []

    def get_inputs(self):
        img = _Meta("image", [1, 3, 512, 512])
        msk = _Meta("mask", [1, 1, 512, 512])
        return [img, msk] if self.image_first else [msk, img]

    def run(self, _, feed):
        self.feeds.append(feed)
        value = 0.5 if self.scale01 else 128.0
        return [np.full((1, 3, 512, 512), value, dtype=np.float32)]


@pytest.mark.parametrize("image_first", [True, False])
def test_lama_feed_is_matched_by_channel_count_not_name(image_first):
    session = FakeLama(image_first=image_first)
    inpaint.inpaint_lama(_with_object(), _box_mask(), session=session)
    feed = session.feeds[0]
    assert feed["image"].shape == (1, 3, 512, 512)
    assert feed["mask"].shape == (1, 1, 512, 512)
    assert feed["image"].dtype == np.float32 and feed["image"].max() <= 1.0
    assert set(np.unique(feed["mask"])) <= {0.0, 1.0}
    assert feed["mask"].mean() > 0  # the box made it into the mask


@pytest.mark.parametrize("scale01", [True, False])
def test_lama_output_is_blended_back_only_inside_the_mask(scale01):
    original = _with_object()
    out = inpaint.inpaint_lama(original, _box_mask(), session=FakeLama(scale01=scale01))
    assert out.size == original.size
    # Inside the box: the model's grey (either output scale handled).
    assert all(abs(c - 128) <= 2 for c in out.getpixel((450, 220)))
    # Far from the box: the original, untouched.
    assert out.getpixel((50, 50)) == original.getpixel((50, 50))


def test_lama_rejects_sessions_with_unexpected_inputs():
    class Odd(FakeLama):
        def get_inputs(self):
            return [_Meta("x", [1, 3, 512, 512])]

    with pytest.raises(RuntimeError):
        inpaint.inpaint_lama(_with_object(), _box_mask(), session=Odd())


def test_lama_model_path_honours_env(monkeypatch, tmp_path):
    monkeypatch.setenv("LAMA_HOME", str(tmp_path / "lama"))
    assert inpaint.lama_model_path() == tmp_path / "lama" / "lama_fp32.onnx"
    monkeypatch.delenv("LAMA_HOME")
    monkeypatch.setenv("REMBG_HOME", str(tmp_path / "r"))
    assert inpaint.lama_model_path() == tmp_path / "r" / "models" / "lama" / "lama_fp32.onnx"


# --- clean_background ---------------------------------------------------------


def test_clean_background_removes_the_object(fake_rembg_mask):
    cleaned, filled, used = inpaint.clean_background(_with_object(), "isnet-general-use", "auto")
    assert used == "shiftmap"
    assert 0.11 < filled < 0.20  # box + dilation
    assert fake_rembg_mask["calls"][0] == ("session", "isnet-general-use")
    # The red is gone.
    r, g, b = cleaned.getpixel((450, 220))
    assert r < 190 and abs(r - g) < 60


def test_clean_background_nothing_to_do(fake_rembg_mask):
    fake_rembg_mask["mask"] = Image.new("L", (W, H), 0)
    with pytest.raises(inpaint.NothingToClean, match="не нашлось объектов"):
        inpaint.clean_background(_gradient(), "m", "opencv")


def test_clean_background_refuses_when_object_is_the_picture(fake_rembg_mask):
    fake_rembg_mask["mask"] = _box_mask((0, 0, W - 1, int(H * 0.8)))  # 80 % of the frame
    with pytest.raises(inpaint.RefusedToClean, match="кадра"):
        inpaint.clean_background(_with_object(), "m", "opencv")


def test_clean_background_resizes_a_mismatched_mask(fake_rembg_mask):
    fake_rembg_mask["mask"] = _box_mask((190, 60, 260, 160), size=(300, 200))  # half-size mask
    cleaned, filled, _ = inpaint.clean_background(_with_object(), "m", "opencv")
    assert cleaned.size == (W, H)
    assert filled > 0.1
