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


def test_inpaint_auto_falls_back_to_opencv_when_lama_fails(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no model")

    monkeypatch.setattr(inpaint, "inpaint_lama", boom)
    out, used = inpaint.inpaint(_with_object(), _box_mask(), backend="auto")
    assert used == "opencv"
    assert out.size == (W, H)


def test_inpaint_lama_backend_does_not_fall_back(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no model")

    monkeypatch.setattr(inpaint, "inpaint_lama", boom)
    with pytest.raises(RuntimeError):
        inpaint.inpaint(_with_object(), _box_mask(), backend="lama")


def test_inpaint_opencv_backend_never_touches_lama(monkeypatch):
    called = []
    monkeypatch.setattr(inpaint, "inpaint_lama", lambda *a, **k: called.append(1))
    _, used = inpaint.inpaint(_with_object(), _box_mask(), backend="opencv")
    assert used == "opencv" and called == []


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
    cleaned, filled, used = inpaint.clean_background(_with_object(), "isnet-general-use", "opencv")
    assert used == "opencv"
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
