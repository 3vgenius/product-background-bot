"""Tests for the GitHub Actions worker.

rembg is replaced with a stub module, so the real ``worker.main()`` runs end to
-end - argument parsing, photo download, segmentation call, background loading,
compositing, JPEG encoding, caption and error reporting - without a 200 MB+
model download.
"""

from __future__ import annotations

import io
import sys
import types

import pytest
from PIL import Image

import worker
from common import telegram_api as tg

BLUE = (0, 0, 255)
RED = (255, 0, 0)


def _jpeg(color, size=(400, 300)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG")
    return buf.getvalue()


def _cutout(size=(400, 300), blob=(120, 90)) -> Image.Image:
    canvas = Image.new("RGBA", size, (0, 0, 0, 0))
    x0, y0 = (size[0] - blob[0]) // 2, (size[1] - blob[1]) // 2
    canvas.paste(Image.new("RGBA", blob, RED + (255,)), (x0, y0))
    return canvas


@pytest.fixture()
def fake_rembg(monkeypatch):
    """Install a stub `rembg` whose remove() returns a known cutout."""
    calls = []

    def new_session(model):
        calls.append(("session", model))
        return f"session:{model}"

    def remove(image, session=None):
        calls.append(("remove", session))
        return _cutout(image.size)

    module = types.ModuleType("rembg")
    module.new_session = new_session
    module.remove = remove
    monkeypatch.setitem(sys.modules, "rembg", module)
    return calls


@pytest.fixture()
def stub_telegram(monkeypatch):
    sent = []
    monkeypatch.setattr(tg, "download_by_file_id", lambda file_id: _jpeg(BLUE))
    monkeypatch.setattr(
        tg,
        "send_document",
        lambda chat_id, blob, filename, caption=None, **kw: sent.append(
            {"chat_id": chat_id, "blob": blob, "filename": filename, "caption": caption}
        ),
    )
    monkeypatch.setattr(
        tg, "send_message", lambda chat_id, text, **kw: sent.append({"chat_id": chat_id, "text": text})
    )
    return sent


@pytest.fixture()
def background_file(tmp_path):
    path = tmp_path / "backgrounds" / "1.png"
    path.parent.mkdir(parents=True)
    Image.new("RGB", (600, 400), BLUE).save(path, format="PNG")
    return "backgrounds/1.png"


def _argv(tmp_path, background, **kw):
    argv = [
        "--chat-id", "555",
        "--user-id", "111",
        "--file-id", "PRODUCT_FILE",
        "--model", "isnet-general-use",
        "--size", "1024",
        "--label", "Мрамор",
    ]
    if background is not None:
        argv += ["--background", background]
    for key, value in kw.items():
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            # store_true flags take no argument; passing "True" is an argparse error.
            if value:
                argv.append(flag)
        else:
            argv += [flag, str(value)]
    return argv


def _close(actual: tuple, expected: tuple, tolerance: int = 8) -> bool:
    """Compare pixels with a tolerance: the output is JPEG, which is lossy."""
    return all(abs(a - e) <= tolerance for a, e in zip(actual[:3], expected))


# --- happy path -------------------------------------------------------------


def test_worker_produces_jpeg_and_sends_it(
    tmp_path, monkeypatch, fake_rembg, stub_telegram, background_file
):
    monkeypatch.chdir(tmp_path)
    code = worker.main(_argv(tmp_path, background_file))

    assert code == 0
    assert fake_rembg == [("session", "isnet-general-use"), ("remove", "session:isnet-general-use")]

    sent = [item for item in stub_telegram if "blob" in item]
    assert len(sent) == 1
    assert sent[0]["chat_id"] == "555"
    assert sent[0]["filename"] == "product.jpg"

    image = Image.open(io.BytesIO(sent[0]["blob"]))
    assert image.format == "JPEG"
    assert image.size == (1024, 1024)
    # Corners are the background, the middle is the product. Compared with a
    # tolerance because the result is JPEG, not a lossless format.
    assert _close(image.getpixel((0, 0)), BLUE), image.getpixel((0, 0))
    assert _close(image.getpixel((512, 512)), RED), image.getpixel((512, 512))

    caption = sent[0]["caption"]
    assert "Мрамор" in caption
    assert "1024" in caption
    assert "isnet-general-use" in caption


def test_worker_honours_custom_size(
    tmp_path, monkeypatch, fake_rembg, stub_telegram, background_file
):
    monkeypatch.chdir(tmp_path)
    assert worker.main(_argv(tmp_path, background_file, size=1600)) == 0
    blob = next(i["blob"] for i in stub_telegram if "blob" in i)
    assert Image.open(io.BytesIO(blob)).size == (1600, 1600)


def test_worker_dry_run_writes_file_instead_of_sending(
    tmp_path, monkeypatch, fake_rembg, stub_telegram, background_file
):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "result.jpg"
    code = worker.main(_argv(tmp_path, background_file, dry_run=True, out=str(out)))

    assert code == 0
    assert out.exists()
    assert Image.open(out).size == (1024, 1024)
    assert not [i for i in stub_telegram if "blob" in i]


def test_worker_falls_back_to_store_when_file_is_absent(
    tmp_path, monkeypatch, fake_rembg, stub_telegram
):
    """Outside a checkout the background comes from the Contents API."""
    from common import store

    monkeypatch.chdir(tmp_path)
    local = store.get_store()
    local.write_bytes("backgrounds/9.png", _jpeg((0, 128, 0), (300, 300)))

    assert worker.main(_argv(tmp_path, "backgrounds/9.png")) == 0
    blob = next(i["blob"] for i in stub_telegram if "blob" in i)
    assert _close(Image.open(io.BytesIO(blob)).getpixel((0, 0)), (0, 128, 0))


# --- failure paths ----------------------------------------------------------


def test_worker_reports_missing_background(tmp_path, monkeypatch, fake_rembg, stub_telegram):
    monkeypatch.chdir(tmp_path)
    code = worker.main(_argv(tmp_path, "backgrounds/missing.png"))
    assert code == 1
    texts = [i["text"] for i in stub_telegram if "text" in i]
    assert any("Не получилось" in t for t in texts)
    assert not [i for i in stub_telegram if "blob" in i]


def test_worker_reports_no_background_given(tmp_path, monkeypatch, fake_rembg, stub_telegram):
    monkeypatch.chdir(tmp_path)
    assert worker.main(_argv(tmp_path, None)) == 1
    assert any("Не получилось" in i["text"] for i in stub_telegram if "text" in i)


def test_worker_reports_empty_segmentation(
    tmp_path, monkeypatch, stub_telegram, background_file
):
    """If the model found nothing, the user must be told, not get a blank square."""
    empty = types.ModuleType("rembg")
    empty.new_session = lambda model: "s"
    empty.remove = lambda image, session=None: Image.new("RGBA", image.size, (0, 0, 0, 0))
    monkeypatch.setitem(sys.modules, "rembg", empty)
    monkeypatch.chdir(tmp_path)

    assert worker.main(_argv(tmp_path, background_file)) == 1
    texts = " ".join(i["text"] for i in stub_telegram if "text" in i)
    assert "Не получилось" in texts


def test_worker_reports_download_failure(
    tmp_path, monkeypatch, fake_rembg, stub_telegram, background_file
):
    def boom(file_id):
        raise tg.TelegramError("getFile", 400, "wrong file identifier")

    monkeypatch.setattr(tg, "download_by_file_id", boom)
    monkeypatch.chdir(tmp_path)

    assert worker.main(_argv(tmp_path, background_file)) == 1
    assert any("Не получилось" in i["text"] for i in stub_telegram if "text" in i)


def test_worker_reports_unreadable_photo(
    tmp_path, monkeypatch, fake_rembg, stub_telegram, background_file
):
    monkeypatch.setattr(tg, "download_by_file_id", lambda file_id: b"definitely not an image")
    monkeypatch.chdir(tmp_path)

    assert worker.main(_argv(tmp_path, background_file)) == 1
    assert any("Не получилось" in i["text"] for i in stub_telegram if "text" in i)


def test_worker_reports_when_result_cannot_be_sent(
    tmp_path, monkeypatch, fake_rembg, stub_telegram, background_file
):
    def boom(*args, **kwargs):
        raise tg.TelegramError("sendDocument", 400, "chat not found")

    monkeypatch.setattr(tg, "send_document", boom)
    monkeypatch.chdir(tmp_path)

    assert worker.main(_argv(tmp_path, background_file)) == 1
    assert any("не смог отправить" in i["text"] for i in stub_telegram if "text" in i)


def test_worker_survives_unexpected_error(
    tmp_path, monkeypatch, stub_telegram, background_file
):
    broken = types.ModuleType("rembg")
    broken.new_session = lambda model: (_ for _ in ()).throw(RuntimeError("onnx blew up"))
    broken.remove = lambda image, session=None: image
    monkeypatch.setitem(sys.modules, "rembg", broken)
    monkeypatch.chdir(tmp_path)

    assert worker.main(_argv(tmp_path, background_file)) == 1
    assert any("внутренняя ошибка" in i["text"] for i in stub_telegram if "text" in i)


# --- argument parsing -------------------------------------------------------


def test_parse_args_defaults():
    args = worker.parse_args(["--chat-id", "1", "--file-id", "f"])
    assert args.background == ""
    assert args.oneoff_file_id == ""
    assert args.model == ""
    assert args.size == 0
    assert args.dry_run is False


def test_parse_args_requires_chat_and_file():
    with pytest.raises(SystemExit):
        worker.parse_args(["--chat-id", "1"])


# --- one-off bookkeeping after delivery ------------------------------------


@pytest.fixture()
def settings_store(tmp_path, monkeypatch):
    """worker._forget_oneoff builds a GithubStore(prefer_local=False); swap in
    a LocalStore rooted at tmp_path so the read-modify-write is real."""
    from common import config, store

    monkeypatch.setattr(config, "GITHUB_TOKEN", "T")
    local = store.LocalStore(tmp_path)
    monkeypatch.setattr(store, "GithubStore", lambda **kw: local)
    return local


def test_worker_clears_oneoff_after_successful_delivery(
    tmp_path, monkeypatch, fake_rembg, stub_telegram, settings_store
):
    from common import store

    store.update_user(111, {"oneoff": "ONEOFF_BG", "model": "isnet-general-use"}, settings_store)
    assert worker.main(_argv(tmp_path, None, oneoff_file_id="ONEOFF_BG")) == 0
    assert stub_telegram[-1]["filename"] == "product.jpg"
    assert store.user_settings(111, store.load_settings(settings_store))["oneoff"] is None


def test_worker_keeps_a_newer_oneoff_set_meanwhile(
    tmp_path, monkeypatch, fake_rembg, stub_telegram, settings_store
):
    from common import store

    # While this job ran, the user already registered a different one-off.
    store.update_user(111, {"oneoff": "NEWER_BG"}, settings_store)
    assert worker.main(_argv(tmp_path, None, oneoff_file_id="ONEOFF_BG")) == 0
    assert store.user_settings(111, store.load_settings(settings_store))["oneoff"] == "NEWER_BG"


def test_worker_does_not_clear_oneoff_when_delivery_failed(
    tmp_path, monkeypatch, fake_rembg, stub_telegram, settings_store
):
    from common import store

    def boom(*a, **k):
        raise tg.TelegramError("sendDocument", 500, "down")

    monkeypatch.setattr(tg, "send_document", boom)
    store.update_user(111, {"oneoff": "ONEOFF_BG"}, settings_store)
    assert worker.main(_argv(tmp_path, None, oneoff_file_id="ONEOFF_BG")) == 1
    assert store.user_settings(111, store.load_settings(settings_store))["oneoff"] == "ONEOFF_BG"


def test_worker_bookkeeping_failure_does_not_fail_the_job(
    tmp_path, monkeypatch, fake_rembg, stub_telegram
):
    from common import config, store

    monkeypatch.setattr(config, "GITHUB_TOKEN", "T")

    def boom(**kw):
        raise store.StoreError("PUT settings.json: HTTP 500")

    monkeypatch.setattr(store, "GithubStore", boom)
    # The result was delivered; a failed clean-up must not turn that into a red run.
    assert worker.main(_argv(tmp_path, None, oneoff_file_id="ONEOFF_BG")) == 0
    assert stub_telegram[-1]["filename"] == "product.jpg"


def test_worker_skips_bookkeeping_without_token(tmp_path, monkeypatch, fake_rembg, stub_telegram):
    from common import config, store

    monkeypatch.setattr(config, "GITHUB_TOKEN", "")
    calls = []
    monkeypatch.setattr(store, "GithubStore", lambda **kw: calls.append(kw))
    assert worker.main(_argv(tmp_path, None, oneoff_file_id="ONEOFF_BG")) == 0
    assert calls == []


# --- task: clean ------------------------------------------------------------


def _clean_argv(**kw):
    argv = [
        "--task", "clean",
        "--chat-id", "555",
        "--user-id", "111",
        "--background", "backgrounds/7_marble.jpg",
        "--background-id", "7",
        "--model", "isnet-general-use",
        "--label", "Мрамор",
    ]
    for key, value in kw.items():
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                argv.append(flag)
        else:
            argv += [flag, str(value)]
    return argv


def _busy_background(size=(600, 400)) -> Image.Image:
    img = Image.new("RGB", size, (200, 190, 180))
    img.paste(Image.new("RGB", (120, 160), (20, 20, 20)), (380, 120))  # a "vase"
    return img


@pytest.fixture()
def clean_env(monkeypatch, tmp_path):
    """A library with one background on disk and in the index, a LocalStore in
    place of GithubStore and an rembg stub whose mask is the dark rectangle."""
    from common import config, store

    repo = config.REPO_DIR
    (repo / "backgrounds").mkdir(parents=True, exist_ok=True)
    _busy_background().save(repo / "backgrounds" / "7_marble.jpg", format="JPEG", quality=95)
    local = store.LocalStore(repo)
    store.save_backgrounds(
        [{"id": 7, "path": "backgrounds/7_marble.jpg", "name": "Мрамор"}], local
    )
    monkeypatch.setattr(config, "GITHUB_TOKEN", "T")
    monkeypatch.setattr(config, "INPAINT_BACKEND", "auto")
    monkeypatch.setattr(store, "GithubStore", lambda **kw: local)

    state = {"mask_box": (380, 120, 499, 279)}

    def remove(image, session=None, only_mask=False, **kw):
        assert only_mask
        mask = Image.new("L", image.size, 0)
        if state["mask_box"]:
            from PIL import ImageDraw

            ImageDraw.Draw(mask).rectangle(state["mask_box"], fill=255)
        return mask

    module = types.ModuleType("rembg")
    module.new_session = lambda model: model
    module.remove = remove
    monkeypatch.setitem(sys.modules, "rembg", module)
    return {"store": local, "repo": repo, "state": state}


@pytest.fixture()
def stub_photo(monkeypatch, stub_telegram):
    monkeypatch.setattr(
        tg,
        "send_photo",
        lambda chat_id, blob, caption=None, reply_markup=None: stub_telegram.append(
            {"chat_id": chat_id, "photo": blob, "caption": caption, "reply_markup": reply_markup}
        ),
    )
    return stub_telegram


def test_clean_task_stores_cleaned_copy_and_keeps_original(clean_env, stub_photo):
    from common import store

    assert worker.main(_clean_argv()) == 0

    items = store.list_backgrounds(clean_env["store"])
    item = items[0]
    assert item["path"] == "backgrounds/7_marble_clean.jpg"
    assert item["original_path"] == "backgrounds/7_marble.jpg"
    assert item["cleaned_at"].endswith("Z")
    assert (clean_env["repo"] / "backgrounds" / "7_marble.jpg").exists()  # original untouched
    cleaned = Image.open(clean_env["repo"] / "backgrounds" / "7_marble_clean.jpg")
    assert cleaned.size == (600, 400)
    assert all(c > 120 for c in cleaned.getpixel((440, 200)))  # the dark vase is gone

    msg = stub_photo[-1]
    assert msg["chat_id"] == "555"
    assert "очищен" in msg["caption"] and "Мрамор" in msg["caption"]
    assert "фрагментами фона" in msg["caption"]
    assert msg["reply_markup"]["inline_keyboard"][0][0]["callback_data"] == "bgorig:7"
    assert Image.open(io.BytesIO(msg["photo"])).format == "JPEG"


def test_clean_task_twice_does_not_nest_suffixes_or_lose_original(clean_env, stub_photo):
    from common import store

    assert worker.main(_clean_argv()) == 0
    # Second pass starts from the already-cleaned file (that is what the bot passes).
    assert worker.main(_clean_argv(background="backgrounds/7_marble_clean.jpg")) == 0
    item = store.list_backgrounds(clean_env["store"])[0]
    assert item["path"] == "backgrounds/7_marble_clean.jpg"
    assert item["original_path"] == "backgrounds/7_marble.jpg"


def test_clean_task_reports_nothing_to_clean(clean_env, stub_photo):
    from common import store

    clean_env["state"]["mask_box"] = None
    assert worker.main(_clean_argv()) == 0
    assert "не нашлось объектов" in stub_photo[-1]["text"]
    assert "original_path" not in store.list_backgrounds(clean_env["store"])[0]


def test_clean_task_refuses_when_mask_covers_most_of_the_frame(clean_env, stub_photo):
    clean_env["state"]["mask_box"] = (0, 0, 599, 330)
    assert worker.main(_clean_argv()) == 1
    assert "Не стал очищать" in stub_photo[-1]["text"]


def test_clean_task_reports_missing_background(clean_env, stub_photo):
    assert worker.main(_clean_argv(background="backgrounds/nope.jpg")) == 1
    assert "not found" in stub_photo[-1]["text"]


def test_clean_task_dry_run_writes_file(clean_env, stub_photo, tmp_path):
    from common import store

    out = tmp_path / "c.jpg"
    assert worker.main(_clean_argv(dry_run=True, out=str(out))) == 0
    assert Image.open(out).size == (600, 400)
    assert "original_path" not in store.list_backgrounds(clean_env["store"])[0]
    assert stub_photo == []


def test_clean_task_reports_store_failure(clean_env, stub_photo, monkeypatch):
    from common import store

    def boom(**kw):
        raise store.StoreError("PUT: HTTP 500")

    monkeypatch.setattr(store, "GithubStore", boom)
    assert worker.main(_clean_argv()) == 1
    assert "не смог сохранить" in stub_photo[-1]["text"]


def test_clean_task_survives_unexpected_error(clean_env, stub_photo, monkeypatch):
    from common import inpaint

    def boom(*a, **k):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(inpaint, "object_mask", boom)
    assert worker.main(_clean_argv()) == 1
    assert "внутренняя ошибка" in stub_photo[-1]["text"]


def test_parse_args_clean_requires_background_and_id():
    with pytest.raises(SystemExit):
        worker.parse_args(["--task", "clean", "--chat-id", "1", "--background", "b"])
    with pytest.raises(SystemExit):
        worker.parse_args(["--task", "clean", "--chat-id", "1", "--background-id", "3"])
    args = worker.parse_args(
        ["--task", "clean", "--chat-id", "1", "--background", "b", "--background-id", "3"]
    )
    assert args.task == "clean" and args.file_id == ""


def test_parse_args_replace_still_requires_file_id():
    with pytest.raises(SystemExit):
        worker.parse_args(["--task", "replace", "--chat-id", "1", "--background", "b"])
