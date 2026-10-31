"""End-to-end tests for the webhook service.

A real Flask test client posts real Telegram update payloads at /webhook. The
only thing stubbed is the outbound Telegram API and the Actions dispatch, so
routing, the allowlist, the secret-token check, command parsing, callback
handling and the store writes are all genuinely executed.
"""

from __future__ import annotations

import itertools
import json
import threading
import time

import pytest

from common import actions, config, store, telegram_api as tg

ALLOWED = 111
OUTSIDER = 999
CHAT = 555
_update_ids = itertools.count(1000)


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BOT_TOKEN", "TOKEN")
    monkeypatch.setattr(config, "WEBHOOK_SECRET", "s3cret")
    monkeypatch.setattr(config, "ALLOWED_USER_IDS", {ALLOWED})
    monkeypatch.setattr(config, "GITHUB_TOKEN", "")
    monkeypatch.setattr(config, "GITHUB_REPO", "acme/repo")
    monkeypatch.setattr(config, "REPO_DIR", tmp_path)
    store.reset_store()

    sent: list[dict] = []
    monkeypatch.setattr(
        tg, "send_message", lambda chat_id, text, reply_markup=None: sent.append(
            {"kind": "message", "chat_id": chat_id, "text": text, "reply_markup": reply_markup}
        )
    )
    monkeypatch.setattr(
        tg, "send_photo", lambda chat_id, blob, caption=None, reply_markup=None: sent.append(
            {"kind": "photo", "chat_id": chat_id, "caption": caption, "bytes": len(blob)}
        )
    )
    monkeypatch.setattr(tg, "answer_callback", lambda *a, **k: sent.append({"kind": "answer", "args": a}))
    monkeypatch.setattr(
        tg, "edit_message", lambda *a, **k: sent.append({"kind": "edit", "args": a, "kwargs": k})
    )

    dispatched: list[dict] = []
    monkeypatch.setattr(
        actions, "dispatch_workflow", lambda inputs, **kw: dispatched.append(dict(inputs))
    )

    import app as app_module

    app_module.app.config["TESTING"] = True
    app_module._seen.clear()
    return {
        "client": app_module.app.test_client(),
        "sent": sent,
        "dispatched": dispatched,
        "tmp_path": tmp_path,
    }


def _post(client, update, secret="s3cret"):
    headers = {"Content-Type": "application/json"}
    if secret is not None:
        headers["X-Telegram-Bot-Api-Secret-Token"] = secret
    return client.post("/webhook", data=json.dumps(update), headers=headers)


def _message(text=None, user_id=ALLOWED, chat_id=CHAT, **extra):
    message = {"message_id": 1, "chat": {"id": chat_id, "type": "private"}, "from": {"id": user_id}}
    if text is not None:
        message["text"] = text
    message.update(extra)
    return {"update_id": next(_update_ids), "message": message}


def _photo(user_id=ALLOWED, chat_id=CHAT, caption=None, file_id="PROD"):
    message = {
        "message_id": 2,
        "chat": {"id": chat_id, "type": "private"},
        "from": {"id": user_id},
        "photo": [
            {"file_id": "small", "file_size": 1000},
            {"file_id": file_id, "file_size": 50000},
        ],
    }
    if caption is not None:
        message["caption"] = caption
    return {"update_id": next(_update_ids), "message": message}


def _callback(data, user_id=ALLOWED):
    return {
        "update_id": next(_update_ids),
        "callback_query": {
            "id": "cb1",
            "from": {"id": user_id},
            "data": data,
            "message": {
                "message_id": 9,
                "chat": {"id": CHAT, "type": "private"},
                "text": "Фон №1",
                "reply_markup": {"inline_keyboard": []},
            },
        },
    }


# --- HTTP surface -----------------------------------------------------------


def test_health_endpoint(env):
    response = env["client"].get("/health")
    assert response.status_code == 200
    assert response.get_data(as_text=True).strip() == "ok"


def test_index_endpoint(env):
    assert env["client"].get("/").status_code == 200


def test_webhook_rejects_missing_secret(env):
    assert _post(env["client"], _message("/start"), secret=None).status_code == 403
    assert env["sent"] == []


def test_webhook_rejects_wrong_secret(env):
    assert _post(env["client"], _message("/start"), secret="wrong").status_code == 403
    assert env["sent"] == []


def test_webhook_rejects_non_json(env):
    response = env["client"].post(
        "/webhook",
        data="not json",
        headers={"X-Telegram-Bot-Api-Secret-Token": "s3cret", "Content-Type": "application/json"},
    )
    assert response.status_code == 400


def test_webhook_returns_200_even_for_unknown_update(env):
    """A non-200 makes Telegram retry, which would double-process a photo."""
    assert _post(env["client"], {"update_id": 77}).status_code == 200


def test_webhook_survives_handler_crash(env, monkeypatch):
    from common import updates

    def boom(_update):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(updates, "handle_update", boom)
    import app as app_module

    monkeypatch.setattr(app_module, "handle_update", boom)
    assert _post(env["client"], _message("/start")).status_code == 200


# --- allowlist --------------------------------------------------------------


def test_outsider_is_refused(env):
    assert _post(env["client"], _message("/start", user_id=OUTSIDER)).status_code == 200
    assert len(env["sent"]) == 1
    assert "ограничен" in env["sent"][0]["text"]


def test_anonymous_update_is_refused(env):
    update = _message("/start")
    del update["message"]["from"]
    assert _post(env["client"], update).status_code == 200
    assert "ограничен" in env["sent"][0]["text"]


def test_empty_allowlist_denies_everyone(env, monkeypatch):
    """An unset allowlist must fail closed, not open."""
    monkeypatch.setattr(config, "ALLOWED_USER_IDS", set())
    assert _post(env["client"], _message("/start")).status_code == 200
    assert "ограничен" in env["sent"][0]["text"]


# --- commands ---------------------------------------------------------------


def test_start_sends_help(env):
    from common.updates import HELP_TEXT

    assert _post(env["client"], _message("/start")).status_code == 200
    assert env["sent"][0]["text"] == HELP_TEXT


def test_command_with_bot_username_is_parsed(env):
    from common.updates import HELP_TEXT

    _post(env["client"], _message("/start@product_bg_bot"))
    assert env["sent"][0]["text"] == HELP_TEXT


def test_unknown_command_is_answered(env):
    _post(env["client"], _message("/frobnicate"))
    assert "Не знаю такой команды" in env["sent"][0]["text"]


def test_plain_text_without_command(env):
    _post(env["client"], _message("привет"))
    assert "Пришлите фото товара" in env["sent"][0]["text"]


def test_backgrounds_command_on_empty_library(env):
    _post(env["client"], _message("/backgrounds"))
    assert "Фонов пока нет" in env["sent"][0]["text"]


def test_settings_command_shows_current_state(env):
    _post(env["client"], _message("/settings"))
    payload = env["sent"][0]
    assert "Настройки" in payload["text"]
    assert payload["reply_markup"]["inline_keyboard"]  # buttons present


def test_cancel_clears_pending_state(env):
    _post(env["client"], _message("/add_background"))
    entry = store.user_settings(ALLOWED, store.load_settings())
    assert entry["pending"] == "add_background"

    _post(env["client"], _message("/cancel"))
    entry = store.user_settings(ALLOWED, store.load_settings())
    assert entry["pending"] is None
    assert entry["oneoff"] is None


def test_bg_without_image_waits_for_the_next_picture(env):
    """Regression: /bg alone used to only print usage and remember nothing, so
    the picture sent next was treated as a product photo and the user was told
    to add a background via /bg - an endless loop on a phone."""
    _post(env["client"], _message("/bg"))
    entry = store.user_settings(ALLOWED, store.load_settings())
    assert entry["pending"] == "add_background"
    assert "следующим сообщением" in env["sent"][0]["text"]


def test_bg_with_name_remembers_it_for_the_next_picture(env):
    _post(env["client"], _message("/bg Мрамор"))
    entry = store.user_settings(ALLOWED, store.load_settings())
    assert entry["pending"] == "add_background"
    assert entry["pending_name"] == "Мрамор"


def test_once_without_image_sets_pending(env):
    _post(env["client"], _message("/once"))
    assert store.user_settings(ALLOWED, store.load_settings())["pending"] == "once"


def test_once_with_image_stores_oneoff(env):
    _post(env["client"], _photo(caption="/once"))
    entry = store.user_settings(ALLOWED, store.load_settings())
    assert entry["oneoff"] == "PROD"
    assert entry["pending"] is None


# --- product photo -> dispatch ----------------------------------------------


def test_product_photo_without_background_is_not_dispatched(env):
    _post(env["client"], _photo())
    assert env["dispatched"] == []
    assert "Фон ещё не выбран" in env["sent"][0]["text"]
    assert "/bg" in env["sent"][0]["text"]


def test_product_photo_dispatches_with_settings(env):
    items = store.list_backgrounds()
    items.append({"id": 3, "name": "Мрамор", "path": "backgrounds/3.png"})
    store.save_backgrounds(items)
    store.update_user(ALLOWED, {"background_id": 3, "model": "birefnet-general", "size": 1600})

    _post(env["client"], _photo())

    assert len(env["dispatched"]) == 1
    job = env["dispatched"][0]
    assert job["chat_id"] == str(CHAT)
    assert job["user_id"] == str(ALLOWED)
    assert job["file_id"] == "PROD"
    assert job["background"] == "backgrounds/3.png"
    assert job["oneoff_file_id"] == ""
    assert job["model"] == "birefnet-general"
    assert job["size"] == "1600"
    assert job["label"] == "Мрамор"
    assert "обрабатываю" in env["sent"][-1]["text"].lower()


def test_picks_the_largest_photo_from_an_album(env):
    items = store.list_backgrounds()
    items.append({"id": 1, "name": "Фон", "path": "backgrounds/1.png"})
    store.save_backgrounds(items)
    store.update_user(ALLOWED, {"background_id": 1})

    _post(env["client"], _photo())
    assert env["dispatched"][0]["file_id"] == "PROD"  # 50000 bytes, not "small"


def test_oneoff_background_wins_and_is_consumed(env):
    items = store.list_backgrounds()
    items.append({"id": 1, "name": "Фон", "path": "backgrounds/1.png"})
    store.save_backgrounds(items)
    store.update_user(ALLOWED, {"background_id": 1, "oneoff": "ONEOFF_BG"})

    _post(env["client"], _photo())

    job = env["dispatched"][0]
    assert job["oneoff_file_id"] == "ONEOFF_BG"
    assert job["background"] == ""
    assert store.user_settings(ALLOWED, store.load_settings())["oneoff"] is None


def test_pending_once_then_product_photo(env):
    _post(env["client"], _photo(caption="/once"))
    _post(env["client"], _photo(file_id="ITEM"))
    assert env["dispatched"][0]["oneoff_file_id"] == "PROD"


def test_dispatch_failure_is_reported(env, monkeypatch):
    items = store.list_backgrounds()
    items.append({"id": 1, "name": "Фон", "path": "backgrounds/1.png"})
    store.save_backgrounds(items)
    store.update_user(ALLOWED, {"background_id": 1})

    def fail(inputs, **kw):
        raise actions.DispatchError("HTTP 404 workflow not found")

    monkeypatch.setattr(actions, "dispatch_workflow", fail)
    _post(env["client"], _photo())
    assert "process.yml" in env["sent"][-1]["text"]


# --- document handling ------------------------------------------------------


def test_image_document_is_accepted(env):
    items = store.list_backgrounds()
    items.append({"id": 1, "name": "Фон", "path": "backgrounds/1.png"})
    store.save_backgrounds(items)
    store.update_user(ALLOWED, {"background_id": 1})

    update = _message()
    update["message"]["document"] = {
        "file_id": "DOC",
        "file_size": 12345,
        "mime_type": "image/png",
    }
    _post(env["client"], update)
    assert env["dispatched"][0]["file_id"] == "DOC"


def test_non_image_document_is_rejected(env):
    items = store.list_backgrounds()
    items.append({"id": 1, "name": "Фон", "path": "backgrounds/1.png"})
    store.save_backgrounds(items)
    store.update_user(ALLOWED, {"background_id": 1})

    update = _message()
    update["message"]["document"] = {"file_id": "X", "mime_type": "application/pdf"}
    _post(env["client"], update)
    assert env["dispatched"] == []
    assert "Пришлите фото товара" in env["sent"][-1]["text"]


# --- callbacks --------------------------------------------------------------


def test_callback_selects_background(env):
    items = store.list_backgrounds()
    items.append({"id": 5, "name": "Дерево", "path": "backgrounds/5.png"})
    store.save_backgrounds(items)

    _post(env["client"], _callback("bg:5"))
    assert store.user_settings(ALLOWED, store.load_settings())["background_id"] == 5


def test_callback_unknown_background_is_rejected(env):
    _post(env["client"], _callback("bg:999"))
    assert store.user_settings(ALLOWED, store.load_settings())["background_id"] is None
    assert any(item["kind"] == "answer" for item in env["sent"])


def test_callback_malformed_id_is_rejected(env):
    _post(env["client"], _callback("bg:abc"))
    assert store.user_settings(ALLOWED, store.load_settings())["background_id"] is None


def test_callback_outsider_is_rejected(env):
    items = store.list_backgrounds()
    items.append({"id": 5, "name": "Дерево", "path": "backgrounds/5.png"})
    store.save_backgrounds(items)

    _post(env["client"], _callback("bg:5", user_id=OUTSIDER))
    assert store.user_settings(ALLOWED, store.load_settings())["background_id"] is None
    assert store.user_settings(OUTSIDER, store.load_settings())["background_id"] is None


def test_callback_switches_model(env):
    _post(env["client"], _callback(f"model:quality"))
    assert store.user_settings(ALLOWED, store.load_settings())["model"] == config.MODEL_QUALITY

    _post(env["client"], _callback("model:fast"))
    assert store.user_settings(ALLOWED, store.load_settings())["model"] == config.MODEL_FAST


def test_callback_rejects_unknown_model(env):
    _post(env["client"], _callback("model:ultra"))
    assert store.user_settings(ALLOWED, store.load_settings())["model"] == config.DEFAULT_MODEL


def test_callback_switches_size(env):
    _post(env["client"], _callback("size:2048"))
    assert store.user_settings(ALLOWED, store.load_settings())["size"] == 2048


def test_callback_rejects_unavailable_size(env):
    _post(env["client"], _callback("size:9999"))
    assert store.user_settings(ALLOWED, store.load_settings())["size"] == config.DEFAULT_SIZE


def test_callback_deletes_background(env):
    items = store.list_backgrounds()
    items.append({"id": 8, "name": "Старый", "path": "backgrounds/8.png"})
    store.save_backgrounds(items)
    env["tmp_path"].joinpath("backgrounds").mkdir(exist_ok=True)
    (env["tmp_path"] / "backgrounds" / "8.png").write_bytes(b"x")

    # First tap only asks; the shared library must not vanish on a mis-tap.
    _post(env["client"], _callback("bgdel:8"))
    assert [i["id"] for i in store.list_backgrounds()] == [8]
    assert (env["tmp_path"] / "backgrounds" / "8.png").exists()
    edit = [m for m in env["sent"] if m["kind"] == "edit"][-1]
    assert "Удалить этот фон" in edit["args"][2]

    _post(env["client"], _callback("bgdelyes:8"))
    assert store.list_backgrounds() == []
    assert not (env["tmp_path"] / "backgrounds" / "8.png").exists()
    edit = [m for m in env["sent"] if m["kind"] == "edit"][-1]
    assert "Удалён" in edit["args"][2]


def test_delete_can_be_cancelled(env):
    items = store.list_backgrounds()
    items.append({"id": 8, "name": "Старый", "path": "backgrounds/8.png"})
    store.save_backgrounds(items)
    store.update_user(ALLOWED, {"background_id": 8})
    _post(env["client"], _callback("bgdel:8"))
    _post(env["client"], _callback("bgdelno:8"))
    assert [i["id"] for i in store.list_backgrounds()] == [8]
    edit = [m for m in env["sent"] if m["kind"] == "edit"][-1]
    assert "Удалить" not in edit["args"][2]
    keyboard = edit["kwargs"]["reply_markup"]
    assert keyboard["inline_keyboard"][0][0]["text"] == "✅ Выбран"


def test_delete_resets_everyones_selection_of_that_background(env):
    items = store.list_backgrounds()
    items.append({"id": 8, "name": "Старый", "path": "backgrounds/8.png"})
    items.append({"id": 9, "name": "Другой", "path": "backgrounds/9.png"})
    store.save_backgrounds(items)
    store.update_user(ALLOWED, {"background_id": 8})
    store.update_user(4242, {"background_id": 8})
    store.update_user(4343, {"background_id": 9})

    _post(env["client"], _callback("bgdelyes:8"))

    settings = store.load_settings()
    assert settings["users"][str(ALLOWED)]["background_id"] is None
    assert settings["users"]["4242"]["background_id"] is None
    assert settings["users"]["4343"]["background_id"] == 9


def test_delete_of_unknown_background_is_an_alert(env):
    _post(env["client"], _callback("bgdelyes:77"))
    answer = [m for m in env["sent"] if m["kind"] == "answer"][-1]
    assert "уже удалён" in answer["args"][1]


def test_select_callback_rebuilds_keyboard_without_appending_text(env):
    items = store.list_backgrounds()
    items.append({"id": 3, "name": "Мрамор", "path": "backgrounds/3.png"})
    store.save_backgrounds(items)
    _post(env["client"], _callback("bg:3"))
    _post(env["client"], _callback("bg:3"))
    edits = [m for m in env["sent"] if m["kind"] == "edit"]
    assert len(edits) == 2
    for e in edits:
        assert e["args"][2] == "№3 — Мрамор"  # no "✅ Выбран пользователем ..." lines piling up
    assert store.user_settings(ALLOWED, store.load_settings())["background_id"] == 3


def test_callback_noop_is_acknowledged(env):
    _post(env["client"], _callback("noop"))
    assert any(item["kind"] == "answer" for item in env["sent"])


def test_callback_unknown_button_is_acknowledged(env):
    _post(env["client"], _callback("wat"))
    assert any(item["kind"] == "answer" for item in env["sent"])


# --- error classification ---------------------------------------------------


def test_real_transport_failure_is_classified_as_telegram_error(env, monkeypatch, caplog):
    """A TLS reset must be caught by the TelegramError branch, not the catch-all.

    Regression: the raw requests exception used to escape telegram_api, so a
    network blip was logged as "unexpected error" and looked like a code bug.
    This stubs requests itself, so the whole chain is exercised for real.
    """
    import logging

    import requests as real_requests

    from common import updates

    # The `env` fixture stubs send_message. Restore the real implementation so
    # the failure happens inside telegram_api exactly as it does in production -
    # otherwise nothing ever reaches the network layer.
    def real_send_message(chat_id, text, reply_markup=None):
        return tg.call(
            "sendMessage",
            {"chat_id": chat_id, "text": text, "reply_markup": reply_markup},
        )

    monkeypatch.setattr(tg, "send_message", real_send_message)

    def boom(url, **kwargs):
        raise real_requests.exceptions.SSLError("TLS/SSL connection has been closed (EOF)")

    monkeypatch.setattr(tg.requests, "post", boom)

    with caplog.at_level(logging.ERROR):
        updates.handle_update(_message("/start"))

    messages = [record.getMessage() for record in caplog.records]
    assert any("Telegram API error" in m for m in messages), messages
    assert not any("unexpected error" in m for m in messages), messages


# --- mode selection ---------------------------------------------------------


def test_mode_defaults_to_polling_without_render_url(monkeypatch):
    import app as app_module

    monkeypatch.delenv("RENDER_EXTERNAL_URL", raising=False)
    monkeypatch.setattr(config, "BOT_MODE", "")
    assert app_module._mode() == "polling"


def test_mode_defaults_to_webhook_on_render(monkeypatch):
    import app as app_module

    monkeypatch.setenv("RENDER_EXTERNAL_URL", "https://x.onrender.com")
    monkeypatch.setattr(config, "BOT_MODE", "")
    assert app_module._mode() == "webhook"


def test_explicit_mode_wins(monkeypatch):
    import app as app_module

    monkeypatch.setenv("RENDER_EXTERNAL_URL", "https://x.onrender.com")
    monkeypatch.setattr(config, "BOT_MODE", "off")
    assert app_module._mode() == "off"


# --- failures must be reported to the user, not only logged -----------------


def test_store_failure_is_reported_to_user(env, monkeypatch):
    """Regression: a failing GitHub write used to be logged and swallowed, so
    /once, /bg and settings silently "did not stick". The user must get a
    message naming the likely cause."""

    def boom(*a, **k):
        raise store.StoreError("PUT settings.json: HTTP 401 Bad credentials")

    monkeypatch.setattr(store, "update_user", boom)
    resp = _post(env["client"], _photo(caption="/once"))
    assert resp.status_code == 200  # never make Telegram redeliver
    assert env["dispatched"] == []
    assert len(env["sent"]) == 1
    text = env["sent"][0]["text"]
    assert env["sent"][0]["chat_id"] == CHAT
    assert "HTTP 401" in text
    assert "GITHUB_TOKEN" in text


def test_unexpected_failure_is_reported_to_user(env, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(store, "update_user", boom)
    resp = _post(env["client"], _message("/once"))
    assert resp.status_code == 200
    assert len(env["sent"]) == 1
    assert "kaboom" in env["sent"][0]["text"]


def test_error_reply_failure_does_not_escape(env, monkeypatch):
    """If even the error message cannot be sent, the webhook still answers 200."""

    def boom(*a, **k):
        raise store.StoreError("PUT settings.json: HTTP 500")

    def dead_api(*a, **k):
        raise tg.TelegramError("sendMessage", 0, "transport failure")

    monkeypatch.setattr(store, "update_user", boom)
    monkeypatch.setattr(tg, "send_message", dead_api)
    resp = _post(env["client"], _message("/once"))
    assert resp.status_code == 200


# --- adding a background: the full path, not just the pending flag ---------


def _png_bytes(w=900, h=600, color=(10, 20, 200)):
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture()
def fake_download(monkeypatch):
    """Serve a real PNG for any file_id; records what was asked for."""
    asked: list[str] = []

    def download(file_id):
        asked.append(file_id)
        return _png_bytes()

    monkeypatch.setattr(tg, "download_by_file_id", download)
    return asked


def test_bg_then_picture_adds_background_and_selects_it(env, fake_download):
    _post(env["client"], _message("/bg Мрамор"))
    _post(env["client"], _photo(file_id="BGFILE"))

    assert fake_download == ["BGFILE"]
    items = store.list_backgrounds()
    assert len(items) == 1
    item = items[0]
    assert item["id"] == 1
    assert item["name"] == "Мрамор"
    assert item["added_by"] == ALLOWED
    assert (env["tmp_path"] / item["path"]).is_file()
    assert item["path"].endswith(".jpg")
    assert (env["tmp_path"] / item["path"]).read_bytes()[:3] == b"\xff\xd8\xff"

    entry = store.user_settings(ALLOWED, store.load_settings())
    assert entry["background_id"] == 1
    assert entry["pending"] is None
    assert entry["pending_name"] is None

    confirmation = env["sent"][-1]["text"]
    assert "Фон №1" in confirmation and "Мрамор" in confirmation
    assert env["dispatched"] == []  # the background picture is not a product


def test_picture_with_bg_caption_still_works(env, fake_download):
    """The old one-message form must keep working for people who knew it."""
    _post(env["client"], _photo(caption="/bg Дерево", file_id="BGFILE"))
    items = store.list_backgrounds()
    assert [i["name"] for i in items] == ["Дерево"]
    assert store.user_settings(ALLOWED, store.load_settings())["background_id"] == 1


def test_caption_on_the_picture_overrides_name_from_bg(env, fake_download):
    _post(env["client"], _message("/bg Старое"))
    _post(env["client"], _photo(caption="Новое", file_id="BGFILE"))
    assert [i["name"] for i in store.list_backgrounds()] == ["Новое"]


def test_product_photo_after_bg_flow_is_dispatched_with_that_background(env, fake_download):
    _post(env["client"], _message("/bg"))
    _post(env["client"], _photo(file_id="BGFILE"))
    _post(env["client"], _photo(file_id="PRODUCT"))

    assert len(env["dispatched"]) == 1
    job = env["dispatched"][0]
    assert job["file_id"] == "PRODUCT"
    assert job["background"] == store.list_backgrounds()[0]["path"]
    assert job["oneoff_file_id"] == ""
    assert job["label"] == "Фон 1"
    assert "Принял" in env["sent"][-1]["text"]


def test_bg_flow_reports_unreadable_picture(env, monkeypatch):
    monkeypatch.setattr(tg, "download_by_file_id", lambda fid: b"not an image at all")
    _post(env["client"], _message("/bg"))
    _post(env["client"], _photo(file_id="BAD"))
    assert store.list_backgrounds() == []
    assert "Не получилось добавить фон" in env["sent"][-1]["text"]
    # The intent is kept so the user can simply send another picture.
    assert store.user_settings(ALLOWED, store.load_settings())["pending"] == "add_background"


def test_cancel_clears_bg_intent(env):
    _post(env["client"], _message("/bg Имя"))
    _post(env["client"], _message("/cancel"))
    entry = store.user_settings(ALLOWED, store.load_settings())
    assert entry["pending"] is None
    assert entry["pending_name"] is None


# --- redelivery and background processing ----------------------------------


def test_duplicate_delivery_is_ignored(env):
    """Telegram redelivers an update when it did not get a timely 2xx. The
    second copy must not produce a second reply (or a second Actions run)."""
    update = _message("/start")
    assert _post(env["client"], update).status_code == 200
    assert _post(env["client"], update).status_code == 200
    assert len(env["sent"]) == 1


def test_duplicate_product_photo_is_dispatched_once(env):
    items = store.list_backgrounds()
    items.append({"id": 3, "name": "Мрамор", "path": "backgrounds/3.png"})
    store.save_backgrounds(items)
    store.update_user(ALLOWED, {"background_id": 3})
    update = _photo(file_id="PRODUCT")
    _post(env["client"], update)
    _post(env["client"], update)
    assert len(env["dispatched"]) == 1


def test_updates_without_id_are_never_deduplicated(env):
    upd = _message("/start")
    del upd["update_id"]
    _post(env["client"], upd)
    _post(env["client"], upd)
    assert len(env["sent"]) == 2


def test_seen_cache_is_bounded(env):
    import app as app_module

    for i in range(app_module._SEEN_LIMIT + 50):
        app_module._already_seen(10_000_000 + i)
    assert len(app_module._seen) == app_module._SEEN_LIMIT
    # The newest id is still known...
    assert app_module._already_seen(10_000_000 + app_module._SEEN_LIMIT + 49) is True
    # ...and the oldest was evicted, so it reads as new again.
    assert 10_000_000 not in app_module._seen
    assert app_module._already_seen(10_000_000) is False


def test_outside_testing_mode_handler_runs_in_background(env, monkeypatch):
    """In production the POST returns before the handler finishes, so a slow
    /bg cannot make Telegram time out and redeliver."""
    import app as app_module

    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def slow_handler(update):
        started.set()
        release.wait(5)
        finished.set()

    monkeypatch.setattr(app_module, "handle_update", slow_handler)
    app_module.app.config["TESTING"] = False
    try:
        t0 = time.monotonic()
        resp = _post(env["client"], _message("/start"))
        elapsed = time.monotonic() - t0
    finally:
        app_module.app.config["TESTING"] = True
    assert resp.status_code == 200
    assert elapsed < 1.0, "request must not wait for the handler"
    assert started.wait(2)
    assert not finished.is_set()
    release.set()
    assert finished.wait(2)


def test_start_and_help_work_even_when_storage_is_down(env, monkeypatch):
    def down(*a, **k):
        raise store.StoreError("GET settings.json: transport failure: ConnectionError")

    monkeypatch.setattr(store, "load_settings", down)
    _post(env["client"], _message("/start"))
    _post(env["client"], _message("/help"))
    texts = [m["text"] for m in env["sent"]]
    assert len(texts) == 2 and all("Я заменяю фон" in t for t in texts)
