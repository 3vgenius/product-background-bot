"""Tests for the Telegram API client.

The important ones here are the transport-failure cases: a TLS reset or timeout
must surface as ``TelegramError``, because every caller in the bot catches that
type and nothing else. If a raw ``requests`` exception escaped, a network blip
would turn into an unhandled traceback instead of a friendly message.
"""

from __future__ import annotations

import pytest
import requests

from common import telegram_api as tg


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text="", content=b"", json_error=False):
        self.status_code = status_code
        self._payload = payload if payload is not None else {"ok": True, "result": {"x": 1}}
        self.text = text
        self.content = content
        self._json_error = json_error

    def json(self):
        # A separate flag, not payload=None: None is also the constructor's
        # "use the default" signal.
        if self._json_error:
            raise ValueError("Expecting value: line 1 column 1")
        return self._payload


@pytest.fixture()
def capture(monkeypatch):
    """Record what was sent and replay a scripted response or exception."""
    seen = {}

    def install(response=None, error=None):
        def fake_post(url, **kwargs):
            seen["url"] = url
            seen["kwargs"] = kwargs
            if error is not None:
                raise error
            return response

        monkeypatch.setattr(tg.requests, "post", fake_post)

    return seen, install


# --- success ----------------------------------------------------------------


def test_call_returns_result_and_strips_none(capture):
    seen, install = capture
    install(FakeResponse(payload={"ok": True, "result": 42}))

    assert tg.call("sendMessage", {"chat_id": 1, "text": "hi", "reply_markup": None}) == 42
    assert seen["url"].endswith("/botTOKEN/sendMessage")
    assert seen["kwargs"]["json"] == {"chat_id": 1, "text": "hi"}


def test_upload_serialises_reply_markup_as_json(capture):
    seen, install = capture
    install(FakeResponse())

    keyboard = {"inline_keyboard": [[{"text": "Выбрать", "callback_data": "bg:1"}]]}
    tg.upload(
        "sendPhoto",
        {"photo": ("p.jpg", b"data", "image/jpeg")},
        {"chat_id": 1, "caption": "Фон №1", "reply_markup": keyboard},
    )
    sent = seen["kwargs"]["data"]
    assert sent["caption"] == "Фон №1"
    assert "bg:1" in sent["reply_markup"]  # serialised, not a Python repr


# --- failure modes ----------------------------------------------------------


def test_non_200_raises_telegram_error(capture):
    _, install = capture
    install(FakeResponse(status_code=400, text='{"ok":false,"description":"bad chat"}'))
    with pytest.raises(tg.TelegramError) as exc:
        tg.call("sendMessage", {"chat_id": 1, "text": "x"})
    assert exc.value.status == 400


def test_ok_false_raises_telegram_error(capture):
    _, install = capture
    install(FakeResponse(payload={"ok": False, "description": "bot was blocked"}))
    with pytest.raises(tg.TelegramError):
        tg.call("sendMessage", {"chat_id": 1, "text": "x"})


def test_non_json_body_raises_telegram_error(capture):
    _, install = capture
    install(FakeResponse(text="<html>gateway</html>", json_error=True))
    with pytest.raises(tg.TelegramError, match="non-JSON"):
        tg.call("sendMessage", {"chat_id": 1, "text": "x"})


@pytest.mark.parametrize(
    "error",
    [
        requests.exceptions.SSLError("TLS/SSL connection has been closed (EOF)"),
        requests.exceptions.ConnectTimeout("timed out"),
        requests.exceptions.ConnectionError("name resolution failed"),
        requests.exceptions.ChunkedEncodingError("connection broken"),
    ],
)
def test_transport_failures_become_telegram_error(capture, error):
    """Regression: these used to escape as raw requests exceptions."""
    _, install = capture
    install(error=error)
    with pytest.raises(tg.TelegramError) as exc:
        tg.call("sendMessage", {"chat_id": 1, "text": "x"})
    assert exc.value.status == 0
    assert "transport failure" in str(exc.value)


@pytest.mark.parametrize(
    "error",
    [requests.exceptions.SSLError("eof"), requests.exceptions.ReadTimeout("slow")],
)
def test_upload_transport_failures_become_telegram_error(capture, error):
    _, install = capture
    install(error=error)
    with pytest.raises(tg.TelegramError):
        tg.upload("sendDocument", {"document": ("f.jpg", b"x", "image/jpeg")}, {"chat_id": 1})


@pytest.mark.parametrize(
    "error", [requests.exceptions.SSLError("eof"), requests.exceptions.ReadTimeout("slow")]
)
def test_download_transport_failures_become_telegram_error(monkeypatch, error):
    monkeypatch.setattr(
        tg, "call", lambda *a, **k: {"file_id": "F", "file_path": "photos/1.jpg"}
    )

    def boom(url, **kwargs):
        raise error

    monkeypatch.setattr(tg.requests, "get", boom)
    with pytest.raises(tg.TelegramError, match="transport failure"):
        tg.download_by_file_id("F")


# --- getFile / download -----------------------------------------------------


def test_download_returns_bytes(monkeypatch):
    monkeypatch.setattr(tg, "call", lambda *a, **k: {"file_path": "photos/1.jpg"})
    monkeypatch.setattr(
        tg.requests, "get", lambda url, **kw: FakeResponse(content=b"JPEGDATA", status_code=200)
    )
    assert tg.download_by_file_id("F") == b"JPEGDATA"


def test_download_without_file_path_raises(monkeypatch):
    monkeypatch.setattr(tg, "call", lambda *a, **k: {})
    with pytest.raises(tg.TelegramError, match="no file_path"):
        tg.download_by_file_id("F")


def test_download_rejects_oversized_file(monkeypatch):
    from common import config

    monkeypatch.setattr(config, "MAX_BYTES", 10)
    monkeypatch.setattr(tg, "call", lambda *a, **k: {"file_path": "photos/1.jpg"})
    monkeypatch.setattr(
        tg.requests, "get", lambda url, **kw: FakeResponse(content=b"x" * 100, status_code=200)
    )
    with pytest.raises(tg.TelegramError, match="too large"):
        tg.download_by_file_id("F")


def test_download_http_error_raises(monkeypatch):
    monkeypatch.setattr(tg, "call", lambda *a, **k: {"file_path": "photos/1.jpg"})
    monkeypatch.setattr(
        tg.requests, "get", lambda url, **kw: FakeResponse(status_code=404, text="not found")
    )
    with pytest.raises(tg.TelegramError):
        tg.download_by_file_id("F")


# --- misc -------------------------------------------------------------------


def test_missing_token_raises(capture, monkeypatch):
    from common import config

    monkeypatch.setattr(config, "BOT_TOKEN", "")
    with pytest.raises(tg.TelegramError, match="TELEGRAM_BOT_TOKEN"):
        tg.call("getMe", {})


def test_answer_callback_swallows_errors(capture):
    """A stale callback must never break the rest of the update."""
    _, install = capture
    install(error=requests.exceptions.SSLError("eof"))
    tg.answer_callback("cb1", "text")  # must not raise


def test_set_webhook_passes_secret_and_updates(capture):
    seen, install = capture
    install(FakeResponse(payload={"ok": True, "result": True}))
    tg.set_webhook("https://x.onrender.com/webhook", "s3cret")
    payload = seen["kwargs"]["json"]
    assert payload["url"] == "https://x.onrender.com/webhook"
    assert payload["secret_token"] == "s3cret"
    assert payload["allowed_updates"] == ["message", "callback_query"]


def test_edit_message_falls_back_to_caption(monkeypatch):
    """editMessageText is rejected on a message that carries a caption."""
    calls = []

    def fake_post(url, **kwargs):
        calls.append(url.rsplit("/", 1)[-1])
        if url.endswith("editMessageText"):
            return FakeResponse(status_code=400, text="message can't be edited")
        return FakeResponse()

    monkeypatch.setattr(tg.requests, "post", fake_post)
    tg.edit_message(1, 2, "new text")
    assert calls == ["editMessageText", "editMessageCaption"]


def test_edit_message_does_not_fall_back_when_text_edit_works(monkeypatch):
    calls = []

    def fake_post(url, **kwargs):
        calls.append(url.rsplit("/", 1)[-1])
        return FakeResponse()

    monkeypatch.setattr(tg.requests, "post", fake_post)
    tg.edit_message(1, 2, "new text")
    assert calls == ["editMessageText"]
