"""Minimal Telegram Bot API client built on `requests`.

aiogram is deliberately not used here: the Render free tier has 512 MB and
0.1 CPU, and the webhook service only needs a handful of Bot API calls. Keeping
dependencies to `flask` + `requests` also removes a whole class of version
conflicts with rembg on the Actions runner.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import requests

from . import config

log = logging.getLogger(__name__)


class TelegramError(RuntimeError):
    def __init__(self, method: str, status: int, body: str):
        super().__init__(f"{method} failed: HTTP {status} {body[:300]}")
        self.method = method
        self.status = status
        self.body = body


def _url(method: str) -> str:
    if not config.BOT_TOKEN:
        raise TelegramError(method, 0, "TELEGRAM_BOT_TOKEN is not set")
    return f"{config.TELEGRAM_API}/bot{config.BOT_TOKEN}/{method}"


def _request(method: str, **kwargs) -> requests.Response:
    """Perform the HTTP call, converting transport failures into TelegramError.

    Without this, a TLS reset / timeout / DNS failure escapes as a raw
    ``requests.exceptions.RequestException``, which none of the callers catch -
    they all handle TelegramError. The distinction matters: a network blip
    should produce a friendly retry message, not an unhandled traceback.
    """
    try:
        return requests.post(_url(method), **kwargs)
    except requests.RequestException as exc:
        raise TelegramError(method, 0, f"transport failure: {exc}") from exc


def _unwrap(method: str, response: requests.Response) -> Any:
    if response.status_code != 200:
        raise TelegramError(method, response.status_code, response.text)
    try:
        body = response.json()
    except ValueError as exc:
        raise TelegramError(method, response.status_code, "non-JSON response") from exc
    if not body.get("ok"):
        raise TelegramError(method, response.status_code, response.text)
    return body.get("result")


def call(method: str, payload: dict[str, Any] | None = None, *, timeout: int = 30) -> Any:
    """POST a JSON payload and return ``result``. Raises on a non-ok response."""
    payload = {k: v for k, v in (payload or {}).items() if v is not None}
    return _unwrap(method, _request(method, json=payload, timeout=timeout))


def upload(
    method: str,
    files: dict[str, tuple[str, bytes, str]],
    data: dict[str, Any],
    *,
    timeout: int = 60,
) -> Any:
    """Multipart upload (sendPhoto / sendDocument)."""
    payload = {}
    for key, value in data.items():
        if value is None:
            continue
        payload[key] = (
            json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
        )
    return _unwrap(method, _request(method, files=files, data=payload, timeout=timeout))


# --- Convenience wrappers ---------------------------------------------------


def send_message(chat_id: int | str, text: str, reply_markup: dict | None = None) -> Any:
    return call(
        "sendMessage",
        {"chat_id": chat_id, "text": text, "reply_markup": reply_markup},
    )


def send_document(
    chat_id: int | str,
    blob: bytes,
    filename: str,
    caption: str | None = None,
    reply_markup: dict | None = None,
) -> Any:
    return upload(
        "sendDocument",
        {"document": (filename, blob, "image/jpeg")},
        {"chat_id": chat_id, "caption": caption, "reply_markup": reply_markup},
    )


def send_photo(
    chat_id: int | str,
    blob: bytes,
    caption: str | None = None,
    reply_markup: dict | None = None,
) -> Any:
    return upload(
        "sendPhoto",
        {"photo": ("preview.jpg", blob, "image/jpeg")},
        {"chat_id": chat_id, "caption": caption, "reply_markup": reply_markup},
    )


def answer_callback(callback_id: str, text: str | None = None, show_alert: bool = False) -> None:
    try:
        call(
            "answerCallbackQuery",
            {"callback_query_id": callback_id, "text": text, "show_alert": show_alert},
            timeout=10,
        )
    except TelegramError:  # a stale callback must never break the update
        log.warning("answerCallbackQuery failed for %s", callback_id, exc_info=True)


def edit_message(
    chat_id: int | str,
    message_id: int,
    text: str,
    reply_markup: dict | None = None,
) -> None:
    """editMessageText, falling back to editMessageCaption for media messages.

    Telegram rejects editMessageText on a message that carries a caption, so a
    photo/document message needs the caption variant. This is the same failure
    mode that was hit in the goofish bot.
    """
    try:
        call(
            "editMessageText",
            {
                "chat_id": chat_id,
                "message_id": message_id,
                "text": text,
                "reply_markup": reply_markup,
            },
            timeout=15,
        )
        return
    except TelegramError as exc:
        log.info("editMessageText failed (%s), retrying as caption", exc)
    call(
        "editMessageCaption",
        {
            "chat_id": chat_id,
            "message_id": message_id,
            "caption": text,
            "reply_markup": reply_markup,
        },
        timeout=15,
    )


def download_by_file_id(file_id: str) -> bytes:
    """Resolve a file_id to bytes via getFile. Subject to the 20 MB Bot API limit."""
    info = call("getFile", {"file_id": file_id}, timeout=30) or {}
    file_path = info.get("file_path")
    if not file_path:
        raise TelegramError("getFile", 200, f"no file_path for {file_id}")
    try:
        response = requests.get(
            f"{config.TELEGRAM_API}/file/bot{config.BOT_TOKEN}/{file_path}", timeout=120
        )
    except requests.RequestException as exc:
        raise TelegramError("download", 0, f"transport failure: {exc}") from exc
    if response.status_code != 200:
        raise TelegramError("download", response.status_code, file_path)
    blob = response.content
    if len(blob) > config.MAX_BYTES:
        raise TelegramError("download", 200, f"file too large: {len(blob)} bytes")
    return blob


def set_webhook(url: str, secret_token: str | None = None) -> Any:
    return call(
        "setWebhook",
        {
            "url": url,
            "secret_token": secret_token or None,
            "allowed_updates": ["message", "callback_query"],
            "max_connections": 20,
        },
        timeout=30,
    )


def delete_webhook(drop_pending: bool = False) -> Any:
    return call("deleteWebhook", {"drop_pending_updates": drop_pending}, timeout=30)


def get_updates(offset: int | None = None, timeout: int = 30) -> list[dict]:
    return (
        call(
            "getUpdates",
            {"offset": offset, "timeout": timeout, "allowed_updates": ["message", "callback_query"]},
            timeout=timeout + 15,
        )
        or []
    )
