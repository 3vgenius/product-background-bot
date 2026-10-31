#!/usr/bin/env python3
"""Render entry point: Telegram webhook receiver (or local polling loop).

Runs on the Render free tier (512 MB / 0.1 CPU). It deliberately does no image
processing — the heavy rembg work happens in ``.github/workflows/process.yml``
on a GitHub-hosted runner with 2 vCPU and 7-8 GB of RAM, which is free.

Modes
-----
webhook  default when ``RENDER_EXTERNAL_URL`` is set (Render).
polling  default otherwise (local development, no public URL needed).
off      serve HTTP only, never touch Telegram (used by tests / health checks).
"""

from __future__ import annotations

import logging
import os
import time

from flask import Flask, request

from common import config
from common import telegram_api as tg
from common.updates import handle_update

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("app")

app = Flask(__name__)


def _mode() -> str:
    if config.BOT_MODE in ("webhook", "polling", "off"):
        return config.BOT_MODE
    return "webhook" if os.getenv("RENDER_EXTERNAL_URL") else "polling"


@app.get("/")
def index() -> tuple[str, int]:
    return "product-background-bot is alive\n", 200


@app.get("/health")
def health() -> tuple[str, int]:
    return "ok\n", 200


@app.post("/webhook")
def webhook() -> tuple[str, int]:
    """Receive one Telegram update.

    Always answers 200: returning an error makes Telegram retry, and a retry of
    an update we already handled would double-process the user's photo.
    """
    if config.WEBHOOK_SECRET:
        provided = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
        if provided != config.WEBHOOK_SECRET:
            log.warning("rejected webhook call with a bad secret token")
            return "forbidden\n", 403
    update = request.get_json(force=True, silent=True)
    if not isinstance(update, dict):
        return "bad request\n", 400
    # Swallow everything: a 500 here makes Telegram redeliver the same update,
    # and a redelivered photo would be processed (and billed) a second time.
    try:
        handle_update(update)
    except Exception:
        log.exception("handler failed for update %s", update.get("update_id"))
    return "ok\n", 200


def _register_webhook() -> None:
    external = os.getenv("RENDER_EXTERNAL_URL") or os.getenv("RENDER_EXTERNAL_HOSTNAME")
    if not external:
        log.error("RENDER_EXTERNAL_URL is not set; cannot register the webhook")
        return
    if not external.startswith("http"):
        external = "https://" + external
    url = external.rstrip("/") + "/webhook"
    for attempt in range(3):
        try:
            tg.set_webhook(url, config.WEBHOOK_SECRET or None)
            log.info("webhook registered at %s", url)
            return
        except tg.TelegramError as exc:
            log.warning("setWebhook attempt %d failed: %s", attempt + 1, exc)
            time.sleep(2 * (attempt + 1))
    log.error("giving up on setWebhook")


def _poll() -> None:
    try:
        tg.delete_webhook(drop_pending=True)
        log.info("webhook cleared, switching to long polling")
    except tg.TelegramError as exc:
        log.warning("deleteWebhook failed: %s", exc)
    offset: int | None = None
    while True:
        try:
            updates = tg.get_updates(offset=offset, timeout=30)
        except tg.TelegramError as exc:
            log.warning("getUpdates failed: %s", exc)
            time.sleep(3)
            continue
        for update in updates:
            offset = int(update.get("update_id", 0)) + 1
            handle_update(update)


def main() -> None:
    mode = _mode()
    log.info("starting in %s mode", mode)

    if not config.BOT_TOKEN:
        log.error("TELEGRAM_BOT_TOKEN is not set")
    if not config.ALLOWED_USER_IDS:
        log.error(
            "ALLOWED_USER_IDS is empty — nobody will be able to use the bot. "
            "Set it to a comma-separated list of numeric Telegram IDs."
        )

    if mode == "webhook":
        if config.BOT_TOKEN:
            _register_webhook()
    elif mode == "polling":
        import threading

        if config.BOT_TOKEN:
            threading.Thread(target=_poll, daemon=True).start()

    port = int(os.getenv("PORT", "10000"))
    # Render proxies to this port; host must be 0.0.0.0 for the container network.
    app.run(host="0.0.0.0", port=port, threaded=True)


if __name__ == "__main__":
    main()
