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
import threading
import time
from collections import OrderedDict

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

# Telegram redelivers an update when the webhook does not answer 2xx quickly.
# Heavy commands (/bg downloads, re-encodes and commits a picture; a product
# photo dispatches a workflow) can take longer than that on a 0.1-CPU dyno, so
# the handler runs in a background thread and the request returns at once.
# Remembering recent update_ids makes an unexpected redelivery a no-op instead
# of a second Actions run.
_SEEN_LIMIT = 2000
_seen: "OrderedDict[int, None]" = OrderedDict()
_seen_lock = threading.Lock()


def _already_seen(update_id: int | None) -> bool:
    if update_id is None:
        return False
    with _seen_lock:
        if update_id in _seen:
            return True
        _seen[update_id] = None
        while len(_seen) > _SEEN_LIMIT:
            _seen.popitem(last=False)
    return False


# One update at a time. settings.json and backgrounds.json are shared files
# edited read-modify-write; two concurrent handlers would race and one change
# would be lost (a photo album arrives as several updates within the same
# second). Throughput is not a concern for a small-team bot, ordering is.
_handler_lock = threading.Lock()


def _run_handler(update: dict) -> None:
    with _handler_lock:
        try:
            handle_update(update)
        except Exception:  # handle_update already swallows; this is belt and braces
            log.exception("handler failed for update %s", update.get("update_id"))


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
    if _already_seen(update.get("update_id")):
        log.info("duplicate delivery of update %s ignored", update.get("update_id"))
        return "ok\n", 200
    # Swallow everything: a 500 here makes Telegram redeliver the same update,
    # and a redelivered photo would be processed (and billed) a second time.
    if app.config.get("TESTING"):
        # Tests assert on side effects right after the POST returns.
        _run_handler(update)
    else:
        threading.Thread(target=_run_handler, args=(update,), daemon=True).start()
    return "ok\n", 200


def _register_commands() -> None:
    """Fill the "/" menu. Cosmetic, so a failure is logged and ignored."""
    from common.updates import BOT_COMMANDS

    try:
        tg.set_my_commands(BOT_COMMANDS)
        log.info("command menu registered (%d commands)", len(BOT_COMMANDS))
    except tg.TelegramError as exc:
        log.warning("setMyCommands failed: %s", exc)


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

    if config.BOT_TOKEN and mode != "off":
        _register_commands()

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
