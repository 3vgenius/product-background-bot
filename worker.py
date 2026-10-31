#!/usr/bin/env python3
"""GitHub Actions entry point: the heavy half of the pipeline.

Runs on a GitHub-hosted runner (2 vCPU / 7-8 GB RAM, free for private repos
within the monthly minutes allowance) — roughly 14x the memory of a Render
Starter instance, which is why BiRefNet is even an option here.

Flow
----
1. download the product photo from Telegram by file_id
2. cut the product out with rembg
3. read the background from the repository checkout (no API call needed)
4. composite and send the JPEG back to the chat

Any failure is reported to the user instead of dying silently in a log.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("worker")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chat-id", required=True)
    parser.add_argument("--user-id", default="")
    parser.add_argument("--file-id", required=True, help="Telegram file_id of the product photo")
    parser.add_argument("--background", default="", help="repo-relative path of a library background")
    parser.add_argument("--oneoff-file-id", default="", help="Telegram file_id of a one-off background")
    parser.add_argument("--model", default="")
    parser.add_argument("--size", type=int, default=0)
    parser.add_argument("--label", default="", help="human-readable background name for the caption")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="do everything except sending the result; write it to --out instead",
    )
    parser.add_argument("--out", default="", help="write the result here instead of sending it")
    return parser.parse_args(argv)


def load_background(args: argparse.Namespace):
    """Return a PIL image for the background, or raise ValueError."""
    from PIL import Image

    from common import store
    from common.imaging import ImageRejected, load_image

    if args.oneoff_file_id:
        from common import telegram_api as tg

        log.info("downloading one-off background %s", args.oneoff_file_id)
        return load_image(tg.download_by_file_id(args.oneoff_file_id))

    if not args.background:
        raise ValueError("no background specified")

    # 1. local checkout (the normal case inside Actions)
    local = Path(args.background)
    if local.exists():
        log.info("background from checkout: %s", local)
        return load_image(local.read_bytes())

    # 2. Contents API fallback (running outside a checkout)
    blob = store.get_store().read_bytes(args.background)
    if blob is None:
        raise ValueError(f"background not found: {args.background}")
    try:
        return load_image(blob)
    except ImageRejected as exc:
        raise ValueError(str(exc)) from exc


def segment(image, model: str):
    """Cut the product out. rembg is imported lazily so --dry-run tests and the
    test suite never pay for it."""
    from rembg import new_session, remove

    started = time.monotonic()
    log.info("loading rembg session: %s", model)
    session = new_session(model)
    log.info("session ready in %.1fs", time.monotonic() - started)

    started = time.monotonic()
    result = remove(image, session=session)
    log.info("inference done in %.1fs", time.monotonic() - started)

    from PIL import Image

    if not isinstance(result, Image.Image):
        import io

        result = Image.open(io.BytesIO(result))
    return result.convert("RGBA")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    from common import config, telegram_api as tg
    from common.imaging import ImageRejected, composite, encode_jpeg, load_image

    model = args.model or config.DEFAULT_MODEL
    size = args.size or config.DEFAULT_SIZE
    started = time.monotonic()

    try:
        log.info("downloading product photo %s", args.file_id)
        raw = tg.download_by_file_id(args.file_id)
        log.info("downloaded %d bytes", len(raw))
        product = load_image(raw)
        log.info("product %dx%d", product.width, product.height)

        cutout = segment(product, model)
        background = load_background(args)
        result = composite(cutout, background, size=size, fill=config.DEFAULT_FILL)
        blob = encode_jpeg(result)
        log.info("result %dx%d, %d bytes", result.width, result.height, len(blob))
    except (ImageRejected, ValueError, tg.TelegramError) as exc:
        log.error("job failed: %s", exc)
        _notify_failure(args.chat_id, str(exc))
        return 1
    except Exception as exc:  # noqa: BLE001 - report anything, then exit non-zero
        log.exception("unexpected failure")
        _notify_failure(args.chat_id, f"внутренняя ошибка: {exc}")
        return 1

    elapsed = int(time.monotonic() - started)
    label = args.label or "фон"
    caption = f"✅ Готово за {elapsed} с\n{label} · {size} × {size} · модель {model}"

    if args.dry_run:
        out = args.out or "result.jpg"
        Path(out).write_bytes(blob)
        log.info("dry run: wrote %s (%d bytes)", out, len(blob))
        return 0

    try:
        tg.send_document(args.chat_id, blob, "product.jpg", caption=caption)
    except tg.TelegramError as exc:
        log.error("failed to send result: %s", exc)
        _notify_failure(args.chat_id, "обработал, но не смог отправить результат в чат")
        return 1
    log.info("sent to chat %s", args.chat_id)
    return 0


def _notify_failure(chat_id: str, text: str) -> None:
    from common import telegram_api as tg

    try:
        tg.send_message(chat_id, f"⚠️ Не получилось обработать фото.\n{text}")
    except tg.TelegramError:
        log.exception("could not notify chat %s about the failure", chat_id)


if __name__ == "__main__":
    sys.exit(main())
