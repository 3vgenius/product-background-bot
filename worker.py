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
    parser.add_argument(
        "--task",
        choices=("replace", "clean"),
        default="replace",
        help="replace = put a product on a background; clean = remove objects from a background",
    )
    parser.add_argument("--chat-id", required=True)
    parser.add_argument("--user-id", default="")
    parser.add_argument("--file-id", default="", help="Telegram file_id of the product photo")
    parser.add_argument("--background-id", default="", help="library id, for --task clean")
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
    args = parser.parse_args(argv)
    if args.task == "replace" and not args.file_id:
        parser.error("--file-id is required for --task replace")
    if args.task == "clean" and not (args.background and args.background_id):
        parser.error("--background and --background-id are required for --task clean")
    return args


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
    if args.task == "clean":
        return clean_main(args)

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
    if args.oneoff_file_id:
        _forget_oneoff(args.user_id, args.oneoff_file_id)
    return 0


def _forget_oneoff(user_id: str, file_id: str) -> None:
    """The result is already in the chat; a failure here must not fail the
    job. Worst case the next product photo reuses the one-off once more."""
    from common import config, store

    if not user_id or not config.GITHUB_TOKEN:
        log.warning("cannot clear one-off: no user id or no GITHUB_TOKEN")
        return
    try:
        # prefer_local=False: the Actions checkout is minutes old; settings
        # must be read fresh so a model/size change made meanwhile survives.
        github = store.GithubStore(prefer_local=False)
        if store.clear_oneoff_if_matches(int(user_id), file_id, github):
            log.info("one-off background cleared for user %s", user_id)
        else:
            log.info("one-off already replaced or cleared for user %s", user_id)
    except Exception as exc:  # noqa: BLE001
        log.warning("could not clear one-off for user %s: %s", user_id, exc)


# --- task: clean --------------------------------------------------------------


def clean_main(args: argparse.Namespace) -> int:
    """Remove the objects from a library background, store the cleaned copy
    next to the original and show the user the result."""
    from common import config, store, telegram_api as tg
    from common.imaging import ImageRejected, encode_jpeg, load_image, make_preview
    from common.inpaint import NothingToClean, RefusedToClean, clean_background

    model = args.model or config.DEFAULT_MODEL
    started = time.monotonic()
    try:
        local = config.REPO_DIR / args.background
        if local.exists():
            raw = local.read_bytes()
        else:
            blob = store.get_store().read_bytes(args.background)
            if blob is None:
                raise ValueError(f"background not found: {args.background}")
            raw = blob
        image = load_image(raw)
        log.info("background %s: %dx%d", args.background, image.width, image.height)
        cleaned, filled, backend = clean_background(image, model, config.INPAINT_BACKEND)
        blob = encode_jpeg(cleaned, quality=config.BACKGROUND_JPEG_QUALITY)
    except NothingToClean as exc:
        log.info("nothing to clean: %s", exc)
        _say(args.chat_id, f"ℹ️ {exc}")
        return 0
    except (RefusedToClean, ImageRejected, ValueError) as exc:
        log.error("clean failed: %s", exc)
        _say(args.chat_id, f"⚠️ Не стал очищать фон.\n{exc}")
        return 1
    except Exception as exc:  # noqa: BLE001
        log.exception("unexpected failure while cleaning")
        _say(args.chat_id, f"⚠️ Не получилось очистить фон: внутренняя ошибка: {exc}")
        return 1

    if args.dry_run:
        out = args.out or "cleaned.jpg"
        Path(out).write_bytes(blob)
        log.info("dry run: wrote %s (%d bytes, %.1f%% filled, %s)", out, len(blob), filled * 100, backend)
        return 0

    try:
        item = _store_cleaned(int(args.background_id), args.background, blob)
    except Exception as exc:  # noqa: BLE001
        log.exception("could not store the cleaned background")
        _say(args.chat_id, f"⚠️ Фон очистил, но не смог сохранить в репозиторий: {exc}")
        return 1

    elapsed = int(time.monotonic() - started)
    engine = "нейросеть LaMa" if backend == "lama" else "OpenCV (запасной алгоритм)"
    caption = (
        f"🧹 Фон №{item['id']} «{item['name']}» очищен за {elapsed} с.\n"
        f"Заполнено {filled:.0%} кадра · {engine}.\n"
        "Товары теперь будут ставиться на эту версию. Не понравилось — верните оригинал."
    )
    keyboard = {
        "inline_keyboard": [
            [{"text": "↩️ Вернуть оригинал", "callback_data": f"bgorig:{item['id']}"}]
        ]
    }
    try:
        tg.send_photo(args.chat_id, make_preview(blob, 1280), caption=caption, reply_markup=keyboard)
    except tg.TelegramError as exc:
        log.error("failed to send the cleaned preview: %s", exc)
        _say(args.chat_id, "Фон очищен и сохранён, но превью отправить не удалось. Посмотрите /backgrounds.")
    log.info("cleaned background %s stored as %s", args.background_id, item["path"])
    return 0


def _store_cleaned(background_id: int, original_path: str, blob: bytes) -> dict:
    """Commit the cleaned picture and point the library entry at it. The
    original stays in the repo under ``original_path`` so it can be restored.
    Reads through the API: the Actions checkout is minutes old."""
    from common import config, store

    github = store.GithubStore(prefer_local=False)
    stem = Path(original_path).stem
    if stem.endswith("_clean"):
        stem = stem[: -len("_clean")]
    new_path = f"{config.BACKGROUNDS_DIR}/{stem}_clean.jpg"
    github.write_bytes(new_path, blob, f"background {background_id} cleaned")

    items = store.list_backgrounds(github)
    item = store.find_background(items, background_id)
    if not item:
        raise ValueError(f"background {background_id} vanished from the index")
    if not item.get("original_path"):
        item["original_path"] = item["path"]
    item["path"] = new_path
    item["cleaned_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    store.save_backgrounds(items, github, message=f"background {background_id} cleaned")
    return item


def _say(chat_id: str, text: str) -> None:
    from common import telegram_api as tg

    try:
        tg.send_message(chat_id, text)
    except tg.TelegramError:
        log.exception("could not message chat %s", chat_id)


def _notify_failure(chat_id: str, text: str) -> None:
    from common import telegram_api as tg

    try:
        tg.send_message(chat_id, f"⚠️ Не получилось обработать фото.\n{text}")
    except tg.TelegramError:
        log.exception("could not notify chat %s about the failure", chat_id)


if __name__ == "__main__":
    sys.exit(main())
