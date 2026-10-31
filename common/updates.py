"""Update handling: commands, callbacks and the product-photo entry point.

Shared by the Render webhook (``app.py``) and the local polling loop, so both
modes behave identically. Everything here is synchronous and cheap: the only
network calls are a few Bot API requests plus at most one workflow dispatch.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from . import actions, config, store
from . import telegram_api as tg

log = logging.getLogger(__name__)

HELP_TEXT = (
    "Я заменяю фон на фотографиях товаров.\n\n"
    "Как пользоваться:\n"
    "1. /bg — затем пришлите картинку фона (или сразу картинку с подписью /bg), она попадёт в общую библиотеку\n"
    "2. /backgrounds — посмотреть библиотеку и выбрать фон себе\n"
    "3. Просто пришлите фото товара — я вырежу его и поставлю на выбранный фон\n\n"
    "Другие команды:\n"
    "/once — разовый фон: затем пришлите картинку фона, она сработает только на следующее фото товара\n"
    "/settings — качество вырезания и размер результата\n"
    "/cancel — сбросить незавершённый сценарий\n\n"
    "Вырезание считается на GitHub Actions, поэтому результат приходит примерно через минуту. "
    "Для лучшего качества отправляйте исходник как файл, а не как фото Telegram."
)

PROCESSING_TEXT = (
    "⏳ Принял, обрабатываю на отдельном сервере. Результат придёт сюда примерно через минуту."
)


class AccessDenied(Exception):
    pass


def authorized(user_id: int | None) -> bool:
    if user_id is None:
        return False
    if not config.ALLOWED_USER_IDS:
        # No allowlist configured means "not configured yet", not "everyone in".
        return False
    return user_id in config.ALLOWED_USER_IDS


# --- small helpers ----------------------------------------------------------


def _image_from_message(message: dict) -> dict | None:
    """Return the largest photo / an image document, or None."""
    photos = message.get("photo") or []
    if photos:
        best = max(photos, key=lambda p: int(p.get("file_size") or 0))
        return {
            "file_id": best["file_id"],
            "size": int(best.get("file_size") or 0),
            "kind": "photo",
        }
    document = message.get("document")
    if document:
        mime = (document.get("mime_type") or "").lower()
        if mime not in ("image/jpeg", "image/png", "image/webp"):
            return None
        return {
            "file_id": document["file_id"],
            "size": int(document.get("file_size") or 0),
            "kind": "document",
        }
    return None


def _text_of(message: dict) -> str:
    return (message.get("text") or message.get("caption") or "").strip()


def _command_of(text: str) -> tuple[str, str]:
    """Split ``/cmd@BotName rest`` into (``cmd``, ``rest``)."""
    if not text.startswith("/"):
        return "", text
    head, _, rest = text.partition(" ")
    cmd = head.split("@", 1)[0].lower().lstrip("/")
    return cmd, rest.strip()


def _preview_bytes(store_obj: store.LocalStore, item: dict) -> bytes | None:
    from . import imaging

    blob = store_obj.read_bytes(item["path"])
    if blob is None:
        return None
    try:
        return imaging.make_preview(blob, 640)
    except imaging.ImageRejected:
        log.warning("background %s is unreadable", item.get("id"))
        return None


# --- background library -----------------------------------------------------


def add_background(user_id: int, file_id: str, name: str = "") -> dict:
    """Download the image, normalise it, commit it to the repo, index it."""
    from . import imaging

    blob = tg.download_by_file_id(file_id)
    normalized = imaging.normalize_background(blob)
    items = store.list_backgrounds()
    background_id = store.next_background_id(items)
    filename = f"{background_id}_{int(time.time())}.png"
    path = f"{config.BACKGROUNDS_DIR}/{filename}"
    store_obj = store.get_store()
    store_obj.write_bytes(
        path, normalized, f"background {background_id} added by {user_id}"
    )
    item = {
        "id": background_id,
        "name": name.strip()[:60] or f"Фон {background_id}",
        "path": path,
        "added_by": user_id,
        "added_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    items.append(item)
    store.save_backgrounds(items, message=f"background {background_id} added")
    store.update_user(
        user_id, {"background_id": background_id, "pending": None, "pending_name": None}
    )
    return item


def delete_background(background_id: int) -> bool:
    store_obj = store.get_store()
    items = store.list_backgrounds()
    item = store.find_background(items, background_id)
    if not item:
        return False
    store_obj.delete(item["path"])
    items = [i for i in items if int(i.get("id", -1)) != background_id]
    store.save_backgrounds(items, message=f"background {background_id} removed")
    return True


# --- rendering --------------------------------------------------------------


def _settings_keyboard(entry: dict) -> dict:
    model_rows = [
        [
            {
                "text": ("✅ " if entry["model"] == config.MODEL_FAST else "") + "⚡ Быстро",
                "callback_data": "model:fast",
            },
            {
                "text": ("✅ " if entry["model"] == config.MODEL_QUALITY else "") + "💎 Качество",
                "callback_data": "model:quality",
            },
        ]
    ]
    size_row = []
    for size in config.SIZE_CHOICES:
        prefix = "✅ " if int(entry["size"]) == size else ""
        size_row.append({"text": f"{prefix}{size}", "callback_data": f"size:{size}"})
    return {
        "inline_keyboard": [
            model_rows[0],
            size_row,
            [{"text": "⬅️ Закрыть", "callback_data": "noop"}],
        ]
    }


def _describe(entry: dict) -> str:
    items = store.list_backgrounds()
    current = store.find_background(items, int(entry["background_id"])) if entry["background_id"] else None
    model_label = "💎 Качество" if entry["model"] == config.MODEL_QUALITY else "⚡ Быстро"
    oneoff = "\n📌 Задан разовый фон — он сработает на следующее фото." if entry["oneoff"] else ""
    return (
        "Настройки\n\n"
        f"Фон: {current['name'] if current else 'не выбран'}\n"
        f"Вырезание: {model_label}\n"
        f"Размер: {entry['size']} × {entry['size']}{oneoff}"
    )


def send_gallery(chat_id: int, user_id: int) -> None:
    store_obj = store.get_store()
    items = store.list_backgrounds()
    settings = store.load_settings()
    entry = store.user_settings(user_id, settings)
    if not items:
        tg.send_message(
            chat_id,
            "Фонов пока нет. Пришлите картинку фона вместе с командой /bg — "
            "она попадёт в общую библиотеку.",
        )
        return
    current_id = entry["background_id"]
    tg.send_message(
        chat_id,
        f"Общих фонов: {len(items)}. Текущий: "
        + (f"№{current_id}" if current_id else "не выбран")
        + ". Нажмите «Выбрать» под нужным.",
    )
    for item in items[-10:]:
        blob = _preview_bytes(store_obj, item)
        keyboard = {
            "inline_keyboard": [
                [
                    {
                        "text": "✅ Выбран"
                        if current_id == item["id"]
                        else "Выбрать",
                        "callback_data": f"bg:{item['id']}",
                    },
                    {"text": "🗑", "callback_data": f"bgdel:{item['id']}"},
                ]
            ]
        }
        caption = f"№{item['id']} — {item['name']}"
        if blob is None:
            tg.send_message(chat_id, f"Не удалось показать {caption}.")
            continue
        try:
            tg.send_photo(chat_id, blob, caption=caption, reply_markup=keyboard)
        except tg.TelegramError:
            log.exception("failed to send preview for background %s", item["id"])
            tg.send_message(chat_id, f"Не удалось показать {caption}.")


# --- dispatch ---------------------------------------------------------------


def process_product(chat_id: int, user_id: int, file_id: str) -> None:
    """Hand the job to GitHub Actions."""
    store_obj = store.get_store()
    settings = store.load_settings()
    entry = store.user_settings(user_id, settings)

    oneoff_file_id = entry.get("oneoff") or ""
    background_path = ""
    background_label = "разовый фон"

    if not oneoff_file_id:
        background_id = entry.get("background_id")
        item = (
            store.find_background(store.list_backgrounds(), int(background_id))
            if background_id
            else None
        )
        if not item:
            tg.send_message(
                chat_id,
                "Фон ещё не выбран, поэтому это фото я не обрабатывал.\n\n"
                "Сделайте одно из трёх и пришлите фото товара ещё раз:\n"
                "• /bg — затем отправьте картинку фона (она сохранится в библиотеку)\n"
                "• /backgrounds — выбрать из уже сохранённых\n"
                "• /once — затем картинку фона, только для одного следующего фото",
            )
            return
        background_path = item["path"]
        background_label = item["name"]

    try:
        actions.dispatch_workflow(
            {
                "chat_id": str(chat_id),
                "user_id": str(user_id),
                "file_id": file_id,
                "background": background_path,
                "oneoff_file_id": oneoff_file_id,
                "model": str(entry.get("model") or config.DEFAULT_MODEL),
                "size": str(entry.get("size") or config.DEFAULT_SIZE),
                "label": background_label[:80],
            }
        )
    except actions.DispatchError as exc:
        log.error("dispatch failed: %s", exc)
        tg.send_message(
            chat_id,
            "Не удалось запустить обработку. Проверьте, что воркфлоу process.yml "
            "уже в ветке main и у токена есть права Actions.",
        )
        return

    # A one-off background is consumed as soon as the job is queued.
    if oneoff_file_id:
        store.save_settings(
            {**settings, "users": {**settings["users"], str(user_id): {**entry, "oneoff": None}}},
            message="one-off background consumed",
        )
    tg.send_message(chat_id, PROCESSING_TEXT)


# --- message handling -------------------------------------------------------


def handle_message(message: dict) -> None:
    sender = message.get("from") or {}
    user_id = sender.get("id")
    chat_id = message["chat"]["id"]
    if not authorized(user_id):
        tg.send_message(
            chat_id,
            "Доступ к боту ограничен. Попросите администратора добавить ваш Telegram ID.",
        )
        return

    text = _text_of(message)
    command, rest = _command_of(text)
    image = _image_from_message(message)
    store_obj = store.get_store()
    settings = store.load_settings()
    entry = store.user_settings(user_id, settings)

    if command == "start" or command == "help":
        tg.send_message(chat_id, HELP_TEXT)
        return

    if command == "cancel":
        store.update_user(user_id, {"pending": None, "pending_name": None, "oneoff": None})
        tg.send_message(chat_id, "Сбросил незавершённый сценарий и разовый фон.")
        return

    if command == "backgrounds":
        send_gallery(chat_id, user_id)
        return

    if command == "settings":
        tg.send_message(chat_id, _describe(entry), reply_markup=_settings_keyboard(entry))
        return

    if command == "bg" or command == "add_background":
        if image is None:
            # Remember the intent: on a phone it is far easier to send the
            # command first and the picture second than to type a caption.
            # Without this, the follow-up picture was treated as a product
            # photo and the user was told to "add a background via /bg" - a loop.
            store.update_user(user_id, {"pending": "add_background", "pending_name": rest[:60]})
            tg.send_message(
                chat_id,
                "Жду картинку фона — пришлите её следующим сообщением (фото или файл). "
                "Она попадёт в общую библиотеку и сразу станет вашим фоном. Отмена: /cancel.",
            )
            return
        try:
            item = add_background(user_id, image["file_id"], rest)
        except (tg.TelegramError, ValueError) as exc:
            tg.send_message(chat_id, f"Не получилось добавить фон: {exc}")
            return
        tg.send_message(
            chat_id,
            f"✅ Фон №{item['id']} «{item['name']}» добавлен в общую библиотеку и выбран для вас.",
        )
        return

    if command == "once":
        if image is None:
            store.update_user(user_id, {"pending": "once"})
            tg.send_message(
                chat_id,
                "Отправьте следующее изображение — это будет разовый фон для одного фото товара. "
                "Отмена: /cancel.",
            )
            return
        store.update_user(user_id, {"oneoff": image["file_id"], "pending": None})
        tg.send_message(
            chat_id,
            "📌 Разовый фон принят. Теперь пришлите фото товара — фон не попадёт в библиотеку.",
        )
        return

    if command:
        tg.send_message(chat_id, "Не знаю такой команды. /start — список того, что я умею.")
        return

    # No command: either a pending flow or a product photo.
    pending = entry.get("pending")
    if image is None:
        tg.send_message(chat_id, "Пришлите фото товара или нажмите /start.")
        return

    if pending == "add_background":
        # A caption on the picture itself wins over the name given with /bg.
        name = text or str(entry.get("pending_name") or "")
        try:
            item = add_background(user_id, image["file_id"], name)
        except (tg.TelegramError, ValueError) as exc:
            tg.send_message(chat_id, f"Не получилось добавить фон: {exc}")
            return
        tg.send_message(
            chat_id,
            f"✅ Фон №{item['id']} «{item['name']}» добавлен в общую библиотеку и выбран для вас. "
            "Теперь пришлите фото товара.",
        )
        return

    if pending == "once":
        store.update_user(user_id, {"oneoff": image["file_id"], "pending": None})
        tg.send_message(
            chat_id,
            "📌 Разовый фон принят. Теперь пришлите фото товара.",
        )
        return

    process_product(chat_id, user_id, image["file_id"])


# --- callback handling ------------------------------------------------------


def handle_callback(query: dict) -> None:
    sender = query.get("from") or {}
    user_id = sender.get("id")
    callback_id = query["id"]
    if not authorized(user_id):
        tg.answer_callback(callback_id, "Доступ запрещён.", show_alert=True)
        return

    data = query.get("data") or ""
    message = query.get("message") or {}
    chat_id = message.get("chat", {}).get("id")
    message_id = message.get("message_id")

    if data == "noop":
        tg.answer_callback(callback_id)
        return

    if data.startswith("bg:"):
        try:
            background_id = int(data.split(":", 1)[1])
        except ValueError:
            tg.answer_callback(callback_id, "Некорректный фон.", show_alert=True)
            return
        if not store.find_background(store.list_backgrounds(), background_id):
            tg.answer_callback(callback_id, "Этот фон больше недоступен.", show_alert=True)
            return
        store.update_user(user_id, {"background_id": background_id})
        tg.answer_callback(callback_id, f"Выбран фон №{background_id}")
        if chat_id and message_id:
            tg.edit_message(
                chat_id,
                message_id,
                (message.get("caption") or message.get("text") or "Фон")
                + f"\n✅ Выбран пользователем {user_id}",
                reply_markup=message.get("reply_markup"),
            )
        return

    if data.startswith("bgdel:"):
        try:
            background_id = int(data.split(":", 1)[1])
        except ValueError:
            tg.answer_callback(callback_id, "Некорректный фон.", show_alert=True)
            return
        removed = delete_background(background_id)
        tg.answer_callback(
            callback_id,
            f"Фон №{background_id} удалён" if removed else "Фон не найден",
            show_alert=not removed,
        )
        return

    if data.startswith("model:"):
        preset = data.split(":", 1)[1]
        model = config.MODELS.get(preset)
        if not model:
            tg.answer_callback(callback_id, "Неизвестный режим.", show_alert=True)
            return
        entry = store.update_user(user_id, {"model": model})
        tg.answer_callback(callback_id, f"Режим: {model}")
        if chat_id and message_id:
            tg.edit_message(
                chat_id, message_id, _describe(entry), reply_markup=_settings_keyboard(entry)
            )
        return

    if data.startswith("size:"):
        try:
            size = int(data.split(":", 1)[1])
        except ValueError:
            tg.answer_callback(callback_id, "Некорректный размер.", show_alert=True)
            return
        if size not in config.SIZE_CHOICES:
            tg.answer_callback(callback_id, "Недоступный размер.", show_alert=True)
            return
        entry = store.update_user(user_id, {"size": size})
        tg.answer_callback(callback_id, f"Размер {size}")
        if chat_id and message_id:
            tg.edit_message(
                chat_id, message_id, _describe(entry), reply_markup=_settings_keyboard(entry)
            )
        return

    tg.answer_callback(callback_id, "Не знаю такой кнопки.", show_alert=True)


# --- entry point ------------------------------------------------------------


def handle_update(update: dict[str, Any]) -> None:
    """Dispatch one Telegram update. Never raises: one bad update must not
    break polling or make Telegram drop the webhook."""
    try:
        if "callback_query" in update:
            handle_callback(update["callback_query"])
        elif "message" in update:
            handle_message(update["message"])
        else:
            log.debug("ignoring update without message/callback: %s", list(update))
    except tg.TelegramError:
        log.exception("Telegram API error while handling update %s", update.get("update_id"))
    except store.StoreError as exc:
        log.exception("storage error while handling update %s", update.get("update_id"))
        _reply_error(
            update,
            "⚠️ Не удалось сохранить данные в репозиторий GitHub.\n"
            f"{_short(exc)}\n\n"
            "Обычно это GITHUB_TOKEN в Render: нужен classic-токен с правами "
            "repo и workflow. Фон/настройка не сохранились, повторите после исправления.",
        )
    except Exception as exc:
        log.exception("unexpected error while handling update %s", update.get("update_id"))
        _reply_error(
            update,
            f"⚠️ Внутренняя ошибка бота: {_short(exc)}\nПодробности в логах Render.",
        )


def _chat_id_of(update: dict[str, Any]) -> int | None:
    """Best-effort chat id for an error reply; None if the update has none."""
    message = update.get("message") or (update.get("callback_query") or {}).get("message") or {}
    chat = message.get("chat") or {}
    return chat.get("id")


def _short(exc: BaseException, limit: int = 300) -> str:
    text = str(exc) or exc.__class__.__name__
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _reply_error(update: dict[str, Any], text: str) -> None:
    """Tell the user something went wrong. Must never raise: this runs inside
    the error handler, and a second failure would only hide the first."""
    chat_id = _chat_id_of(update)
    if chat_id is None:
        return
    try:
        tg.send_message(chat_id, text)
    except Exception:  # noqa: BLE001 - last line of defence
        log.warning("could not deliver error message to chat %s", chat_id)
