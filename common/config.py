"""Central configuration.

Everything is read from the environment so the same code runs in three places:
  * Render free web service  (webhook receiver / dispatcher)
  * GitHub Actions runner    (heavy rembg inference)
  * local Docker Compose     (development, polling mode)

Nothing in this module raises on import: missing values surface as empty
strings / empty sets and are validated where they are actually used. That keeps
the modules importable from the test suite without a full environment.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path


def _clean(value: str | None) -> str:
    return (value or "").strip()


_SECRET_ALLOWED = re.compile(r"^[A-Za-z0-9_-]{1,256}$")


def normalize_webhook_secret(raw: str | None) -> str:
    """Return a value Telegram accepts for ``setWebhook(secret_token=...)``.

    Telegram only allows ``A-Z a-z 0-9 _ -`` (1-256 chars). Render's
    ``generateValue: true`` produces strings with other characters, and
    ``setWebhook`` then fails with "secret token contains illegal characters"
    -- which silently leaves the bot without a webhook. Instead of failing, map
    any non-conforming value to its SHA-256 hex digest: deterministic, 64
    chars, hex only. Both registration and request verification read the same
    normalised value, so they stay consistent.
    """
    value = _clean(raw)
    if not value or _SECRET_ALLOWED.match(value):
        return value
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def parse_id_list(raw: str | None) -> set[int]:
    """Parse ``"123, 456"`` into ``{123, 456}``.

    Non-numeric entries are ignored instead of crashing the process: a typo in
    one ID must not take the whole bot down.
    """
    ids: set[int] = set()
    for chunk in (raw or "").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            ids.add(int(chunk))
        except ValueError:
            continue
    return ids


# --- Telegram ---------------------------------------------------------------
BOT_TOKEN = _clean(os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("BOT_TOKEN"))
TELEGRAM_API = _clean(os.getenv("TELEGRAM_API_BASE")) or "https://api.telegram.org"

# Shared secret sent by Telegram in X-Telegram-Bot-Api-Secret-Token. When set,
# updates without it are rejected, so strangers cannot forge webhook calls.
WEBHOOK_SECRET = normalize_webhook_secret(os.getenv("WEBHOOK_SECRET"))

ALLOWED_USER_IDS = parse_id_list(os.getenv("ALLOWED_USER_IDS"))

# webhook | polling | off. Render sets RENDER_EXTERNAL_URL, which is the only
# reliable way to know we are behind a public HTTPS endpoint.
BOT_MODE = _clean(os.getenv("BOT_MODE")).lower()

# --- GitHub -----------------------------------------------------------------
GITHUB_TOKEN = _clean(os.getenv("GITHUB_TOKEN"))
GITHUB_REPO = _clean(os.getenv("GITHUB_REPO")) or "3vgenius/product-background-bot"
GITHUB_API = _clean(os.getenv("GITHUB_API_BASE")) or "https://api.github.com"
# Branch the workflow is dispatched against and that the store writes to.
GITHUB_BRANCH = _clean(os.getenv("GITHUB_BRANCH")) or "main"
WORKFLOW_FILE = _clean(os.getenv("WORKFLOW_FILE")) or "process.yml"

# --- Storage ----------------------------------------------------------------
# Repository checkout root. In GitHub Actions this is the working directory and
# holds the real background files, so the worker never needs the Contents API.
REPO_DIR = Path(_clean(os.getenv("REPO_DIR")) or ".").resolve()

# Read local files before asking the GitHub API. Correct only where the
# checkout is the current branch head: a GitHub Actions runner (GITHUB_ACTIONS
# is "true" there). On Render the checkout is frozen at deploy time while
# writes go to GitHub, so local-first reads would be stale.
_prefer_local_raw = _clean(os.getenv("STORE_PREFER_LOCAL")).lower()
if _prefer_local_raw in ("1", "true", "yes"):
    STORE_PREFER_LOCAL = True
elif _prefer_local_raw in ("0", "false", "no"):
    STORE_PREFER_LOCAL = False
else:
    STORE_PREFER_LOCAL = _clean(os.getenv("GITHUB_ACTIONS")).lower() == "true"

BACKGROUNDS_DIR = "backgrounds"
BACKGROUNDS_INDEX = "backgrounds.json"
SETTINGS_FILE = "settings.json"

# --- Image processing -------------------------------------------------------
MAX_BYTES = 19 * 1024 * 1024  # Bot API download limit is 20 MB
MAX_PIXELS = 25_000_000
# Backgrounds are decoded fully in the Render process before being thumbnailed
# down, so they get a tighter cap: at 25 MP one decode peaks near 275 MB, and
# Flask runs threaded, so two concurrent uploads would blow the 512 MB free-tier
# limit. 12 MP is well above anything normalize_background keeps (max_side 3000
# = 9 MP), so this costs no quality.
MAX_BACKGROUND_PIXELS = 12_000_000
ALLOWED_FORMATS = frozenset({"JPEG", "PNG", "WEBP"})

DEFAULT_SIZE = int(_clean(os.getenv("OUTPUT_SIZE")) or 1024)
DEFAULT_FILL = float(_clean(os.getenv("OBJECT_FILL")) or 0.8)
BACKGROUND_JPEG_QUALITY = 90

# Background cleaning (object removal). LaMa ONNX export; downloaded once on
# the Actions runner and kept in the model cache. INPAINT_BACKEND: auto (LaMa,
# falling back to OpenCV), lama, or opencv.
LAMA_ONNX_URL = _clean(os.getenv("LAMA_ONNX_URL")) or (
    "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx"
)
INPAINT_BACKEND = _clean(os.getenv("INPAINT_BACKEND")).lower() or "auto"
JPEG_QUALITY = int(_clean(os.getenv("JPEG_QUALITY")) or 95)

# Model presets exposed as the "fast / quality" button in the bot.
# NOTE: rembg's own default model is now `bria-rmbg`, whose weights require a
# paid agreement for commercial use. We always pass an explicit model name so
# that default is never picked up by accident.
MODEL_FAST = _clean(os.getenv("MODEL_FAST")) or "isnet-general-use"
MODEL_QUALITY = _clean(os.getenv("MODEL_QUALITY")) or "birefnet-general"
DEFAULT_MODEL = _clean(os.getenv("DEFAULT_MODEL")) or MODEL_FAST

MODELS: dict[str, str] = {
    "fast": MODEL_FAST,
    "quality": MODEL_QUALITY,
}

SIZE_CHOICES = (1024, 1600, 2048)
