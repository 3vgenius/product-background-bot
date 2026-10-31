"""Repository-backed storage.

The repository itself is the database — the same trick the goofish parser uses
for ``filters.json``. There is no SQLite and no persistent disk, which is what
makes the Render *free* tier viable (Render has no free background worker, and a
paid disk would cost more than the compute).

Two backends:

``GithubStore``
    Reads/writes through the Contents API. Used by the Render webhook, which
    has no writable persistent filesystem.
``LocalStore``
    Reads/writes files on disk. Used by the GitHub Actions runner (the repo is
    already checked out) and by local development / the test suite.

``GithubStore`` reads a local file first only when ``prefer_local`` is on. That
is right on a GitHub Actions runner (the checkout *is* the branch head) and
wrong on Render: the checkout there is frozen at deploy time, while writes go
to GitHub, so a local-first read returns stale settings until the next deploy.
Production symptom of that bug: "/once, then a product photo, and the bot asks
for /once again".
"""

from __future__ import annotations

import base64
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

import requests

from . import config

log = logging.getLogger(__name__)

_READ_CACHE_TTL = 5.0  # seconds


class StoreError(RuntimeError):
    pass


# Render auto-deploys the web service on every push to the tracked branch.
# Data commits made by the bot itself (settings, backgrounds) must not trigger
# a rebuild: each one cost a 1-2 minute deploy, during which an in-flight
# update could be killed with the old instance. Render honours this phrase.
_SKIP_DEPLOY = "[skip render]"


def _commit_message(message: str) -> str:
    message = message.strip()
    return message if _SKIP_DEPLOY in message else f"{message} {_SKIP_DEPLOY}"


class LocalStore:
    """Filesystem-backed store rooted at ``config.REPO_DIR``."""

    name = "local"

    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root is not None else config.REPO_DIR
        self._lock = threading.Lock()

    # -- helpers ---------------------------------------------------------
    def _path(self, rel: str) -> Path:
        root = self.root.resolve()
        target = (self.root / rel).resolve()
        # Guard against "../" escaping the checkout via a crafted background path.
        # is_relative_to, not str.startswith: "/tmp/xy" startswith "/tmp/x" is
        # true, which would let a sibling directory through.
        if target != root and not target.is_relative_to(root):
            raise StoreError(f"path escapes repository root: {rel}")
        return target

    # -- API -------------------------------------------------------------
    def read_bytes(self, rel: str) -> bytes | None:
        path = self._path(rel)
        if not path.exists():
            return None
        return path.read_bytes()

    def write_bytes(self, rel: str, blob: bytes, message: str = "") -> None:
        path = self._path(rel)
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(blob)

    def delete(self, rel: str) -> bool:
        path = self._path(rel)
        with self._lock:
            if path.exists():
                path.unlink()
                return True
            return False

    def read_json(self, rel: str, default: Any) -> Any:
        raw = self.read_bytes(rel)
        if raw is None:
            return default
        try:
            return json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            log.warning("corrupt JSON at %s, falling back to default", rel)
            return default

    def write_json(self, rel: str, obj: Any, message: str = "") -> None:
        blob = json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")
        self.write_bytes(rel, blob, message)


class GithubStore(LocalStore):
    """Contents-API store with a local-read fast path."""

    name = "github"

    def __init__(
        self,
        repo: str | None = None,
        token: str | None = None,
        branch: str | None = None,
        root: Path | None = None,
        session: requests.Session | None = None,
        prefer_local: bool | None = None,
    ):
        super().__init__(root)
        self.repo = repo or config.GITHUB_REPO
        self.token = token or config.GITHUB_TOKEN
        self.branch = branch or config.GITHUB_BRANCH
        self.session = session or requests.Session()
        self.prefer_local = (
            config.STORE_PREFER_LOCAL if prefer_local is None else bool(prefer_local)
        )
        self._cache: dict[str, tuple[float, bytes]] = {}

    # -- internals -------------------------------------------------------
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def _contents_url(self, rel: str) -> str:
        return f"{config.GITHUB_API}/repos/{self.repo}/contents/{rel}"

    def _request(self, method: str, url: str, what: str, **kwargs: Any) -> requests.Response:
        """All GitHub traffic goes through here so that a DNS failure, TLS
        reset or timeout surfaces as StoreError (which callers handle and
        report) instead of a raw requests exception (which they do not)."""
        try:
            return self.session.request(method, url, headers=self._headers(), **kwargs)
        except requests.RequestException as exc:
            raise StoreError(f"{what}: transport failure: {exc.__class__.__name__}: {exc}") from exc

    def _api_get(self, rel: str) -> bytes | None:
        url = f"{self._contents_url(rel)}?ref={self.branch}"
        response = self._request("GET", url, f"GET {rel}", timeout=30)
        if response.status_code == 404:
            return None
        if response.status_code != 200:
            raise StoreError(f"GET {rel}: HTTP {response.status_code} {response.text[:200]}")
        body = response.json()
        if body.get("encoding") == "base64" and body.get("content"):
            return base64.b64decode(body["content"])
        # Files between 1 MB and 100 MB come back with encoding "none" and an
        # empty content field; the blob endpoint serves them (base64) by sha.
        # Every real background picture is in that range.
        sha = body.get("sha")
        if not sha:
            raise StoreError(f"GET {rel}: unexpected encoding {body.get('encoding')} and no sha")
        return self._api_blob(rel, sha)

    def _api_blob(self, rel: str, sha: str) -> bytes:
        url = f"{config.GITHUB_API}/repos/{self.repo}/git/blobs/{sha}"
        response = self._request("GET", url, f"BLOB {rel}", timeout=60)
        if response.status_code != 200:
            raise StoreError(f"BLOB {rel}: HTTP {response.status_code} {response.text[:200]}")
        body = response.json()
        if body.get("encoding") != "base64":
            raise StoreError(f"BLOB {rel}: unexpected encoding {body.get('encoding')}")
        return base64.b64decode(body.get("content") or "")

    def _api_sha(self, rel: str) -> str | None:
        url = f"{self._contents_url(rel)}?ref={self.branch}"
        response = self._request("GET", url, f"HEAD {rel}", timeout=30)
        if response.status_code == 404:
            return None
        if response.status_code != 200:
            raise StoreError(f"HEAD {rel}: HTTP {response.status_code} {response.text[:200]}")
        return response.json().get("sha")

    def _api_put(self, rel: str, blob: bytes, message: str, sha: str | None) -> None:
        payload: dict[str, Any] = {
            "message": _commit_message(message or f"update {rel}"),
            "content": base64.b64encode(blob).decode("ascii"),
            "branch": self.branch,
        }
        if sha:
            payload["sha"] = sha
        response = self._request("PUT", self._contents_url(rel), f"PUT {rel}", json=payload, timeout=60)
        if response.status_code not in (200, 201):
            raise StoreError(f"PUT {rel}: HTTP {response.status_code} {response.text[:200]}")

    # -- API -------------------------------------------------------------
    def read_bytes(self, rel: str) -> bytes | None:
        # 1. local checkout — only where the checkout is the branch head
        #    (GitHub Actions). On Render it is frozen at deploy time.
        if self.prefer_local:
            local = super().read_bytes(rel)
            if local is not None:
                return local
        # 2. short-lived cache — repeated reads inside one webhook call
        cached = self._cache.get(rel)
        if cached and (time.monotonic() - cached[0]) < _READ_CACHE_TTL:
            return cached[1]
        blob = self._api_get(rel)
        if blob is not None:
            self._cache[rel] = (time.monotonic(), blob)
        return blob

    def write_bytes(self, rel: str, blob: bytes, message: str = "") -> None:
        if not self.token:
            raise StoreError("GITHUB_TOKEN is not set; cannot write to the repository")
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                self._api_put(rel, blob, message, self._api_sha(rel))
                self._cache.pop(rel, None)
                return
            except StoreError as exc:
                # 409/422 = the file changed under us (two users at once). Retry.
                last_error = exc
                log.warning("write %s attempt %d failed: %s", rel, attempt + 1, exc)
                time.sleep(0.5 * (attempt + 1))
        raise StoreError(f"failed to write {rel} after 3 attempts: {last_error}")

    def delete(self, rel: str) -> bool:
        if not self.token:
            raise StoreError("GITHUB_TOKEN is not set; cannot delete from the repository")
        sha = self._api_sha(rel)
        if not sha:
            return False
        response = self._request(
            "DELETE",
            self._contents_url(rel),
            f"DELETE {rel}",
            json={"message": _commit_message(f"delete {rel}"), "sha": sha, "branch": self.branch},
            timeout=60,
        )
        if response.status_code != 200:
            raise StoreError(f"DELETE {rel}: HTTP {response.status_code} {response.text[:200]}")
        self._cache.pop(rel, None)
        local = self._path(rel)
        if local.exists():
            local.unlink()
        return True


_store: LocalStore | None = None


def get_store(force: str | None = None) -> LocalStore:
    """Return the process-wide store.

    ``force`` is used by tests; otherwise GitHub is used when a token and repo
    are configured, falling back to the local filesystem.
    """
    global _store
    if force == "local":
        return LocalStore()
    if force == "github":
        return GithubStore()
    if _store is None:
        if config.GITHUB_TOKEN and config.GITHUB_REPO:
            _store = GithubStore()
            log.info("storage backend: github (%s@%s)", config.GITHUB_REPO, config.GITHUB_BRANCH)
        else:
            _store = LocalStore()
            log.info("storage backend: local (%s)", config.REPO_DIR)
    return _store


def reset_store() -> None:
    global _store
    _store = None


# --- Domain helpers ---------------------------------------------------------
# Kept here so app.py and worker.py share one definition of the on-disk schema.


def list_backgrounds(store: LocalStore | None = None) -> list[dict]:
    store = store or get_store()
    data = store.read_json(config.BACKGROUNDS_INDEX, [])
    return [item for item in data if isinstance(item, dict) and "id" in item]


def save_backgrounds(items: list[dict], store: LocalStore | None = None, message: str = "") -> None:
    store = store or get_store()
    store.write_json(config.BACKGROUNDS_INDEX, items, message or "update background library")


def next_background_id(items: list[dict]) -> int:
    return max((int(i.get("id", 0)) for i in items), default=0) + 1


def find_background(items: list[dict], background_id: int) -> dict | None:
    for item in items:
        if int(item.get("id", -1)) == background_id:
            return item
    return None


def load_settings(store: LocalStore | None = None) -> dict:
    store = store or get_store()
    data = store.read_json(config.SETTINGS_FILE, {})
    if not isinstance(data, dict):
        return {}
    data.setdefault("users", {})
    if not isinstance(data["users"], dict):
        data["users"] = {}
    return data


def save_settings(settings: dict, store: LocalStore | None = None, message: str = "") -> None:
    store = store or get_store()
    store.write_json(config.SETTINGS_FILE, settings, message or "update settings")


def user_settings(user_id: int, settings: dict) -> dict:
    users = settings.setdefault("users", {})
    entry = users.get(str(user_id))
    if not isinstance(entry, dict):
        entry = {
            "background_id": None,
            "model": config.DEFAULT_MODEL,
            "size": config.DEFAULT_SIZE,
            "pending": None,
            "oneoff": None,
        }
        users[str(user_id)] = entry
    entry.setdefault("background_id", None)
    entry.setdefault("model", config.DEFAULT_MODEL)
    entry.setdefault("size", config.DEFAULT_SIZE)
    entry.setdefault("pending", None)
    entry.setdefault("oneoff", None)
    return entry


def update_user(user_id: int, patch: dict, store: LocalStore | None = None) -> dict:
    """Read-modify-write one user's settings. Returns the updated entry."""
    store = store or get_store()
    settings = load_settings(store)
    entry = user_settings(user_id, settings)
    entry.update(patch)
    save_settings(settings, store, f"settings: user {user_id}")
    return entry


def clear_oneoff_if_matches(user_id: int, file_id: str, store: LocalStore | None = None) -> bool:
    """Forget the user's one-off background, but only if it is still ``file_id``.

    Called by the worker after a successful delivery. The user may already have
    set a *new* one-off while the job was running; that one must survive.
    Reads through the API (never the Actions checkout, which is minutes old).
    """
    store = store or get_store()
    settings = load_settings(store)
    entry = user_settings(user_id, settings)
    if entry.get("oneoff") != file_id:
        return False
    entry["oneoff"] = None
    save_settings(settings, store, message=f"one-off background used by {user_id}")
    return True
