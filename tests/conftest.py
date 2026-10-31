"""Test-wide guard rails.

CI sandboxes usually have a real GITHUB_TOKEN and sometimes a real bot token in
the environment. Clearing them for every test means a test that forgets to stub
something fails loudly instead of quietly calling the GitHub or Telegram API.
"""

from __future__ import annotations

import pytest

from common import config, store


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path_factory):
    monkeypatch.setattr(config, "GITHUB_TOKEN", "")
    monkeypatch.setattr(config, "GITHUB_REPO", "acme/repo")
    monkeypatch.setattr(config, "GITHUB_BRANCH", "main")
    monkeypatch.setattr(config, "WEBHOOK_SECRET", "s3cret")
    monkeypatch.setattr(config, "BOT_TOKEN", "TOKEN")
    monkeypatch.setattr(config, "REPO_DIR", tmp_path_factory.mktemp("repo"))
    store.reset_store()
    yield
    store.reset_store()
