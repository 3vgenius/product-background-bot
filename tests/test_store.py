"""Tests for the repository-backed storage layer.

``LocalStore`` is exercised against a real temporary directory. ``GithubStore``
is exercised against a fake ``requests.Session`` so the Contents API payloads
(base64 body, sha handling, conflict retry) are verified without network access.
"""

from __future__ import annotations

import base64
import json

import pytest

from common import config, store


# --- LocalStore -------------------------------------------------------------


def test_local_roundtrip_json(tmp_path):
    obj = store.LocalStore(tmp_path)
    obj.write_json("a.json", {"x": 1, "имя": "фон"})
    assert obj.read_json("a.json", None) == {"x": 1, "имя": "фон"}


def test_local_read_missing_returns_default(tmp_path):
    assert store.LocalStore(tmp_path).read_json("nope.json", {"d": 1}) == {"d": 1}
    assert store.LocalStore(tmp_path).read_bytes("nope.bin") is None


def test_local_corrupt_json_falls_back(tmp_path):
    obj = store.LocalStore(tmp_path)
    obj.write_bytes("bad.json", b"{not json")
    assert obj.read_json("bad.json", []) == []


def test_local_roundtrip_bytes_and_delete(tmp_path):
    obj = store.LocalStore(tmp_path)
    obj.write_bytes("backgrounds/1.png", b"\x89PNG\r\n")
    assert obj.read_bytes("backgrounds/1.png") == b"\x89PNG\r\n"
    assert obj.delete("backgrounds/1.png") is True
    assert obj.delete("backgrounds/1.png") is False


def test_local_rejects_path_traversal(tmp_path):
    obj = store.LocalStore(tmp_path)
    with pytest.raises(store.StoreError):
        obj.read_bytes("../escape.txt")
    with pytest.raises(store.StoreError):
        obj.write_bytes("../escape.txt", b"x")


def test_local_rejects_sibling_prefix_directory(tmp_path):
    """/tmp/x must not accept a path resolving into /tmp/xy."""
    root = tmp_path / "x"
    root.mkdir()
    sibling = tmp_path / "xy"
    sibling.mkdir()
    obj = store.LocalStore(root)
    with pytest.raises(store.StoreError):
        obj.read_bytes("../xy/secret.txt")


def test_local_accepts_nested_paths_inside_root(tmp_path):
    obj = store.LocalStore(tmp_path)
    obj.write_bytes("backgrounds/2_x.png", b"data")
    assert obj.read_bytes("backgrounds/2_x.png") == b"data"


# --- GithubStore ------------------------------------------------------------


class FakeResponse:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text or json.dumps(self._payload)

    def json(self):
        return self._payload


class FakeSession:
    """Records requests and replays a scripted list of responses per method."""

    def __init__(self, responses):
        self.responses = {k: list(v) for k, v in responses.items()}
        self.calls = []

    def _next(self, method):
        queue = self.responses.get(method, [])
        if not queue:
            raise AssertionError(f"unexpected {method} call")
        return queue.pop(0)

    def request(self, method, url, headers=None, timeout=None, json=None):
        # GithubStore funnels everything through session.request; keep the
        # per-method scripts so the tests read naturally.
        if method == "GET":
            return self.get(url, headers=headers, timeout=timeout)
        if method == "PUT":
            return self.put(url, headers=headers, json=json, timeout=timeout)
        if method == "DELETE":
            return self.delete(url, headers=headers, json=json, timeout=timeout)
        raise AssertionError(f"unexpected method {method}")

    def get(self, url, headers=None, timeout=None):
        self.calls.append(("get", url))
        return self._next("get")

    def put(self, url, headers=None, json=None, timeout=None):
        self.calls.append(("put", url, json))
        return self._next("put")

    def delete(self, url, headers=None, json=None, timeout=None):
        self.calls.append(("delete", url, json))
        return self._next("delete")


def _github_store(tmp_path, session, prefer_local=False):
    return store.GithubStore(
        repo="acme/repo",
        token="TOKEN",
        branch="main",
        root=tmp_path,
        session=session,
        prefer_local=prefer_local,
    )


def test_github_read_decodes_base64(tmp_path):
    payload = {"encoding": "base64", "content": base64.b64encode(b"hello").decode()}
    session = FakeSession({"get": [FakeResponse(200, payload)]})
    obj = _github_store(tmp_path, session)
    assert obj.read_bytes("settings.json") == b"hello"
    assert session.calls[0][1].endswith("/repos/acme/repo/contents/settings.json?ref=main")


def test_github_read_404_returns_none(tmp_path):
    session = FakeSession({"get": [FakeResponse(404)]})
    assert _github_store(tmp_path, session).read_bytes("missing.json") is None


def test_github_read_raises_on_server_error(tmp_path):
    session = FakeSession({"get": [FakeResponse(500, text="boom")]})
    with pytest.raises(store.StoreError):
        _github_store(tmp_path, session).read_bytes("x.json")


def test_github_read_prefers_local_checkout_on_actions_runner(tmp_path):
    """The Actions worker must not pay for an API call it does not need."""
    (tmp_path / "backgrounds").mkdir()
    (tmp_path / "backgrounds" / "1.png").write_bytes(b"local-bytes")
    session = FakeSession({})  # any call raises
    got = _github_store(tmp_path, session, prefer_local=True).read_bytes("backgrounds/1.png")
    assert got == b"local-bytes"
    assert session.calls == []


def test_github_read_ignores_stale_local_checkout_on_web_service(tmp_path):
    """Regression for the worst production bug: on Render the checkout is
    frozen at deploy time, writes go to GitHub, and local-first reads returned
    the deploy-time settings.json. /once "did not stick" until Render happened
    to redeploy from the bot's own commit."""
    (tmp_path / "settings.json").write_text('{"users": {"1": {"oneoff": null}}}')
    fresh = {"encoding": "base64", "content": base64.b64encode(b'{"users": {"1": {"oneoff": "F"}}}').decode()}
    session = FakeSession({"get": [FakeResponse(200, fresh)]})
    got = _github_store(tmp_path, session).read_bytes("settings.json")
    assert json.loads(got)["users"]["1"]["oneoff"] == "F"
    assert len(session.calls) == 1


def test_prefer_local_defaults_follow_environment(monkeypatch, tmp_path):
    """Off on Render, on inside GitHub Actions, overridable either way."""
    import importlib

    from common import config

    def reload_with(env):
        for k in ("STORE_PREFER_LOCAL", "GITHUB_ACTIONS"):
            monkeypatch.delenv(k, raising=False)
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        return importlib.reload(config).STORE_PREFER_LOCAL

    try:
        assert reload_with({}) is False
        assert reload_with({"GITHUB_ACTIONS": "true"}) is True
        assert reload_with({"GITHUB_ACTIONS": "true", "STORE_PREFER_LOCAL": "0"}) is False
        assert reload_with({"STORE_PREFER_LOCAL": "1"}) is True
    finally:
        reload_with({})


def test_github_read_falls_back_to_blob_api_for_large_files(tmp_path):
    """Contents API returns encoding "none" and empty content for 1-100 MB
    files - i.e. for every real background picture."""
    big = b"\x89PNG" + b"x" * 2000
    listing = {"encoding": "none", "content": "", "sha": "blobsha", "size": len(big)}
    blob = {"encoding": "base64", "content": base64.b64encode(big).decode()}
    session = FakeSession({"get": [FakeResponse(200, listing), FakeResponse(200, blob)]})
    assert _github_store(tmp_path, session).read_bytes("backgrounds/1.png") == big
    assert session.calls[1][1].endswith("/repos/acme/repo/git/blobs/blobsha")


def test_github_read_large_file_without_sha_is_an_error(tmp_path):
    listing = {"encoding": "none", "content": ""}
    session = FakeSession({"get": [FakeResponse(200, listing)]})
    with pytest.raises(store.StoreError):
        _github_store(tmp_path, session).read_bytes("backgrounds/1.png")


def test_github_blob_api_error_is_a_store_error(tmp_path):
    listing = {"encoding": "none", "content": "", "sha": "blobsha"}
    session = FakeSession({"get": [FakeResponse(200, listing), FakeResponse(403, {"message": "nope"})]})
    with pytest.raises(store.StoreError, match="BLOB"):
        _github_store(tmp_path, session).read_bytes("backgrounds/1.png")


def test_data_commits_skip_render_auto_deploy(tmp_path):
    """Every bot commit used to rebuild the Render service (1-2 min each)."""
    assert store._commit_message("update settings") == "update settings [skip render]"
    assert store._commit_message("x [skip render]") == "x [skip render]"
    session = FakeSession(
        {"get": [FakeResponse(404, {})], "put": [FakeResponse(201, {})]}
    )
    _github_store(tmp_path, session).write_bytes("settings.json", b"{}", "settings: user 1")
    assert session.calls[-1][2]["message"].endswith("[skip render]")
    session = FakeSession({"get": [FakeResponse(200, {"sha": "s"})], "delete": [FakeResponse(200, {})]})
    _github_store(tmp_path, session).delete("backgrounds/1.png")
    assert "[skip render]" in session.calls[-1][2]["message"]


def test_github_read_is_cached(tmp_path):
    payload = {"encoding": "base64", "content": base64.b64encode(b"v1").decode()}
    session = FakeSession({"get": [FakeResponse(200, payload)]})
    obj = _github_store(tmp_path, session)
    assert obj.read_bytes("settings.json") == b"v1"
    assert obj.read_bytes("settings.json") == b"v1"
    assert len([c for c in session.calls if c[0] == "get"]) == 1


def test_github_write_sends_base64_and_sha(tmp_path):
    session = FakeSession(
        {
            "get": [FakeResponse(200, {"sha": "abc123"})],
            "put": [FakeResponse(200, {"content": {"sha": "new"}})],
        }
    )
    obj = _github_store(tmp_path, session)
    obj.write_bytes("settings.json", b'{"a": 1}', "update settings")

    method, url, body = next(c for c in session.calls if c[0] == "put")
    assert url.endswith("/repos/acme/repo/contents/settings.json")
    assert body["sha"] == "abc123"
    assert body["branch"] == "main"
    assert body["message"] == "update settings [skip render]"
    assert base64.b64decode(body["content"]) == b'{"a": 1}'


def test_github_write_creates_file_without_sha(tmp_path):
    session = FakeSession(
        {"get": [FakeResponse(404)], "put": [FakeResponse(201, {"content": {"sha": "n"}})]}
    )
    _github_store(tmp_path, session).write_bytes("new.json", b"{}")
    body = next(c for c in session.calls if c[0] == "put")[2]
    assert "sha" not in body


def test_github_write_retries_on_conflict(tmp_path):
    """Two users pressing a button at once -> 409 -> re-read sha -> success."""
    session = FakeSession(
        {
            "get": [FakeResponse(200, {"sha": "old"}), FakeResponse(200, {"sha": "fresh"})],
            "put": [FakeResponse(409, text="conflict"), FakeResponse(200, {})],
        }
    )
    _github_store(tmp_path, session).write_bytes("settings.json", b"{}")
    body = next(c for c in session.calls if c[0] == "put" and c[2].get("sha") == "fresh")[2]
    assert base64.b64decode(body["content"]) == b"{}"


def test_github_write_gives_up_after_three_attempts(tmp_path):
    session = FakeSession(
        {
            "get": [FakeResponse(200, {"sha": "s"})] * 3,
            "put": [FakeResponse(409, text="conflict")] * 3,
        }
    )
    with pytest.raises(store.StoreError):
        _github_store(tmp_path, session).write_bytes("settings.json", b"{}")


def test_github_write_needs_token(tmp_path, monkeypatch):
    # Without this the store would fall back to config.GITHUB_TOKEN, which is
    # set in CI/sandboxes - and then talk to the real API.
    monkeypatch.setattr(config, "GITHUB_TOKEN", "")
    obj = store.GithubStore(repo="acme/repo", token="", branch="main", root=tmp_path)
    with pytest.raises(store.StoreError, match="GITHUB_TOKEN"):
        obj.write_bytes("x.json", b"{}")


def test_github_delete_sends_sha(tmp_path):
    session = FakeSession(
        {"get": [FakeResponse(200, {"sha": "delme"})], "delete": [FakeResponse(200, {})]}
    )
    assert _github_store(tmp_path, session).delete("backgrounds/1.png") is True
    body = next(c for c in session.calls if c[0] == "delete")[2]
    assert body["sha"] == "delme"


def test_github_delete_missing_returns_false(tmp_path):
    session = FakeSession({"get": [FakeResponse(404)]})
    assert _github_store(tmp_path, session).delete("nope.png") is False


# --- backend selection ------------------------------------------------------


def test_get_store_falls_back_to_local_without_token(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "GITHUB_TOKEN", "")
    monkeypatch.setattr(config, "REPO_DIR", tmp_path)
    store.reset_store()
    assert isinstance(store.get_store(), store.LocalStore)
    store.reset_store()


def test_get_store_uses_github_when_configured(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "GITHUB_TOKEN", "TOKEN")
    monkeypatch.setattr(config, "GITHUB_REPO", "acme/repo")
    store.reset_store()
    assert isinstance(store.get_store(), store.GithubStore)
    store.reset_store()


# --- domain helpers ---------------------------------------------------------


@pytest.fixture()
def local(tmp_path):
    return store.LocalStore(tmp_path)


def test_background_index_starts_empty(local):
    assert store.list_backgrounds(local) == []


def test_next_background_id_increments(local):
    assert store.next_background_id([]) == 1
    assert store.next_background_id([{"id": 4}, {"id": 2}]) == 5


def test_add_find_roundtrip(local):
    items = store.list_backgrounds(local)
    background_id = store.next_background_id(items)
    items.append({"id": background_id, "name": "Белый", "path": "backgrounds/1.png"})
    store.save_backgrounds(items, local)

    assert store.list_backgrounds(local) == [
        {"id": 1, "name": "Белый", "path": "backgrounds/1.png"}
    ]
    assert store.find_background(store.list_backgrounds(local), 1)["name"] == "Белый"
    assert store.find_background(store.list_backgrounds(local), 99) is None


def test_list_backgrounds_skips_garbage_entries(local):
    local.write_json(config.BACKGROUNDS_INDEX, [{"id": 1, "path": "p"}, "junk", 42, {}])
    assert [b["id"] for b in store.list_backgrounds(local)] == [1]


def test_user_settings_creates_defaults(local):
    settings = store.load_settings(local)
    entry = store.user_settings(111, settings)
    assert entry["background_id"] is None
    assert entry["model"] == config.DEFAULT_MODEL
    assert entry["size"] == config.DEFAULT_SIZE
    assert entry["pending"] is None
    assert entry["oneoff"] is None


def test_update_user_persists(local):
    store.update_user(111, {"background_id": 7}, local)
    entry = store.user_settings(111, store.load_settings(local))
    assert entry["background_id"] == 7
    # Another user is untouched.
    assert store.user_settings(222, store.load_settings(local))["background_id"] is None


def test_load_settings_recovers_from_corruption(local):
    local.write_bytes(config.SETTINGS_FILE, b"not json")
    settings = store.load_settings(local)
    assert settings == {"users": {}}


def test_load_settings_heals_bad_shape(local):
    local.write_json(config.SETTINGS_FILE, {"users": "oops"})
    assert store.load_settings(local) == {"users": {}}


def test_github_transport_failures_are_store_errors(tmp_path):
    """A DNS failure / TLS reset / timeout must surface as StoreError so the
    user gets a message, not as a raw requests exception nobody catches.
    Found by running the real server against an unreachable api.github.com."""
    import requests

    class DeadSession:
        def request(self, *a, **k):
            raise requests.ConnectionError("Max retries exceeded with url: /repos/x")

    s = store.GithubStore(repo="acme/repo", token="T", branch="main", root=tmp_path, session=DeadSession())
    with pytest.raises(store.StoreError, match="transport failure"):
        s.read_bytes("settings.json")
    with pytest.raises(store.StoreError, match="transport failure"):
        s.write_bytes("settings.json", b"{}", "m")
    with pytest.raises(store.StoreError, match="transport failure"):
        s.delete("backgrounds/1.jpg")
