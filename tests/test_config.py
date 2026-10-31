"""Webhook secret normalisation.

Telegram's setWebhook(secret_token=...) accepts only A-Z a-z 0-9 _ - (1-256
chars). Render's ``generateValue: true`` can emit other characters, and in
production that produced ``400 secret token contains illegal characters`` and
a bot with no webhook at all.
"""

from __future__ import annotations

import hashlib
import re

import pytest

from common.config import normalize_webhook_secret

TELEGRAM_RE = re.compile(r"^[A-Za-z0-9_-]{1,256}$")


@pytest.mark.parametrize(
    "value",
    ["s3cret", "abc-DEF_123", "x", "a" * 256],
)
def test_valid_secret_is_kept_verbatim(value):
    assert normalize_webhook_secret(value) == value


def test_empty_secret_stays_empty():
    assert normalize_webhook_secret(None) == ""
    assert normalize_webhook_secret("   ") == ""


@pytest.mark.parametrize(
    "value",
    [
        "Yo1ll+Z8g/4rzhOkWjR1mA==",  # base64 (what Render generated)
        "has space",
        "comma,dot.",
        "a" * 257,  # too long
        "пароль",  # non-ASCII
    ],
)
def test_invalid_secret_is_mapped_to_hex(value):
    out = normalize_webhook_secret(value)
    assert TELEGRAM_RE.match(out), out
    assert out == hashlib.sha256(value.encode()).hexdigest()
    assert len(out) == 64


def test_normalisation_is_deterministic():
    """Registration (setWebhook) and verification (/webhook header check)
    both read config.WEBHOOK_SECRET; the mapping must be stable across
    processes and restarts, not random."""
    assert normalize_webhook_secret("a+b/c=") == normalize_webhook_secret("a+b/c=")


def test_surrounding_whitespace_is_stripped_before_validation():
    assert normalize_webhook_secret("  s3cret\n") == "s3cret"
