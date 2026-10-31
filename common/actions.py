"""Triggering the heavy job on GitHub Actions.

The webhook service does no image work at all: it hands the ``file_id`` to a
workflow run and returns in well under a second. That is what keeps the Render
free tier (512 MB / 0.1 CPU) usable.
"""

from __future__ import annotations

import logging

import requests

from . import config

log = logging.getLogger(__name__)


class DispatchError(RuntimeError):
    pass


def dispatch_workflow(inputs: dict[str, str], *, branch: str | None = None) -> None:
    """POST a workflow_dispatch. All input values must be strings."""
    if not config.GITHUB_TOKEN:
        raise DispatchError("GITHUB_TOKEN is not set")
    branch = branch or config.GITHUB_BRANCH
    url = (
        f"{config.GITHUB_API}/repos/{config.GITHUB_REPO}"
        f"/actions/workflows/{config.WORKFLOW_FILE}/dispatches"
    )
    payload = {
        "ref": branch,
        "inputs": {k: str(v) for k, v in inputs.items() if v is not None},
    }
    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {config.GITHUB_TOKEN}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        json=payload,
        timeout=30,
    )
    if response.status_code not in (200, 201, 204):
        raise DispatchError(
            f"dispatch failed: HTTP {response.status_code} {response.text[:300]}"
        )
    log.info("dispatched %s on %s: %s", config.WORKFLOW_FILE, branch, payload["inputs"])


def minutes_remaining() -> int | None:
    """Best-effort Actions minutes left this month (needs a token with actions:read)."""
    try:
        response = requests.get(
            f"{config.GITHUB_API}/users/{config.GITHUB_REPO.split('/')[0]}/settings/billing/actions",
            headers={
                "Authorization": f"Bearer {config.GITHUB_TOKEN}",
                "Accept": "application/vnd.github+json",
            },
            timeout=15,
        )
        if response.status_code != 200:
            return None
        body = response.json()
        total = body.get("included_minutes", 0)
        used = body.get("total_minutes_used", 0)
        return int(total - used)
    except (requests.RequestException, ValueError, TypeError):
        return None
