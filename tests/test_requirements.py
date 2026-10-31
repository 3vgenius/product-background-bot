"""Render installs only requirements.txt. Every third-party module the web
process imports must be listed there - a missing one shows up as
"No module named ..." in production, not in tests (which run with the
worker's fuller environment).

Regression for: Pillow used by /bg on Render but only listed in
requirements-worker.txt.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB_FILES = [ROOT / "app.py", *sorted((ROOT / "common").glob("*.py"))]
WORKER_ONLY = {"inpaint"}  # imported by worker.py only; never by the web process

# import name -> pip distribution name (lower case), for the common mismatches
DIST = {"PIL": "pillow", "cv2": "opencv-python-headless", "yaml": "pyyaml"}


def _requirements(path: Path) -> set[str]:
    names = set()
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            names.add(re.split(r"[\[<>=!~ ]", line, 1)[0].lower())
    return names


def _top_level_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_web_process_imports_are_in_requirements_txt():
    required = _requirements(ROOT / "requirements.txt")
    stdlib = set(sys.stdlib_module_names) | {"__future__"}
    missing = {}
    for path in WEB_FILES:
        if path.stem in WORKER_ONLY:
            continue
        for name in _top_level_imports(path):
            if name in stdlib or name == "common":
                continue
            dist = DIST.get(name, name).lower()
            if dist not in required:
                missing.setdefault(dist, []).append(path.name)
    assert not missing, f"used by the Render web process but not in requirements.txt: {missing}"


def test_web_process_never_imports_worker_only_modules():
    for path in WEB_FILES:
        if path.stem in WORKER_ONLY:
            continue
        assert not (_top_level_imports(path) & WORKER_ONLY), path.name
    tree = ast.parse((ROOT / "common" / "updates.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[-1] in WORKER_ONLY:
            raise AssertionError("updates.py must not import the inpainting module")
