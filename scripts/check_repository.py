"""Run lightweight checks that do not require optional GIS dependencies."""

from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IGNORED_PARTS = {".git", ".venv", "venv", "__pycache__"}


def iter_files(pattern: str):
    for path in ROOT.rglob(pattern):
        if not IGNORED_PARTS.intersection(path.parts):
            yield path


def main() -> int:
    errors: list[str] = []

    for path in iter_files("*.py"):
        try:
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        except (OSError, SyntaxError, UnicodeError) as exc:
            errors.append(f"Python: {path.relative_to(ROOT)}: {exc}")

    for path in iter_files("*.json"):
        try:
            json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError, UnicodeError) as exc:
            errors.append(f"JSON: {path.relative_to(ROOT)}: {exc}")

    if errors:
        print("Repository checks failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Repository checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
