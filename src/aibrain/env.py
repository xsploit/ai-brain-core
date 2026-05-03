from __future__ import annotations

import os
from pathlib import Path


def load_env_file(path: str | Path | None, *, override: bool = False) -> None:
    if path is None:
        return
    env_path = Path(path)
    if not env_path.exists():
        raise FileNotFoundError(f"Env file not found: {env_path}")
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[7:].strip()
        if not key:
            continue
        value = _strip_inline_comment(value.strip())
        if (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in {'"', "'"}
        ):
            value = value[1:-1]
        if override or key not in os.environ:
            os.environ[key] = value


def _strip_inline_comment(value: str) -> str:
    quote: str | None = None
    for index, char in enumerate(value):
        if char in {'"', "'"}:
            quote = None if quote == char else char if quote is None else quote
        if char == "#" and quote is None and index > 0 and value[index - 1].isspace():
            return value[:index].rstrip()
    return value
