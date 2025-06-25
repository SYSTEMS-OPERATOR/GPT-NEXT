"""Sentinel helpers for graceful failure."""

import sys
from pathlib import Path


def panic(message: str) -> None:
    """Print error with emoji and exit."""
    print(f"\N{bomb} {message}", file=sys.stderr)
    raise SystemExit(1)


def sanitize_path(path: str) -> str:
    """Resolve ``path`` safely or terminate on failure."""
    try:
        safe_path = Path(path).expanduser().resolve()
    except Exception as exc:  # pragma: no cover - extremely rare failures
        panic(f"Invalid path: {exc}")
    return str(safe_path)
