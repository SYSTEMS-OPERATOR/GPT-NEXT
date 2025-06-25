"""Sentinel helpers for graceful failure."""

import sys


def panic(message: str) -> None:
    """Print error with emoji and exit."""
    print(f"\N{bomb} {message}", file=sys.stderr)
    raise SystemExit(1)
