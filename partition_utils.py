"""Shared helpers for working with integer partition identifiers."""

import ast
from typing import Optional, Tuple


def parse_partition_id(value) -> Optional[Tuple[int, ...]]:
    """Parse a string representation of an integer partition safely."""
    if isinstance(value, tuple) and all(isinstance(part, int) for part in value):
        return value
    if not isinstance(value, str):
        return None

    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return None

    if not isinstance(parsed, tuple):
        return None
    if not all(isinstance(part, int) and part > 0 for part in parsed):
        return None
    if tuple(sorted(parsed, reverse=True)) != parsed:
        return None

    return parsed
