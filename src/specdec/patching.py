"""Reversible in-place patching.

Repo rule: every model patch snapshots what it changes and restores on exit.
All attribute swaps go through this context manager; nothing patches by hand.
"""

from typing import Any


class Patched:
    """Collects attribute swaps and restores them in reverse order on exit."""

    def __init__(self) -> None:
        self._undo: list[tuple[Any, str, Any, bool]] = []

    def attr(self, holder: Any, name: str, value: Any) -> None:
        had = hasattr(holder, name)
        old = getattr(holder, name) if had else None
        self._undo.append((holder, name, old, had))
        setattr(holder, name, value)

    def __enter__(self) -> "Patched":
        return self

    def __exit__(self, *exc) -> None:
        while self._undo:
            holder, name, old, had = self._undo.pop()
            if had:
                setattr(holder, name, old)
            else:
                delattr(holder, name)
