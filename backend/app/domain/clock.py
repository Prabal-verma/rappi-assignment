"""A single source of 'now'.

Seeds are written relative to this date and every horizon is measured from
it, so a scenario behaves identically whenever it is run.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

_FROZEN: date | None = None


def today() -> date:
    return _FROZEN or datetime.now(timezone.utc).date()


def now() -> datetime:
    return datetime.now(timezone.utc)


def days_from_today(n: int) -> date:
    return today() + timedelta(days=n)


def days_between(start: date, end: date) -> int:
    return (end - start).days


def freeze(d: date | None) -> None:
    """Pin the clock. Used by tests that assert on absolute dates."""
    global _FROZEN
    _FROZEN = d
