"""
The result broken down by calendar period.

A strategy that made its whole result in one year, or loses on every weekday but
one, shows it here. Trades are assigned by their exit time, in UTC.
"""

from dataclasses import dataclass
from datetime import datetime, UTC
from typing import Any, Literal

from .metrics import Summary, summarize

Period = Literal['year', 'month', 'weekday']

_WEEKDAYS = ('Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun')


@dataclass(slots=True, frozen=True)
class PeriodRow:
    """
    The result of one calendar period.

    :ivar period: The kind of period
    :ivar label: The period: ``2025``, ``2025-03`` or ``Mon``
    :ivar summary: Statistics of the trades that exited in the period
    """

    period: Period
    label: str
    summary: Summary

    def to_dict(self) -> dict[str, Any]:
        """:return: JSON-ready form"""
        return {'period': self.period, 'label': self.label, **self.summary.to_dict()}


def _label(time_ms: int, period: Period) -> tuple[int, str]:
    moment = datetime.fromtimestamp(time_ms / 1000.0, UTC)
    if period == 'year':
        return moment.year, f"{moment.year}"
    if period == 'month':
        return moment.year * 12 + moment.month, f"{moment.year}-{moment.month:02d}"
    return moment.weekday(), _WEEKDAYS[moment.weekday()]


def by_period(values: list[float], times: list[int], period: Period) -> list[PeriodRow]:
    """
    Break the result down by calendar period.

    :param values: Per-trade values
    :param times: Exit time of each trade, Unix milliseconds
    :param period: The kind of period
    :return: One row per period that has trades, in calendar order
    """
    grouped: dict[tuple[int, str], list[float]] = {}
    for value, time_ms in zip(values, times):
        grouped.setdefault(_label(time_ms, period), []).append(value)
    return [PeriodRow(period=period, label=label, summary=summarize(members))
            for (_, label), members in sorted(grouped.items())]
