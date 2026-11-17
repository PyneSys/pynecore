"""
Timing rules of a live bar stream, shared by the stream and the script runner.

A leaf module: :mod:`~pynecore.core.live_runner` imports
:mod:`~pynecore.core.script_runner`, so a rule both of them apply cannot live in either.
"""

from datetime import datetime, UTC

# noinspection PyProtectedMember
from pynecore.lib.timeframe import _in_seconds, _process_tf

__all__ = ['close_grace_seconds', 'is_catch_up', 'is_stale_update', 'scheduled_close_ms']


def scheduled_close_ms(open_ms: int, timeframe: str) -> int:
    """Return the scheduled close of a bar opening at ``open_ms``, which is the next bar's open.

    A monthly bar closes at the start of the UTC calendar month after its months;
    every other timeframe has a fixed length.

    :param open_ms: The bar's open, Unix milliseconds.
    :param timeframe: Timeframe in TradingView format.
    :return: The close, Unix milliseconds (exclusive).
    """
    modifier, multiplier = _process_tf(timeframe)
    if modifier == 'M':
        opened = datetime.fromtimestamp(open_ms / 1000, UTC)
        months = opened.year * 12 + opened.month - 1 + multiplier
        closes = datetime(months // 12, months % 12 + 1, 1, tzinfo=UTC)
        return int(closes.timestamp() * 1000)
    return open_ms + _in_seconds(timeframe) * 1000


def close_grace_seconds(tf_seconds: float) -> float:
    """Return how long past a bar's close the stream waits for that close.

    Half a period, kept between 15 and 30 seconds: long enough for a venue to publish
    the closed bar to its history endpoint and for one query to fetch it, short enough
    that a long timeframe does not sit on a missing bar.

    :param tf_seconds: The timeframe period in seconds.
    :return: The grace in seconds.
    """
    return max(15.0, min(tf_seconds * 0.5, 30.0))


def is_catch_up(bar_open_ms: int, timeframe: str, now_ms: float) -> bool:
    """Tell whether a closed bar is delivered late enough to be a catch-up bar.

    A bar is fresh while its successor could still be closing on time: up to the
    successor's scheduled close plus the close grace. Past that the bar is a replay of
    what the stream missed (an outage, a slow start), not the live edge.

    :param bar_open_ms: Opening time of the closed bar, in UNIX milliseconds.
    :param timeframe: Timeframe in TradingView format.
    :param now_ms: The current time, in UNIX milliseconds.
    :return: ``True`` when the bar closed too long ago to be the live edge.
    """
    successor_close_ms = scheduled_close_ms(scheduled_close_ms(bar_open_ms, timeframe), timeframe)
    return now_ms - successor_close_ms > close_grace_seconds(_in_seconds(timeframe)) * 1000.0


def is_stale_update(bar_open_ms: int, timeframe: str, now_ms: float) -> bool:
    """Tell whether an intra-bar update is delivered too late to be the live edge.

    A forming bar is the live edge until its own close's deadline: its scheduled close
    plus the close grace. An update processed past that belongs to a slot that has
    closed, typically one a lagging consumer took from a backlog.

    :param bar_open_ms: Opening time of the forming bar, in UNIX milliseconds.
    :param timeframe: Timeframe in TradingView format.
    :param now_ms: The current time, in UNIX milliseconds.
    :return: ``True`` when the update's slot closed too long ago to be the live edge.
    """
    close_ms = scheduled_close_ms(bar_open_ms, timeframe)
    return now_ms - close_ms > close_grace_seconds(_in_seconds(timeframe)) * 1000.0
