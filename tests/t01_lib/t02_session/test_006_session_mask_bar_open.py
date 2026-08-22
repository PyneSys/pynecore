"""
The intraday session mask admits a bar by its OPENING time only.

MEASURED on TradingView (BINANCE:BTCUSDT, ``time(timeframe.period, s)`` over
a full trading day, all seven weekdays enabled):

| chart | session     | bars in session                     |
|-------|-------------|-------------------------------------|
| 30    | "0930-1000" | 09:30                               |
| 30    | "0945-1015" | 10:00                               |
| 30    | "0915-0945" | 09:30                               |
| 30    | "0931-0959" | none                                |
| 30    | "1545-1615" | 16:00                               |
| 30    | "1550-1555" | none                                |
| 30    | "1000-1200" | 10:00, 10:30, 11:00, 11:30          |
| 240   | "0930-1600" | 12:00                               |
| 240   | "0100-0300" | none                                |
| 240   | "2200-0200" | 00:00                               |
| D     | "0930-1600" | none                                |
| D     | "2200-0200" | 00:00                               |

Every row is ``open <= bar_open < close``; an overlap rule would additionally
admit 09:30 for "0945-1015", 09:00 for "0915-0945", 09:30 for both "0931-0959"
and "1550-1555", 15:30 for "1545-1615", 08:00 for the 4h "0930-1600", 20:00 for
the 4h "2200-0200", and the daily bar for "0930-1600".
"""
from datetime import datetime, UTC

from pynecore.lib import _parse_session_string, _is_bar_in_session


def _ms(hour: int, minute: int = 0, day: int = 6) -> int:
    """A 2025-01-06 (Monday) UTC wall clock, in milliseconds."""
    return int(datetime(2025, 1, day, hour, minute, tzinfo=UTC).timestamp() * 1000)


def _hits(session: str, bar_opens: list[tuple[int, int]]) -> list[tuple[int, int]]:
    infos = _parse_session_string(session + ":1234567", "UTC")
    return [hm for hm in bar_opens if _is_bar_in_session(_ms(*hm), infos)]


def __test_half_hour_chart_mask__():
    """Every measured 30-minute row, over that day's half-hour grid."""
    grid = [(h, m) for h in range(24) for m in (0, 30)]
    assert _hits("0930-1000", grid) == [(9, 30)]
    assert _hits("0945-1015", grid) == [(10, 0)]
    assert _hits("0915-0945", grid) == [(9, 30)]
    assert _hits("0931-0959", grid) == []
    assert _hits("1545-1615", grid) == [(16, 0)]
    assert _hits("1550-1555", grid) == []
    assert _hits("1000-1200", grid) == [(10, 0), (10, 30), (11, 0), (11, 30)]


def __test_bar_longer_than_the_session__():
    """A 4-hour and a daily bar are admitted by their opening time too."""
    four_hour = [(h, 0) for h in range(0, 24, 4)]
    assert _hits("0930-1600", four_hour) == [(12, 0)]
    assert _hits("0100-0300", four_hour) == []
    assert _hits("2200-0200", four_hour) == [(0, 0)]

    daily = [(0, 0)]
    assert _hits("0930-1600", daily) == []
    assert _hits("1200-1300", daily) == []
    assert _hits("2200-0200", daily) == [(0, 0)]
