from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from pynecore.cli.commands.run import _exchange_display_time


def __test_exchange_display_time_uses_exchange_timezone__():
    timestamp = int(datetime(2026, 5, 12, 15, 49, tzinfo=UTC).timestamp())

    display_time = _exchange_display_time(timestamp, ZoneInfo("America/New_York"))

    assert display_time == datetime(2026, 5, 12, 11, 49)
