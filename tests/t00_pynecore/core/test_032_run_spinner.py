from datetime import UTC, datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from pynecore.cli.commands import run as run_command
from pynecore.cli.commands.run import (
    CustomTimeElapsedColumn,
    ExchangeClockColumn,
    _exchange_display_time,
)


def __test_exchange_display_time_uses_exchange_timezone__():
    timestamp = int(datetime(2026, 5, 12, 15, 49, tzinfo=UTC).timestamp())

    display_time = _exchange_display_time(timestamp, ZoneInfo("America/New_York"))

    assert display_time == datetime(2026, 5, 12, 11, 49)


def __test_exchange_clock_column_omits_year_and_uses_current_time__(monkeypatch):
    timestamp = datetime(2026, 5, 12, 15, 49, tzinfo=UTC).timestamp()
    monkeypatch.setattr(run_command.time, "time", lambda: timestamp)

    rendered = ExchangeClockColumn(ZoneInfo("America/New_York")).render(SimpleNamespace())

    assert rendered.plain == "Live — 05-12 11:49:00"


def __test_elapsed_column_formats_tenths__():
    rendered = CustomTimeElapsedColumn().render(SimpleNamespace(elapsed=3723.456))

    assert rendered.plain == "01:02:03.5"
