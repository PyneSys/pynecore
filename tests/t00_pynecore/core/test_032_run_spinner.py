from datetime import UTC, datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from pynecore.cli.commands import run as run_command
from pynecore.cli.commands.run import (
    CustomTimeElapsedColumn,
    ExchangeClockColumn,
    _broker_metrics_text,
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


def __test_broker_metrics_text_uses_bid_for_long_unrealized_pnl__():
    position = SimpleNamespace(
        netprofit=10.0,
        openprofit=0.0,
        open_trades=[SimpleNamespace(size=2.0, entry_price=100.0)],
    )

    text = _broker_metrics_text(
        position, balance={"EUR": 996.21}, preferred_currency="EUR",
        bid=102.0, ask=103.0, fallback_price=None,
    )

    assert text == "Eq [cyan]996.21 EUR[/] UPnL [green]+4.00[/]"


def __test_broker_metrics_text_uses_ask_for_short_unrealized_pnl__():
    position = SimpleNamespace(
        netprofit=0.0,
        openprofit=0.0,
        open_trades=[SimpleNamespace(size=-3.0, entry_price=100.0)],
    )

    text = _broker_metrics_text(
        position, balance={"USDT": 1000.0}, preferred_currency="USDT",
        bid=98.0, ask=99.0, fallback_price=None,
    )

    assert text == "Eq [cyan]1,000.00 USDT[/] UPnL [green]+3.00[/]"


def __test_broker_metrics_text_uses_position_openprofit_without_open_trades__():
    position = SimpleNamespace(
        netprofit=0.0,
        openprofit=-12.34,
        open_trades=[],
    )

    text = _broker_metrics_text(
        position, balance={"EUR": 996.21}, preferred_currency="EUR",
        bid=102.0, ask=103.0, fallback_price=None,
    )

    assert text == "Eq [cyan]996.21 EUR[/] UPnL [red]-12.34[/]"
