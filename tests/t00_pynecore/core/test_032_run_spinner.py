from datetime import UTC, datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from pynecore.cli.commands import run as run_command
from pynecore.cli.commands.run import (
    CustomTimeElapsedColumn,
    ExchangeClockColumn,
    _broker_metrics_text,
    _exchange_display_time,
    _format_run_completion_summary,
)


def __test_exchange_display_time_uses_exchange_timezone__():
    """``_exchange_display_time`` converts a UTC timestamp to the exchange's local wall time."""
    timestamp = int(datetime(2026, 5, 12, 15, 49, tzinfo=UTC).timestamp())

    display_time = _exchange_display_time(timestamp, ZoneInfo("America/New_York"))

    assert display_time == datetime(2026, 5, 12, 11, 49)


def __test_exchange_clock_column_omits_year_and_uses_current_time__(monkeypatch):
    """``ExchangeClockColumn`` renders current exchange time as ``Live — MM-DD HH:MM:SS``."""
    timestamp = datetime(2026, 5, 12, 15, 49, tzinfo=UTC).timestamp()
    monkeypatch.setattr(run_command.time, "time", lambda: timestamp)

    rendered = ExchangeClockColumn(ZoneInfo("America/New_York")).render(SimpleNamespace())

    assert rendered.plain == "Live — 05-12 11:49:00"


def __test_elapsed_column_formats_tenths__():
    """``CustomTimeElapsedColumn`` formats elapsed seconds as ``HH:MM:SS.t`` with tenths."""
    rendered = CustomTimeElapsedColumn().render(SimpleNamespace(elapsed=3723.456))

    assert rendered.plain == "01:02:03.5"


def __test_broker_metrics_text_uses_bid_for_long_unrealized_pnl__():
    """``_broker_metrics_text`` prices a long position's unrealized PnL against the bid."""
    position = SimpleNamespace(
        size=2.0,
        avg_price=100.0,
        netprofit=10.0,
        openprofit=0.0,
        open_trades=[SimpleNamespace(size=2.0, entry_price=100.0)],
    )

    text = _broker_metrics_text(
        position, exchange_position=None,
        balance={"EUR": 996.21}, preferred_currency="EUR",
        price_decimals=5, bid=102.0, ask=103.0, fallback_price=None,
    )

    assert text == (
        "Eq [cyan]996.21 EUR[/] Pos [cyan]2[/] "
        "Entry [cyan]100.00000[/] UPnL [green]+4.00[/]"
    )


def __test_broker_metrics_text_uses_ask_for_short_unrealized_pnl__():
    """``_broker_metrics_text`` prices a short position's unrealized PnL against the ask."""
    position = SimpleNamespace(
        size=-3.0,
        avg_price=100.0,
        netprofit=0.0,
        openprofit=0.0,
        open_trades=[SimpleNamespace(size=-3.0, entry_price=100.0)],
    )

    text = _broker_metrics_text(
        position, exchange_position=None,
        balance={"USDT": 1000.0}, preferred_currency="USDT",
        price_decimals=2, bid=98.0, ask=99.0, fallback_price=None,
    )

    assert text == (
        "Eq [cyan]1,000.00 USDT[/] Pos [cyan]-3[/] "
        "Entry [cyan]100.00[/] UPnL [green]+3.00[/]"
    )


def __test_broker_metrics_text_uses_position_openprofit_without_open_trades__():
    """``_broker_metrics_text`` shows only equity when flat with no open trades."""
    position = SimpleNamespace(
        size=0.0,
        avg_price=0.0,
        netprofit=0.0,
        openprofit=-12.34,
        open_trades=[],
    )

    text = _broker_metrics_text(
        position, exchange_position=None,
        balance={"EUR": 996.21}, preferred_currency="EUR",
        price_decimals=5, bid=102.0, ask=103.0, fallback_price=None,
    )

    assert text == "Eq [cyan]996.21 EUR[/]"


def __test_broker_metrics_text_prefers_exchange_position_snapshot__():
    """``_broker_metrics_text`` favors the exchange position snapshot over the local position."""
    position = SimpleNamespace(
        size=0.0,
        avg_price=0.0,
        netprofit=0.0,
        openprofit=0.0,
        open_trades=[],
    )
    exchange_position = SimpleNamespace(
        side="long",
        size=100.0,
        entry_price=1.17293,
        unrealized_pnl=-0.01,
    )

    text = _broker_metrics_text(
        position, exchange_position=exchange_position,
        balance={"EURd": 999.43}, preferred_currency="EURd",
        price_decimals=5, bid=1.17377, ask=1.17384, fallback_price=None,
    )

    assert text == (
        "Eq [cyan]999.43 EURd[/] Pos [cyan]100[/] "
        "Entry [cyan]1.17293[/] UPnL [red]-0.01[/]"
    )


def __test_broker_metrics_text_signs_short_exchange_position__():
    """``_broker_metrics_text`` renders a short exchange position with a negated size."""
    exchange_position = SimpleNamespace(
        side="short",
        size=25.0,
        entry_price=1.2000,
        unrealized_pnl=3.21,
    )

    text = _broker_metrics_text(
        None, exchange_position=exchange_position,
        balance={"EUR": 1000.0}, preferred_currency="EUR",
        price_decimals=5, bid=1.1900, ask=1.1901, fallback_price=None,
    )

    assert text == (
        "Eq [cyan]1,000.00 EUR[/] Pos [cyan]-25[/] "
        "Entry [cyan]1.20000[/] UPnL [green]+3.21[/]"
    )


def __test_completion_summary_reports_flat_position_and_equity__():
    """A graceful stop with no open exchange position summarizes as ``position=flat``."""
    position = SimpleNamespace(size=0.0)

    summary = _format_run_completion_summary(
        "completed",
        position=position,
        exchange_position=None,
        balance={"USDT": 49998.5},
        preferred_currency="USDT",
    )

    assert summary == "run stopped (completed) position=flat equity=49,998.50 USDT"


def __test_completion_summary_signs_short_exchange_position__():
    """The completion summary negates a short exchange position's size."""
    exchange_position = SimpleNamespace(side="short", size=25.0)

    summary = _format_run_completion_summary(
        "interrupted",
        position=None,
        exchange_position=exchange_position,
        balance={"EUR": 1000.0},
        preferred_currency="EUR",
    )

    assert summary == "run stopped (interrupted) position=-25 equity=1,000.00 EUR"


def __test_completion_summary_survives_missing_state__():
    """With no position or balance the summary is just the reason line — never crashes."""
    summary = _format_run_completion_summary(
        "manual intervention required",
        position=None,
        exchange_position=None,
        balance=None,
        preferred_currency=None,
    )

    assert summary == "run stopped (manual intervention required)"
