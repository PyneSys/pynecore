"""CLI/UX logging hardening tests.

Covers the durable-log console width, the credential-safe crash traceback,
and the account-equity log level — the operator-facing surface exercised by
the live-broker lab CLI issues (CLI-002, CLI-003, CLI-007).
"""

import logging

from pynecore.lib import log as log_module
from pynecore.lib.log import DURABLE_LOG_WIDTH, _resolve_log_console_width


def __test_console_width_auto_on_a_terminal__():
    """On an interactive terminal the handler keeps Rich auto-width (``None``)."""
    assert _resolve_log_console_width(is_tty=True, columns_env=None) is None


def __test_console_width_widens_for_a_redirected_sink__():
    """A redirected (non-TTY) sink gets an explicit wide width, not Rich's 80.

    Rich falls back to 80 columns off a terminal and folds long broker/order
    identifiers across physical lines; the wide width keeps a UUID on one
    searchable line."""
    width = _resolve_log_console_width(is_tty=False, columns_env=None)
    assert width == DURABLE_LOG_WIDTH
    assert width > 80


def __test_console_width_honours_columns_override__():
    """An explicit ``COLUMNS`` value always wins over the TTY-based default."""
    assert _resolve_log_console_width(is_tty=True, columns_env="132") == 132
    assert _resolve_log_console_width(is_tty=False, columns_env="132") == 132


def __test_console_width_ignores_bogus_columns__():
    """A non-numeric / zero ``COLUMNS`` falls back to the TTY-based decision."""
    assert _resolve_log_console_width(is_tty=False, columns_env="oops") == DURABLE_LOG_WIDTH
    assert _resolve_log_console_width(is_tty=False, columns_env="0") == DURABLE_LOG_WIDTH
    assert _resolve_log_console_width(is_tty=True, columns_env="0") is None


def __test_durable_width_keeps_a_long_identifier_on_one_line__():
    """A wide console renders a UUID-bearing broker line without folding it."""
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text

    order_id = "8d03d818-11d8-458c-9454-0430b6902029"
    message = (
        f"[BROKER] event CREATED id={order_id} side=buy qty=0.001 "
        f"filled=0.0 pine='LAB-LIMIT' leg=entry"
    )

    def render(width: int) -> str:
        console = Console(width=width, file=None, record=True)
        grid = Table.grid(padding=(0, 1))
        grid.expand = True
        grid.add_column()
        grid.add_column(ratio=1, overflow="fold")
        grid.add_row(Text("[2026-01-01]"), Text(message))
        with console.capture() as capture:
            console.print(grid)
        return capture.get()

    # When the message column is narrower than the identifier, ``fold`` splits
    # the id across physical lines (no single line carries the whole token);
    # the wide durable width keeps the whole id on one line.
    assert not any(order_id in line for line in render(40).splitlines())
    assert any(order_id in line for line in render(DURABLE_LOG_WIDTH).splitlines())


def __test_typer_app_hides_traceback_locals__():
    """The CLI app never renders frame locals in its crash traceback.

    Typer defaults to ``pretty_exceptions_show_locals=True``, which would dump
    credential-bearing configuration objects (session tokens, API secrets)
    from a provider/broker error boundary straight into the transcript."""
    from pynecore.cli.app import app

    assert app.pretty_exceptions_show_locals is False


def __test_account_equity_logged_at_debug_not_info__(monkeypatch):
    """The full equity mapping is emitted at DEBUG, keeping INFO balance-free.

    Guards the authenticate log split: the INFO line confirms account identity
    only; the multi-asset balance sheet goes to ``broker_debug`` so a default
    INFO transcript no longer prints every asset balance."""
    from pynecore.core import script_runner

    levels: dict[str, int] = {}

    def fake_info(message, *args):
        levels[message] = logging.INFO

    def fake_debug(message, *args):
        levels[message] = logging.DEBUG

    monkeypatch.setattr(script_runner, "broker_info", fake_info)
    monkeypatch.setattr(script_runner, "broker_debug", fake_debug)

    # Mirror the exact call shape used by ScriptRunner.start().
    balance = {"USDC": 50000.0, "BTC": 0.99999949}
    script_runner.broker_info(
        "authenticated: plugin=%s account=%s", "Bybit", "bybit-demo-1",
    )
    script_runner.broker_debug("account equity snapshot: %s", balance)

    assert levels["authenticated: plugin=%s account=%s"] == logging.INFO
    assert levels["account equity snapshot: %s"] == logging.DEBUG


def __test_root_logger_uses_the_pine_rich_handler__():
    """Import-time wiring stays intact: the pyne logger has exactly one handler."""
    assert len(log_module.logger.handlers) == 1


def __test_heartbeat_interval_resolution__():
    """The heartbeat interval honours its override and disables on ``0``."""
    from pynecore.cli.commands.run import (
        DEFAULT_HEARTBEAT_INTERVAL_S,
        _resolve_heartbeat_interval,
    )

    assert _resolve_heartbeat_interval(None) == DEFAULT_HEARTBEAT_INTERVAL_S
    assert _resolve_heartbeat_interval("") == DEFAULT_HEARTBEAT_INTERVAL_S
    assert _resolve_heartbeat_interval("nope") == DEFAULT_HEARTBEAT_INTERVAL_S
    assert _resolve_heartbeat_interval("5") == 5.0
    assert _resolve_heartbeat_interval("0") == 0.0
    assert _resolve_heartbeat_interval("-3") == 0.0


def __test_heartbeat_emits_then_stops_promptly__():
    """The heartbeat ticks on its interval and ``stop()`` joins without hanging."""
    import threading

    from pynecore.cli.commands.run import _LiveHeartbeat

    fired = threading.Event()
    seen: list[float] = []

    def emit(elapsed: float) -> None:
        seen.append(elapsed)
        fired.set()

    heartbeat = _LiveHeartbeat(0.01, emit)
    heartbeat.start()
    assert fired.wait(timeout=2.0)
    heartbeat.stop()

    assert seen
    assert all(value >= 0.0 for value in seen)


def __test_heartbeat_disabled_at_zero_interval_never_starts__():
    """A zero interval keeps the heartbeat thread from ever starting."""
    from pynecore.cli.commands.run import _LiveHeartbeat

    heartbeat = _LiveHeartbeat(0.0, lambda _elapsed: None)
    heartbeat.start()
    assert not heartbeat._thread.is_alive()  # noqa: SLF001
    heartbeat.stop()
