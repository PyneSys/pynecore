import asyncio
import json
import logging
import os
import signal
import threading
import time
import sys
import tomllib

from pathlib import Path
from datetime import datetime, UTC, tzinfo
from collections.abc import Callable
from typing import Any

from typer import Option, Argument, secho, Exit, colors
from rich.progress import (Progress, SpinnerColumn, TextColumn, BarColumn,
                           ProgressColumn, Task, TimeElapsedColumn, TimeRemainingColumn)
from rich.text import Text
from rich.console import Console

from ..app import app, app_state
from ..pluggable import PluggableCommand

from ...utils.rich.date_column import DateColumn
from pynecore.core.ohlcv import ChartBarWindow, OHLCVReader
from pynecore.core.data_converter import DataConverter, DataFormatError, ConversionError
from pynecore.core.data_path import DATA_EXTENSIONS, data_extension, with_data_extension
from pynecore.core.aggregator import validate_aggregation
from pynecore.lib.log import logger as pyne_logger
from pynecore.lib.timeframe import in_seconds

from pynecore.core.broker.exceptions import BrokerManualInterventionError
from pynecore.core.broker.interrupt_guard import request_interrupt
from pynecore.lib.log import broker_info, broker_warning
from pynecore.core.syminfo import SymInfo, mintick_decimals
from pynecore.core.script_runner import (ScriptRunner, DataRequirements, SecurityRequirement,
                                         history_trades_option_error, profile_option_error)
from pynecore.core.profiler import ProfilerUnavailable
from pynecore.errorcodes import InvalidOptionError, MissingFileError
from pynecore.core.error_location import INTERNAL_ERROR_REPORT, internal
from pynecore.core.errors import PyneError
from pynecore.pynesys.compiler import PyneComp
from pynecore.core.provider_string import is_provider_string
from pynecore.core.provider_download import (
    DownloadListener, ProviderHistory, ProviderSetupError, download_provider_history_resilient,
    open_live_stream, parse_time_value, require_live_provider,
)
from pynecore.core.plugin import ProviderError
from ...cli.utils.api_error_handler import APIErrorHandler
from ...cli.utils.profile_report import print_profile_report

__all__ = ['backtest_security_path', 'check_data_period', 'resolve_chart_timeframe',
           'resolve_data', 'resolve_script', 'resolve_time_window',
           'run_backtest_with_progress']


#: How long a stopped live run waits for the fills of market orders it already
#: dispatched, so the journal owns what the venue executed. Well inside the
#: supervisor's shutdown grace.
_STOP_FILL_SETTLE_S = 15.0

#: Task name given to :func:`_drain_loop_tasks`, used to keep a drain from
#: cancelling a sibling drain (only possible when the teardown is invoked twice
#: on a loop whose first drain outran its wait — the drain must never abort
#: another drain).
_DRAIN_NAME = "_drain_loop_tasks"


async def _drain_loop_tasks() -> None:
    """Cancel and await every task on the running loop except this one.

    Runs on the broker event loop itself (scheduled by :func:`_drain_then_stop`).
    Mirrors the standard ``asyncio.runners._cancel_all_tasks`` shutdown: request
    cancellation on all sibling tasks, then await them so each finishes
    unwinding — closing its WebSocket transport and cancelling its own child
    tasks — before the loop is stopped and closed. ``return_exceptions=True``
    keeps a task that re-raises something other than ``CancelledError`` from
    aborting the drain.
    """
    current = asyncio.current_task()
    pending = [
        task for task in asyncio.all_tasks()
        if task is not current and task.get_name() != _DRAIN_NAME
    ]
    if not pending:
        return
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)


def _drain_then_stop(loop: "Any") -> None:
    """Start the task drain on the loop, then stop the loop once it completes.

    Runs *on* the loop thread (scheduled with ``call_soon_threadsafe``), which
    is what makes the teardown self-contained: the coroutine is created only at
    the moment a task can be built from it, and ``loop.stop`` fires from the
    drain's completion callback rather than from the caller. A caller that
    stops waiting therefore leaves the teardown running to completion instead
    of stranding a half-scheduled drain — the loop still stops, just later.

    :param loop: The broker event loop being torn down.
    """
    task = loop.create_task(_drain_loop_tasks(), name=_DRAIN_NAME)

    def _on_drained(drain: "Any") -> None:
        if not drain.cancelled():
            # Retrieve any exception so it is not reported as never-retrieved.
            _ = drain.exception()
        loop.stop()

    task.add_done_callback(_on_drained)


def _shutdown_broker_event_loop(
    loop: "Any",
    thread: "threading.Thread | None",
    timeout: float,
) -> bool:
    """
    Tear down the broker event-loop pump thread, then close the loop.

    The loop runs ``run_forever`` on ``thread``; ``loop.stop()`` must be
    scheduled onto the loop thread with ``call_soon_threadsafe`` because the
    caller lives on a different (Pine script) thread. ``loop.close()`` is only
    safe once ``run_forever`` has actually returned — closing a still-running
    loop raises ``RuntimeError: Cannot close a running event loop``. This joins
    the thread first and only closes when the loop has genuinely stopped, so a
    slow-to-drain pump can never crash the CLI exit.

    :param loop: The broker event loop to stop and close.
    :param thread: The thread running ``loop.run_forever`` (``None`` if the
        pump never started).
    :param timeout: Maximum seconds to wait for the loop thread to exit.
        Non-positive means wait forever, matching the ``--shutdown-timeout``
        contract honoured by :mod:`pynecore.core.live_runner`.
    :return: ``True`` if the loop was closed; ``False`` if it was still running
        after ``timeout`` (left open, not closed, to avoid the
        close-while-running crash).
    """
    # Cancel and await every task still living on the loop BEFORE stopping it.
    # The broker's private order-event stream (``run_event_stream`` →
    # ``watch_orders``) and its WebSocket receive/ping loops are long-lived
    # tasks that the graceful public-data disconnect does not reach. Closing the
    # loop while they are pending destroys them mid-await, producing
    # ``Task was destroyed but it is pending`` warnings and
    # ``RuntimeError: Event loop is closed`` at process exit — the exact failure
    # a strategy exception (which skips the normal completion summary) surfaces.
    # Draining here, while ``run_forever`` is still turning the loop, lets each
    # task observe its cancellation and unwind cleanly (closing its socket)
    # instead of being abandoned. The drain and the subsequent ``loop.stop`` are
    # chained together *on the loop thread* so that giving up on the wait below
    # never strands a partially scheduled teardown.
    join_timeout = timeout if timeout > 0 else None
    try:
        if thread is not None and thread.is_alive() and loop.is_running():
            loop.call_soon_threadsafe(_drain_then_stop, loop)
        else:
            loop.call_soon_threadsafe(loop.stop)
    except RuntimeError:
        # Loop already stopped or closed — nothing to signal.
        pass
    if thread is not None:
        thread.join(timeout=join_timeout)
    # Gate close() on the loop having actually stopped rather than assuming the
    # join succeeded within ``timeout``. A join timeout leaves the thread alive
    # and the loop running; closing it then raises RuntimeError.
    if loop.is_running():
        return False
    loop.close()
    return True

console = Console()

#: Default seconds between live-run heartbeat lines when the interactive
#: spinner is suppressed (durable log / non-TTY). Overridable via
#: ``PYNE_HEARTBEAT_INTERVAL``; ``0`` disables the heartbeat entirely.
DEFAULT_HEARTBEAT_INTERVAL_S = 30.0


def _resolve_heartbeat_interval(raw: str | None) -> float:
    """Parse the ``PYNE_HEARTBEAT_INTERVAL`` override.

    :param raw: The raw environment value, if any.
    :return: A positive interval in seconds, or ``0.0`` to disable.
    """
    if raw is None or raw == "":
        return DEFAULT_HEARTBEAT_INTERVAL_S
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_HEARTBEAT_INTERVAL_S
    return value if value > 0.0 else 0.0


class _LiveHeartbeat:
    """Emit a periodic "still running" line while the live loop is quiet.

    When the interactive spinner is suppressed (durable log / non-TTY), a
    long silent phase — waiting for the next bar or a resting order to move —
    leaves the transcript with no output for tens of seconds, so an operator
    cannot tell a healthy wait from a hang. This ticker logs a heartbeat on a
    fixed cadence using an :class:`threading.Event` wait (event-driven, no
    busy-poll), so ``stop()`` returns promptly instead of after a full sleep.
    """

    def __init__(self, interval_s: float, emit: 'Any') -> None:
        self._interval = interval_s
        self._emit = emit
        self._stop = threading.Event()
        self._started_at = time.monotonic()
        self._thread = threading.Thread(
            target=self._run, name="live-heartbeat", daemon=True,
        )

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            elapsed = time.monotonic() - self._started_at
            self._emit(elapsed)

    def start(self) -> None:
        if self._interval > 0.0:
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)


class CustomTimeElapsedColumn(ProgressColumn):
    """Custom time elapsed column showing tenths of a second."""

    def render(self, task: Task) -> Text:
        """Render the time elapsed with tenths of a second."""
        elapsed = task.elapsed
        if elapsed is None:
            return Text("--:--:--.-", style="cyan")

        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = elapsed % 60

        return Text(f"{hours:02d}:{minutes:02d}:{seconds:04.1f}", style="cyan")


class CustomTimeRemainingColumn(ProgressColumn):
    """Custom time remaining column showing milliseconds."""

    def render(self, task: Task) -> Text:
        """Render the time remaining with milliseconds."""
        remaining = task.time_remaining
        if remaining is None:
            return Text("--:--.-", style="cyan")

        minutes = int(remaining // 60)
        seconds = remaining % 60

        return Text(f"{minutes:02d}:{seconds:06.3f}", style="cyan")


def _exchange_display_time(timestamp: int | float, display_tz: tzinfo) -> datetime:
    """Return an exchange-local naive timestamp for terminal display."""
    return datetime.fromtimestamp(timestamp, UTC).astimezone(display_tz).replace(tzinfo=None)


class ExchangeClockColumn(ProgressColumn):
    """Live exchange clock for the terminal spinner."""

    def __init__(self, display_tz: tzinfo):
        super().__init__()
        self.display_tz = display_tz

    def render(self, task: Task) -> Text:
        display_time = _exchange_display_time(time.time(), self.display_tz)
        return Text(f"Live — {display_time:%m-%d %H:%M:%S}", style="white")


def _format_broker_value(value: float, *, signed: bool = False) -> str:
    """Format broker spinner account metrics."""
    if signed:
        return f"{value:+,.2f}"
    return f"{value:,.2f}"


def _select_broker_balance(
        balance: dict[str, float] | None,
        preferred_currency: str | None,
) -> tuple[str, float] | None:
    """Pick the balance row to display in the live broker spinner."""
    if not balance:
        return None
    if preferred_currency and preferred_currency in balance:
        return preferred_currency, balance[preferred_currency]
    if len(balance) == 1:
        currency, value = next(iter(balance.items()))
        return currency, value
    currency = sorted(balance)[0]
    return currency, balance[currency]


def _coerce_finite_float(value: Any) -> float | None:
    """Coerce a dynamic position attribute to a finite float.

    Returns ``None`` when the value is missing, non-numeric, or NaN — so the
    spinner display logic can simply skip it rather than crash the live loop
    on a stray type.

    :param value: A loosely-typed attribute read via ``getattr``.
    :return: The finite float value, or ``None``.
    """
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result else None


def _broker_metrics_text(
        position: Any,
        exchange_position: Any,
        balance: dict[str, float] | None,
        preferred_currency: str | None,
        price_decimals: int,
        bid: float | None,
        ask: float | None,
        fallback_price: float | None,
) -> str:
    """Return spinner text for broker equity and unrealized PnL."""
    selected_balance = _select_broker_balance(balance, preferred_currency)
    if selected_balance is None:
        return ""
    currency, equity = selected_balance

    position_text = ""
    unrealized = 0.0
    if exchange_position is not None:
        exchange_side = str(getattr(exchange_position, 'side', '') or '').lower()
        exchange_size = float(getattr(exchange_position, 'size', 0.0) or 0.0)
        if exchange_side == 'short':
            exchange_size = -abs(exchange_size)
        elif exchange_side == 'long':
            exchange_size = abs(exchange_size)
        elif abs(exchange_size) < 1e-12:
            exchange_size = 0.0

        if abs(exchange_size) > 1e-12:
            position_text = f"Pos [cyan]{exchange_size:g}[/]"
            entry_price = float(getattr(exchange_position, 'entry_price', 0.0) or 0.0)
            if entry_price > 0.0:
                position_text += f" Entry [cyan]{entry_price:.{price_decimals}f}[/]"
        unrealized = float(getattr(exchange_position, 'unrealized_pnl', 0.0) or 0.0)
    elif position is not None:
        position_size = float(getattr(position, 'size', 0.0) or 0.0)
        if abs(position_size) > 1e-12:
            avg_price = _coerce_finite_float(getattr(position, 'avg_price', None))
            position_text = f"Pos [cyan]{position_size:g}[/]"
            if avg_price is not None and avg_price > 0.0:
                position_text += f" Entry [cyan]{avg_price:.{price_decimals}f}[/]"

        unrealized = float(getattr(position, 'openprofit', 0.0) or 0.0)
        open_trades = list(getattr(position, 'open_trades', []) or [])
        if open_trades:
            unrealized = 0.0
            for trade in open_trades:
                size = float(getattr(trade, 'size', 0.0) or 0.0)
                entry_price = float(getattr(trade, 'entry_price', 0.0) or 0.0)
                mark = bid if size >= 0.0 else ask
                if mark is None:
                    mark = fallback_price
                if mark is None:
                    continue
                unrealized += (float(mark) - entry_price) * size

    pnl_style = "green" if unrealized >= 0.0 else "red"
    parts = [f"Eq [cyan]{_format_broker_value(equity)} {currency}[/]"]
    if position_text:
        parts.append(position_text)
        parts.append(f"UPnL [{pnl_style}]{_format_broker_value(unrealized, signed=True)}[/]")
    return " ".join(parts)


def _format_run_completion_summary(
        reason: str,
        position: Any,
        exchange_position: Any,
        balance: dict[str, float] | None,
        preferred_currency: str | None,
) -> str:
    """Build the one-line broker-run completion summary.

    Emitted when a ``--broker`` run stops (graceful shutdown, ``Ctrl-C`` /
    ``SIGTERM`` interrupt, or manual-intervention halt) so the operator gets
    an explicit closing line stating the final position and account equity,
    rather than a transcript that simply ends.

    :param reason: Short why-it-stopped label (``completed``, ``interrupted``…).
    :param position: The Pine ``position`` object (paper/fallback size source).
    :param exchange_position: The broker position snapshot, when available.
    :param balance: The account equity mapping (currency -> equity).
    :param preferred_currency: ``syminfo.currency`` used to pick the equity row.
    :return: A single log line summarizing the final state.
    """
    parts = [f"run stopped ({reason})"]

    size: float | None = None
    if exchange_position is not None:
        raw = _coerce_finite_float(getattr(exchange_position, 'size', None))
        if raw is not None:
            side = str(getattr(exchange_position, 'side', '') or '').lower()
            if side == 'short':
                raw = -abs(raw)
            elif side == 'long':
                raw = abs(raw)
            size = raw
    elif position is not None:
        size = _coerce_finite_float(getattr(position, 'size', None))

    if size is not None:
        if abs(size) < 1e-12:
            parts.append("position=flat")
        else:
            parts.append(f"position={size:g}")

    selected = _select_broker_balance(balance, preferred_currency)
    if selected is not None:
        currency, equity = selected
        parts.append(f"equity={_format_broker_value(equity)} {currency}")

    return " ".join(parts)


def _parse_time_value(value: str | None, *, allow_bars: bool = False) -> datetime | int | None:
    """
    Parse a --from or --to parameter value (see :func:`~pynecore.core.provider_download.parse_time_value`).

    :param value: The raw string value.
    :param allow_bars: If True, allow negative numbers as bar counts.
    :return: A timezone-aware datetime, a negative int (bar count), or None.
    :raises Exit: When the value is invalid
    """
    try:
        return parse_time_value(value, allow_bars=allow_bars)
    except ValueError as e:
        secho(f"Error: {e}", err=True, fg=colors.RED)
        raise Exit(1)


class _RichDownloadListener(DownloadListener):
    """Shows a provider download in the terminal: spinners, a progress bar, warnings."""

    def __init__(self) -> None:
        self._progress: Progress | None = None
        self._task: Any = None

    def symbol_info_started(self) -> None:
        progress = Progress(SpinnerColumn(finished_text="[green]✓"),
                            TextColumn("{task.description}"))
        progress.start()
        self._task = progress.add_task("Fetching symbol info...", total=1)
        self._progress = progress

    def symbol_info_finished(self, ok: bool) -> None:
        self._stop(ok)

    def download_started(self, time_from: datetime) -> None:
        progress = Progress(
            SpinnerColumn(finished_text="[green]✓"),
            TextColumn("{task.description}"),
            DateColumn(),
            BarColumn(),
            TimeElapsedColumn(),
            "/",
            TimeRemainingColumn(),
        )
        progress.start()
        self._task = progress.add_task(
            "Downloading OHLCV data...", total=1, start_time=time_from,
        )
        self._progress = progress

    def download_attempt(self, time_from: datetime, total_seconds: int) -> None:
        if self._progress is not None:
            self._progress.update(self._task, total=total_seconds, completed=0,
                                  start_time=time_from)

    def download_progress(self, elapsed_seconds: int) -> None:
        if self._progress is not None:
            self._progress.update(self._task, completed=elapsed_seconds)

    def download_finished(self, ok: bool) -> None:
        self._stop(False)

    def warning(self, message: str) -> None:
        secho(message, err=True, fg=colors.YELLOW)

    def retry_wait(self, error: BaseException, delay: float, attempt: int) -> None:
        secho(
            f"Provider temporarily unavailable ({error}); waiting {int(delay)}s "
            f"before retry #{attempt} (press Ctrl-C to abort)...",
            err=True, fg=colors.YELLOW,
        )

    def _stop(self, completed: bool) -> None:
        """End the current display, ticking its task off when it completed."""
        if self._progress is None:
            return
        if completed:
            self._progress.update(self._task, completed=1)
        self._progress.stop()
        self._progress = None


def _download_provider_data_resilient(
        provider_str: str, time_from_str: str | None, *, retry_transient: bool,
) -> ProviderHistory:
    """Download a provider's history for ``pyne run``, with a clean one-line error.

    See :func:`~pynecore.core.provider_download.download_provider_history_resilient`.
    A failure it raises exits with its message instead of a traceback, mirroring
    ``pyne data download``.

    :param provider_str: The provider string (e.g. ``ctrader:pepperstoneuk:BTCUSD@1``).
    :param time_from_str: The ``--from`` value (date, days, or ``-bars``).
    :param retry_transient: Whether to wait-and-retry on transient errors —
        ``True`` only for long-running ``--broker`` / ``--live`` runs.
    :return: The downloaded history.
    :raises Exit: When the download fails
    """
    try:
        return download_provider_history_resilient(
            provider_str, time_from_str,
            data_dir=app_state.data_dir, config_dir=app_state.config_dir,
            retry_transient=retry_transient, listener=_RichDownloadListener(),
        )
    except ProviderSetupError as e:
        secho(str(e), err=True, fg=colors.RED)
        raise Exit(1)
    except (ProviderError, ValueError) as e:
        secho(f"Error: {e}", err=True, fg=colors.RED)
        raise Exit(1)



def _restart_in_place() -> None:
    """Re-execute this process with its original command line, keeping its PID.

    Used after an operator resumed the run's quarantine (``pyne broker resume``):
    the bot re-syncs with the venue through the regular startup path (adoption,
    replay, reconcile), while a supervisor (systemd, a container runtime) sees
    no exit and restarts nothing.
    """
    logging.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()
    os.execv(sys.executable, sys.orig_argv)


def _pin_timenow(last_bar_time: int | None) -> None:
    """Pin ``timenow`` to the final bar of a bounded historical replay.

    A backtest that reads the real clock is not reproducible: a script gating
    its entries on ``time >= timenow - N days`` measures a different bar set on
    every run, so its result changes from one day to the next and can never be
    matched against a stored reference again. Anchoring the clock to the last
    bar of the replayed window makes the run deterministic — the instant that
    run's world ends. Live runs never call this and keep the system clock.

    Both channels are needed: the environment variable reaches the
    ``request.security`` subprocesses, which import ``pynecore.lib`` fresh under
    the ``spawn`` start method, while the attribute covers this process, which
    imported it long before the data window was known.

    :param last_bar_time: UNIX time (ms) of the window's last bar; ``None``
        (an empty window) leaves the clock alone.
    """
    if last_bar_time is None:
        return
    from pynecore import lib as _lib
    os.environ['PYNE_TIMENOW_MS'] = str(last_bar_time)
    _lib._timenow_ms = last_bar_time


def _print_data_requirements(requirements: DataRequirements, script_name: str) -> None:
    """Print a plain, human-readable summary of a script's data dependencies.

    Used by ``--list-data``: only non-empty sections are shown, with ASCII
    markers (``->``, ``!``) so the output stays terminal-safe.

    :param requirements: The classified buckets from
        :meth:`ScriptRunner.list_data_requirements`.
    :param script_name: Display name of the script (for the header line).
    """
    def _tags(sr: SecurityRequirement) -> str:
        out = ""
        if sr.is_ltf:
            out += " (lower timeframe)"
        if sr.ignore_invalid_symbol:
            out += " [ignore_invalid_symbol]"
        if sr.from_library:
            out += " (from library)"
        return out

    lines: list[str] = [
        f"Data requirements for {script_name} "
        f"(chart: {requirements.chart_symbol} @ {requirements.chart_tf})"
    ]

    total = (len(requirements.chart_main) + len(requirements.same_symbol_other_tf)
             + len(requirements.cross_symbol) + len(requirements.dynamic))
    if total == 0:
        lines.append("")
        lines.append("No external data required — this script uses only the "
                     "chart data you pass as DATA.")
        console.print("\n".join(lines), markup=False, highlight=False)
        return

    if requirements.chart_main:
        lines.append("")
        lines.append("Chart / main data (served from the DATA you pass):")
        for r in requirements.chart_main:
            lines.append(f"  -> {r.symbol} @ {r.timeframe}{_tags(r)}")

    if requirements.same_symbol_other_tf:
        lines.append("")
        lines.append("Same symbol, other timeframe (resampled from the chart base data):")
        for r in requirements.same_symbol_other_tf:
            if r.has_security_mapping:
                suffix = " [mapping present]"
            elif r.derived_from_chart:
                suffix = " [served from the chart data, nothing to supply]"
            else:
                suffix = f" [needs --security '{r.timeframe}=<base file>' in backtest]"
            lines.append(f"  -> {r.symbol} @ {r.timeframe}{_tags(r)}{suffix}")

    if requirements.cross_symbol:
        lines.append("")
        lines.append("Cross-symbol data (separate download / symbol_map / --security required):")
        for r in requirements.cross_symbol:
            if r.has_security_mapping:
                suffix = " [--security mapping present]"
            elif r.has_global_map:
                status = "ok" if r.mapped_file_exists else "missing"
                fname = Path(r.mapped_file).name if r.mapped_file else "?"
                suffix = (f" [symbol_map -> {r.mapped_provider}:"
                          f"{r.mapped_native_symbol} -> {fname} ({status})]")
            else:
                suffix = " ! no mapping"
            lines.append(f"  -> {r.symbol} @ {r.timeframe}{_tags(r)}{suffix}")
            if r.has_global_map and not r.mapped_file_exists and r.download_suggestion:
                lines.append(f"       download: {r.download_suggestion}")
            elif not r.has_global_map and r.file_suggestions:
                lines.append(f"       existing data matching ticker: "
                             f"{', '.join(r.file_suggestions)}")

    if requirements.dynamic:
        lines.append("")
        lines.append("Dynamic data (symbol or timeframe only known at runtime):")
        lines.append(f"  ! {len(requirements.dynamic)} request.security() "
                     f"call(s) with a runtime symbol/timeframe")
        lines.append("    -> cannot be listed statically; inspect the script source")

    console.print("\n".join(lines), markup=False, highlight=False)


def resolve_script(script: Path, api_key: str | None) -> tuple[Path, bool]:
    """
    Find a script by name or path and compile it when it is Pine Script.

    A bare name is looked up in the workdir's ``scripts`` directory; without a suffix
    ``.pine`` is tried first, then ``.py``. A ``.pine`` script is compiled to the
    ``.py`` beside it when the API configuration has a key and the source changed.

    :param script: Script name or path
    :param api_key: PyneSys API key overriding the configuration file
    :return: The ``.py`` script to run, and whether it was found as the ``.pine`` source,
             so an error names the Pine line
    :raises Exit: When the script does not exist or the configuration is invalid
    """
    # Expand script path
    if len(script.parts) == 1:
        script = app_state.scripts_dir / script

    # If no script suffix, try .pine 1st
    if script.suffix == "":
        script = script.with_suffix(".pine")
    # If doesn't exist, try .py
    if not script.exists():
        script = script.with_suffix(".py")

    # Check if script exists
    if not script.exists():
        secho(f"Error: {MissingFileError(path=str(script)).report()}", fg="red", err=True)
        raise Exit(1)

    from_pine = script.suffix == ".pine"

    # Handle .pine files - compile them first
    if from_pine:
        # Read api.toml configuration
        api_config = {}
        try:
            with open(app_state.config_dir / 'api.toml', 'rb') as f:
                api_config = tomllib.load(f)['api']
        except KeyError:
            console.print("[red]Invalid API config file (api.toml)![/red]")
            raise Exit(1)
        except FileNotFoundError:
            pass

        # Override API key if provided
        if api_key:
            api_config['api_key'] = api_key

        # Override API URL if provided via environment variable
        api_url = os.getenv("PYNESYS_API_URL")
        if api_url:
            api_config['base_url'] = api_url

        if api_config.get('api_key'):
            # Create the compiler instance
            compiler = PyneComp(**api_config)

            # Determine output path for compiled file
            out_path = script.with_suffix(".py")

            # Check if compilation is needed
            if compiler.needs_compilation(script, out_path):
                with APIErrorHandler(console):
                    with Progress(
                            SpinnerColumn(finished_text="[green]✓"),
                            TextColumn("[progress.description]{task.description}"),
                            console=console
                    ) as progress:
                        task = progress.add_task("Compiling Pine Script...", total=1)

                        # Compile the .pine file
                        compiler.compile(script, out_path)

                        progress.update(task, completed=1)

            # Update script to point to the compiled file
            script = out_path

        # Go back to normal .py file
        else:
            script = script.with_suffix(".py")
            # Check if script exists
            if not script.exists():
                secho(f"Error: {MissingFileError(path=str(script)).report()}", fg="red",
                      err=True)
                raise Exit(1)

    return script, from_pine


def resolve_data(data: str, time_from: str | None, *,
                 retry_transient: bool = False) -> tuple[Path, SymInfo, ProviderHistory | None]:
    """
    Resolve the chart data of a run: download it from a provider or find the file.

    A provider string downloads the history and the symbol information. A file name
    is looked up in the workdir's ``data`` directory and converted to OHLCV when it
    is in another format.

    :param data: Data file or provider string
    :param time_from: The ``--from`` value, which sizes a provider download
    :param retry_transient: Keep retrying a provider through transient outages
    :return: The OHLCV file, its symbol information, and the provider download when
        the data came from a provider
    :raises Exit: When the data cannot be found, converted or downloaded
    """
    provider_data = None
    if is_provider_string(data):
        # Provider mode: download historical warmup data + syminfo. A
        # long-running --broker/--live run rides out transient broker outages
        # (maintenance, lost route) instead of dying mid-startup; a one-shot
        # backtest and any permanent failure (unknown symbol, bad credentials)
        # still fail fast with a clean one-line error, not a traceback.
        provider_data = _download_provider_data_resilient(
            data, time_from, retry_transient=retry_transient,
        )
        data_path, syminfo = provider_data.ohlcv_path, provider_data.syminfo
    else:
        # File mode: resolve path, convert if needed
        data_path = Path(data)

        if len(data_path.parts) == 1:
            data_path = app_state.data_dir / data_path

        # A name without a known data extension (e.g. ``bybit_BTCUSD.P_1D``, whose
        # ``.P_1D`` belongs to the symbol) resolves to its first existing data
        # sibling. A file that exists under the name as given is a source file of
        # its own, its format detected by content.
        if not data_extension(data_path):
            candidates = [with_data_extension(data_path, ext) for ext in DATA_EXTENSIONS]
            data_path = next(
                (c for c in candidates if c.exists()),
                data_path if data_path.is_file() else candidates[0],
            )

        if data_extension(data_path) != ".ohlcv":
            try:
                converter = DataConverter()
                if converter.is_conversion_required(data_path):
                    detected_symbol, detected_provider = DataConverter.guess_symbol_from_filename(data_path)
                    if not detected_symbol:
                        detected_symbol = with_data_extension(data_path, '').name.upper()
                    with Progress(
                            SpinnerColumn(finished_text="[green]✓"),
                            TextColumn("[progress.description]{task.description}"),
                            console=console
                    ) as progress:
                        task = progress.add_task(f"Converting {data_path.suffix} to OHLCV format...", total=1)
                        converter.convert_to_ohlcv(
                            data_path, provider=detected_provider,
                            symbol=detected_symbol, force=True
                        )
                        data_path = with_data_extension(data_path, ".ohlcv")
                        progress.update(task, completed=1)
                else:
                    data_path = with_data_extension(data_path, ".ohlcv")
            except PyneError as e:
                secho(f"Error: {e.report()}", fg="red", err=True)
                raise Exit(1)
            except (DataFormatError, ConversionError) as e:
                secho(f"Conversion failed: {e}", fg="red", err=True)
                secho("Please convert the file manually:", fg="red")
                secho(f"pyne data convert-from {data_path}", fg="yellow")
                raise Exit(1)

        if not data_path.exists():
            secho(f"Error: {MissingFileError(path=str(data_path)).report()}", fg="red", err=True)
            raise Exit(1)

        toml_path = data_path.with_suffix(".toml")
        try:
            syminfo = SymInfo.load_toml(toml_path)
        except FileNotFoundError:
            secho(f"Error: {MissingFileError(path=str(toml_path)).report()}", fg="red", err=True)
            raise Exit(1)
        except PyneError as e:
            secho(f"Error: {e.report()}", fg="red", err=True)
            raise Exit(1)

    return data_path, syminfo, provider_data


def resolve_chart_timeframe(syminfo: SymInfo, timeframe: str | None) -> str | None:
    """
    Apply the ``--timeframe`` option to the chart.

    A chart timeframe above the data's own aggregates the data, or runs the bar
    magnifier when the strategy asks for it; ``syminfo.period`` becomes the chart
    timeframe.

    :param syminfo: Symbol information of the data; its period is updated
    :param timeframe: The ``--timeframe`` value, or ``None``
    :return: The data's own timeframe when it differs from the chart's, else ``None``
    :raises Exit: When the timeframe is invalid or cannot be aggregated to
    """
    if timeframe:
        chart_tf: str = timeframe.upper()
        try:
            in_seconds(chart_tf)
        except (ValueError, AssertionError):
            secho(f"Invalid timeframe: {chart_tf}. Must be a valid TradingView format "
                  f"(e.g. '1', '5', '60', '1D', '1W', '1M').", fg="red", err=True)
            raise Exit(1)

        data_tf = syminfo.period
        if chart_tf != data_tf:
            try:
                validate_aggregation(data_tf, chart_tf)
            except ValueError as e:
                secho(str(e), fg="red", err=True)
                raise Exit(1)
            # Override syminfo period to the chart timeframe
            syminfo.period = chart_tf
            return data_tf
    return None


def check_data_period(reader: OHLCVReader, data_file_period: str | None, data_path: Path) -> None:
    """
    Reject a symbol information file whose period disagrees with the data file.

    :param reader: The open data file
    :param data_file_period: The period the symbol information declares
    :param data_path: The data file, for the message
    :raises Exit: When the two periods differ
    """
    # The file header owns the data's period; a sidecar that disagrees is
    # stale and would silently drive the magnifier / aggregation decisions off
    # the wrong resolution. Legacy files declare no header period.
    if reader.period is not None and data_file_period:
        try:
            header_sec = in_seconds(reader.period)
            toml_sec = in_seconds(data_file_period)
        except (ValueError, AssertionError):
            header_sec = toml_sec = None
        if header_sec is not None and header_sec != toml_sec:
            secho(f"Stale syminfo: TOML period '{data_file_period}' does not match "
                  f"the period '{reader.period}' declared by {data_path.name}.",
                  fg="red", err=True)
            raise Exit(1)


def resolve_time_window(reader: OHLCVReader, time_from: str | None, time_to: str | None,
                        provider_data: ProviderHistory | None) -> tuple[datetime, datetime]:
    """
    The time window of a run, from the ``--from`` / ``--to`` options and the data.

    :param reader: The open data file
    :param time_from: The ``--from`` value; ``None`` in provider mode, where the
        download already applied it
    :param time_to: The ``--to`` value
    :param provider_data: The provider download, or ``None`` in file mode
    :return: Start and end of the window
    """
    # Parse time range
    time_from_dt = _parse_time_value(time_from) if time_from else None
    time_to_dt = _parse_time_value(time_to) if time_to else None

    if not time_from_dt:
        # Provider bar-count mode pins the start to the N-th last
        # real bar; otherwise use the file's natural start.
        if provider_data is not None and provider_data.time_from_ts is not None:
            time_from_dt = datetime.fromtimestamp(
                provider_data.time_from_ts / 1000, UTC,
            )
        else:
            time_from_dt = reader.start_datetime
    if not time_to_dt:
        time_to_dt = reader.end_datetime
    assert isinstance(time_from_dt, datetime) and isinstance(time_to_dt, datetime)
    return time_from_dt, time_to_dt


def backtest_security_path(value: str) -> str:
    """
    The data file a ``--security`` mapping names in a backtest.

    :param value: File stem or path; a bare name is looked up in the workdir's
        ``data`` directory, and may carry the ``.ohlcv`` extension
    :return: The file path without extension, as the runner takes it
    :raises Exit: When the OHLCV file does not exist
    """
    sec_path = Path(value)
    if len(sec_path.parts) == 1:
        sec_path = app_state.data_dir / sec_path
    ohlcv_check = with_data_extension(sec_path, '.ohlcv')
    if not ohlcv_check.exists():
        secho(f"Error: {MissingFileError(path=str(ohlcv_check)).report()}", fg="red", err=True)
        raise Exit(1)
    return str(with_data_extension(sec_path, ''))


def run_backtest_with_progress(runner: ScriptRunner, time_from_display: datetime,
                               time_to_display: datetime, total_seconds: int,
                               on_bar: Callable[[], None] | None = None) -> None:
    """
    Run a backtest to its end under a progress bar of the time window.

    :param runner: The script runner
    :param time_from_display: Start of the window, naive UTC
    :param time_to_display: End of the window, naive UTC
    :param total_seconds: Length of the window
    :param on_bar: Called after every bar, once its orders and trades are settled
    """
    with Progress(
            SpinnerColumn(finished_text="[green]✓"),
            TextColumn("{task.description}"),
            DateColumn(time_from_display),
            BarColumn(),
            CustomTimeElapsedColumn(),
            "/",
            CustomTimeRemainingColumn(),
    ) as progress:
        task = progress.add_task(
            description="Running script...",
            total=total_seconds,
        )

        # Single slot for progress updates. The callback runs once per
        # BAR while the bar only ever redraws on a whole-second change,
        # so all a queue could offer is a backlog the reader throws
        # away. One producer, one consumer, and a list store is a single
        # bytecode under the GIL -- overwriting needs no lock.
        last_time: list[datetime | None] = [None]
        stop_event = threading.Event()

        def progress_worker():
            """Worker thread that updates progress bar at 30Hz"""
            last_update = 0
            while not stop_event.is_set():
                try:
                    current_time = last_time[0]
                    if current_time is not None:
                        if current_time == datetime.max:
                            current_time = time_to_display
                        elapsed_seconds = int(
                            (current_time - time_from_display).total_seconds())
                        if elapsed_seconds != last_update:
                            progress.update(task, completed=elapsed_seconds)
                            last_update = elapsed_seconds
                except Exception:  # noqa
                    pass

                time.sleep(1 / 30)

        # Start worker thread
        worker = threading.Thread(target=progress_worker, daemon=True)
        worker.start()

        def cb_progress(current_time: datetime | None):
            """Callback that keeps only the newest timestamp"""
            last_time[0] = current_time
            if on_bar is not None and current_time != datetime.max:
                on_bar()

        try:
            runner.run(on_progress=cb_progress)

            progress.update(task, completed=total_seconds)
        finally:
            stop_event.set()
            worker.join(timeout=0.1)
            progress.refresh()


@app.command(cls=PluggableCommand)
def run(
        script: Path = Argument(..., dir_okay=False, file_okay=True, help="Script to run (.py or .pine)"),
        data: str = Argument(...,
                             help="Data file (*.ohlcv, *.csv) or provider string "
                                  "(e.g. ccxt:BYBIT:BTC/USDT:USDT@1D)"),
        time_from: str | None = Option(None, '--from', '-f',
                                       metavar="[DATE|DAYS|-BARS]",
                                       help="Start: date in UTC (2025-01-01), days back from now (30), "
                                            "or -N bars back (-500). Default: -500 bars in provider mode."),
        time_to: str | None = Option(None, '--to', '-t',
                                     metavar="[DATE|DAYS]",
                                     help="End: date in UTC or days back from now "
                                          "(default: end of data or now)"),
        plot_path: Path | None = Option(None, "--plot", "-pp",
                                        help="Path to save the plot data",
                                        rich_help_panel="Out Path Options"),
        no_plot: bool = Option(False, "--no-plot",
                               help="Do not write the plot CSV at all (it is written by default). "
                                    "Useful in live mode, where the file grows without bound.",
                               rich_help_panel="Out Path Options"),
        strat_path: Path | None = Option(None, "--strat", "-sp",
                                         help="Path to save the strategy statistics",
                                         rich_help_panel="Out Path Options"
                                         ),
        trade_path: Path | None = Option(None, "--trade", "-tp",
                                         help="Path to save the trade data",
                                         rich_help_panel="Out Path Options"),
        viz: bool = Option(False, "--viz", "-vz",
                           help="Write plot/drawing visual data (NDJSON)",
                           rich_help_panel="Out Path Options"),
        viz_path: Path | None = Option(None, "--viz-path",
                                       help="Viz NDJSON path (implies --viz)",
                                       rich_help_panel="Out Path Options"),
        viz_journal: bool = Option(False, "--viz-journal",
                                   help="Record per-bar drawing create/update/delete events "
                                        "(implies --viz)",
                                   rich_help_panel="Out Path Options"),
        api_key: str | None = Option(None, "--api-key", "-a",
                                     help="PyneSys API key for compilation (overrides configuration file)",
                                     envvar="PYNESYS_API_KEY",
                                     rich_help_panel="Compilation Options"),
        live: bool = Option(False, "--live", "-l",
                            help="Continue with live data after historical phase "
                                 "(provider mode only)"),
        broker: bool = Option(False, "--broker",
                              help="Enable live broker trading — requires a provider plugin that "
                                   "subclasses BrokerPlugin. Implies --live.",
                              rich_help_panel="Live Options"),
        history_trades: bool = Option(False, "--history-trades",
                                      help="Live paper trading: let the strategy trade on the "
                                           "historical bars too, and continue that position live. "
                                           "By default a live run starts flat when it goes live. "
                                           "Not available with --broker.",
                                      rich_help_panel="Live Options"),
        run_label: str | None = Option(None, "--run-label",
                                       help="Optional label to distinguish parallel instances of the "
                                            "same strategy+account+symbol+timeframe. Stored in the "
                                            "broker run_id as ``...#<label>``.",
                                       rich_help_panel="Live Options"),
        shutdown_timeout: float = Option(120.0, "--shutdown-timeout",
                                         help="Max seconds to wait for graceful shutdown "
                                              "(0 = wait forever)",
                                         rich_help_panel="Live Options"),
        no_log_ohlcv: bool = Option(False, "--no-log-ohlcv",
                                    help="Disable per-bar OHLCV log lines in live mode "
                                         "(default: enabled).",
                                    rich_help_panel="Live Options"),
        security: list[str] | None = Option(None, "--security", "-sec",
                                            help='Security data: "TIMEFRAME=data_name" or '
                                                 '"SYMBOL:TIMEFRAME=data_name"',
                                            rich_help_panel="Security Options"),
        list_data: bool = Option(False, "--list-data",
                                 help="List the OHLCV data this script needs (from "
                                      "request.security calls) and exit without running.",
                                 rich_help_panel="Security Options"),
        timeframe: str | None = Option(None, "--timeframe", "-tf",
                                       help="Chart timeframe (TradingView format, e.g. '60', '1D'). "
                                            "When larger than data timeframe: aggregates on-the-fly, "
                                            "or activates bar magnifier if strategy uses "
                                            "use_bar_magnifier=true."),
        profile: bool = Option(False, "--profile",
                               help="Profile the run: the time and execution count of every "
                                    "line, and the run's time by category. Writes "
                                    "<script>_profile.json next to the other outputs. "
                                    "Backtests only; needs Python 3.13+.",
                               rich_help_panel="Profiling Options"),
        profile_top: int = Option(15, "--profile-top", min=0, metavar="N",
                                  help="Number of lines listed in the profile summary",
                                  rich_help_panel="Profiling Options"),
        show_traceback: bool = Option(False, "--traceback",
                                      help="When the script stops with a coded error "
                                           "(R0010, ...), print the Python traceback too, "
                                           "not only the error's line"),

):
    """
    Run a script (.py or .pine)

    The system automatically searches for the workdir folder in the current and parent directories.
    If not found, it creates or uses a workdir folder in the current directory.

    If [bold]script[/] path is a name without full path, it will be searched in the [italic]"workdir/scripts"[/] directory.
    Similarly, if [bold]data[/] path is a name without full path, it will be searched in the [italic]"workdir/data"[/] directory.
    The [bold]plot_path[/], [bold]strat_path[/], and [bold]trade_path[/] outputs are saved in the [italic]"workdir/output"[/]
    directory by default. An explicit path is used as given, so a bare file name is written to the current directory.

    [bold]Data Source:[/bold]
    The [bold]data[/] argument accepts either a file path or a provider string:
    \b
      File mode:     pyne run script.py data.csv
      Provider mode: pyne run script.py ccxt:BYBIT:BTC/USDT:USDT@1D -f -500

    In provider mode, historical data is downloaded automatically. The --from/-f parameter
    accepts: date in UTC (2025-01-01), days back (30), or -N bars back (-500). Default: -500 bars.

    [bold]Pine Script Support:[/bold]
    Pine Script (.pine) files are automatically compiled to Python (.py) before execution.
    A valid [bold]PyneSys API[/bold] key is required. Get one at [blue]https://pynesys.io[/blue].

    [bold]Data Formats:[/bold]
    Supports CSV, TXT, JSON, and OHLCV data files. Non-OHLCV files are automatically converted.
    """  # noqa

    script, from_pine = resolve_script(script, api_key)

    # --- Data resolution: provider string or file path ---
    provider_mode = is_provider_string(data)

    option_error: InvalidOptionError | None = None
    if live and not provider_mode:
        option_error = InvalidOptionError(option='--live', why="it needs a provider string as "
                                                              "the data")
    elif history_trades and (broker or not live):
        option_error = history_trades_option_error()
    elif profile and (live or broker):
        option_error = profile_option_error()
    if option_error is not None:
        secho(f"Error: {option_error.report()}", err=True, fg=colors.RED)
        raise Exit(1)

    data_path, syminfo, provider_data = resolve_data(
        data, time_from, retry_transient=(live or broker),
    )

    # --- Output paths ---
    if no_plot and plot_path:
        secho("--no-plot and --plot are mutually exclusive!", fg="red", err=True)
        raise Exit(1)
    if plot_path and plot_path.suffix != ".csv":
        plot_path = plot_path.with_suffix(".csv")
    if not plot_path and not no_plot:
        plot_path = app_state.output_dir / f"{script.stem}.csv"

    if strat_path and strat_path.suffix != ".csv":
        strat_path = strat_path.with_suffix(".csv")
    if not strat_path:
        strat_path = app_state.output_dir / f"{script.stem}_strat.csv"

    if trade_path and trade_path.suffix != ".csv":
        trade_path = trade_path.with_suffix(".csv")
    if not trade_path:
        trade_path = app_state.output_dir / f"{script.stem}_trade.csv"

    # --viz-path / --viz-journal both imply --viz
    if viz_path or viz_journal:
        viz = True
    if viz and not viz_path:
        viz_path = app_state.output_dir / f"{script.stem}_viz.ndjson"

    profile_path: Path | None = (app_state.output_dir / f"{script.stem}_profile.json"
                                 if profile else None)

    # Validate and process --timeframe option
    # ``syminfo.period`` is the CHART timeframe from here on — the magnifier path
    # below deliberately overwrites it. Keep the data file's own declared period so
    # it can still be checked against the file header once the reader is open.
    data_file_period = syminfo.period
    magnifier_source_tf = resolve_chart_timeframe(syminfo, timeframe)
    magnifier_mode = magnifier_source_tf is not None

    # --- Open data and run ---
    with OHLCVReader(data_path) as reader:
        check_data_period(reader, data_file_period, data_path)
        time_from_dt, time_to_dt = resolve_time_window(
            reader, None if provider_mode else time_from, time_to, provider_data,
        )
        # The reader window and the live-stream dedup boundary are both in the
        # OHLCV unit, Unix milliseconds.
        time_from_ts = int(time_from_dt.timestamp() * 1000)
        time_to_ts = int(time_to_dt.timestamp() * 1000)

        # Naive UTC for display purposes
        time_from_display = time_from_dt.astimezone(UTC).replace(tzinfo=None)
        time_to_display = time_to_dt.astimezone(UTC).replace(tzinfo=None)

        total_seconds = int((time_to_display - time_from_display).total_seconds())

        # Get the iterator using the correct UTC timestamps
        size = reader.get_size(time_from_ts, time_to_ts)
        # Pine anchors ``last_bar_time`` on historical bars to the chart's final
        # bar, known up front from the data window.
        last_bar_time = reader.get_last_bar_time(time_from_ts, time_to_ts)
        magnifier_iter = None

        # The run's chart bar window. The bar loop's bars come out of it, and so
        # do the bars a security context's child walks to replay its own
        # historical rounds (``ChartBarWindow.bars`` is the single production of
        # both), which is what makes the two identical by construction.
        chart_bar_window = ChartBarWindow(reader.path, time_from_ts, time_to_ts)

        if magnifier_mode:
            # Sub-TF data goes to magnifier; ohlcv_iter is unused (replaced in ScriptRunner)
            magnifier_iter = reader.read_from(time_from_ts, time_to_ts)
            ohlcv_iter = iter([])
        else:
            ohlcv_iter = chart_bar_window.bars()
        lossless_volume = reader.lossless_volume
        lossless_prices = reader.lossless_prices

        # --broker implies --live.
        if broker:
            live = True

        # Broker mode: verify plugin capability up front.
        broker_plugin = None
        broker_event_loop = None
        broker_event_loop_thread = None
        broker_store = None
        broker_store_ctx = None
        # Whether the live loop reached its explicit completion summary. When
        # it did not (an early raise, a crash mid-run) the outer ``finally``
        # narrates the cleanup outcome + next step so a failed run never ends
        # without a visible cleanup status.
        broker_run_reached_summary = False
        # Set when the run stopped because an operator resumed its quarantine;
        # the process then re-executes itself after the teardown below.
        restart_after_quarantine_resume = False
        # Live OHLCV consumer generator — captured here so the outer
        # ``finally`` can close it explicitly *before* the broker event
        # loop is torn down. Without that, the consumer's own ``finally``
        # only runs when the runner is GC'd (after the function returns),
        # by which point ``broker_event_loop`` is already closed and the
        # producer thread's ``run_coroutine_threadsafe`` future is stuck
        # on a dead loop — its ``thread.join(shutdown_timeout + 5)`` then
        # blocks the whole ~125 s before the process actually exits.
        live_iter = None
        if broker:
            if not provider_data:
                secho("--broker requires a provider string (ccxt:EXCHANGE:SYMBOL@TIMEFRAME).",
                      err=True, fg=colors.RED)
                raise Exit(1)
            from pynecore.core.plugin.broker import BrokerPlugin
            if not isinstance(provider_data.provider_instance, BrokerPlugin):
                secho(
                    f"Plugin '{provider_data.parsed_string.provider}' is not a BrokerPlugin "
                    f"— broker mode requires an exchange-backed plugin.",
                    err=True, fg=colors.RED,
                )
                raise Exit(1)
            broker_plugin = provider_data.provider_instance

            # Apply cross-broker runtime defaults (workdir/config/brokers.toml).
            # The policies are broker-agnostic; living here keeps the user-facing
            # knobs in one place and out of every plugin's own config.
            from pynecore.core.broker.defaults import load_broker_defaults
            broker_defaults = load_broker_defaults(app_state.config_dir)
            broker_plugin.on_unexpected_cancel = broker_defaults.on_unexpected_cancel
            broker_plugin.on_inventory_conflict = broker_defaults.on_inventory_conflict

            # Probe the plugin against the BrokerPlugin authoring contract
            # before any storage or engine state exists. Authentication has
            # already run (``download_provider_history`` drove it), so the
            # account-id lifecycle is checkable here — and the broker
            # storage below derives the run identity from it.
            from pynecore.core.broker.validation import validate_plugin_contract
            contract_errors, contract_warnings = validate_plugin_contract(
                broker_plugin, require_account_id=True,
            )
            for warning in contract_warnings:
                broker_warning("%s", warning)
            if contract_errors:
                secho(
                    "Broker plugin contract violation(s):\n"
                    + "\n".join(f"  - {e}" for e in contract_errors),
                    err=True, fg=colors.RED,
                )
                raise Exit(1)

            broker_event_loop = asyncio.new_event_loop()
            # Drive the loop on a dedicated daemon thread. Broker plugin
            # coroutines are submitted from the (synchronous) Pine script
            # thread via ``run_coroutine_threadsafe``, which requires the
            # target loop to actually be running — without this pump every
            # broker call would park forever.
            broker_event_loop_thread = threading.Thread(
                target=broker_event_loop.run_forever,
                daemon=True,
                name="broker-event-loop",
            )
            broker_event_loop_thread.start()

            # Open the unified broker storage and register a new run instance.
            # The plugin's account_id is already populated by this point —
            # download_provider_history has driven authentication through the
            # provider side, and the plugin stashes the identifier during
            # its session setup.
            from pynecore.core.broker.run_identity import RunIdentity
            from pynecore.core.broker.storage import BrokerStore
            store_path = app_state.workdir / "output" / "logs" / "broker.sqlite"
            broker_store = BrokerStore(
                store_path, plugin_name=broker_plugin.plugin_name,
            )
            identity = RunIdentity(
                strategy_id=script.stem,
                symbol=str(syminfo.ticker),
                timeframe=str(syminfo.period or ""),
                account_id=broker_plugin.account_id,
                label=run_label,
            )
            try:
                broker_store_ctx = broker_store.open_run(
                    identity,
                    script_source=script.read_text(encoding='utf-8'),
                    script_path=script,
                )
            except RuntimeError as e:
                secho(str(e), err=True, fg=colors.RED)
                broker_store.close()
                raise Exit(1)
            except BaseException:
                # open_run wraps its INSERT in a transaction: any failure before
                # it returns rolls the row back (no orphaned LIVE row) but leaves
                # the SQLite connection open. Close it so a startup crash here
                # cannot leak the store, then re-raise the real error.
                broker_store.close()
                raise

        # The broker run is now open; every subsequent startup step
        # (security parsing, live iterator chaining, ScriptRunner import)
        # must run under the same try/finally that also wraps runner.run(),
        # otherwise an early raise would leave the active runs row with a
        # NULL ``ended_ts_ms`` and block the next startup of the same bot
        # until the stale-cleanup window (5 min) expires.
        try:
            # Validate live mode capability up front (the live iterator is
            # created only AFTER ``Loading PyneCore`` finishes, so its eager
            # ``WS connect`` log doesn't race the spinner).
            if live and provider_data:
                try:
                    require_live_provider(provider_data)
                except ProviderSetupError as e:
                    secho(str(e), err=True, fg=colors.RED)
                    raise Exit(1)
                assert provider_data.parsed_string.timeframe is not None
                size = 0

            # Parse security data mappings. Two strict modes:
            #
            #   --live: value is a plugin-native symbol that the chart
            #           provider can serve (e.g. ``EURUSD`` on Capital.com).
            #           The security subprocess opens its own provider,
            #           downloads warmup in-memory, and streams live —
            #           no ``.ohlcv`` file is involved.
            #
            #   backtest: value is a static ``.ohlcv`` file (as before).
            #
            # Cross-mode values are rejected so a silent mismatch can't
            # produce wrong results.
            from pynecore.core.plugin.live_provider import PluginSymbol
            security_data: dict[str, str | Path | PluginSymbol] | None = None
            if security:
                sec_map: dict[str, str | Path | PluginSymbol] = {}
                for entry in security:
                    if '=' not in entry:
                        secho(
                            f"Invalid --security format: '{entry}'. "
                            f"Expected 'TIMEFRAME=value' or 'SYMBOL:TIMEFRAME=value'",
                            fg="red", err=True,
                        )
                        raise Exit(1)
                    key, value = entry.split('=', 1)

                    if live:
                        # Live mode: value must be a plugin-native symbol,
                        # not a file path. Native symbols can contain ``/``
                        # (e.g. CCXT ``binance:ETH/USDT`` or ``BTC/USDT:USDT``),
                        # so only reject values that *look* like a path: start
                        # with a path separator/relative prefix or carry the
                        # ``.ohlcv`` extension.
                        looks_like_path = (
                            value.startswith(('/', './', '../', '~/'))
                            or value.endswith('.ohlcv')
                        )
                        if looks_like_path:
                            secho(
                                f"--security value '{value}' looks like a file path. "
                                f"In --live mode the value must be a plugin-native "
                                f"symbol (e.g. EURUSD or binance:ETH/USDT).",
                                fg="red", err=True,
                            )
                            raise Exit(1)
                        # The chart's provider is the one that will serve
                        # the security warmup + live stream. Derive the
                        # timeframe from the key: ``SYMBOL:TIMEFRAME`` or
                        # bare ``TIMEFRAME``.
                        assert provider_data is not None  # --live already required provider mode
                        sec_tf = key.rsplit(':', 1)[-1] if ':' in key else key
                        sec_map[key] = PluginSymbol(
                            provider_name=provider_data.parsed_string.provider,
                            symbol=value,
                            timeframe=sec_tf,
                            config=getattr(provider_data.provider_instance, 'config', None),
                            ohlcv_dir=app_state.data_dir,
                        )
                    else:
                        # Backtest mode: value is a file stem or path.
                        sec_map[key] = backtest_security_path(value)
                security_data = sec_map

            # Add lib directory to Python path for library imports
            lib_dir = app_state.scripts_dir / "lib"
            lib_path_added = False
            if lib_dir.exists() and lib_dir.is_dir():
                sys.path.insert(0, str(lib_dir))
                lib_path_added = True

            if not live:
                _pin_timenow(last_bar_time)

            # Show loading spinner while importing
            with Progress(
                    SpinnerColumn(finished_text="[green]✓"),
                    TextColumn("{task.description}"),
            ) as loading_progress:
                loading_task = loading_progress.add_task("Loading PyneCore...", total=1)

                try:
                    # Create script runner (this is where the import happens).
                    # In live mode we pass the chart provider so security
                    # contexts without an explicit ``--security`` mapping can
                    # be auto-translated through the plugin's ``symbol_map``
                    # / ``normalize_symbol``.
                    chart_provider_name = None
                    chart_provider_instance = None
                    if live and provider_data is not None:
                        chart_provider_name = provider_data.parsed_string.provider
                        chart_provider_instance = provider_data.provider_instance
                    runner = ScriptRunner(script, ohlcv_iter, syminfo, last_bar_index=size - 1,
                                          last_bar_time=last_bar_time, live=live,
                                          live_history_trades=history_trades,
                                          history_as_of=(provider_data.as_of_ts
                                                         if provider_data else None),
                                          plot_path=plot_path, strat_path=strat_path, trade_path=trade_path,
                                          viz_path=viz_path if viz else None, viz_journal=viz_journal,
                                          security_data=security_data,
                                          magnifier_iter=magnifier_iter,
                                          magnifier_source_tf=magnifier_source_tf,
                                          chart_bar_window=chart_bar_window,
                                          broker_plugin=broker_plugin,
                                          broker_event_loop=broker_event_loop,
                                          broker_store_ctx=broker_store_ctx,
                                          broker_stop_settle_s=_STOP_FILL_SETTLE_S,
                                          log_ohlcv=live and not no_log_ohlcv,
                                          chart_provider_name=chart_provider_name,
                                          chart_provider_instance=chart_provider_instance,
                                          time_from=time_from_dt,
                                          chart_data_path=data_path,
                                          lossless_volume=lossless_volume,
                                          lossless_prices=lossless_prices,
                                          config_dir=app_state.config_dir,
                                          profile_path=profile_path)
                except ProfilerUnavailable as e:
                    secho(f"Error: {e}", err=True, fg=colors.RED)
                    raise Exit(1)
                finally:
                    # Remove lib directory from Python path
                    if lib_path_added:
                        sys.path.remove(str(lib_dir))

                # Mark as completed
                loading_progress.update(loading_task, completed=1)

            # --list-data: report the data this script needs (from
            # request.security calls) and exit before consuming any bar.
            if list_data:
                requirements = runner.list_data_requirements(
                    chart_symbol=f"{syminfo.prefix}:{syminfo.ticker}",
                    chart_tf=str(syminfo.period),
                    security_keys=set(security_data or {}),
                )
                _print_data_requirements(requirements, script.name)
                raise Exit(0)

            # Now that the script is loaded, start the live OHLCV stream.
            # ``live_ohlcv_generator`` eager-starts the WS connect (and
            # blocks until subscribed), so doing it AFTER the spinner keeps
            # the ``[BROKER] WS connect …`` lines below ``✓ Loading PyneCore``
            # in the user-visible startup log.
            if live and provider_data:
                import itertools
                live_iter = open_live_stream(
                    provider_data,
                    last_historical_timestamp=time_to_ts,
                    shutdown_timeout=shutdown_timeout,
                    event_loop=broker_event_loop,
                    # Broker mode: a warmup-connect failure must surface its
                    # real cause here, not be masked by start_broker()'s
                    # reconcile ("live connection not established").
                    raise_on_connect_failure=broker_plugin is not None,
                    # A stop landing in the bar wait settles the in-flight fills
                    # before the stream's teardown disconnects the provider
                    on_interrupt=(runner.settle_broker_stop
                                  if broker_plugin is not None else None),
                )
                runner.ohlcv_iter = itertools.chain(runner.ohlcv_iter, live_iter)

            # Start broker-side I/O (watch_orders task + startup reconcile).
            # No-op when ``broker_plugin`` is None.
            runner.start_broker()

            if live:
                # Share the Pine logger's Console with the live Progress so
                # `logger.info(...)` lines render above the spinner rather
                # than colliding with it. The PineRichHandler owns the only
                # Console; reusing it keeps Rich's Live-intercept coherent.
                live_console = None
                for _h in pyne_logger.handlers:
                    _h_console = getattr(_h, 'console', None)
                    if _h_console is not None:
                        live_console = _h_console
                        break

                # Latest quote snapshot — the tick hook updates these so the
                # spinner text carries the current bid/ask even between bar
                # closes.
                spinner_state: dict[str, Any] = {
                    'bid': None,
                    'ask': None,
                    'price': None,
                    'last_mid': None,
                    'arrow': ' ',
                }

                # Derive fixed price decimals from ``syminfo.mintick`` so the
                # spinner prices keep a constant width (``1.16830`` /
                # ``1.16837`` instead of ``1.1683`` / ``1.16837``). Matches
                # the same computation in ScriptRunner for the OHLCV log,
                # including the 2-decimal fallback for missing/zero mintick.
                _mintick = getattr(syminfo, 'mintick', 0.0) or 0.0
                price_decimals = mintick_decimals(_mintick) if _mintick > 0 else 2

                def _spinner_text() -> str:
                    bid = spinner_state['bid']
                    ask = spinner_state['ask']
                    d = price_decimals
                    position_obj = getattr(runner.script, 'position', None)
                    # Paper trading (``--live`` without ``--broker``) has no
                    # broker balance, so synthesise one from the simulator's
                    # equity — otherwise ``_broker_metrics_text`` bails early
                    # and the spinner shows no position/PnL at all.
                    balance = runner.broker_balance
                    if balance is None and position_obj is not None:
                        eq = _coerce_finite_float(getattr(position_obj, 'equity', None))
                        if eq is not None:
                            balance = {getattr(syminfo, 'currency', None) or '': eq}
                    metrics = _broker_metrics_text(
                        position_obj,
                        runner.broker_position_snapshot,
                        balance,
                        getattr(syminfo, 'currency', None),
                        d,
                        bid,
                        ask,
                        spinner_state['price'],
                    )
                    suffix = f"  {metrics}" if metrics else ""
                    if bid is not None and ask is not None:
                        arrow = spinner_state['arrow']
                        return (f"[green]{bid:.{d}f}[/] {arrow} "
                                f"[red]{ask:.{d}f}[/]{suffix}")
                    if bid is not None:
                        return f"[green]{bid:.{d}f}[/]{suffix}"
                    return f"Live streaming...{suffix}"

                # ``PYNE_NO_LIVE_SPINNER`` suppresses the per-tick spinner so
                # systemd journals / Docker logs only carry whole log lines
                # (``[BROKER]``, ``[OHLCV]``, Pine ``log.*``) instead of the
                # spinner refresh stream. Rich already disables live rendering
                # on a non-TTY, but explicit opt-out works regardless of how
                # the harness wires stdout.
                _spinner_disabled = (
                        os.environ.get("PYNE_NO_LIVE_SPINNER", "").lower()
                        not in ("", "0", "false", "no", "off")
                )
                if _spinner_disabled:
                    broker_info("live spinner disabled (PYNE_NO_LIVE_SPINNER)")

                # Live mode: spinner instead of progress bar (no known end time)
                # ``transient=True`` clears the live spinner row when the
                # ``with`` block exits, so the user-visible tail is just the
                # ``[BROKER]`` log lines — no orphaned ``⠧ Live —``
                # snapshot frozen below the final halt entry.
                with Progress(
                        SpinnerColumn(),
                        ExchangeClockColumn(runner.tz),
                        CustomTimeElapsedColumn(),
                        TextColumn("{task.description}"),
                        console=live_console,
                        transient=True,
                        disable=_spinner_disabled,
                        refresh_per_second=10,
                ) as progress:
                    task = progress.add_task(description="Live streaming...", total=None)

                    def _apply_quote(new_bid: float | None,
                                     new_ask: float | None) -> None:
                        """Update bid/ask + midline arrow in spinner_state."""
                        if new_bid is not None and new_ask is not None:
                            new_mid = (new_bid + new_ask) / 2
                            last_mid = spinner_state['last_mid']
                            if last_mid is not None:
                                if new_mid > last_mid:
                                    spinner_state['arrow'] = '[green]▲[/]'
                                elif new_mid < last_mid:
                                    spinner_state['arrow'] = '[red]▼[/]'
                            spinner_state['last_mid'] = new_mid
                        if new_bid is not None:
                            spinner_state['bid'] = new_bid
                        if new_ask is not None:
                            spinner_state['ask'] = new_ask

                    def cb_progress_live(current_time: datetime | None):
                        if current_time is not None:
                            progress.update(task, description=_spinner_text())

                    def cb_tick_live(candle):
                        extra = candle.extra_fields or {}
                        spinner_state['price'] = candle.close
                        # Prefer the live quote snapshot over ``candle.close``:
                        # on a closed OHLC bar ``candle.close`` is the previous
                        # period's bid-side close, not the current quote. Without
                        # a snapshot the bar's own ask side is the quote.
                        ask = extra.get('ask_close')
                        if ask is None and candle.ask is not None:
                            ask = candle.ask.close
                        _apply_quote(extra.get('bid_close', candle.close), ask)
                        progress.update(task, description=_spinner_text())

                    stop_reason = "completed"
                    # Translate SIGTERM (the signal a supervisor / lab harness
                    # sends to stop the process) into the same graceful
                    # ``KeyboardInterrupt`` path as Ctrl-C, so a stopped run
                    # still tears down cleanly and prints its completion +
                    # cleanup summary instead of dying at return_code=-15 with
                    # no visible cleanup outcome. Both signals go through the
                    # interrupt guard: a stop that arrives while the sync
                    # engine is dispatching is raised only once that sync
                    # completes, so an order already on the wire is recorded
                    # instead of landing behind the run's back.
                    _prev_signals: dict[int, Any] = {}
                    if broker_plugin is not None:
                        def _on_stop_signal(_signum, _frame):
                            request_interrupt()
                        for _signum in (signal.SIGTERM, signal.SIGINT):
                            try:
                                _prev_signals[_signum] = signal.getsignal(_signum)
                                signal.signal(_signum, _on_stop_signal)
                            except (ValueError, OSError):
                                # signal.signal only works on the main thread; a
                                # non-main-thread run keeps the default disposition.
                                _prev_signals.pop(_signum, None)
                    # When the spinner is suppressed (durable log / non-TTY)
                    # a quiet phase would otherwise print nothing for tens of
                    # seconds; a periodic heartbeat keeps the transcript alive
                    # so a healthy wait is distinguishable from a hang.
                    heartbeat = None
                    if _spinner_disabled:
                        _hb_interval = _resolve_heartbeat_interval(
                            os.environ.get("PYNE_HEARTBEAT_INTERVAL"),
                        )
                        if _hb_interval > 0.0:
                            heartbeat = _LiveHeartbeat(
                                _hb_interval,
                                lambda elapsed: broker_info(
                                    "still running — %.0fs elapsed, waiting "
                                    "for market data / next event", elapsed,
                                ),
                            )
                            heartbeat.start()
                    try:
                        runner.run(on_progress=cb_progress_live,
                                   on_tick=cb_tick_live)
                    except KeyboardInterrupt:
                        # Tear down the spinner BEFORE the follow-up log line
                        # so it doesn't print over the bottom of the screen
                        # — Rich captures the live region as a snapshot below
                        # every log line, otherwise leaving an orphan
                        # ``⠧ Live — …`` row in the transcript.
                        progress.stop()
                        if runner.quarantine_resume_restart:
                            stop_reason = "quarantine resumed"
                            restart_after_quarantine_resume = True
                            broker_warning(
                                "live streaming stopped — the operator resumed "
                                "the quarantine; restarting in place to re-sync "
                                "with the venue"
                            )
                        else:
                            stop_reason = "interrupted"
                            broker_warning("live streaming stopped (interrupted)")
                        # The runner waited for the fills before its teardown
                        # stopped the event stream (see ``broker_stop_settle_s``)
                        if runner.broker_stop_settled is False:
                            # Timed out, or cut short by a repeated stop
                            broker_warning(
                                "stopped with a dispatched market order still "
                                "waiting for its fill; the next run reconciles "
                                "it from the venue",
                            )
                    except BrokerManualInterventionError:
                        # The ``[BROKER] ERROR sync engine halted by …`` line
                        # logged from ``OrderSyncEngine._record_halt`` already
                        # carries the reason + context. Just append a single
                        # follow-up hint in the same Pine log format so the
                        # operator knows the strategy is no longer running.
                        progress.stop()
                        stop_reason = "manual intervention required"
                        broker_warning(
                            "live streaming stopped — manual intervention "
                            "required, resolve the broker-side state and "
                            "restart"
                        )
                    finally:
                        if heartbeat is not None:
                            heartbeat.stop()
                        for _signum, _previous in _prev_signals.items():
                            try:
                                signal.signal(_signum, _previous)
                            except (ValueError, OSError):
                                pass

                # Explicit end-of-run summary so a successful broker command
                # confirms it stopped and reports its final position/equity
                # instead of an unadorned end of transcript.
                if broker_plugin is not None:
                    broker_info("%s", _format_run_completion_summary(
                        stop_reason,
                        getattr(runner.script, 'position', None),
                        runner.broker_position_snapshot,
                        runner.broker_balance,
                        getattr(syminfo, 'currency', None),
                    ))
                    broker_run_reached_summary = True

            else:
                run_backtest_with_progress(runner, time_from_display, time_to_display,
                                           total_seconds)

                if profile_path is not None:
                    print_profile_report(json.loads(profile_path.read_text(encoding='utf-8')),
                                         profile_path, profile_top, console)
        except PyneError as e:
            # The script's own error: one line, as `pyneaot run` writes it
            if show_traceback:
                console.print_exception(show_locals=False)
            secho(f"Error: {e.report(pine=from_pine)}", err=True, fg=colors.RED)
            raise Exit(1)
        except Exception as e:
            error = internal(e)
            if error is None:
                raise
            # PyneCore's own failure: the traceback is what a report needs
            console.print_exception(show_locals=False)
            secho(f"Error: {error.report()}", err=True, fg=colors.RED)
            secho(INTERNAL_ERROR_REPORT, err=True, fg=colors.RED)
            raise Exit(1)
        finally:
            # A broker run that never reached its completion summary ended
            # abnormally (early raise / crash mid-run). Narrate that the
            # teardown below still runs and what to do next, so a failed run
            # does not end without a visible cleanup outcome.
            if broker_plugin is not None and not broker_run_reached_summary:
                broker_warning(
                    "run ended before a clean stop — closing broker storage "
                    "and stopping the event loop; review the last [BROKER] "
                    "entries and re-run to reconcile before trading again"
                )
            # Close the live OHLCV consumer first so its ``finally`` can
            # signal ``stop_event`` and join the producer thread WHILE the
            # broker event loop is still alive — otherwise the producer's
            # ``run_coroutine_threadsafe`` future never completes and the
            # join times out at ``shutdown_timeout + 5`` seconds.
            if live_iter is not None:
                try:
                    live_iter.close()
                except RuntimeError:
                    # Generator.close() only raises RuntimeError when the
                    # generator swallows GeneratorExit and yields again —
                    # harmless during teardown.
                    pass
            # Close the broker storage run cleanly — happy-path teardown.
            # Crash paths (SIGKILL, OOM) are handled by the storage's
            # stale-run cleanup.
            if broker_store_ctx is not None:
                broker_store_ctx.close()
            if broker_store is not None:
                broker_store.close()
            # Stop the broker event-loop pump thread before process exit, then
            # close the loop only once its thread has actually exited
            # ``run_forever``. Closing a still-running loop raises
            # ``RuntimeError: Cannot close a running event loop``.
            if broker_event_loop is not None:
                if not _shutdown_broker_event_loop(
                    broker_event_loop, broker_event_loop_thread, shutdown_timeout,
                ):
                    broker_warning(
                        "Broker event loop did not stop within %.0fs; "
                        "leaving it open to avoid a close-while-running crash.",
                        shutdown_timeout,
                    )
        if restart_after_quarantine_resume:
            _restart_in_place()
