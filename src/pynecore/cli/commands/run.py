import asyncio
import os
import queue
import signal
import threading
import time
import sys
import tomllib

from pathlib import Path
from dataclasses import replace as dc_replace
from datetime import datetime, timedelta, UTC, tzinfo
from typing import Any

from typer import Option, Argument, secho, Exit, colors
from rich.progress import (Progress, SpinnerColumn, TextColumn, BarColumn,
                           ProgressColumn, Task, TimeElapsedColumn, TimeRemainingColumn)
from rich.text import Text
from rich.console import Console

from ..app import app, app_state
from ..pluggable import PluggableCommand

from ...utils.rich.date_column import DateColumn
from pynecore.core.ohlcv_file import OHLCVReader
from pynecore.core.data_converter import DataConverter, DataFormatError, ConversionError
from pynecore.core.aggregator import validate_aggregation
from pynecore.lib.log import logger as pyne_logger
from pynecore.lib.timeframe import in_seconds

from pynecore.core.broker.exceptions import BrokerManualInterventionError
from pynecore.lib.log import broker_info, broker_warning
from pynecore.core.syminfo import SymInfo, mintick_decimals
from pynecore.core.script_runner import ScriptRunner, DataRequirements, SecurityRequirement
from pynecore.pynesys.compiler import PyneComp
from pynecore.core.provider_string import ProviderString, is_provider_string, parse_provider_string
from pynecore.core.live_runner import live_ohlcv_generator
from ...cli.utils.api_error_handler import APIErrorHandler

__all__ = []


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
    Parse a --from or --to parameter value.

    :param value: The raw string value.
    :param allow_bars: If True, allow negative numbers as bar counts.
    :return: A datetime, a negative int (bar count), or None.
    """
    if value is None:
        return None
    value: str = value.strip()

    # Negative number = bar count (only for --from in provider mode)
    if allow_bars and value.startswith('-'):
        try:
            bars = int(value)
            return bars
        except ValueError:
            pass

    # Positive number = days back
    try:
        days = int(value)
        if days < 0:
            secho("Error: Days cannot be negative (use negative numbers only with provider mode for bar count)",
                  err=True, fg=colors.RED)
            raise Exit(1)
        return (datetime.now(UTC) - timedelta(days=days)).replace(second=0, microsecond=0)
    except ValueError:
        pass

    # Date string
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        secho(f"Error: Invalid date or number: '{value}'", err=True, fg=colors.RED)
        raise Exit(1)


class _ProviderData:
    """Result of provider data download, including the provider instance for live mode."""

    def __init__(self, ohlcv_path: Path, syminfo: 'SymInfo', parsed_string: ProviderString,
                 provider_instance=None, time_from_ts: int | None = None):
        self.ohlcv_path = ohlcv_path
        self.syminfo = syminfo
        self.provider_instance = provider_instance
        self.parsed_string: ProviderString = parsed_string
        # Exact start timestamp that yields exactly the requested bar
        # count in ``-N bars`` mode — None when the caller should use
        # the file's natural start (date/days mode, no bar target).
        self.time_from_ts = time_from_ts


def _download_provider_data(provider_str: str, time_from_str: str | None) -> _ProviderData:
    """
    Download historical data from a provider and return the result.

    :param provider_str: Provider string (e.g. "ccxt:BYBIT:BTC/USDT:USDT@1D").
    :param time_from_str: The --from parameter value (date, days, or -bars).
    :return: _ProviderData with ohlcv_path, syminfo, and provider instance.
    """
    from pynecore.core.plugin import load_plugin, ProviderPlugin
    from pynecore.core.config import ensure_config
    from pynecore.lib.timeframe import in_seconds

    # Load the provider plugin first so we know whether it is multi-broker,
    # which controls how the provider string is split (broker vs. symbol).
    provider_name = provider_str.split(':', 1)[0].lower()
    provider_class = load_plugin(provider_name)
    if not issubclass(provider_class, ProviderPlugin):
        secho(f"Plugin '{provider_name}' is not a data provider.", err=True, fg=colors.RED)
        raise Exit(1)

    ps = parse_provider_string(provider_str, require_timeframe=True,
                               multi_broker=provider_class.multi_broker)
    # Store the normalized (lowercased) provider name so that later
    # case-sensitive ``load_plugin()`` lookups for security/auto-rate
    # contexts succeed even when the user typed the provider in a
    # different case (e.g. ``CCXT:...``).
    ps = dc_replace(ps, provider=provider_name)

    # Default to -500 bars if --from not specified in provider mode
    if not time_from_str:
        time_from_str = "-500"

    time_from_value = _parse_time_value(time_from_str, allow_bars=True)
    time_to_dt = datetime.now(UTC).replace(second=0, microsecond=0)

    # Convert bar count to time range. ``bar_count`` being set signals
    # the "-N bars" mode — we then guarantee at least N *real* bars
    # (exchange-provided, non-gap-fill) after download, extending the
    # from-timestamp on miss. Date/days ranges are left untouched.
    tf_seconds = in_seconds(ps.timeframe)
    bar_count: int | None = None
    if isinstance(time_from_value, int) and time_from_value < 0:
        bc = abs(time_from_value)
        bar_count = bc
        # Pad the request by one bar to absorb the still-forming current
        # bar that closed-bars-only providers (e.g. Capital.com) filter
        # out of history responses. Without this, every ``-N`` run
        # against a now-aligned end-time would burn a wasted retry pass
        # (``real_bars == N - 1`` on first attempt → retry → success).
        time_from_dt = time_to_dt - timedelta(seconds=tf_seconds * (bc + 1))
    else:
        assert isinstance(time_from_value, datetime)
        time_from_dt = time_from_value

    # Load config
    config = None
    config_cls: type | None = getattr(provider_class, 'Config', None)
    if config_cls is not None:
        config = ensure_config(config_cls,
                               app_state.config_dir / 'plugins' / f'{provider_name}.toml')

    # Create provider instance. ``provider_symbol`` re-folds the broker into
    # the symbol for multi-broker providers, which split it off internally.
    provider_instance: ProviderPlugin = provider_class(
        symbol=ps.provider_symbol, timeframe=ps.timeframe,
        ohlcv_dir=app_state.data_dir, config=config
    )

    # Fetch symbol info
    with Progress(SpinnerColumn(finished_text="[green]✓"), TextColumn("{task.description}")) as progress:
        task = progress.add_task("Fetching symbol info...", total=1)
        syminfo = provider_instance.get_symbol_info(force_update=not provider_instance.is_symbol_info_exists())
        progress.update(task, completed=1)

    # Download OHLCV data (always fresh in provider mode). In bar-count
    # mode we may re-download with an extended ``from`` until we hit the
    # target — some feeds omit minutes with no ticks (CFD quiet hours,
    # illiquid futures), and ``--from -500`` must mean 500 real bars.
    # The Progress wrapper lives outside the retry loop so a gap-driven
    # second pass updates the same spinner instead of stamping a
    # duplicate ``Downloading OHLCV data...`` line.
    max_retries = 4
    with Progress(
            SpinnerColumn(finished_text="[green]✓"),
            TextColumn("{task.description}"),
            DateColumn(),
            BarColumn(),
            TimeElapsedColumn(),
            "/",
            TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task(
            "Downloading OHLCV data...", total=1, start_time=time_from_dt,
        )
        for attempt in range(max_retries + 1):
            with provider_instance as ohlcv_writer:
                ohlcv_writer.seek(0)
                ohlcv_writer.truncate()

                time_from_dl = time_from_dt.replace(tzinfo=None) if time_from_dt.tzinfo else time_from_dt
                time_to_dl = time_to_dt.replace(tzinfo=None) if time_to_dt.tzinfo else time_to_dt

                total_seconds = int((time_to_dl - time_from_dl).total_seconds())

                progress.update(
                    task, total=total_seconds, completed=0,
                    start_time=time_from_dl,
                )

                def cb_progress(current_time: datetime):
                    elapsed_seconds = int((current_time - time_from_dl).total_seconds())
                    progress.update(task, completed=elapsed_seconds)

                provider_instance.download_ohlcv(time_from_dl, time_to_dl, on_progress=cb_progress)

            if bar_count is None:
                break

            # Count real bars (``volume >= 0``) in the requested range —
            # gap-fill rows (``volume == -1``) emitted by the OHLCV writer
            # don't count against the target.
            with OHLCVReader(provider_instance.ohlcv_path) as r:  # type: ignore[arg-type]
                real_bars = sum(1 for _ in r.read_from(
                    int(time_from_dt.timestamp()),
                    int(time_to_dt.timestamp()),
                    skip_gaps=True,
                ))

            if real_bars >= bar_count:
                break
            if attempt == max_retries:
                secho(
                    f"Warning: requested {bar_count} bars, only got {real_bars} "
                    f"real bars after {max_retries + 1} attempts (feed has "
                    f"sparse coverage).",
                    fg=colors.YELLOW,
                )
                break

            # Extend the range by the shortfall plus a proportional buffer so
            # we likely finish in one extra pass instead of retrying again.
            missing = bar_count - real_bars
            gap_ratio = missing / real_bars if real_bars else 1.0
            extra_bars = int(missing * (1.0 + gap_ratio)) + 10
            time_from_dt -= timedelta(seconds=tf_seconds * extra_bars)

    assert provider_instance.ohlcv_path is not None

    # For bar-count mode, pin the start timestamp to the N-th last real
    # bar so the reader serves *exactly* N bars, not the over-fetched
    # surplus we used to guarantee coverage through gaps.
    exact_from_ts: int | None = None
    if bar_count is not None:
        with OHLCVReader(provider_instance.ohlcv_path) as r:  # type: ignore[arg-type]
            real_ts = [b.timestamp for b in r.read_from(
                0, int(time_to_dt.timestamp()), skip_gaps=True,
            )]
        if len(real_ts) >= bar_count:
            exact_from_ts = real_ts[-bar_count]

    return _ProviderData(
        ohlcv_path=provider_instance.ohlcv_path,
        syminfo=syminfo,
        provider_instance=provider_instance,
        parsed_string=ps,
        time_from_ts=exact_from_ts,
    )


#: Startup provider-download retry backoff (seconds): the first wait and the
#: ceiling the exponential growth saturates at. Internal tunables — deliberately
#: not user config: a live bot should keep trying until the broker returns, and
#: ~1 minute is a sane poll cadence through a multi-hour maintenance window.
_PROVIDER_RETRY_BASE_DELAY = 2.0
_PROVIDER_RETRY_MAX_DELAY = 60.0


def _wait_before_retry(delay: float) -> None:
    """Block for ``delay`` seconds interruptibly, without a bare sleep.

    Waits on a never-set :class:`threading.Event`, an event-driven,
    deadline-bounded primitive: a ``Ctrl-C`` (``KeyboardInterrupt`` in the main
    thread) breaks out of the backoff at once instead of stalling for the full
    delay, and there is no busy-poll.

    :param delay: Maximum seconds to wait.
    """
    threading.Event().wait(timeout=delay)


def _download_provider_data_resilient(
        provider_str: str, time_from_str: str | None, *, retry_transient: bool,
) -> _ProviderData:
    """Download provider data, riding out transient broker outages.

    For a long-running run (``--broker`` / ``--live``) a *transient* provider
    failure — broker maintenance, a lost route, a dropped connection — must not
    strand the bot: we wait with capped exponential backoff and keep retrying
    until the feed returns or the operator interrupts (``Ctrl-C``). A
    *permanent* failure (unknown symbol, bad credentials, wrong account mode),
    and every failure of a one-shot backtest, still exits immediately with a
    clean one-line message instead of a traceback — mirroring ``pyne data
    download``.

    Retrying is intentionally unbounded for the transient class: a bot is meant
    to run until told to stop, so giving up after N attempts would re-introduce
    the very crash we are fixing for any outage longer than the backoff budget.
    The safeguard against looping on a misclassified error is that the transient
    set is narrow and source-specific (see ``is_retryable_provider_error`` and
    the cTrader ``_CONNECTION_CLASS_CODES``), and every wait is logged.

    :param provider_str: The provider string (e.g. ``ctrader:pepperstoneuk:BTCUSD@1``).
    :param time_from_str: The ``--from`` value (date, days, or ``-bars``).
    :param retry_transient: Whether to wait-and-retry on transient errors —
        ``True`` only for long-running ``--broker`` / ``--live`` runs.
    :return: The downloaded provider data.
    """
    from pynecore.core.plugin import ProviderError, is_retryable_provider_error

    delay = _PROVIDER_RETRY_BASE_DELAY
    attempt = 0
    while True:
        try:
            return _download_provider_data(provider_str, time_from_str)
        except ProviderError as e:
            if not (retry_transient and is_retryable_provider_error(e)):
                secho(f"Error: {e}", err=True, fg=colors.RED)
                raise Exit(1)
            attempt += 1
            secho(
                f"Provider temporarily unavailable ({e}); waiting {int(delay)}s "
                f"before retry #{attempt} (press Ctrl-C to abort)...",
                err=True, fg=colors.YELLOW,
            )
            _wait_before_retry(delay)
            delay = min(delay * 2.0, _PROVIDER_RETRY_MAX_DELAY)


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
            suffix = (" [mapping present]" if r.has_security_mapping
                      else f" [needs --security '{r.timeframe}=<base file>' in backtest]")
            lines.append(f"  -> {r.symbol} @ {r.timeframe}{_tags(r)}{suffix}")

    if requirements.cross_symbol:
        lines.append("")
        lines.append("Cross-symbol data (separate download / --security mapping required):")
        for r in requirements.cross_symbol:
            suffix = (" [mapping present]" if r.has_security_mapping
                      else " ! no --security mapping")
            lines.append(f"  -> {r.symbol} @ {r.timeframe}{_tags(r)}{suffix}")

    if requirements.dynamic:
        lines.append("")
        lines.append("Dynamic data (symbol or timeframe only known at runtime):")
        lines.append(f"  ! {len(requirements.dynamic)} request.security() "
                     f"call(s) with a runtime symbol/timeframe")
        lines.append("    -> cannot be listed statically; inspect the script source")

    console.print("\n".join(lines), markup=False, highlight=False)


@app.command(cls=PluggableCommand)
def run(
        script: Path = Argument(..., dir_okay=False, file_okay=True, help="Script to run (.py or .pine)"),
        data: str = Argument(...,
                             help="Data file (*.ohlcv, *.csv) or provider string "
                                  "(e.g. ccxt:BYBIT:BTC/USDT:USDT@1D)"),
        time_from: str | None = Option(None, '--from', '-f',
                                       metavar="[DATE|DAYS|-BARS]",
                                       help="Start: date (2025-01-01), days back (30), "
                                            "or -N bars back (-500). Default: -500 bars in provider mode."),
        time_to: str | None = Option(None, '--to', '-t',
                                     metavar="[DATE|DAYS]",
                                     help="End: date or days from start (default: end of data or now)"),
        plot_path: Path | None = Option(None, "--plot", "-pp",
                                        help="Path to save the plot data",
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

):
    """
    Run a script (.py or .pine)

    The system automatically searches for the workdir folder in the current and parent directories.
    If not found, it creates or uses a workdir folder in the current directory.

    If [bold]script[/] path is a name without full path, it will be searched in the [italic]"workdir/scripts"[/] directory.
    Similarly, if [bold]data[/] path is a name without full path, it will be searched in the [italic]"workdir/data"[/] directory.
    The [bold]plot_path[/], [bold]strat_path[/], and [bold]trade_path[/] work the same way - if they are names without full paths,
    they will be saved in the [italic]"workdir/output"[/] directory.

    [bold]Data Source:[/bold]
    The [bold]data[/] argument accepts either a file path or a provider string:
    \b
      File mode:     pyne run script.py data.csv
      Provider mode: pyne run script.py ccxt:BYBIT:BTC/USDT:USDT@1D -f -500

    In provider mode, historical data is downloaded automatically. The --from/-f parameter
    accepts: date (2025-01-01), days back (30), or -N bars back (-500). Default: -500 bars.

    [bold]Pine Script Support:[/bold]
    Pine Script (.pine) files are automatically compiled to Python (.py) before execution.
    A valid [bold]PyneSys API[/bold] key is required. Get one at [blue]https://pynesys.io[/blue].

    [bold]Data Formats:[/bold]
    Supports CSV, TXT, JSON, and OHLCV data files. Non-OHLCV files are automatically converted.
    """  # noqa

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
        secho(f"Script file '{script}' not found!", fg="red", err=True)
        raise Exit(1)

    # Handle .pine files - compile them first
    if script.suffix == ".pine":
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
                secho(f"Script file '{script}' not found!", fg="red", err=True)
                raise Exit(1)

    # --- Data resolution: provider string or file path ---
    provider_mode = is_provider_string(data)
    provider_data = None

    if live and not provider_mode:
        secho("Error: --live is only available in provider mode.", err=True, fg=colors.RED)
        raise Exit(1)

    if provider_mode:
        # Provider mode: download historical warmup data + syminfo. A
        # long-running --broker/--live run rides out transient broker outages
        # (maintenance, lost route) instead of dying mid-startup; a one-shot
        # backtest and any permanent failure (unknown symbol, bad credentials)
        # still fail fast with a clean one-line error, not a traceback.
        provider_data = _download_provider_data_resilient(
            data, time_from, retry_transient=(live or broker),
        )
        data_path, syminfo = provider_data.ohlcv_path, provider_data.syminfo
    else:
        # File mode: resolve path, convert if needed
        data_path = Path(data)

        if len(data_path.parts) == 1:
            data_path = app_state.data_dir / data_path

        if data_path.suffix == "":
            ohlcv_path = data_path.with_suffix(".ohlcv")
            csv_path = data_path.with_suffix(".csv")
            if ohlcv_path.exists():
                data_path = ohlcv_path
            elif csv_path.exists():
                data_path = csv_path
            else:
                data_path = ohlcv_path

        if data_path.suffix != ".ohlcv":
            try:
                converter = DataConverter()
                if converter.is_conversion_required(data_path):
                    detected_symbol, detected_provider = DataConverter.guess_symbol_from_filename(data_path)
                    if not detected_symbol:
                        detected_symbol = data_path.stem.upper()
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
                        data_path = data_path.with_suffix(".ohlcv")
                        progress.update(task, completed=1)
                else:
                    data_path = data_path.with_suffix(".ohlcv")
            except (DataFormatError, ConversionError) as e:
                secho(f"Conversion failed: {e}", fg="red", err=True)
                secho("Please convert the file manually:", fg="red")
                secho(f"pyne data convert-from {data_path}", fg="yellow")
                raise Exit(1)

        if not data_path.exists():
            secho(f"Data file not found: {data_path.name}", fg="red", err=True)
            raise Exit(1)

        try:
            syminfo = SymInfo.load_toml(data_path.with_suffix(".toml"))
        except FileNotFoundError:
            secho(f"Symbol info file '{data_path.with_suffix('.toml')}' not found!", fg="red", err=True)
            raise Exit(1)

    # --- Output paths ---
    if plot_path and plot_path.suffix != ".csv":
        plot_path = plot_path.with_suffix(".csv")
    if not plot_path:
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

    # Validate and process --timeframe option
    magnifier_mode = False
    magnifier_source_tf: str | None = None
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
            magnifier_mode = True
            magnifier_source_tf = data_tf

    # --- Open data and run ---
    with OHLCVReader(data_path) as reader:
        # Parse time range
        time_from_dt = _parse_time_value(time_from) if time_from and not provider_mode else None
        time_to_dt = _parse_time_value(time_to) if time_to else None

        if not time_from_dt:
            # Provider bar-count mode pins the start to the N-th last
            # real bar; otherwise use the file's natural start.
            if provider_data is not None and provider_data.time_from_ts is not None:
                time_from_dt = datetime.fromtimestamp(
                    provider_data.time_from_ts, UTC,
                )
            else:
                time_from_dt = reader.start_datetime
        if not time_to_dt:
            time_to_dt = reader.end_datetime

        assert isinstance(time_from_dt, datetime) and isinstance(time_to_dt, datetime)
        time_from_ts = int(time_from_dt.timestamp())
        time_to_ts = int(time_to_dt.timestamp())

        # Remove timezone for display purposes
        time_from_display = time_from_dt.replace(tzinfo=None)
        time_to_display = time_to_dt.replace(tzinfo=None)

        total_seconds = int((time_to_display - time_from_display).total_seconds())

        # Get the iterator using the correct UTC timestamps
        size = reader.get_size(time_from_ts, time_to_ts)
        # Pine anchors ``last_bar_time`` on historical bars to the chart's final
        # bar, known up front from the data window. Scan back over the writer's
        # gap-fill tail (``volume == -1`` records; ``not (volume < 0)`` keeps
        # NaN-volume real bars) for the last real bar of the window.
        last_bar_time = None
        start_pos, end_pos = reader.get_positions(time_from_ts, time_to_ts)
        for pos in range(end_pos - 1, start_pos - 1, -1):
            window_tail_bar = reader.read(pos)
            if not (window_tail_bar.volume < 0):
                last_bar_time = int(window_tail_bar.timestamp * 1000)
                break
        magnifier_iter = None
        if magnifier_mode:
            # Sub-TF data goes to magnifier; ohlcv_iter is unused (replaced in ScriptRunner)
            magnifier_iter = reader.read_from(time_from_ts, time_to_ts)
            ohlcv_iter = iter([])
        else:
            ohlcv_iter = reader.read_from(time_from_ts, time_to_ts)

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
            # already run (``_download_provider_data`` drove it), so the
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
            # _download_provider_data has driven authentication through the
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
                from pynecore.core.plugin.live_provider import LiveProviderPlugin

                if not isinstance(provider_data.provider_instance, LiveProviderPlugin):
                    secho(f"Plugin '{provider_data.parsed_string.provider}' does not support live data.",
                          err=True, fg=colors.RED)
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
                        sec_path = Path(value)
                        if len(sec_path.parts) == 1:
                            sec_path = app_state.data_dir / sec_path
                        # ``value`` may carry the ``.ohlcv`` data extension or be a
                        # bare stem. Only ``.ohlcv`` is meaningful — a dot inside the
                        # name belongs to the symbol (e.g. a perpetual ``BTCUSDT.P``),
                        # so append/strip by name instead of ``with_suffix`` which
                        # would clobber the symbol's own dotted tail.
                        if sec_path.name.endswith('.ohlcv'):
                            sec_path = sec_path.with_name(sec_path.name[:-len('.ohlcv')])
                        ohlcv_check = sec_path.with_name(sec_path.name + '.ohlcv')
                        if not ohlcv_check.exists():
                            secho(
                                f"Security data not found: {ohlcv_check}",
                                fg="red", err=True,
                            )
                            raise Exit(1)
                        sec_map[key] = str(sec_path)
                security_data = sec_map

            # Add lib directory to Python path for library imports
            lib_dir = app_state.scripts_dir / "lib"
            lib_path_added = False
            if lib_dir.exists() and lib_dir.is_dir():
                sys.path.insert(0, str(lib_dir))
                lib_path_added = True

            # Set live mode flags before ScriptRunner creation
            if live:
                from pynecore import lib as _lib
                _lib._is_live = True
                _lib._strategy_suppressed = True

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
                                          last_bar_time=last_bar_time,
                                          plot_path=plot_path, strat_path=strat_path, trade_path=trade_path,
                                          viz_path=viz_path if viz else None, viz_journal=viz_journal,
                                          security_data=security_data,
                                          magnifier_iter=magnifier_iter,
                                          magnifier_source_tf=magnifier_source_tf,
                                          broker_plugin=broker_plugin,
                                          broker_event_loop=broker_event_loop,
                                          broker_store_ctx=broker_store_ctx,
                                          log_ohlcv=live and not no_log_ohlcv,
                                          chart_provider_name=chart_provider_name,
                                          chart_provider_instance=chart_provider_instance,
                                          time_from=time_from_dt,
                                          chart_data_path=data_path)
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
                assert provider_data.parsed_string.timeframe is not None
                live_iter = live_ohlcv_generator(
                    provider=provider_data.provider_instance,
                    symbol=provider_data.parsed_string.symbol,
                    timeframe=provider_data.parsed_string.timeframe,
                    syminfo=syminfo,
                    last_historical_timestamp=time_to_ts,
                    shutdown_timeout=shutdown_timeout,
                    event_loop=broker_event_loop,
                    # Broker mode: a warmup-connect failure must surface its
                    # real cause here, not be masked by start_broker()'s
                    # reconcile ("live connection not established").
                    raise_on_connect_failure=broker_plugin is not None,
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
                        # period's bid-side close, not the current quote.
                        _apply_quote(
                            extra.get('bid_close', candle.close),
                            extra.get('ask_close'),
                        )
                        progress.update(task, description=_spinner_text())

                    stop_reason = "completed"
                    # Translate SIGTERM (the signal a supervisor / lab harness
                    # sends to stop the process) into the same graceful
                    # ``KeyboardInterrupt`` path as Ctrl-C, so a stopped run
                    # still tears down cleanly and prints its completion +
                    # cleanup summary instead of dying at return_code=-15 with
                    # no visible cleanup outcome.
                    _prev_sigterm = None
                    _sigterm_installed = False
                    if broker_plugin is not None:
                        def _on_sigterm(_signum, _frame):
                            raise KeyboardInterrupt
                        try:
                            _prev_sigterm = signal.getsignal(signal.SIGTERM)
                            signal.signal(signal.SIGTERM, _on_sigterm)
                            _sigterm_installed = True
                        except (ValueError, OSError):
                            # signal.signal only works on the main thread; a
                            # non-main-thread run keeps the default disposition.
                            _sigterm_installed = False
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
                        stop_reason = "interrupted"
                        broker_warning("live streaming stopped (interrupted)")
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
                        if _sigterm_installed:
                            try:
                                signal.signal(signal.SIGTERM, _prev_sigterm)
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
                # Batch mode: progress bar with time range
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

                    # Create queue for progress updates
                    progress_queue = queue.Queue()
                    stop_event = threading.Event()

                    def progress_worker():
                        """Worker thread that updates progress bar at 30Hz"""
                        last_update = 0
                        while not stop_event.is_set():
                            try:
                                # Drain all pending updates
                                current_time = None
                                while True:
                                    try:
                                        current_time = progress_queue.get_nowait()
                                    except queue.Empty:
                                        break

                                # Update progress if we have new data
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
                        """Callback that just puts timestamp in queue"""
                        try:
                            progress_queue.put_nowait(current_time)
                        except queue.Full:
                            pass

                    try:
                        runner.run(on_progress=cb_progress)

                        progress_queue.put(time_to_display)
                        time.sleep(0.05)

                        progress.update(task, completed=total_seconds)
                    finally:
                        stop_event.set()
                        worker.join(timeout=0.1)
                        progress.refresh()
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
