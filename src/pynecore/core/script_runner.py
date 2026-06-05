from typing import Iterable, Iterator, Callable, TYPE_CHECKING, Any, cast
from types import ModuleType
import asyncio
import sys
from pathlib import Path
from datetime import datetime, UTC

from pynecore import lib
from pynecore.lib.log import broker_info, ohlcv_info, sim_info
from pynecore.types.ohlcv import OHLCV
from pynecore.types.na import na_float
from pynecore.core.syminfo import SymInfo, mintick_decimals
from pynecore.core.csv_file import CSVWriter
from pynecore.core.strategy_stats import calculate_strategy_statistics, write_strategy_statistics_csv
from pynecore.core.var_snapshot import VarSnapshot

from pynecore.types import script_type
from pynecore.core.plugin.live_provider import PluginSymbol

if TYPE_CHECKING:
    from multiprocessing.process import BaseProcess
    from zoneinfo import ZoneInfo
    from pynecore.core.script import script
    from pynecore.lib.strategy import Trade, SimPosition
    from pynecore.core.broker.position import BrokerPosition
    from pynecore.core.plugin.broker import BrokerPlugin
    from pynecore.core.plugin.live_provider import LiveProviderPlugin
    from pynecore.core.broker.sync_engine import OrderSyncEngine
    from pynecore.core.broker.storage import RunContext
    from pynecore.core.broker.models import ScriptRequirements

__all__ = [
    'import_script',
    'ScriptRunner',
    'LIVE_TRANSITION',
]

LIVE_TRANSITION = OHLCV(timestamp=-1, open=-1, high=-1, low=-1, close=-1, volume=-1)
"""Sentinel inserted between historical and live OHLCV data in the iterator."""


def _close_price_or_none() -> float | None:
    """Best-effort current bar close, ``None`` before any bar is ingested.

    The runner rebinds ``lib.close`` to a float on every bar; at startup
    (and during a pre-bar refresh window) it still holds the
    :class:`~pynecore.types.source.Source` sentinel placeholder. Returning
    ``None`` in that case lets the broker engine's partial-bracket WATCH
    phase short-circuit cleanly until a real price lands.
    """
    val = getattr(lib, 'close', None)
    if isinstance(val, (int, float)):
        return float(val)
    return None


def import_script(script_path: Path) -> ModuleType:
    """
    Import the script
    """
    from importlib import import_module
    import re

    # Check for @pyne magic doc comment before importing (prevents import errors)
    # Without this user may get strange errors which are very hard to debug
    try:
        with open(script_path, 'r') as f:
            # Read only the first few lines to check for docstring
            content = f.read(1024)  # Read first 1KB, should be enough for docstring check

        # Check if file starts with a docstring containing @pyne
        if not re.search(r'^(""".*?@pyne.*?"""|\'\'\'.*?@pyne.*?\'\'\')',
                         content, re.DOTALL | re.MULTILINE):
            raise ImportError(
                f"Script '{script_path}' must have a magic doc comment containing "
                f"'@pyne' at the beginning of the file!"
            )
    except (OSError, IOError) as e:
        raise ImportError(f"Could not read script file '{script_path}': {e}")

    # Add script's directory to Python path temporarily
    sys.path.insert(0, str(script_path.parent))
    try:
        # Import hook is registered at pynecore package import time (see pynecore/__init__.py),
        # so any subsequent import goes through PyneLoader and AST transformers.
        module = import_module(script_path.stem)
    finally:
        # Remove the directory from path
        sys.path.pop(0)

    if not hasattr(module, 'main'):
        raise ImportError(f"Script '{script_path}' must have a 'main' function to run!")

    return module


def _round_price(price: float):
    """
    Round price to 6 significant digits to clean float32 storage artifacts
    without destroying sub-mintick precision.

    TradingView does NOT round OHLC data to syminfo.mintick — scripts see
    the raw data (e.g. close=4.125 even when mintick=0.01). The float32
    OHLCV format introduces small errors (e.g. 4.12 → 4.1199998856) that
    this function cleans by rounding to 6 significant digits (float32 has ~7).
    """
    if price == 0.0:
        return 0.0
    from math import log10, floor
    magnitude = floor(log10(abs(price)))
    precision = 5 - magnitude  # 6 significant digits
    return round(price, precision)


# noinspection PyShadowingNames,PyUnusedLocal
def _set_lib_properties(ohlcv: OHLCV, bar_index: int, tz: 'ZoneInfo', lib: ModuleType):
    """
    Set lib properties from OHLCV
    """
    if TYPE_CHECKING:  # This is needed for the type checker to work
        from .. import lib

    lib.bar_index = lib.last_bar_index = bar_index

    lib.open = _round_price(ohlcv.open)
    lib.high = _round_price(ohlcv.high)
    lib.low = _round_price(ohlcv.low)
    lib.close = _round_price(ohlcv.close)

    lib.volume = ohlcv.volume
    lib.extra_fields = ohlcv.extra_fields if ohlcv.extra_fields else {}

    # Pine's ``bid``/``ask`` only carry real values on the ``"1T"`` (tick) feed; on every
    # other timeframe TradingView reports ``na``. PyneCore does not support tick data, so
    # they are always ``na`` — matching TV behaviour on bar timeframes.
    lib.bid = lib.ask = na_float

    lib.hl2 = (lib.high + lib.low) / 2.0
    lib.hlc3 = (lib.high + lib.low + lib.close) / 3.0
    lib.ohlc4 = (lib.open + lib.high + lib.low + lib.close) / 4.0
    lib.hlcc4 = (lib.high + lib.low + 2 * lib.close) / 4.0

    dt = lib._datetime = datetime.fromtimestamp(ohlcv.timestamp, UTC).astimezone(tz)
    lib._time = lib.last_bar_time = int(dt.timestamp() * 1000)  # PineScript representation of time


# noinspection PyUnusedLocal
def _set_lib_syminfo_properties(syminfo: SymInfo):
    """
    Set syminfo library properties from this object
    """
    for slot_name in syminfo.__slots__:  # type: ignore
        value = getattr(syminfo, slot_name)
        if value is not None:
            try:
                setattr(lib.syminfo, slot_name, value)
            except AttributeError:
                pass

    lib.syminfo.root = syminfo.ticker
    lib.syminfo.tickerid = syminfo.prefix + ':' + syminfo.ticker
    lib.syminfo.ticker = lib.syminfo.tickerid

    lib.syminfo._opening_hours = syminfo.opening_hours
    lib.syminfo._session_starts = syminfo.session_starts
    lib.syminfo._session_ends = syminfo.session_ends

    if syminfo.type == 'crypto':
        decimals = 6 if syminfo.basecurrency == 'BTC' else 4
        lib.syminfo._size_round_factor = 10 ** decimals
    else:
        lib.syminfo._size_round_factor = 1


# noinspection PyProtectedMember
def _reset_lib_vars():
    """
    Reset lib variables to be able to run other scripts
    """
    from ..types.source import Source

    lib.open = Source("open")
    lib.high = Source("high")
    lib.low = Source("low")
    lib.close = Source("close")
    lib.volume = Source("volume")
    lib.bid = Source("bid")
    lib.ask = Source("ask")
    lib.hl2 = Source("hl2")
    lib.hlc3 = Source("hlc3")
    lib.ohlc4 = Source("ohlc4")
    lib.hlcc4 = Source("hlcc4")

    lib._time = 0
    lib._datetime = datetime.fromtimestamp(0, UTC)

    lib.extra_fields = {}
    lib._lib_semaphore = False
    lib._is_live = False
    lib._strategy_suppressed = False

    lib.barstate.isfirst = True
    lib.barstate.islast = False
    lib.barstate.isconfirmed = True
    lib.barstate.ishistory = True
    lib.barstate.isrealtime = False
    lib.barstate.isnew = False
    lib.barstate.islastconfirmedhistory = False

    from ..lib import request
    request._reset_request_state()


class ScriptRunner:
    """
    Script runner
    """

    __slots__ = ('script_module', 'script', 'ohlcv_iter', 'syminfo', 'update_syminfo_every_run',
                 'bar_index', 'tz', 'plot_writer', 'strat_writer', 'trades_writer', 'last_bar_index',
                 'equity_curve', 'first_price', 'last_price',
                 '_script_path', '_security_data', '_magnifier_iter',
                 '_chart_provider_name', '_chart_provider_instance',
                 '_time_from', '_sec_syminfos', '_signal_rate_sources_fn',
                 '_broker_plugin', '_order_sync_engine', '_broker_event_loop',
                 '_engine_event_stream_future',
                 '_broker_store_ctx', '_log_ohlcv', '_price_decimals',
                 'broker_balance', '_sim_logged_open_ids')

    # noinspection PyProtectedMember
    def __init__(self, script_path: Path, ohlcv_iter: Iterable[OHLCV], syminfo: SymInfo, *,
                 plot_path: Path | None = None, strat_path: Path | None = None,
                 trade_path: Path | None = None,
                 update_syminfo_every_run: bool = False, last_bar_index=0,
                 inputs: dict[str, Any] | None = None,
                 security_data: 'dict[str, str | Path | PluginSymbol] | None' = None,
                 magnifier_iter: Iterable[OHLCV] | None = None,
                 broker_plugin: 'BrokerPlugin | None' = None,
                 broker_event_loop: 'asyncio.AbstractEventLoop | None' = None,
                 broker_store_ctx: 'RunContext | None' = None,
                 log_ohlcv: bool = False,
                 chart_provider_name: str | None = None,
                 chart_provider_instance: Any = None,
                 time_from: datetime | None = None):
        """
        Initialize the script runner

        :param script_path: The path to the script to run
        :param ohlcv_iter: Iterator of OHLCV data
        :param syminfo: Symbol information
        :param plot_path: Path to save the plot data
        :param strat_path: Path to save the strategy results
        :param trade_path: Path to save the trade data of the strategy
        :param update_syminfo_every_run: If it is needed to update the syminfo lib in every run,
                                         needed for parallel script executions
        :param last_bar_index: Last bar index, the index of the last bar of the historical data
        :param inputs: Optional dictionary of input values to pass to the script,
                       overrides values from .toml files
        :param security_data: Optional dict mapping ``"[SYMBOL:]TIMEFRAME"`` keys to
                              OHLCV file paths for request.security() contexts.
                              Examples: ``{"1D": "path/to/daily.ohlcv"}`` or
                              ``{"AAPL:1H": "path/to/aapl_1h.ohlcv"}``
        :param magnifier_iter: Optional sub-timeframe OHLCV iterator for bar magnifier mode.
                               When provided with use_bar_magnifier=true, order fills are checked
                               against each sub-bar for more accurate backtesting.
        :param broker_plugin: If set, the runner operates in **broker (live trading) mode**:
                              ``script.position`` is replaced by a :class:`BrokerPosition`,
                              ``strategy.*`` orders are dispatched through an
                              :class:`OrderSyncEngine`, and the simulator's order processing
                              is bypassed. The plugin also drives the OHLCV stream
                              (a :class:`BrokerPlugin` extends :class:`LiveProviderPlugin`).
        :param broker_event_loop: The shared ``asyncio`` event loop on which the broker plugin
                                  runs. Passed to the :class:`OrderSyncEngine` so that
                                  broker coroutines can be awaited from the runner thread
                                  via ``run_coroutine_threadsafe``.
        :param broker_store_ctx: Optional :class:`RunContext` from the unified
                                 :class:`BrokerStore`. When provided the engine persists
                                 envelope identity and parked-verification entries through
                                 it, and the runner heartbeats this context on every sync
                                 so crash detection works. ``None`` means no persistence
                                 (tests, backtests) — the ``run_tag`` is then derived
                                 locally from the plugin's ``account_id``. Caller owns
                                 the lifecycle: ``close()`` on shutdown.
        :raises ImportError: If the script does not have a 'main' function
        :raises ImportError: If the 'main' function is not decorated with @script.[indicator|strategy|library]
        :raises OSError: If the plot file could not be opened
        """
        self._script_path = script_path
        self._security_data = security_data or {}
        self._magnifier_iter = magnifier_iter
        self._log_ohlcv = log_ohlcv
        # Chart provider hooks — used in live mode by ``_resolve_security_data``
        # to translate Pine-style cross-symbol security keys to plugin-native
        # symbols (via ``provider.resolve_symbol``) when the user did not
        # supply an explicit ``--security`` mapping.
        self._chart_provider_name: str | None = chart_provider_name
        self._chart_provider_instance: Any = chart_provider_instance
        # Chart-side ``--from`` (already datetime). Forwarded into every
        # live-mode :class:`PluginSymbol` so each security context's warmup
        # window inherits the chart's look-back range instead of the
        # hard-coded subprocess default.
        self._time_from: datetime | None = time_from
        # Cache for pre-fetched ``SymInfo`` per live-mode security sec_id —
        # populated by ``_prefetch_sec_syminfos`` and consumed by the
        # currency-rate plumbing on the chart side. Empty in backtest mode.
        self._sec_syminfos: 'dict[str, SymInfo]' = {}
        # Optional per-bar driver for ``__auto_rate_*`` rate-source
        # subprocesses. Installed by ``create_chart_protocol`` when any
        # auto-rate sec_ids exist; left as ``None`` for backtests / runs
        # without ``currency=`` conversions, so the bar loop short-circuits.
        self._signal_rate_sources_fn: 'Callable[[], None] | None' = None

        # Import lib module to set syminfo properties before script import
        from .. import lib

        # Set syminfo properties BEFORE importing the script
        # This ensures that timestamp() calls in default parameters use the correct timezone
        _set_lib_syminfo_properties(syminfo)

        # Set programmatic inputs before script import so they override .toml values
        if inputs:
            from .script import _programmatic_inputs
            _programmatic_inputs.update(inputs)

        # Now import the script (default parameters will use correct timezone)
        self.script_module = import_script(script_path)

        if not hasattr(self.script_module.main, 'script'):
            raise ImportError(f"The 'main' function must be decorated with "
                              f"@script.[indicator|strategy|library] to run!")

        self.script: script = self.script_module.main.script

        # Broker (live trading) mode setup.
        # Done before ohlcv_iter is consumed so the engine is ready before run_iter.
        self._broker_plugin: 'BrokerPlugin | None' = broker_plugin
        self._broker_event_loop: 'asyncio.AbstractEventLoop | None' = broker_event_loop
        self._broker_store_ctx: 'RunContext | None' = broker_store_ctx
        self._order_sync_engine: 'OrderSyncEngine | None' = None
        self._engine_event_stream_future: Any = None
        self.broker_balance: dict[str, float] | None = None
        # Identities of open SimPosition trades already announced via
        # ``[SIM]`` logging — so each fill is narrated once in paper mode.
        self._sim_logged_open_ids: set[int] = set()
        if broker_plugin is not None:
            from pynecore.core.broker.position import BrokerPosition
            from pynecore.core.broker.run_identity import RunIdentity
            from pynecore.core.broker.sync_engine import OrderSyncEngine
            # Swap the simulator position for a live tracker. The
            # @script.strategy(...) decorator already attached a SimPosition;
            # in live broker mode the exchange is authoritative, so the
            # simulator is dropped entirely.
            self.script.position = BrokerPosition()
            if broker_store_ctx is not None:
                # Persistence-backed run: the CLI already opened a RunContext
                # via BrokerStore.open_run(), which computed the canonical
                # run_tag from the full RunIdentity.
                run_tag = broker_store_ctx.run_tag
            else:
                # No-persistence fallback (tests, single-shot backtests):
                # derive the run_tag locally so every sub-path still has a
                # stable id. The fallback identity uses the plugin's
                # ``account_id`` (``"default"`` when the plugin has not been
                # authenticated), matching what the persistence path would
                # compute.
                identity = RunIdentity(
                    strategy_id=script_path.stem,
                    symbol=str(syminfo.ticker),
                    timeframe=str(syminfo.period or ""),
                    account_id=broker_plugin.account_id,
                    label=None,
                )
                run_tag = identity.make_run_tag(
                    script_path.read_text(encoding='utf-8'),
                )
            self._order_sync_engine = OrderSyncEngine(
                broker=broker_plugin,
                position=self.script.position,  # type: ignore[arg-type]
                symbol=str(syminfo.ticker),
                run_tag=run_tag,
                event_loop=broker_event_loop,
                mintick=float(syminfo.mintick) if syminfo.mintick else 0.01,
                # Tick-grid factors for the native fail-safe rounding
                # (mintick == minmove / pricescale). Only forwarded when the
                # symbol carries a real mintick; otherwise the ``0`` sentinel
                # keeps the manager from snapping levels to the synthetic
                # 0.01 fallback grid above.
                minmove=float(syminfo.minmove) if syminfo.mintick else 0.0,
                pricescale=int(syminfo.pricescale) if syminfo.mintick else 0,
                store_ctx=broker_store_ctx,
                # Mirror exchange position state every bar. The exchange is
                # the source of truth — without per-sync reconciliation, an
                # externally-closed position (manual web-UI close, broker
                # liquidation) would never propagate back to ``position.size``,
                # leaving Pine convinced the bot is still in a trade and
                # blocking all subsequent entries.
                reconcile_every_n_syncs=1,
            )
            # Plugin-side access to the storage run: the Capital.com plugin
            # uses this for ``find_by_ref`` lookups, order upserts and audit
            # event logging without having the context threaded through every
            # ``execute_*`` signature.
            broker_plugin.store_ctx = broker_store_ctx

            # §2.6.7 native fail-safe actuator. The engine's
            # ``drive_native_failsafe`` (run once per ``sync``) drains the
            # worst-SL state machine into this dispatcher; without it the
            # fail-safe is state-only and no protective stop is ever placed
            # at the broker — for single-row partial brackets too. The
            # dispatcher is a pure PUT-or-raise actuator: the engine records
            # a put-success on a normal return and a put-failure on any
            # exception (see ``OrderSyncEngine.set_native_bracket_dispatcher``),
            # so this closure must not touch the record_* hooks. The plugin
            # PUT is async and must run on the broker loop, so it is marshalled
            # through the engine's own ``_run_async`` (identical loop + timeout
            # to every other broker call). Only wired when the plugin actually
            # provides the actuator — other plugins simply stay state-only.
            _failsafe_publish = getattr(
                broker_plugin, 'publish_native_failsafe_sl', None,
            )
            if _failsafe_publish is not None:
                _engine = cast('OrderSyncEngine', self._order_sync_engine)

                # noinspection PyProtectedMember
                def _native_failsafe_dispatcher(snapshot):
                    _engine._run_async(_failsafe_publish(snapshot))

                _engine.set_native_bracket_dispatcher(
                    _native_failsafe_dispatcher,
                )

            # §2.6.7 native fail-safe recovery feed (the reverse channel of
            # the dispatcher above). The plugin's reconcile pass observes the
            # broker-side bracket levels per live position; this sink routes
            # them into the engine so a parent stuck in DEGRADING — a restart
            # replay, or a PUT retry whose success the broker could not confirm
            # directly — flips back to HEALTHY once the desired worst-SL is
            # observed in place. Without it the stale-window timer escalates
            # DEGRADING -> DEGRADED in seconds and blocks new entries / brackets
            # until a manual reset. The reconcile pass runs on the broker
            # event-loop thread, so the sink is the engine's thread-safe
            # ``enqueue_native_bracket_observed`` (it queues; the main thread
            # applies it in ``drive_native_failsafe``) — calling
            # ``record_native_bracket_observed`` directly here would race the
            # main-thread worst-SL machinery. Installed unconditionally: the
            # attribute defaults to ``None`` on the base, plugins opt in by
            # calling it, and the engine drops snapshots for refs it does not
            # track at drain time.
            broker_plugin.native_failsafe_observed_sink = (
                cast('OrderSyncEngine', self._order_sync_engine).enqueue_native_bracket_observed
            )

        self.ohlcv_iter = ohlcv_iter
        self.syminfo = syminfo
        self.update_syminfo_every_run = update_syminfo_every_run
        self.last_bar_index = last_bar_index
        # Pre-increment scheme: bumped at the start of each bar's processing
        # (warmup, live, security loops). Starting at -1 keeps the first
        # processed bar at index 0 — matches Pine ``bar_index`` semantics.
        self.bar_index = -1

        # Precompute price decimals from ``syminfo.mintick`` so live OHLCV
        # log lines keep a constant column width (fix-width ``%.*f``). The
        # Pine ``format.mintick`` path in ``lib.string.tostring`` strips
        # trailing zeros and would jitter the width, which is why we don't
        # route through it here.
        #
        # The decimal count comes from ``str(mintick)`` (Python's shortest
        # round-trip repr), so ``0.05`` yields ``2`` without exposing float
        # dust. ``pricescale`` cannot be used: for fractional tick grids the
        # generated symbol info stores ``pricescale = round(1 / mintick)``
        # with ``minmove = 1`` (e.g. ``mintick=0.05`` -> ``pricescale=20``),
        # so ``len(str(pricescale)) - 1`` would under-count decimals. When
        # ``mintick`` is missing/zero we fall back to 2 decimals (the broker
        # path uses a synthetic ``0.01`` tick for the same case).
        _mintick = getattr(syminfo, 'mintick', 0.0) or 0.0
        self._price_decimals = mintick_decimals(_mintick) if _mintick > 0 else 2

        self.tz = lib._parse_timezone(syminfo.timezone)

        # Initialize tracking variables for statistics
        self.equity_curve: list[float] = []
        self.first_price: float | None = None
        self.last_price: float | None = None

        self.plot_writer = CSVWriter(
            plot_path, float_fmt=f".8g"
        ) if plot_path else None
        self.strat_writer = CSVWriter(strat_path, headers=(
            "Metric",
            f"All {syminfo.currency}", "All %",
            f"Long {syminfo.currency}", "Long %",
            f"Short {syminfo.currency}", "Short %",
        )) if strat_path else None
        self.trades_writer = CSVWriter(trade_path, headers=(
            "Trade #", "Bar Index", "Type", "Signal", "Date/Time", f"Price {syminfo.currency}",
            "Contracts", f"Profit {syminfo.currency}", "Profit %", f"Cumulative profit {syminfo.currency}",
            "Cumulative profit %", f"Run-up {syminfo.currency}", "Run-up %", f"Drawdown {syminfo.currency}",
            "Drawdown %",
        )) if trade_path else None

    # === Broker startup ====================================================

    # noinspection PyProtectedMember
    def start_broker(self) -> None:
        """Start broker-side I/O after construction.

        Two side effects, both intentionally kept out of ``__init__`` so the
        caller can finish ``Loading PyneCore`` (script import + runner setup)
        before any broker logs appear:

        1. Schedule :meth:`OrderSyncEngine.run_event_stream` on the broker
           event loop. Without this task, fill events never reach
           :meth:`BrokerPosition.record_fill` and ``position.size`` stays
           at 0 — the script then keeps re-entering on every flat-only
           branch tick because it never sees its own already-open position.
        2. Run the startup reconcile. Adopts the exchange's authoritative
           state (``get_position`` → ``BrokerPosition.size``/``avg_price``,
           ``get_open_orders`` → ``_order_mapping``) before the first bar
           runs. Without this, a fresh process restart with an open
           exchange position would see ``position_size == 0`` in Pine and
           re-enter — opening a *second* position alongside the existing
           one.

        No-op when not in broker mode.
        """
        if self._order_sync_engine is None:
            return
        engine = cast('OrderSyncEngine', self._order_sync_engine)
        # Plugin ``connect()`` (run during ``live_ohlcv_generator``) may have
        # mutated the ``envelopes`` / ``pending_verifications`` tables via
        # ``_retire_startup_orphans``. The engine cached both replays in its
        # ``__init__``, so refresh the in-memory anchors here BEFORE the
        # first dispatch to avoid popping a stale ``bar_ts_ms`` that resurrects
        # a just-retired ``client_order_id`` onto a row whose ``closed_ts_ms``
        # is still set.
        engine.refresh_anchors_from_store()
        loop = self._broker_event_loop
        if loop is not None:
            self._engine_event_stream_future = asyncio.run_coroutine_threadsafe(
                engine.run_event_stream(),
                loop,
            )
        # Defensive-close pending markers from prior process instances
        # must be re-armed (or dropped, if the FILL already settled)
        # BEFORE the startup reconcile so the reconcile snapshot reflects
        # the in-flight-close set the engine should preserve through
        # ``_active_intents``. Without the replay a fresh process could
        # treat a flat exchange as an external flatten and re-enter on
        # the next bar against a position the previous instance was
        # already closing defensively.
        engine._replay_pending_defensive_closes()
        engine.reconcile()

    # === Order-processing dispatch =========================================

    def _process_orders(self, position) -> None:
        """Run one order-processing step.

        In backtest mode this invokes the :class:`SimPosition` simulator
        (OHLC fill detection, slippage, OCA, margin). In broker mode it
        hands the pending Pine order book to the :class:`OrderSyncEngine`,
        which dispatches real exchange calls and routes any fills that
        arrived asynchronously through :meth:`BrokerPosition.record_fill`.
        """
        if self._order_sync_engine is not None:
            cast('OrderSyncEngine', self._order_sync_engine).sync(
                int(lib.last_bar_time),
                last_price=_close_price_or_none(),
            )
            # Heartbeat the storage run on every sync — the RunContext
            # rate-limits internally to ``HEARTBEAT_INTERVAL_MS``, so the
            # actual UPDATE fires at most once per minute regardless of
            # sync frequency. SIGKILL / OOM then gets cleaned on the next
            # open_run() via the stale-run threshold.
            if self._broker_store_ctx is not None:
                self._broker_store_ctx.heartbeat()
        else:
            position.process_orders()

    def _process_orders_magnified(self, position, sub_bars, candle) -> None:
        """Backtest sub-bar order processing; in broker mode, the exchange
        is the source of truth — magnification is irrelevant and the engine
        runs a plain sync."""
        if self._order_sync_engine is not None:
            cast('OrderSyncEngine', self._order_sync_engine).sync(
                int(lib.last_bar_time),
                last_price=_close_price_or_none(),
            )
            if self._broker_store_ctx is not None:
                self._broker_store_ctx.heartbeat()
        else:
            position.process_orders_magnified(sub_bars, candle)

    def _log_sim_fills(self, position) -> None:
        """Narrate paper-trading fills in ``--live`` mode without a broker.

        The :class:`SimPosition` fills orders locally and silently. This is the
        simulator counterpart of the ``[BROKER]`` order narration: ``[SIM]``
        lines so the operator sees entries and exits as they happen. Exits come
        from ``new_closed_trades`` (refreshed by the simulator every bar);
        entries are announced once per open trade, tracked by object identity.

        :param position: The active :class:`SimPosition`.
        """
        d = self._price_decimals
        for t in position.new_closed_trades:
            side = "long" if t.size > 0 else "short"
            sim_info(
                "EXIT %s %s qty=%g entry=%.*f exit=%.*f pnl=%+.2f",
                side, t.exit_id or t.entry_id or "", abs(t.size),
                d, float(t.entry_price), d, float(t.exit_price), float(t.profit),
            )
        current_ids: set[int] = set()
        for t in position.open_trades:
            current_ids.add(id(t))
            if id(t) not in self._sim_logged_open_ids:
                side = "long" if t.size > 0 else "short"
                sim_info(
                    "ENTRY %s %s qty=%g @ %.*f",
                    side, t.entry_id or "", abs(t.size), d, float(t.entry_price),
                )
        self._sim_logged_open_ids = current_ids

    def _process_deferred_margin_call(self, position) -> None:
        """Simulator-only. The exchange handles margin in broker mode, so
        any deferred margin handling is a no-op there."""
        if self._order_sync_engine is None:
            position.process_deferred_margin_call()

    @property
    def _broker_mode(self) -> bool:
        return self._order_sync_engine is not None

    @property
    def broker_position_snapshot(self) -> 'Any | None':
        if self._order_sync_engine is None:
            return None
        return cast('OrderSyncEngine', self._order_sync_engine).exchange_position

    # noinspection PyProtectedMember
    def run_iter(self, on_progress: Callable[[datetime], None] | None = None,
                 on_tick: Callable[[OHLCV], None] | None = None) \
            -> Iterator[tuple[OHLCV, dict[str, Any]] | tuple[OHLCV, dict[str, Any], list['Trade']]]:
        """
        Run the script on the data

        :param on_progress: Callback to call on every iteration
        :param on_tick: Optional per-update live callback (see :meth:`run`).
        :return: Return a dictionary with all data the sctipt plotted
        :raises AssertionError: If the 'main' function does not return a dictionary
        """
        from .. import lib
        from ..lib import _parse_timezone, barstate, string
        from pynecore.core import function_isolation
        from . import script

        is_strat = self.script.script_type == script_type.strategy

        # Reset bar_index — pre-increment scheme starts at -1.
        self.bar_index = -1
        # Reset function isolation
        function_isolation.reset()

        # Set script data
        lib._script = self.script  # Store script object in lib

        # Broker mode: refuse to start if the script needs capabilities the
        # exchange doesn't offer. Fail fast — never on the first bar.
        if self._broker_plugin is not None:
            from pynecore.core.broker.validation import validate_at_startup
            from pynecore.core.broker.exceptions import (
                AuthenticationError,
                ExchangeCapabilityError,
            )
            caps = self._broker_plugin.get_capabilities()
            reqs = getattr(self.script, '_broker_requirements', None)
            if reqs is not None:
                pyramiding = int(getattr(self.script, 'pyramiding', 1) or 1)
                errors = validate_at_startup(cast('ScriptRequirements', reqs), caps, pyramiding=pyramiding)
                if errors:
                    raise ExchangeCapabilityError(
                        "Script requirements not met by exchange:\n"
                        + "\n".join(f"  - {e}" for e in errors)
                    )

            # Auth check: fail fast on bad credentials rather than on the
            # first order attempt. A single get_balance() call is cheap and
            # every exchange supports it. An AuthenticationError here is
            # terminal — reconnect can never recover wrong keys.
            coro = self._broker_plugin.get_balance()
            try:
                if self._broker_event_loop is None:
                    balance = asyncio.run(coro)
                else:
                    balance = asyncio.run_coroutine_threadsafe(
                        coro, self._broker_event_loop,
                    ).result(timeout=30.0)
            except AuthenticationError as exc:
                raise AuthenticationError(
                    "Broker authentication failed at startup — cannot begin "
                    f"trading: {exc.reason}",
                    reason=exc.reason,
                ) from exc

            broker_info(
                "authenticated: plugin=%s account=%s equity=%s",
                type(self._broker_plugin).__name__,
                self._broker_plugin.account_id,
                balance,
            )
            self.broker_balance = balance

        # Update syminfo lib properties if needed
        if not self.update_syminfo_every_run:
            _set_lib_syminfo_properties(self.syminfo)
            self.tz = _parse_timezone(lib.syminfo.timezone)

        # Open plot writer if we have one
        if self.plot_writer:
            self.plot_writer.open()

        # If the script is a strategy, we open strategy output files too
        if is_strat:
            # Open trade writer if we have one
            if self.trades_writer:
                self.trades_writer.open()

        # Clear plot data
        lib._plot_data.clear()

        # Trade counter
        trade_num = 0

        # Position shortcut
        position = self.script.position

        # --- Security contexts setup ---
        sec_contexts: dict[str, dict] | None = getattr(
            self.script_module, '__security_contexts__', None
        )
        sec_processes: 'dict[str, BaseProcess]' = {}
        sec_cleanup_fn: Callable[[], None] | None = None
        sec_states = None
        sec_sync_block = None
        sec_result_blocks = None

        # --- Currency rate provider (default) ---
        # Always install a provider so ``request.currency_rate()`` works
        # without a ``request.security()`` context — e.g. when the chart
        # symbol itself is a currency pair (``lib.close`` is the rate) or
        # when only legacy file-backed rate sources are supplied via
        # ``security_data``. Replaced below inside the ``if sec_contexts``
        # branch with a provider that also reads sec ResultBlocks.
        from .currency import CurrencyRateProvider
        from ..lib import request
        _legacy_file_paths: dict[str, str | Path] = {}
        for _key, _val in self._security_data.items():
            if isinstance(_val, (str, Path)):
                _legacy_file_paths[_key] = _val
        request._currency_provider = CurrencyRateProvider(
            security_data=_legacy_file_paths,
            chart_syminfo=self.syminfo,
        )

        try:
            if sec_contexts:
                import os
                max_security = int(os.environ.get('PYNESYS_MAX_SECURITY_CONTEXTS', '64'))
                if len(sec_contexts) > max_security:
                    raise RuntimeError(
                        f"Script requests too many securities: {len(sec_contexts)} "
                        f"(limit: {max_security}). "
                        f"Set PYNESYS_MAX_SECURITY_CONTEXTS to change the limit."
                    )

                from .security import (
                    setup_security_states, create_chart_protocol,
                    inject_protocol, cleanup_shared_memory, Lookahead,
                )
                from .security_process import security_process_main
                from multiprocessing import Process

                # Detect same-context: symbol+TF identical to chart
                chart_ticker = str(lib.syminfo.ticker)
                chart_tf = str(lib.syminfo.period)
                same_context_ids: set[str] = set()
                for sec_id, ctx in sec_contexts.items():
                    sym = ctx.get('symbol')
                    tf = str(ctx.get('timeframe', chart_tf))
                    if sym is not None and str(sym) == chart_ticker and tf == chart_tf:
                        same_context_ids.add(sec_id)

                # Separate static (symbol known) and deferred (symbol=None) contexts
                # Same-context ids are excluded from both (no process needed)
                static_contexts = {}
                deferred_sec_ids: set[str] = set()
                for sec_id, ctx in sec_contexts.items():
                    if sec_id in same_context_ids:
                        continue
                    if ctx.get('symbol') is not None:
                        static_contexts[sec_id] = ctx
                    else:
                        deferred_sec_ids.add(sec_id)

                # Resolve OHLCV paths for static contexts only
                sec_ohlcv_paths = (
                    self._resolve_security_data(static_contexts) if static_contexts else {}
                )
                # Pre-fetch syminfo for every live-mode PluginSymbol entry
                # from the chart process, so the chart-side currency-rate
                # plumbing sees ``(basecurrency, currency)`` before any
                # subprocess starts, and the subprocess can skip its own
                # ``update_symbol_info()`` REST call. Pass ``sec_contexts``
                # so failures on ``ignore_invalid_symbol=True`` contexts
                # downgrade to None instead of aborting startup.
                sec_ohlcv_paths = self._prefetch_sec_syminfos(
                    sec_ohlcv_paths, sec_contexts=sec_contexts,
                )

                # Auto-spawn rate-source contexts for ``currency=X`` requests
                # that no existing context already covers. Mutates
                # ``sec_contexts`` / ``static_contexts`` / ``sec_ohlcv_paths``
                # in place so the rest of the setup treats the new entries
                # like any other PluginSymbol context.
                self._autospawn_rate_sources(
                    sec_contexts, static_contexts, sec_ohlcv_paths, chart_tf,
                )

                # Track ignored sec_ids (ignore_invalid_symbol=True, no data)
                ignored_sec_ids: set[str] = set()
                for sec_id, path in sec_ohlcv_paths.items():
                    if path is None:
                        ignored_sec_ids.add(sec_id)

                # No-process IDs: both same-context and ignored. Kept mutable
                # so the deferred-resolve callback can append late-discovered
                # ignored symbols (``ignore_invalid_symbol=True`` whose live
                # syminfo lookup fails) — without that, the chart-side
                # ``__sec_signal__`` would wait on a process that was never
                # spawned. ``create_chart_protocol`` captures by reference.
                no_process_ids: set[str] = set(same_context_ids | ignored_sec_ids)

                sec_states, sec_sync_block, sec_result_blocks = setup_security_states(
                    sec_contexts, chart_tf, self.tz, chart_symbol=chart_ticker,
                )

                # Currency rate provider — built after the SyncBlock exists so
                # security-context lookups can read the latest pickled close
                # from the matching ``ResultBlock``. Only **rate-source**
                # sec contexts are exposed as FX pairs: arbitrary user
                # ``request.security()`` expressions are not assumed to
                # yield close, so reading their ResultBlock as an exchange
                # rate would silently misuse indicator values as FX rates.
                legacy_file_paths: dict[str, str | Path] = {}
                for _key, _val in self._security_data.items():
                    if isinstance(_val, (str, Path)):
                        legacy_file_paths[_key] = _val
                rate_source_syminfos: dict[str, SymInfo] = {}
                for _sid, _ps in sec_ohlcv_paths.items():
                    if (isinstance(_ps, PluginSymbol) and _ps.is_rate_source
                            and _ps.syminfo is not None):
                        rate_source_syminfos[_sid] = _ps.syminfo
                request._currency_provider = CurrencyRateProvider(
                    security_data=legacy_file_paths,
                    chart_syminfo=self.syminfo,
                    sec_syminfos=rate_source_syminfos,
                    sync_block=sec_sync_block,
                )

                all_sec_ids = list(sec_contexts.keys())
                script_path_str = str(self._script_path.resolve())
                sec_result_locks = {
                    sid: state.result_lock for sid, state in sec_states.items()
                }

                def _spawn_security_process(sid: str, data_source):
                    sec_state = sec_states[sid]  # noqa - guaranteed non-None inside if sec_contexts
                    proc = Process(
                        target=security_process_main,
                        args=(
                            sid,
                            script_path_str,
                            data_source,
                            sec_sync_block.name,  # noqa
                            all_sec_ids,
                            sec_state.data_ready,
                            sec_state.advance_event,
                            sec_state.done_event,
                            sec_state.stop_event,
                            sec_state.is_ltf,
                            sec_result_locks,
                        ),
                        daemon=True,
                    )
                    proc.start()
                    sec_processes[sid] = proc

                # Callback for lazy resolution of deferred security contexts
                def _deferred_resolve(sid: str, symbol: str, timeframe: str | None):
                    if sid not in deferred_sec_ids:
                        return
                    deferred_sec_ids.discard(sid)
                    # Resolve actual timeframe
                    current_chart_tf = str(lib.syminfo.period)
                    resolved_tf = timeframe if timeframe else current_chart_tf
                    # Update SecurityState with correct timeframe info
                    sec_state = sec_states[sid]  # noqa - guaranteed non-None inside if sec_contexts
                    sec_state.timeframe = resolved_tf
                    same_tf = (resolved_tf == current_chart_tf)
                    sec_state.same_timeframe = same_tf
                    if same_tf:
                        sec_state.resampler = None
                    elif sec_state.resampler is None:
                        from .resampler import Resampler
                        sec_state.resampler = Resampler.get_resampler(resolved_tf)
                    # Now that the real symbol/timeframe are known, decide
                    # whether the live HTF transport applies. ``setup_security_states``
                    # built the aggregator under the assumption ``sym is None``
                    # ⇒ same-symbol; reverse that decision if the resolved symbol
                    # is cross-symbol, or attach one if the timeframe just
                    # promoted from chart-TF to HTF.
                    is_same_symbol = (chart_ticker is None
                                      or str(symbol) == chart_ticker)
                    needs_aggregator = (not same_tf) and is_same_symbol
                    if needs_aggregator and sec_state.htf_aggregator is None:
                        from .htf_aggregator import HTFAggregator
                        sec_state.htf_aggregator = HTFAggregator(resolved_tf, self.tz)
                    elif not needs_aggregator and sec_state.htf_aggregator is not None:
                        sec_state.htf_aggregator = None
                    elif (needs_aggregator
                          and sec_state.htf_aggregator is not None
                          and sec_state.htf_aggregator.timeframe != resolved_tf):
                        # Timeframe resolved to something different from the
                        # placeholder used at setup — rebuild for the right TF.
                        from .htf_aggregator import HTFAggregator
                        sec_state.htf_aggregator = HTFAggregator(resolved_tf, self.tz)
                    # Cross-symbol HTF + lookahead_on: developing bar cannot be
                    # aggregated from chart OHLCV (wrong instrument). Chart-side
                    # read returns ``na`` for every chart bar inside an open HTF
                    # period; the subprocess still advances on closed cross-symbol
                    # HTF bars, so close[1] at the period boundary delivers the
                    # just-closed close.
                    sec_state.na_on_developing = (
                            (not same_tf)
                            and (not is_same_symbol)
                            and sec_state.lookahead is Lookahead.ON
                    )
                    # Resolve OHLCV path and spawn process
                    resolve_ctx = {
                        'symbol': symbol,
                        'timeframe': resolved_tf,
                        'ignore_invalid_symbol': cast('dict[str, dict]', sec_contexts)[sid].get(
                            'ignore_invalid_symbol', False
                        ),
                    }
                    resolved = self._resolve_security_data({sid: resolve_ctx})
                    resolved = self._prefetch_sec_syminfos(
                        resolved, sec_contexts={sid: resolve_ctx},
                    )
                    resolved_path = resolved[sid]
                    sec_ohlcv_paths[sid] = resolved_path
                    if resolved_path is not None:
                        _spawn_security_process(sid, resolved_path)
                    else:
                        # ``ignore_invalid_symbol=True`` downgraded the live
                        # syminfo lookup to ``None``; mark the sid as
                        # no-process so ``__sec_signal__`` short-circuits
                        # instead of waiting on a child that was never
                        # spawned.
                        no_process_ids.add(sid)

                # Lazy spawn callback for static contexts
                def _lazy_spawn(sid: str):
                    resolved_path = sec_ohlcv_paths.get(sid)
                    if resolved_path is not None and sid not in no_process_ids:
                        _spawn_security_process(sid, resolved_path)

                # Eager-spawn auto-rate-source contexts. These hidden
                # ``__auto_rate_*`` sec_ids carry the FX feed for
                # ``request.security(..., currency=...)`` requests; no Pine
                # statement calls ``__sec_signal__`` for them, so the lazy
                # path never fires. Without an immediate spawn the
                # subprocess never starts, its :class:`ResultBlock` stays
                # empty, and ``CurrencyRateProvider`` reads ``NaN`` for
                # every conversion.
                for _sid, _ps in sec_ohlcv_paths.items():
                    if (isinstance(_ps, PluginSymbol) and _ps.is_rate_source
                            and _sid not in no_process_ids):
                        _spawn_security_process(_sid, _ps)

                # Build currency conversion map from security contexts.
                # Live-mode PluginSymbol sources expose syminfo via the
                # chart-side prefetch (``self._sec_syminfos``); file-mode
                # sources still load it from the sibling ``.toml``.
                currency_conversions: dict[str, tuple[str, str]] = {}
                for sec_id, ctx in sec_contexts.items():
                    target_cur = ctx.get('currency')
                    if target_cur is None:
                        continue
                    target_cur_str = str(target_cur)
                    if not target_cur_str or target_cur_str.lower() in ('', 'na', 'nan'):
                        continue
                    sec_si = self._sec_syminfos.get(sec_id)
                    if sec_si is None:
                        ohlcv_path = sec_ohlcv_paths.get(sec_id)
                        if isinstance(ohlcv_path, str):
                            sec_toml = Path(ohlcv_path).with_suffix('.toml')
                            if sec_toml.exists():
                                sec_si = SymInfo.load_toml(sec_toml)
                    if sec_si is not None and sec_si.currency:
                        currency_conversions[sec_id] = (sec_si.currency, target_cur_str)

                frozen_same_ctx = frozenset(same_context_ids)
                # Collect hidden ``__auto_rate_*`` sec_ids so the chart
                # loop can tick their subprocesses each bar — no Pine call
                # signals them, and without per-bar advance their
                # ResultBlock stays empty and ``CurrencyRateProvider``
                # returns NaN for every conversion.
                auto_rate_sec_ids = frozenset(
                    sid for sid, ps in sec_ohlcv_paths.items()
                    if isinstance(ps, PluginSymbol) and ps.is_rate_source
                    and sid not in no_process_ids
                )
                (signal_fn, write_fn, read_fn, wait_fn,
                 sec_cleanup_fn, signal_rate_sources_fn) = create_chart_protocol(
                    sec_states, sec_sync_block,
                    deferred_resolve_fn=_deferred_resolve if deferred_sec_ids else None,
                    lazy_spawn_fn=_lazy_spawn if static_contexts else None,
                    same_context_ids=frozen_same_ctx,
                    no_process_ids=no_process_ids,
                    result_blocks=sec_result_blocks if same_context_ids else None,
                    currency_conversions=currency_conversions or None,
                    sec_processes=sec_processes,
                    auto_rate_sec_ids=auto_rate_sec_ids,
                )
                inject_protocol(self.script_module, signal_fn, write_fn, read_fn, wait_fn,
                                same_context=frozen_same_ctx)
                self._signal_rate_sources_fn = signal_rate_sources_fn

            # --timeframe mode: magnifier_iter provides sub-TF data
            if self._magnifier_iter is not None:
                if is_strat and self.script.use_bar_magnifier:
                    # Bar magnifier: accurate order fills at sub-bar resolution
                    yield from self._run_iter_magnified(
                        barstate, position, script_mod=script,
                        is_strat=is_strat, on_progress=on_progress, string=string,
                    )
                    return
                else:
                    # On-the-fly aggregation: aggregate sub-TF to chart TF
                    from .bar_magnifier import BarMagnifier
                    chart_tf = str(lib.syminfo.period)
                    magnifier = BarMagnifier(self._magnifier_iter, chart_tf, tz=self.tz)
                    self.ohlcv_iter = (w.aggregated for w in magnifier)

            # Initialize calc_on_order_fills snapshot (for COOF or live mode).
            # Pine TV semantics: `calc_on_order_fills` is silently disabled when
            # `process_orders_on_close=True` (TV reverts to a single script calculation
            # per bar in that combo), so the snapshot stays unused in that case.
            var_snapshot: VarSnapshot | None = None
            is_live = lib._is_live
            # Indicators always run on every tick; strategies only if calc_on_every_tick
            run_on_every_tick = not is_strat or self.script.calc_on_every_tick
            if (is_strat and self.script.calc_on_order_fills
                    and not self.script.process_orders_on_close):
                var_snapshot = VarSnapshot(self.script_module, script._registered_libraries)
            elif is_live and run_on_every_tick:
                var_snapshot = VarSnapshot(self.script_module, script._registered_libraries)

            # --- Helper closures for DRY ---
            registered_libraries = script._registered_libraries
            signal_rate_sources_fn = self._signal_rate_sources_fn

            # noinspection PyProtectedMember
            def _run_libs_and_main():
                # Advance hidden ``__auto_rate_*`` subprocesses before
                # libraries/main run so any ``request.currency_rate`` /
                # ``currency=`` conversion looks up a freshly-written
                # close from the rate-source ResultBlock instead of NaN.
                if signal_rate_sources_fn is not None:
                    # noinspection PyCallingNonCallable
                    signal_rate_sources_fn()
                lib._lib_semaphore = True
                for _title, main_func in registered_libraries:
                    main_func()
                lib._lib_semaphore = False
                r = self.script_module.main()
                if r is not None:
                    assert isinstance(r, dict), "The 'main' function must return a dictionary!"
                    lib._plot_data.update(r)

            # noinspection PyProtectedMember
            def _write_bar_output(bar_candle):
                nonlocal trade_num
                if self.plot_writer and lib._plot_data:
                    ef = {} if bar_candle.extra_fields is None else dict(bar_candle.extra_fields)
                    ef.update(lib._plot_data)
                    self.plot_writer.write_ohlcv(bar_candle._replace(extra_fields=ef))

                if is_strat and self.trades_writer and position:
                    for t in position.new_closed_trades:
                        trade_num += 1
                        self.trades_writer.write(
                            trade_num, t.entry_bar_index,
                            "Entry long" if t.size > 0 else "Entry short",
                            t.entry_comment if t.entry_comment else t.entry_id,
                            string.format_time(t.entry_time),  # type: ignore
                            t.entry_price, abs(t.size), t.profit,
                            f"{t.profit_percent:.2f}", t.cum_profit,
                            f"{t.cum_profit_percent:.2f}", t.max_runup,
                            f"{t.max_runup_percent:.2f}", t.max_drawdown,
                            f"{t.max_drawdown_percent:.2f}",
                        )
                        self.trades_writer.write(
                            trade_num, t.exit_bar_index,
                            "Exit long" if t.size > 0 else "Exit short",
                            t.exit_comment if t.exit_comment else t.exit_id,
                            string.format_time(t.exit_time),  # type: ignore
                            t.exit_price, abs(t.size), t.profit,
                            f"{t.profit_percent:.2f}", t.cum_profit,
                            f"{t.cum_profit_percent:.2f}", t.max_runup,
                            f"{t.max_runup_percent:.2f}", t.max_drawdown,
                            f"{t.max_drawdown_percent:.2f}",
                        )

            # noinspection PyProtectedMember
            def _coof_loop():
                """COOF re-execution loop: process orders, re-execute on fills."""
                # Broker mode: no synchronous fill-driven re-execution — exchange
                # fills arrive asynchronously and are routed on the next sync.
                if self._broker_mode:
                    self._process_orders(position)
                    return
                sim = cast('SimPosition', position)
                old_fills = sim._fill_counter
                sim.process_orders()
                new_fills = sim._fill_counter
                while new_fills > old_fills:
                    if var_snapshot.has_vars:  # type: ignore
                        var_snapshot.restore()  # type: ignore
                    function_isolation.reset()
                    _run_libs_and_main()
                    old_fills = new_fills
                    sim.process_orders()
                    new_fills = sim._fill_counter

            # noinspection PyProtectedMember
            def _coof_magnified_loop(sub_bars_list, aggregated_candle):
                """COOF re-execution loop with magnified order processing."""
                if self._broker_mode:
                    self._process_orders(position)
                    return
                sim = cast('SimPosition', position)
                old_fills = sim._fill_counter
                sim.process_orders_magnified(sub_bars_list, aggregated_candle)
                new_fills = sim._fill_counter
                while new_fills > old_fills:
                    if var_snapshot.has_vars:  # type: ignore
                        var_snapshot.restore()  # type: ignore
                    function_isolation.reset()
                    _run_libs_and_main()
                    old_fills = new_fills
                    sim.process_orders_magnified(sub_bars_list, aggregated_candle)
                    new_fills = sim._fill_counter

            # --- Peek-ahead pattern: historical bars ---
            # LIVE_TRANSITION doubles as end-of-data sentinel → next() always returns OHLCV
            ohlcv_iterator = iter(self.ohlcv_iter)
            next_item = next(ohlcv_iterator, LIVE_TRANSITION)
            first_live_update: OHLCV | None = None
            # Tracks the last warmup-bar timestamp so the live loop can tell
            # whether the first live update is a new bar or an intra-bar
            # tick of the warmup's last bar (e.g. the still-open bar that
            # ``download_ohlcv`` brought in as historical).
            last_warmup_timestamp: int | None = None
            warmup_bars_processed = 0

            if is_live and self._broker_plugin is not None:
                broker_info("warmup phase started — replaying historical bars")

            while next_item is not LIVE_TRANSITION:
                candle = next_item
                next_item = next(ohlcv_iterator, LIVE_TRANSITION)

                # Pre-increment: bar_index becomes the index of the bar we
                # are about to process (first bar -> 0).
                self.bar_index += 1
                last_warmup_timestamp = candle.timestamp
                warmup_bars_processed += 1

                # Update syminfo lib properties if needed
                if self.update_syminfo_every_run:
                    _set_lib_syminfo_properties(self.syminfo)
                    self.tz = _parse_timezone(lib.syminfo.timezone)

                # Last bar detection
                if is_live:
                    barstate.islast = False
                    barstate.islastconfirmedhistory = (next_item is LIVE_TRANSITION)
                else:
                    barstate.islast = (next_item is LIVE_TRANSITION)

                # Update lib properties
                _set_lib_properties(candle, self.bar_index, self.tz, lib)

                # Store first price for buy & hold calculation
                if self.first_price is None:
                    self.first_price = lib.close  # type: ignore
                self.last_price = lib.close  # type: ignore

                # calc_on_order_fills path: snapshot, process, re-execute on fills
                if var_snapshot and position and not lib._strategy_suppressed:
                    if var_snapshot.has_vars:
                        var_snapshot.save()
                    _coof_loop()
                    if var_snapshot.has_vars:
                        var_snapshot.restore()
                elif is_strat and position and not lib._strategy_suppressed:
                    self._process_orders(position)

                # Execute libraries + script
                _run_libs_and_main()

                # Pine `process_orders_on_close=true` — extra fill attempt at the bar
                # close for current-bar orders, before the next bar's open arrives.
                # No COOF re-run here: Pine disables `calc_on_order_fills` when this
                # flag is set (var_snapshot is None whenever both are true).
                # Simulator-only; in broker mode the exchange owns fill timing.
                if (is_strat and position and not self._broker_mode
                        and not lib._strategy_suppressed
                        and self.script.process_orders_on_close):
                    cast('SimPosition', position).process_orders_at_close()

                # Process deferred margin calls
                if is_strat and position and not lib._strategy_suppressed:
                    self._process_deferred_margin_call(position)

                # Write output
                _write_bar_output(candle)

                # Yield
                if not is_strat:
                    yield candle, lib._plot_data
                elif position:
                    yield candle, lib._plot_data, position.new_closed_trades

                lib._plot_data.clear()

                if is_strat and position:
                    current_equity = float(position.equity) if position.equity \
                        else self.script.initial_capital
                    self.equity_curve.append(current_equity)

                if on_progress and lib._datetime is not None:
                    on_progress(lib._datetime.replace(tzinfo=None))

                barstate.isfirst = False

            if is_live and self._broker_plugin is not None:
                broker_info(
                    "warmup phase complete — %d bar(s) processed",
                    warmup_bars_processed,
                )

            # --- Live mode: transition and intra-bar loop ---
            # Flip the historical→live flags and emit the transition log
            # **before** blocking on the first WS bar. Otherwise the log
            # appears to fire only when the first live update arrives,
            # which can be a full period later (or never if the WS push
            # for the boundary bar is dedup-eaten upstream) — making the
            # transition look gated on data instead of on the warmup
            # boundary it actually represents.
            if next_item is LIVE_TRANSITION and is_live:
                barstate.ishistory = False
                barstate.isrealtime = True
                barstate.islastconfirmedhistory = False
                lib._strategy_suppressed = False

                # Promote ``request.security()`` contexts into live mode so
                # ``lookahead_on`` switches to the developing-bar transport
                # (see ``security.SecurityState.is_live``).
                if sec_states is not None:
                    for _sec_state in sec_states.values():
                        _sec_state.is_live = True

                if self._broker_mode:
                    # ``bar_index`` and ``lib._time`` are still pointing at
                    # the last warmup bar (e.g. 499) — this log line marks
                    # the transition AT that boundary; the next live bar
                    # arrival will pre-increment to 500.
                    broker_info("live trading active")

                # Flush output at transition point.
                if self.plot_writer:
                    self.plot_writer.flush()
                if self.trades_writer:
                    self.trades_writer.flush()

                first_live_update = next(ohlcv_iterator, None)

            if first_live_update is not None:
                import itertools

                # Seed with the last warmup bar's timestamp so that an
                # incoming live update with the same timestamp (common when
                # ``download_ohlcv`` returned the still-open current bar)
                # is recognised as a continuation of the last warmup bar
                # instead of a fresh one.
                last_bar_timestamp: int | None = last_warmup_timestamp
                sub_bars: list[OHLCV] = []

                live_stream = itertools.chain([first_live_update], ohlcv_iterator)
                for bar_update in live_stream:
                    # An async halt latched on the broker event-loop thread
                    # (e.g. ``UnexpectedCancelError`` from a polling plugin)
                    # must surface NOW — before ``[OHLCV]`` is logged or any
                    # state advances. Without this, a halt set mid-bar would
                    # only fire at the next bar close (via
                    # ``apply_async_events``), spilling a bogus OHLCV log line
                    # for a bar the bot is no longer trading.
                    if self._order_sync_engine is not None:
                        cast('OrderSyncEngine', self._order_sync_engine).raise_if_halted()

                    candle = bar_update
                    is_new_bar = (candle.timestamp != last_bar_timestamp)

                    if is_new_bar:
                        # Pre-increment on bar open; intra-bar ticks for the
                        # same bar reuse the index already assigned here.
                        self.bar_index += 1

                    barstate.islast = True
                    barstate.isconfirmed = bar_update.is_closed
                    barstate.isnew = is_new_bar

                    _set_lib_properties(candle, self.bar_index, self.tz, lib)

                    if self.first_price is None:
                        self.first_price = lib.close  # type: ignore
                    self.last_price = lib.close  # type: ignore

                    # Fire per-update tick hook (bid/ask spinner, other UI).
                    if on_tick is not None:
                        on_tick(candle)

                    if is_new_bar and not bar_update.is_closed:
                        # ── Bar open (first intra-bar tick) ──
                        sub_bars = [candle]
                        if run_on_every_tick:
                            if var_snapshot and var_snapshot.has_vars:
                                var_snapshot.save()
                            # Broker sync runs before the script so orders queued by the
                            # previous tick dispatch now, and async fills from watch_orders
                            # become visible to this script run via record_fill.
                            if is_strat and position and self._broker_mode \
                                    and not lib._strategy_suppressed:
                                self._process_orders(position)
                            _run_libs_and_main()
                        last_bar_timestamp = candle.timestamp

                    elif not bar_update.is_closed:
                        # ── Subsequent intra-bar tick ──
                        sub_bars.append(candle)
                        if run_on_every_tick:
                            if var_snapshot and var_snapshot.has_vars:
                                var_snapshot.restore()
                            function_isolation.reset()
                            if is_strat and position and self._broker_mode \
                                    and not lib._strategy_suppressed:
                                self._process_orders(position)
                            _run_libs_and_main()

                    elif bar_update.is_closed:
                        # ── Bar close ──
                        if is_new_bar:
                            sub_bars = []
                            if var_snapshot and var_snapshot.has_vars:
                                var_snapshot.save()
                        else:
                            sub_bars.append(candle)
                            if run_on_every_tick:
                                if var_snapshot and var_snapshot.has_vars:
                                    var_snapshot.restore()
                                function_isolation.reset()

                        # Strategy not running on ticks: bar close is first execution
                        if not run_on_every_tick:
                            barstate.isnew = True

                        # Per-bar OHLCV log (live mode; opt-out via --no-log-ohlcv).
                        # Logged at bar close *before* strategy processing so
                        # the on-screen log order — `[OHLCV] ... → [BROKER]
                        # dispatching ENTRY ... → [BROKER] fill ...` — matches
                        # the actual event order. Logging after the strategy
                        # ran would make orders appear before the bar that
                        # caused them.
                        if self._log_ohlcv:
                            extra = candle.extra_fields or {}
                            spread = extra.get('spread')
                            d = self._price_decimals
                            if spread is not None:
                                ohlcv_info(
                                    "O=%.*f H=%.*f L=%.*f C=%.*f "
                                    "spread=%.*f V=%.0f",
                                    d, candle.open, d, candle.high,
                                    d, candle.low, d, candle.close,
                                    d, spread,
                                    candle.volume,
                                )
                            else:
                                ohlcv_info(
                                    "O=%.*f H=%.*f L=%.*f C=%.*f V=%.0f",
                                    d, candle.open, d, candle.high,
                                    d, candle.low, d, candle.close,
                                    candle.volume,
                                )

                        if self._broker_mode:
                            # Broker mode: run the script FIRST (this bar's
                            # close queues new orders) and THEN sync the
                            # exchange so dispatch happens *on the same bar*.
                            # Calling sync first would dispatch the previous
                            # close's queue here, adding one full bar of
                            # stale latency to every entry/exit. TV live
                            # semantics: a market order placed at bar close
                            # fills near the next bar's open price (sub-second
                            # in practice). Pine sub-bar magnification and
                            # synchronous COOF re-execution don't apply —
                            # the exchange is the source of truth.
                            #
                            # Async fills (from ``watch_orders``) are
                            # drained *before* the script so the new bar's
                            # script sees the updated ``position.size``
                            # immediately rather than one bar later.
                            if self._order_sync_engine is not None:
                                cast('OrderSyncEngine', self._order_sync_engine).apply_async_events()
                            # Risk management hooks (broker-side parity with
                            # the sim's ``process_orders`` rollover/halt block):
                            # mark-to-market the open P&L so the equity-based
                            # drawdown / intraday-loss predicates use a fresh
                            # price; roll over the day counters before the
                            # script runs (so a day-rollover halt prevents a
                            # new entry from queueing); and enforce post-bar
                            # rules before the sync so the queued risk-close
                            # ships in the same dispatch cycle.
                            if is_strat and position:
                                bpos = cast('BrokerPosition', position)
                                bpos.update_unrealized_pnl(float(lib.close))
                                # noinspection PyProtectedMember
                                bpos._handle_bar_open_risk()
                            lib._plot_data.clear()
                            _run_libs_and_main()
                            if is_strat and position:
                                # noinspection PyProtectedMember
                                cast('BrokerPosition', position)._enforce_post_bar_risk()
                                self._process_orders(position)
                        else:
                            # Backtest: simulator first (fills the previous
                            # close's queue at this bar's open price), then
                            # script executes at this bar's close.
                            if is_strat and position:
                                if sub_bars:
                                    if var_snapshot and var_snapshot.has_vars:
                                        _coof_magnified_loop(sub_bars, candle)
                                        var_snapshot.restore()
                                    else:
                                        self._process_orders_magnified(position, sub_bars, candle)
                                else:
                                    if var_snapshot and var_snapshot.has_vars:
                                        _coof_loop()
                                        var_snapshot.restore()
                                    else:
                                        self._process_orders(position)

                            # Paper-trading narration: the simulator just
                            # filled the previous bar's queued orders — log
                            # them so live sim mode has the same per-fill
                            # visibility as broker mode's ``[BROKER]`` lines.
                            if is_strat and position:
                                self._log_sim_fills(position)

                            lib._plot_data.clear()
                            _run_libs_and_main()

                        if is_strat and position:
                            self._process_deferred_margin_call(position)

                        # Commit state for next bar
                        if var_snapshot and var_snapshot.has_vars:
                            var_snapshot.save()

                        # Output (only on closed bars)
                        _write_bar_output(candle)

                        if not is_strat:
                            yield candle, lib._plot_data
                        elif position:
                            yield candle, lib._plot_data, position.new_closed_trades

                        lib._plot_data.clear()

                        if is_strat and position:
                            current_equity = float(position.equity) if position.equity \
                                else self.script.initial_capital
                            self.equity_curve.append(current_equity)

                        last_bar_timestamp = candle.timestamp
                        barstate.isfirst = False

                        # Live strategy stats: rewrite stats file after each bar
                        if is_strat and self.strat_writer and position:
                            self._write_live_strategy_stats(position)

                        if on_progress and lib._datetime is not None:
                            on_progress(lib._datetime.replace(tzinfo=None))

            elif on_progress:
                on_progress(datetime.max)

        except GeneratorExit:
            pass

        finally:  # Python reference counter will close this even if the iterator is not exhausted
            if is_strat and position:
                # Export remaining open trades before closing
                if self.trades_writer and position.open_trades:
                    for trade in position.open_trades:
                        trade_num += 1  # Continue numbering from closed trades
                        # Export the entry part
                        self.trades_writer.write(
                            trade_num,
                            trade.entry_bar_index,
                            "Entry long" if trade.size > 0 else "Entry short",
                            trade.entry_id,
                            string.format_time(trade.entry_time),  # type: ignore
                            trade.entry_price,
                            abs(trade.size),
                            0.0,  # No profit yet for open trades
                            "0.00",  # No profit percent yet
                            0.0,  # No cumulative profit change
                            "0.00",  # No cumulative profit percent change
                            0.0,  # No max runup yet
                            "0.00",  # No max runup percent yet
                            0.0,  # No max drawdown yet
                            "0.00",  # No max drawdown percent yet
                        )

                        # Export the exit part with "Open" signal (TradingView compatibility)
                        # This simulates automatic closing at the end of backtest
                        # Use the last price from the iteration
                        exit_price = self.last_price

                        if exit_price is not None:
                            # Calculate profit/loss using the same formula as Position._fill_order
                            # For closing, size is negative of the position.
                            # `* syminfo.pointvalue` converts price-delta to account-currency
                            # so the synthetic "Open" exit reports USD consistently with closed
                            # trades on futures (pv != 1). For pv = 1 this is a no-op.
                            pv = self.syminfo.pointvalue
                            closing_size = -trade.size
                            pnl = -closing_size * (exit_price - trade.entry_price) * pv
                            entry_value = abs(trade.size) * trade.entry_price * pv
                            pnl_percent = (pnl / entry_value) * 100 if entry_value != 0 else 0

                            self.trades_writer.write(
                                trade_num,
                                self.bar_index,  # Last bar index processed
                                "Exit long" if trade.size > 0 else "Exit short",
                                "Open",  # TradingView uses "Open" signal for automatic closes
                                string.format_time(lib._time),  # type: ignore
                                exit_price,
                                abs(trade.size),
                                pnl,
                                f"{pnl_percent:.2f}",
                                pnl,  # Same as profit for last trade
                                f"{pnl_percent:.2f}",
                                max(0.0, pnl),  # Runup
                                f"{max(0, pnl_percent):.2f}",
                                max(0.0, -pnl),  # Drawdown
                                f"{max(0, -pnl_percent):.2f}",
                            )

                # Write strategy statistics
                if self.strat_writer and position:
                    try:
                        # Open strat writer and write statistics
                        self.strat_writer.open()

                        # Calculate comprehensive statistics
                        stats = calculate_strategy_statistics(
                            position,
                            self.script.initial_capital,
                            self.equity_curve if self.equity_curve else None,
                            self.first_price,
                            self.last_price
                        )

                        write_strategy_statistics_csv(stats, self.strat_writer)
                        self.strat_writer.close()

                    finally:
                        # Close strat writer
                        self.strat_writer.close()

            # Close the plot writer
            if self.plot_writer:
                self.plot_writer.close()
            # Close the trade writer
            if self.trades_writer:
                self.trades_writer.close()

            # Shutdown security processes
            if sec_processes and sec_states is not None:
                for state in sec_states.values():
                    state.stop_event.set()
                    state.advance_event.set()  # wake up if waiting
                for p in sec_processes.values():
                    p.join(timeout=5)
                    if p.is_alive():
                        p.terminate()
                if callable(sec_cleanup_fn):
                    sec_cleanup_fn: Callable
                    sec_cleanup_fn()
                if sec_sync_block and sec_result_blocks:
                    from .security import cleanup_shared_memory
                    cleanup_shared_memory(sec_sync_block, sec_result_blocks)

            # Cancel the broker event-stream task scheduled in __init__.
            # Done before loop teardown so the watch_orders generator gets
            # a chance to clean up its HTTP session.
            if self._engine_event_stream_future is not None:
                self._engine_event_stream_future.cancel()
                self._engine_event_stream_future = None

            # Reset library variables
            _reset_lib_vars()
            # Reset function isolation
            function_isolation.reset()

    # noinspection PyProtectedMember
    def _run_iter_magnified(self, barstate, position, script_mod, is_strat, on_progress, string):
        from .. import lib
        """
        Magnified bar iteration: iterate sub-TF windows, process orders at sub-bar
        resolution, execute script once per chart bar.
        """
        from .bar_magnifier import BarMagnifier
        # Needed for COOF re-execution path (already loaded by run_iter, safe to re-import)
        from pynecore.core import function_isolation

        chart_tf = str(lib.syminfo.period)
        assert self._magnifier_iter is not None
        magnifier = BarMagnifier(self._magnifier_iter, chart_tf, tz=self.tz)

        trade_num = 0

        # Initialize calc_on_order_fills snapshot for magnified path.
        # Pine TV semantics: `calc_on_order_fills` is silently disabled when
        # `process_orders_on_close=True` (TV reverts to a single script calculation
        # per bar in that combo), so the snapshot stays unused in that case.
        var_snapshot: VarSnapshot | None = None
        if (is_strat and self.script.calc_on_order_fills
                and not self.script.process_orders_on_close):
            var_snapshot = VarSnapshot(self.script_module, script_mod._registered_libraries)

        for window in magnifier:
            # Pre-increment: bar_index becomes the index of the current
            # aggregated chart bar.
            self.bar_index += 1

            barstate.islast = window.is_last_window

            # Set lib OHLCV to the aggregated chart-bar values (what the script sees)
            _set_lib_properties(window.aggregated, self.bar_index, self.tz, lib)

            # Store first price for buy & hold calculation
            if self.first_price is None:
                self.first_price = lib.close  # type: ignore

            # Update last price
            self.last_price = lib.close  # type: ignore

            # Process orders against each sub-bar for accurate fills
            if var_snapshot and position:
                if var_snapshot.has_vars:
                    var_snapshot.save()

                old_fills = position._fill_counter
                position.process_orders_magnified(window.sub_bars, window.aggregated)
                new_fills = position._fill_counter

                while new_fills > old_fills:
                    if var_snapshot.has_vars:
                        var_snapshot.restore()
                    function_isolation.reset()
                    lib._lib_semaphore = True
                    for library_title, main_func in script_mod._registered_libraries:
                        main_func()
                    lib._lib_semaphore = False
                    self.script_module.main()
                    old_fills = new_fills
                    position.process_orders_magnified(window.sub_bars, window.aggregated)
                    new_fills = position._fill_counter

                if var_snapshot.has_vars:
                    var_snapshot.restore()
            elif position:
                position.process_orders_magnified(window.sub_bars, window.aggregated)

            # Execute registered library main functions before main script
            lib._lib_semaphore = True
            for library_title, main_func in script_mod._registered_libraries:
                main_func()
            lib._lib_semaphore = False

            # Run the script
            res = self.script_module.main()

            # Pine `process_orders_on_close=true` — extra fill attempt at the bar
            # close for current-bar orders. No COOF re-run: Pine disables
            # `calc_on_order_fills` when this flag is set (var_snapshot is None
            # whenever both are true).
            if position and self.script.process_orders_on_close:
                position.process_orders_at_close()

            # Process deferred margin calls (after script runs, before results)
            if position:
                position.process_deferred_margin_call()

            # Update plot data with the results
            if res is not None:
                assert isinstance(res, dict), "The 'main' function must return a dictionary!"
                lib._plot_data.update(res)

            # Write plot data to CSV if we have a writer
            if self.plot_writer and lib._plot_data:
                extra_fields = {} if window.aggregated.extra_fields is None \
                    else dict(window.aggregated.extra_fields)
                extra_fields.update(lib._plot_data)
                updated_candle = window.aggregated._replace(extra_fields=extra_fields)
                self.plot_writer.write_ohlcv(updated_candle)

            # Yield results
            if not is_strat:
                yield window.aggregated, lib._plot_data
            elif position:
                yield window.aggregated, lib._plot_data, position.new_closed_trades

            # Save trade data
            if is_strat and self.trades_writer and position:
                for trade in position.new_closed_trades:
                    trade_num += 1
                    self.trades_writer.write(
                        trade_num,
                        trade.entry_bar_index,
                        "Entry long" if trade.size > 0 else "Entry short",
                        trade.entry_comment if trade.entry_comment else trade.entry_id,
                        string.format_time(trade.entry_time),  # type: ignore
                        trade.entry_price,
                        abs(trade.size),
                        trade.profit,
                        f"{trade.profit_percent:.2f}",
                        trade.cum_profit,
                        f"{trade.cum_profit_percent:.2f}",
                        trade.max_runup,
                        f"{trade.max_runup_percent:.2f}",
                        trade.max_drawdown,
                        f"{trade.max_drawdown_percent:.2f}",
                    )
                    self.trades_writer.write(
                        trade_num,
                        trade.exit_bar_index,
                        "Exit long" if trade.size > 0 else "Exit short",
                        trade.exit_comment if trade.exit_comment else trade.exit_id,
                        string.format_time(trade.exit_time),  # type: ignore
                        trade.exit_price,
                        abs(trade.size),
                        trade.profit,
                        f"{trade.profit_percent:.2f}",
                        trade.cum_profit,
                        f"{trade.cum_profit_percent:.2f}",
                        trade.max_runup,
                        f"{trade.max_runup_percent:.2f}",
                        trade.max_drawdown,
                        f"{trade.max_drawdown_percent:.2f}",
                    )

            # Clear plot data
            lib._plot_data.clear()

            # Track equity curve for strategies
            if is_strat and position:
                current_equity = float(position.equity) if position.equity else self.script.initial_capital
                self.equity_curve.append(current_equity)

            # Call the progress callback
            if on_progress and lib._datetime is not None:
                on_progress(lib._datetime.replace(tzinfo=None))

            # It is no longer the first bar
            barstate.isfirst = False

        if on_progress:
            on_progress(datetime.max)

    def _resolve_security_data(self, contexts: dict) -> 'dict[str, str | PluginSymbol | None]':
        """
        Resolve a data source for each security context.

        Walks the user-provided ``security_data`` dictionary first, matching
        on ``"SYMBOL:TF"``, then ``"SYMBOL"``, then ``"TF"`` keys. Falls
        through to two mode-specific behaviours when no explicit mapping
        exists:

        - **Live mode** (chart provider available): builds a
          :class:`PluginSymbol` for the security subprocess by translating
          the Pine-style symbol through ``chart_provider_instance.resolve_symbol``
          (which consults the plugin's ``config.symbol_map`` TOML table
          first, falling back to ``normalize_symbol``).
        - **Backtest mode** (no chart provider): raises ``ValueError`` —
          a security context cannot be resolved without either an explicit
          ``--security`` file mapping or ``ignore_invalid_symbol``.

        :param contexts: The ``__security_contexts__`` dict from the script module
        :return: Dict mapping sec_id to an OHLCV file path (``str``), a
                 :class:`PluginSymbol` for live-mode subprocesses, or
                 ``None`` when the context was opted out via
                 ``ignore_invalid_symbol``.
        :raises ValueError: If no data found and ignore_invalid_symbol is not True
        """
        from dataclasses import replace as dc_replace
        result: dict[str, str | PluginSymbol | None] = {}
        for sec_id, ctx in contexts.items():
            symbol = str(ctx.get('symbol', ''))
            timeframe = str(ctx.get('timeframe', ''))

            entry: str | Path | PluginSymbol | None = None
            # Try exact "SYMBOL:TF" match, then symbol-only, then TF-only.
            key = f"{symbol}:{timeframe}"
            if key in self._security_data:
                entry = self._security_data[key]
            elif symbol in self._security_data:
                entry = self._security_data[symbol]
            elif timeframe in self._security_data:
                entry = self._security_data[timeframe]

            if isinstance(entry, PluginSymbol):
                if entry.time_from is None and self._time_from is not None:
                    entry = dc_replace(entry, time_from=self._time_from)
                result[sec_id] = cast('PluginSymbol', entry)
                continue
            if entry is not None:
                result[sec_id] = self._ensure_ohlcv_ext(entry)
                continue

            # No explicit mapping — fall back to chart-provider resolution
            # in live mode.
            if self._chart_provider_instance is not None and self._chart_provider_name:
                native_symbol = self._chart_provider_instance.resolve_symbol(symbol)
                result[sec_id] = PluginSymbol(
                    provider_name=self._chart_provider_name,
                    symbol=native_symbol,
                    timeframe=timeframe,
                    config=getattr(self._chart_provider_instance, 'config', None),
                    time_from=self._time_from,
                    ohlcv_dir=self._chart_ohlcv_dir(),
                )
                continue

            # No data found — check if ignore_invalid_symbol is set
            if ctx.get('ignore_invalid_symbol'):
                result[sec_id] = None
                continue

            raise ValueError(
                f"No OHLCV data found for security context "
                f"(symbol={symbol!r}, timeframe={timeframe!r}). "
                f"Provide data via the security_data parameter, e.g.: "
                f"security_data={{'{symbol}': 'path/to/data.ohlcv'}}"
            )
        return result

    def _prefetch_sec_syminfos(
            self,
            sec_data: 'dict[str, str | PluginSymbol | None]',
            sec_contexts: dict | None = None,
    ) -> 'dict[str, str | PluginSymbol | None]':
        """Pre-fetch :class:`SymInfo` for every live-mode security context.

        Builds a temporary :class:`LiveProviderPlugin` instance for each
        :class:`PluginSymbol` entry and calls ``update_symbol_info()`` once
        from the chart process. The result is cached on ``self._sec_syminfos``
        (used by the currency-rate plumbing) and folded back into the
        returned :class:`PluginSymbol` so the subprocess does not have to
        repeat the REST round-trip on startup.

        File-mode entries (backtest) are returned unchanged.

        :param sec_data: Per-sec_id resolved data sources (mutated to None
            for sec_ids whose REST lookup fails and whose context opted in
            via ``ignore_invalid_symbol=True``).
        :param sec_contexts: ``__security_contexts__`` dict — consulted to
            honor ``ignore_invalid_symbol`` when a symbol fails to resolve.
            When ``None``, every failure propagates as an exception.
        """
        from dataclasses import replace as dc_replace
        from pynecore.core.plugin.live_provider import LiveProviderPlugin
        from pynecore.core.plugin import load_plugin

        out: dict[str, str | PluginSymbol | None] = {}
        for sec_id, entry in sec_data.items():
            if not isinstance(entry, PluginSymbol):
                out[sec_id] = entry
                continue
            if entry.syminfo is not None:
                self._sec_syminfos[sec_id] = entry.syminfo
                out[sec_id] = entry
                continue
            provider_cls = load_plugin(entry.provider_name)
            if not issubclass(provider_cls, LiveProviderPlugin):
                raise RuntimeError(
                    f"Plugin '{entry.provider_name}' is not a live provider; "
                    f"cannot drive cross-symbol live request.security."
                )
            ignore_invalid = bool(
                sec_contexts and sec_contexts.get(sec_id, {}).get('ignore_invalid_symbol')
            )
            # Constructor and ``update_symbol_info`` both share the
            # ``ignore_invalid_symbol`` downgrade: some live providers (e.g.
            # CCXT) validate the exchange prefix in ``__init__`` and raise
            # before the symbol-info call ever runs.
            # noinspection PyBroadException
            try:
                provider = provider_cls(
                    symbol=entry.symbol,
                    timeframe=entry.timeframe,
                    ohlcv_dir=entry.ohlcv_dir,
                    config=entry.config,
                )
                syminfo = provider.update_symbol_info()
            except Exception:  # noqa: BLE001
                if not ignore_invalid:
                    raise
                # ``ignore_invalid_symbol=True``: downgrade to the
                # backtest-mode "no data" sentinel so the rest of the
                # pipeline treats this context as ignored.
                out[sec_id] = None
                continue
            self._sec_syminfos[sec_id] = syminfo
            out[sec_id] = dc_replace(entry, syminfo=syminfo)
        return out

    def _autospawn_rate_sources(
            self,
            sec_contexts: dict,
            static_contexts: dict,
            sec_ohlcv_paths: 'dict[str, str | PluginSymbol | None]',
            chart_tf: str,
    ) -> None:
        """Discover and spawn rate-source contexts for unresolved ``currency=X`` pairs.

        For every security context whose ``currency`` parameter would
        require a ``(basecurrency, target_currency)`` exchange-rate lookup
        not already covered by the chart pair or by an existing security
        context, builds a hidden rate-source :class:`PluginSymbol` (with
        ``is_rate_source=True``) and adds it to ``sec_contexts`` /
        ``static_contexts`` / ``sec_ohlcv_paths``. The chart's own provider
        instance is used to validate the constructed pair symbol via
        ``update_symbol_info()`` — invalid symbols are skipped silently
        (the rate downstream simply remains ``NaN``).

        Backtest runs (no chart-side live provider) leave everything
        untouched; the legacy ``.toml`` lookup keeps working.
        """
        if self._chart_provider_instance is None or not self._chart_provider_name:
            return

        chart_pair: tuple[str, str] | None = None
        if self.syminfo.basecurrency:
            chart_pair = (self.syminfo.basecurrency, self.syminfo.currency)

        # Only the chart pair (whose ``lib.close`` is the live rate) and
        # other explicit rate sources count as "already covered". User
        # security contexts are *not* assumed to expose close — their
        # ResultBlock carries the user's ``request.security()`` expression
        # result, which can be anything (e.g. ``ta.sma(close, 20)``, ``high``,
        # a tuple). Treating those as FX rates would silently misuse
        # indicator values as exchange rates.
        existing_pairs: set[tuple[str, str]] = set()
        if chart_pair is not None:
            existing_pairs.add(chart_pair)
            existing_pairs.add((chart_pair[1], chart_pair[0]))
        for _sid, ps in sec_ohlcv_paths.items():
            if (isinstance(ps, PluginSymbol) and ps.is_rate_source
                    and ps.syminfo and ps.syminfo.basecurrency):
                existing_pairs.add((ps.syminfo.basecurrency, ps.syminfo.currency))
                existing_pairs.add((ps.syminfo.currency, ps.syminfo.basecurrency))

        # Collect pairs that need an auto-rate-source.
        needed_pairs: set[tuple[str, str]] = set()
        for sid, ctx in sec_contexts.items():
            target_cur = ctx.get('currency')
            if target_cur is None:
                continue
            target_str = str(target_cur)
            if not target_str or target_str.lower() in ('na', 'nan', ''):
                continue
            si = self._sec_syminfos.get(sid)
            if si is None or not si.currency:
                continue
            from_cur, to_cur = si.currency, target_str
            if from_cur == to_cur:
                continue
            if (from_cur, to_cur) in existing_pairs:
                continue
            needed_pairs.add((from_cur, to_cur))

        if not needed_pairs:
            return

        from pynecore.core.plugin import load_plugin
        from pynecore.core.plugin.live_provider import LiveProviderPlugin

        provider_cls = load_plugin(self._chart_provider_name)
        if not issubclass(provider_cls, LiveProviderPlugin):
            return
        config = getattr(self._chart_provider_instance, 'config', None)

        symbol_map = getattr(config, 'symbol_map', None) or {}

        def _try_pair(a: str, b: str) -> 'tuple[str, SymInfo] | None':
            """Try to resolve ``construct_pair_symbol(a, b)``; return the
            ``(native_symbol, syminfo)`` tuple if the provider exposes the
            currency pair (in either direction), else ``None``.
            """
            pk = cast('type[LiveProviderPlugin]', provider_cls).construct_pair_symbol(a, b)
            ns = self._chart_provider_instance.resolve_symbol(pk)
            # noinspection PyBroadException
            try:
                tp = provider_cls(
                    symbol=ns,
                    timeframe=chart_tf,
                    ohlcv_dir=self._chart_ohlcv_dir(),
                    config=config,
                )
                pair_si = tp.update_symbol_info()
            except Exception:  # noqa: BLE001
                return None
            act = (pair_si.basecurrency, pair_si.currency)
            if act != (a, b) and act != (b, a):
                return None
            return ns, pair_si

        for from_cur, to_cur in sorted(needed_pairs):
            # A prior iteration may have already spawned a rate source for
            # the inverse direction of this pair; ``CurrencyRateProvider``
            # inverts rates transparently, so a second feed for the same
            # underlying pair would just duplicate WS subscriptions.
            if (from_cur, to_cur) in existing_pairs:
                continue
            # Try the direct ``from_cur + to_cur`` construction first. If the
            # provider exposes only the inverse pair (e.g. ``EURUSD`` is live
            # but the script requested USD→EUR), fall back to the inverse
            # construction — ``CurrencyRateProvider`` already inverts rates
            # from a reverse-direction source. The fallback is skipped when a
            # ``symbol_map`` already maps the direct Pine key, so user-provided
            # explicit mappings are trusted as-is.
            direct_pinekey = provider_cls.construct_pair_symbol(from_cur, to_cur)
            resolved = _try_pair(from_cur, to_cur)
            if resolved is None and direct_pinekey not in symbol_map:
                resolved = _try_pair(to_cur, from_cur)
            if resolved is None:
                continue
            native_symbol, syminfo = resolved
            auto_sec_id = f"__auto_rate_{from_cur}_{to_cur}__"
            if auto_sec_id in sec_contexts:
                continue
            ps = PluginSymbol(
                provider_name=self._chart_provider_name,
                symbol=native_symbol,
                timeframe=chart_tf,
                config=config,
                time_from=self._time_from,
                syminfo=syminfo,
                is_rate_source=True,
                ohlcv_dir=self._chart_ohlcv_dir(),
            )
            sec_contexts[auto_sec_id] = {
                'symbol': native_symbol,
                'timeframe': chart_tf,
            }
            static_contexts[auto_sec_id] = sec_contexts[auto_sec_id]
            sec_ohlcv_paths[auto_sec_id] = ps
            self._sec_syminfos[auto_sec_id] = syminfo
            existing_pairs.add((from_cur, to_cur))
            existing_pairs.add((to_cur, from_cur))

    def _chart_ohlcv_dir(self) -> 'Path | None':
        """Return the OHLCV data directory of the chart provider, if any.

        Cross-symbol live :class:`PluginSymbol` entries forward this to the
        subprocess so the child provider can locate workdir-side resources
        that live next to the data dir — most notably per-exchange config
        overrides in ``<workdir>/config/plugins/<provider>.toml`` (e.g. the
        ``[binance]`` section of ``ccxt.toml``). Without it, the subprocess
        provider runs with default exchange config while the chart side
        runs with the override, breaking auth and market-type selection
        for the cross-symbol feeds.
        """
        if self._chart_provider_instance is None:
            return None
        ohlcv_path = getattr(self._chart_provider_instance, 'ohlcv_path', None)
        if ohlcv_path is None:
            return None
        return Path(cast('str | Path', ohlcv_path)).parent

    @staticmethod
    def _ensure_ohlcv_ext(path: str | Path) -> str:
        """Add .ohlcv extension if not present."""
        p = Path(path)
        if p.suffix != '.ohlcv':
            ohlcv_path = p.with_suffix('.ohlcv')
            if ohlcv_path.exists():
                return str(ohlcv_path)
        return str(path)

    def _write_live_strategy_stats(self, position):
        """Rewrite strategy stats file with current state (live mode, after each bar)."""
        if self.strat_writer is None:
            return
        from .strategy_stats import calculate_strategy_statistics, write_strategy_statistics_csv
        # noinspection PyBroadException
        try:
            self.strat_writer.open()
            stats = calculate_strategy_statistics(
                position, self.script.initial_capital,
                self.equity_curve if self.equity_curve else None,
                self.first_price, self.last_price,
            )
            write_strategy_statistics_csv(stats, self.strat_writer)
            self.strat_writer.close()
        except Exception:
            # noinspection PyBroadException
            try:
                self.strat_writer.close()
            except Exception:
                pass

    def run(self, on_progress: Callable[[datetime], None] | None = None,
            on_tick: Callable[[OHLCV], None] | None = None):
        """
        Run the script on the data

        :param on_progress: Callback to call on every iteration
        :param on_tick: Optional callback invoked on every live OHLCV update
                        (intra-bar tick + closed bar). Receives the OHLCV
                        candle. Only fires in live mode, after the historical
                        phase has transitioned. Used by the CLI to render
                        bid/ask in the progress spinner.
        :raises AssertionError: If the 'main' function does not return a dictionary
        """
        for _ in self.run_iter(on_progress=on_progress, on_tick=on_tick):
            pass
