from typing import Iterable, Iterator, Callable, TYPE_CHECKING, Any
from types import ModuleType
import sys
from pathlib import Path
from datetime import datetime, UTC

from pynecore import lib
from pynecore.lib.log import broker_info, ohlcv_info
from pynecore.types.ohlcv import OHLCV
from pynecore.core.syminfo import SymInfo
from pynecore.core.csv_file import CSVWriter
from pynecore.core.strategy_stats import calculate_strategy_statistics, write_strategy_statistics_csv
from pynecore.core.var_snapshot import VarSnapshot

from pynecore.types import script_type

if TYPE_CHECKING:
    from zoneinfo import ZoneInfo  # noqa
    from pynecore.core.script import script
    from pynecore.lib.strategy import Trade, Position  # noqa
    from pynecore.core.plugin.broker import BrokerPlugin
    from pynecore.core.broker.sync_engine import OrderSyncEngine
    from pynecore.core.broker.storage import RunContext

__all__ = [
    'import_script',
    'ScriptRunner',
    'LIVE_TRANSITION',
]

LIVE_TRANSITION = OHLCV(timestamp=-1, open=-1, high=-1, low=-1, close=-1, volume=-1)
"""Sentinel inserted between historical and live OHLCV data in the iterator."""


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

    lib.hl2 = (lib.high + lib.low) / 2.0
    lib.hlc3 = (lib.high + lib.low + lib.close) / 3.0
    lib.ohlc4 = (lib.open + lib.high + lib.low + lib.close) / 4.0
    lib.hlcc4 = (lib.high + lib.low + 2 * lib.close) / 4.0

    dt = lib._datetime = datetime.fromtimestamp(ohlcv.timestamp, UTC).astimezone(tz)
    lib._time = lib.last_bar_time = int(dt.timestamp() * 1000)  # PineScript representation of time


# noinspection PyUnusedLocal
def _set_lib_syminfo_properties(syminfo: SymInfo, lib: ModuleType):
    """
    Set syminfo library properties from this object
    """
    if TYPE_CHECKING:  # This is needed for the type checker to work
        from .. import lib

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


# noinspection PyUnusedLocal,PyProtectedMember
def _reset_lib_vars(lib: ModuleType):
    """
    Reset lib variables to be able to run other scripts
    :param lib:
    :return:
    """
    if TYPE_CHECKING:  # This is needed for the type checker to work
        from .. import lib
    from ..types.source import Source

    lib.open = Source("open")
    lib.high = Source("high")
    lib.low = Source("low")
    lib.close = Source("close")
    lib.volume = Source("volume")
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
                 '_broker_plugin', '_order_sync_engine', '_broker_event_loop',
                 '_broker_store_ctx', '_log_ohlcv', '_price_decimals')

    # noinspection PyProtectedMember
    def __init__(self, script_path: Path, ohlcv_iter: Iterable[OHLCV], syminfo: SymInfo, *,
                 plot_path: Path | None = None, strat_path: Path | None = None,
                 trade_path: Path | None = None,
                 update_syminfo_every_run: bool = False, last_bar_index=0,
                 inputs: dict[str, Any] | None = None,
                 security_data: dict[str, str | Path] | None = None,
                 magnifier_iter: Iterable[OHLCV] | None = None,
                 broker_plugin: 'BrokerPlugin | None' = None,
                 broker_event_loop: Any = None,
                 broker_store_ctx: 'RunContext | None' = None,
                 log_ohlcv: bool = False):
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

        # Import lib module to set syminfo properties before script import
        from .. import lib

        # Set syminfo properties BEFORE importing the script
        # This ensures that timestamp() calls in default parameters use the correct timezone
        _set_lib_syminfo_properties(syminfo, lib)

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
        self._broker_event_loop = broker_event_loop
        self._broker_store_ctx: 'RunContext | None' = broker_store_ctx
        self._order_sync_engine: 'OrderSyncEngine | None' = None
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
                store_ctx=broker_store_ctx,
            )
            # Plugin-side access to the storage run: the Capital.com plugin
            # uses this for ``find_by_ref`` lookups, order upserts and audit
            # event logging without having the context threaded through every
            # ``execute_*`` signature.
            broker_plugin.store_ctx = broker_store_ctx

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
        mintick = getattr(syminfo, 'mintick', 0) or 0
        if mintick > 0:
            tick_str = f"{mintick:.20f}".rstrip('0').rstrip('.')
            self._price_decimals = (
                len(tick_str.split('.')[1]) if '.' in tick_str else 0
            )
        else:
            self._price_decimals = 2

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
            self._order_sync_engine.sync(int(lib.last_bar_time))
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
            self._order_sync_engine.sync(int(lib.last_bar_time))
            if self._broker_store_ctx is not None:
                self._broker_store_ctx.heartbeat()
        else:
            position.process_orders_magnified(sub_bars, candle)

    def _process_deferred_margin_call(self, position) -> None:
        """Simulator-only. The exchange handles margin in broker mode, so
        any deferred margin handling is a no-op there."""
        if self._order_sync_engine is None:
            position.process_deferred_margin_call()

    @property
    def _broker_mode(self) -> bool:
        return self._order_sync_engine is not None

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
            import asyncio
            from pynecore.core.broker.validation import validate_at_startup
            from pynecore.core.broker.exceptions import (
                AuthenticationError,
                ExchangeCapabilityError,
            )
            caps = self._broker_plugin.get_capabilities()
            reqs = getattr(self.script, '_broker_requirements', None)
            if reqs is not None:
                errors = validate_at_startup(reqs, caps)
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

        # Update syminfo lib properties if needed
        if not self.update_syminfo_every_run:
            _set_lib_syminfo_properties(self.syminfo, lib)
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

        # --- Currency rate provider setup ---
        from .currency import CurrencyRateProvider
        from ..lib import request
        if self._security_data:
            request._currency_provider = CurrencyRateProvider(
                self._security_data, chart_syminfo=self.syminfo,
            )
        else:
            request._currency_provider = CurrencyRateProvider(
                {}, chart_syminfo=self.syminfo,
            )

        # --- Security contexts setup ---
        sec_contexts: dict[str, dict] | None = getattr(
            self.script_module, '__security_contexts__', None
        )
        sec_processes: list = []
        sec_cleanup_fn: Callable[[], None] | None = None
        sec_states = None
        sec_sync_block = None
        sec_result_blocks = None

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
                    inject_protocol, cleanup_shared_memory,
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

                # Track ignored sec_ids (ignore_invalid_symbol=True, no data)
                ignored_sec_ids: set[str] = set()
                for sec_id, path in sec_ohlcv_paths.items():
                    if path is None:
                        ignored_sec_ids.add(sec_id)

                # No-process IDs: both same-context and ignored
                no_process_ids = frozenset(same_context_ids | ignored_sec_ids)

                sec_states, sec_sync_block, sec_result_blocks = setup_security_states(
                    sec_contexts, chart_tf, self.tz,
                )

                all_sec_ids = list(sec_contexts.keys())
                script_path_str = str(self._script_path.resolve())

                def _spawn_security_process(sid: str, data_path: str):
                    sec_state = sec_states[sid]  # noqa - guaranteed non-None inside if sec_contexts
                    proc = Process(
                        target=security_process_main,
                        args=(
                            sid,
                            script_path_str,
                            data_path,
                            sec_sync_block.name,  # noqa
                            all_sec_ids,
                            sec_state.data_ready,
                            sec_state.advance_event,
                            sec_state.done_event,
                            sec_state.stop_event,
                            sec_state.is_ltf,
                        ),
                        daemon=True,
                    )
                    proc.start()
                    sec_processes.append(proc)

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
                    # Resolve OHLCV path and spawn process
                    resolve_ctx = {'symbol': symbol, 'timeframe': resolved_tf}
                    resolved = self._resolve_security_data({sid: resolve_ctx})
                    resolved_path = resolved[sid]
                    sec_ohlcv_paths[sid] = resolved_path
                    if resolved_path is not None:
                        _spawn_security_process(sid, resolved_path)

                # Lazy spawn callback for static contexts
                def _lazy_spawn(sid: str):
                    resolved_path = sec_ohlcv_paths.get(sid)
                    if resolved_path is not None and sid not in no_process_ids:
                        _spawn_security_process(sid, resolved_path)

                # Build currency conversion map from security contexts
                currency_conversions: dict[str, tuple[str, str]] = {}
                for sec_id, ctx in sec_contexts.items():
                    target_cur = ctx.get('currency')
                    if target_cur is not None:
                        target_cur_str = str(target_cur)
                        if target_cur_str and target_cur_str.lower() not in ('', 'na', 'nan'):
                            ohlcv_path = sec_ohlcv_paths.get(sec_id)
                            if ohlcv_path:
                                sec_toml = Path(ohlcv_path).with_suffix('.toml')
                                if sec_toml.exists():
                                    sec_si = SymInfo.load_toml(sec_toml)
                                    currency_conversions[sec_id] = (
                                        sec_si.currency, target_cur_str
                                    )

                frozen_same_ctx = frozenset(same_context_ids)
                signal_fn, write_fn, read_fn, wait_fn, sec_cleanup_fn = create_chart_protocol(
                    sec_states, sec_sync_block,
                    deferred_resolve_fn=_deferred_resolve if deferred_sec_ids else None,
                    lazy_spawn_fn=_lazy_spawn if static_contexts else None,
                    same_context_ids=frozen_same_ctx,
                    no_process_ids=no_process_ids,
                    result_blocks=sec_result_blocks if same_context_ids else None,
                    currency_conversions=currency_conversions or None,
                )
                inject_protocol(self.script_module, signal_fn, write_fn, read_fn, wait_fn,
                                same_context=frozen_same_ctx)

            # --timeframe mode: magnifier_iter provides sub-TF data
            if self._magnifier_iter is not None:
                if is_strat and self.script.use_bar_magnifier:
                    # Bar magnifier: accurate order fills at sub-bar resolution
                    yield from self._run_iter_magnified(
                        lib, barstate, position, script_mod=script,
                        is_strat=is_strat, on_progress=on_progress, string=string,
                    )
                    return
                else:
                    # On-the-fly aggregation: aggregate sub-TF to chart TF
                    from .bar_magnifier import BarMagnifier
                    chart_tf = str(lib.syminfo.period)
                    magnifier = BarMagnifier(self._magnifier_iter, chart_tf, tz=self.tz)
                    self.ohlcv_iter = (w.aggregated for w in magnifier)

            # Initialize calc_on_order_fills snapshot (for COOF or live mode)
            var_snapshot: VarSnapshot | None = None
            is_live = lib._is_live
            # Indicators always run on every tick; strategies only if calc_on_every_tick
            run_on_every_tick = not is_strat or self.script.calc_on_every_tick
            if is_strat and self.script.calc_on_order_fills:
                var_snapshot = VarSnapshot(self.script_module, script._registered_libraries)
            elif is_live and run_on_every_tick:
                var_snapshot = VarSnapshot(self.script_module, script._registered_libraries)

            # --- Helper closures for DRY ---
            registered_libraries = script._registered_libraries

            # noinspection PyProtectedMember
            def _run_libs_and_main():
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
                old_fills = position._fill_counter
                position.process_orders()
                new_fills = position._fill_counter
                while new_fills > old_fills:
                    if var_snapshot.has_vars:  # type: ignore
                        var_snapshot.restore()  # type: ignore
                    function_isolation.reset()
                    _run_libs_and_main()
                    old_fills = new_fills
                    position.process_orders()
                    new_fills = position._fill_counter

            # noinspection PyProtectedMember
            def _coof_magnified_loop(sub_bars_list, aggregated_candle):
                """COOF re-execution loop with magnified order processing."""
                if self._broker_mode:
                    self._process_orders(position)
                    return
                old_fills = position._fill_counter
                position.process_orders_magnified(sub_bars_list, aggregated_candle)
                new_fills = position._fill_counter
                while new_fills > old_fills:
                    if var_snapshot.has_vars:  # type: ignore
                        var_snapshot.restore()  # type: ignore
                    function_isolation.reset()
                    _run_libs_and_main()
                    old_fills = new_fills
                    position.process_orders_magnified(sub_bars_list, aggregated_candle)
                    new_fills = position._fill_counter

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

            while next_item is not LIVE_TRANSITION:
                candle = next_item
                next_item = next(ohlcv_iterator, LIVE_TRANSITION)

                # Pre-increment: bar_index becomes the index of the bar we
                # are about to process (first bar -> 0).
                self.bar_index += 1
                last_warmup_timestamp = candle.timestamp

                # Update syminfo lib properties if needed
                if self.update_syminfo_every_run:
                    _set_lib_syminfo_properties(self.syminfo, lib)
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

            # --- Live mode: transition and intra-bar loop ---
            # After the historical loop, if LIVE_TRANSITION was hit, get the first live bar
            if next_item is LIVE_TRANSITION and is_live:
                first_live_update = next(ohlcv_iterator, None)

            if first_live_update is not None:
                import itertools

                # Transition: historical → live
                barstate.ishistory = False
                barstate.isrealtime = True
                barstate.islastconfirmedhistory = False
                lib._strategy_suppressed = False

                if self._broker_mode:
                    # ``bar_index`` and ``lib._time`` are still pointing at
                    # the last warmup bar (e.g. 499) — this log line marks
                    # the transition AT that boundary; the next live bar
                    # arrival will pre-increment to 500.
                    broker_info(
                        "live trading active - strategy no longer suppressed",
                    )

                # Flush output at transition point
                if self.plot_writer:
                    self.plot_writer.flush()
                if self.trades_writer:
                    self.trades_writer.flush()

                # Seed with the last warmup bar's timestamp so that an
                # incoming live update with the same timestamp (common when
                # ``download_ohlcv`` returned the still-open current bar)
                # is recognised as a continuation of the last warmup bar
                # instead of a fresh one.
                last_bar_timestamp: int | None = last_warmup_timestamp
                sub_bars: list[OHLCV] = []

                live_stream = itertools.chain([first_live_update], ohlcv_iterator)
                for bar_update in live_stream:
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

                        # Order processing: magnified if sub_bars available
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

                        # Final script execution for the closed bar
                        lib._plot_data.clear()
                        _run_libs_and_main()

                        if is_strat and position:
                            self._process_deferred_margin_call(position)

                        # Commit state for next bar
                        if var_snapshot and var_snapshot.has_vars:
                            var_snapshot.save()

                        # Per-bar OHLCV log (live mode; opt-out via --no-log-ohlcv).
                        if self._log_ohlcv:
                            extra = candle.extra_fields or {}
                            ask_close = extra.get('ask_close')
                            spread = extra.get('spread')
                            d = self._price_decimals
                            if ask_close is not None:
                                ohlcv_info(
                                    "O=%.*f H=%.*f L=%.*f C=%.*f V=%.0f  "
                                    "ask=%.*f  spread=%.*f",
                                    d, candle.open, d, candle.high,
                                    d, candle.low, d, candle.close,
                                    candle.volume,
                                    d, ask_close,
                                    d, spread if spread is not None else 0.0,
                                )
                            else:
                                ohlcv_info(
                                    "O=%.*f H=%.*f L=%.*f C=%.*f V=%.0f",
                                    d, candle.open, d, candle.high,
                                    d, candle.low, d, candle.close,
                                    candle.volume,
                                )

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
                            # For closing, size is negative of the position
                            closing_size = -trade.size
                            pnl = -closing_size * (exit_price - trade.entry_price)
                            pnl_percent = (pnl / (trade.entry_price * abs(trade.size))) * 100 \
                                if trade.entry_price != 0 else 0

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
                for p in sec_processes:
                    p.join(timeout=5)
                    if p.is_alive():
                        p.terminate()
                if callable(sec_cleanup_fn):
                    sec_cleanup_fn: Callable
                    sec_cleanup_fn()
                if sec_sync_block and sec_result_blocks:
                    from .security import cleanup_shared_memory
                    cleanup_shared_memory(sec_sync_block, sec_result_blocks)

            # Reset library variables
            _reset_lib_vars(lib)
            # Reset function isolation
            function_isolation.reset()

    # noinspection PyProtectedMember
    def _run_iter_magnified(self, lib, barstate, position, script_mod, is_strat, on_progress, string):
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

        # Initialize calc_on_order_fills snapshot for magnified path
        var_snapshot: VarSnapshot | None = None
        if is_strat and self.script.calc_on_order_fills:
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

    def _resolve_security_data(self, contexts: dict) -> dict[str, str | None]:
        """
        Resolve OHLCV file paths for each security context.

        Matches each context's (symbol, timeframe) to the user-provided
        ``security_data`` dictionary using ``"SYMBOL:TF"`` or ``"TF"`` keys.

        :param contexts: The ``__security_contexts__`` dict from the script module
        :return: Dict mapping sec_id to resolved OHLCV file path (None if ignored)
        :raises ValueError: If no data found and ignore_invalid_symbol is not True
        """
        result: dict[str, str | None] = {}
        for sec_id, ctx in contexts.items():
            symbol = str(ctx.get('symbol', ''))
            timeframe = str(ctx.get('timeframe', ''))

            # Try exact "SYMBOL:TF" match
            key = f"{symbol}:{timeframe}"
            if key in self._security_data:
                result[sec_id] = self._ensure_ohlcv_ext(self._security_data[key])
                continue

            # Try symbol-only match (without timeframe)
            if symbol in self._security_data:
                result[sec_id] = self._ensure_ohlcv_ext(self._security_data[symbol])
                continue

            # Try timeframe-only match
            if timeframe in self._security_data:
                result[sec_id] = self._ensure_ohlcv_ext(self._security_data[timeframe])
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
