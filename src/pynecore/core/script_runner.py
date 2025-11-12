from typing import Iterable, Iterator, Callable, TYPE_CHECKING, Any
from types import ModuleType
import sys
from pathlib import Path
from datetime import datetime, UTC

from pynecore.types.ohlcv import OHLCV
from pynecore.core.syminfo import SymInfo
from pynecore.core.csv_file import CSVWriter
from pynecore.core.strategy_stats import calculate_strategy_statistics, write_strategy_statistics_csv

from pynecore.types import script_type

if TYPE_CHECKING:
    from zoneinfo import ZoneInfo  # noqa
    from pynecore.core.script import script
    from pynecore.lib.strategy import Trade  # noqa

__all__ = [
    'import_script',
    'ScriptRunner',
]


def import_script(script_path: Path) -> ModuleType:
    """
    Import the script
    """
    from importlib import import_module
    import re
    # Import hook only before importing the script, to make import hook being used only for Pyne scripts
    # (this makes 1st run faster, than if it would be a top-level import)
    from . import import_hook  # noqa

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
        # This will use the import system, including our hook
        module = import_module(script_path.stem)
    finally:
        # Remove the directory from path
        sys.path.pop(0)

    if not hasattr(module, 'main'):
        raise ImportError(f"Script '{script_path}' must have a 'main' function to run!")

    return module


def _round_price(price: float, lib: ModuleType):
    """
    Round price to the nearest tick
    """
    if TYPE_CHECKING:  # This is needed for the type checker to work
        from .. import lib
    syminfo = lib.syminfo
    scaled = round(price * syminfo.pricescale)
    return scaled / syminfo.pricescale


# noinspection PyShadowingNames
def _set_lib_properties(ohlcv: OHLCV, bar_index: int, tz: 'ZoneInfo', lib: ModuleType):
    """
    Set lib properties from OHLCV
    """
    if TYPE_CHECKING:  # This is needed for the type checker to work
        from .. import lib

    lib.bar_index = lib.last_bar_index = bar_index

    lib.open = _round_price(ohlcv.open, lib)
    lib.high = _round_price(ohlcv.high, lib)
    lib.low = _round_price(ohlcv.low, lib)
    lib.close = _round_price(ohlcv.close, lib)

    lib.volume = ohlcv.volume

    lib.hl2 = (lib.high + lib.low) / 2.0
    lib.hlc3 = (lib.high + lib.low + lib.close) / 3.0
    lib.ohlc4 = (lib.open + lib.high + lib.low + lib.close) / 4.0
    lib.hlcc4 = (lib.high + lib.low + 2 * lib.close) / 4.0

    dt = lib._datetime = datetime.fromtimestamp(ohlcv.timestamp, UTC).astimezone(tz)
    lib._time = lib.last_bar_time = int(dt.timestamp() * 1000)  # PineScript representation of time


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
    lib.syminfo.ticker = syminfo.prefix + ':' + syminfo.ticker

    lib.syminfo._opening_hours = syminfo.opening_hours
    lib.syminfo._session_starts = syminfo.session_starts
    lib.syminfo._session_ends = syminfo.session_ends

    if syminfo.type == 'crypto':
        decimals = 6 if syminfo.basecurrency == 'BTC' else 4  # TODO: is it correct?
        lib.syminfo._size_round_factor = 10 ** decimals
    else:
        lib.syminfo._size_round_factor = 1


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

    lib._lib_semaphore = False

    lib.barstate.isfirst = True
    lib.barstate.islast = False


class ScriptRunner:
    """
    Script runner
    """

    __slots__ = ('script_module', 'script', 'ohlcv_iter', 'syminfo', 'update_syminfo_every_run',
                 'bar_index', 'tz', 'plot_writer', 'strat_writer', 'trades_writer', 'last_bar_index',
                 'equity_curve', 'first_price', 'last_price')

    def __init__(self, script_path: Path, ohlcv_iter: Iterable[OHLCV], syminfo: SymInfo, *,
                 plot_path: Path | None = None, strat_path: Path | None = None,
                 trade_path: Path | None = None,
                 update_syminfo_every_run: bool = False, last_bar_index=0):
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
        :raises ImportError: If the script does not have a 'main' function
        :raises ImportError: If the 'main' function is not decorated with @script.[indicator|strategy|library]
        :raises OSError: If the plot file could not be opened
        """
        # Import lib module to set syminfo properties before script import
        from .. import lib

        # Set syminfo properties BEFORE importing the script
        # This ensures that timestamp() calls in default parameters use the correct timezone
        _set_lib_syminfo_properties(syminfo, lib)

        # Now import the script (default parameters will use correct timezone)
        self.script_module = import_script(script_path)

        if not hasattr(self.script_module.main, 'script'):
            raise ImportError(f"The 'main' function must be decorated with "
                              f"@script.[indicator|strategy|library] to run!")

        self.script: script = self.script_module.main.script

        # noinspection PyProtectedMember
        from ..lib import _parse_timezone

        self.ohlcv_iter = ohlcv_iter
        self.syminfo = syminfo
        self.update_syminfo_every_run = update_syminfo_every_run
        self.last_bar_index = last_bar_index
        self.bar_index = 0

        self.tz = _parse_timezone(syminfo.timezone)

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

    # noinspection PyProtectedMember
    def run_iter(self, on_progress: Callable[[datetime], None] | None = None) \
            -> Iterator[tuple[OHLCV, dict[str, Any]] | tuple[OHLCV, dict[str, Any], list['Trade']]]:
        """
        Run the script on the data

        :param on_progress: Callback to call on every iteration
        :return: Return a dictionary with all data the sctipt plotted
        :raises AssertionError: If the 'main' function does not return a dictionary
        """
        from .. import lib
        from ..lib import _parse_timezone, barstate, string
        from pynecore.core import function_isolation
        from . import script

        is_strat = self.script.script_type == script_type.strategy

        # Reset bar_index
        self.bar_index = 0
        # Reset function isolation
        function_isolation.reset()

        # Set script data
        lib._script = self.script  # Store script object in lib

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

        try:
            for candle in self.ohlcv_iter:
                # Update syminfo lib properties if needed, other ScriptRunner instances may have changed them
                if self.update_syminfo_every_run:
                    _set_lib_syminfo_properties(self.syminfo, lib)
                    self.tz = _parse_timezone(lib.syminfo.timezone)

                if self.bar_index == self.last_bar_index:
                    barstate.islast = True

                # Update lib properties
                _set_lib_properties(candle, self.bar_index, self.tz, lib)

                # Store first price for buy & hold calculation
                if self.first_price is None:
                    self.first_price = lib.close  # type: ignore

                # Update last price
                self.last_price = lib.close  # type: ignore

                # Process limit orders
                if is_strat and position:
                    position.process_orders()

                # Execute registered library main functions before main script
                lib._lib_semaphore = True
                for library_title, main_func in script._registered_libraries:
                    main_func()
                lib._lib_semaphore = False

                # Run the script
                res = self.script_module.main()

                # Update plot data with the results
                if res is not None:
                    assert isinstance(res, dict), "The 'main' function must return a dictionary!"
                    lib._plot_data.update(res)

                # Write plot data to CSV if we have a writer
                if self.plot_writer and lib._plot_data:
                    # Create a new dictionary combining extra_fields (if any) with plot data
                    extra_fields = {} if candle.extra_fields is None else dict(candle.extra_fields)
                    extra_fields.update(lib._plot_data)
                    # Create a new OHLCV instance with updated extra_fields
                    updated_candle = candle._replace(extra_fields=extra_fields)
                    self.plot_writer.write_ohlcv(updated_candle)

                # Yield plot data to be able to process in a subclass
                if not is_strat:
                    yield candle, lib._plot_data
                elif position:
                    yield candle, lib._plot_data, position.new_closed_trades

                # Save trade data if we have a writer
                if is_strat and self.trades_writer and position:
                    for trade in position.new_closed_trades:
                        trade_num += 1  # Start from 1
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

                # Update bar index
                self.bar_index += 1
                # It is no longer the first bar
                barstate.isfirst = False

            if on_progress:
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
                                self.bar_index - 1,  # Last bar index
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

            # Reset library variables
            _reset_lib_vars(lib)
            # Reset function isolation
            function_isolation.reset()

    def run(self, on_progress: Callable[[datetime], None] | None = None):
        """
        Run the script on the data

        :param on_progress: Callback to call on every iteration
        :raises AssertionError: If the 'main' function does not return a dictionary
        """
        for _ in self.run_iter(on_progress=on_progress):
            pass
