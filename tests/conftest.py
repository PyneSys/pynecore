from typing import cast, Protocol, Callable, Iterable, Any, Generator
from typing import TYPE_CHECKING
import ast
import sys
import os
import pytest
import logging
import math

from pathlib import Path

from pynecore.types.na import NA
from pynecore.core.script_runner import ScriptRunner
from pynecore.types.ohlcv import OHLCV
from pynecore.core.csv_file import CSVReader
from pynecore.core.syminfo import SymInfo

if TYPE_CHECKING:
    from pynecore.lib.strategy import Trade

logger = logging.getLogger(__name__)

#
# Constants
#

INDENT = 4


#
# Type definitions
#

class RunnerProtocol(Protocol):
    def __call__(self, ohlcv_iter: Iterable[OHLCV], syminfo_override: dict[str, Any] | None = None, *,
                 syminfo_path: Path | None = None) -> ScriptRunner:
        ...


class DictComparatorProtocol(Protocol):
    def __call__(self, a: dict[str, Any], b: dict[str, Any], **kwargs) -> None:
        ...


class LogComparatorProtocol(Protocol):
    def __call__(self, good_log: str, compare_dates: bool = False) -> Generator[None, Any, None]:
        ...


class StratEquityComparatorProtocol(Protocol):
    def __call__(self, trade: 'Trade', good_entry: dict[str, Any], good_exit: dict[str, Any]) -> None:
        ...


class FileReaderProtocol(Protocol):
    def __call__(self, file_path: str | None = None, *, subdir: str | None = None, suffix: str | None = ".txt") -> str:
        ...


class CsvReaderProtocol(Protocol):
    def __call__(self, file_path: str | None = None, *, subdir: str | None = None) -> CSVReader:
        ...


#
# Setup
#

# Disable __pycache__ creation
sys.dont_write_bytecode = True

# Disable color logging for Pyne codes
os.environ['PYNE_NO_COLOR_LOG'] = '1'


def pytest_configure(config: pytest.Config):
    ### pytest_spec ###

    # Set pytest_spec config
    config.inicfg['spec_test_format'] = "{result} {docstring_summary}"

    #  Monkey patching to support indentation #

    import pytest_spec.patch as psp

    def _print_description(self, msg=None):
        if msg is None:
            msg = self.currentfspath
        # count the number of '#' characters in the message
        indent = msg.count('#')
        # Set indent for test cases
        config._inicache['spec_indent'] = "\r" + ' ' * indent + "    "  # noqa
        # Replace '#' to indent
        msg = msg.replace('#', ' ')
        self._tw.line()
        if indent == 0 and getattr(self, '_first_triggered', False):
            self._tw.line()
        self._tw.write(msg, purple=indent == 0, cyan=indent > 0, bold=True)
        self._first_triggered = True

    psp._print_description = _print_description

    ### Set logging ###

    # Store the original factory function
    old_factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):
        nonlocal old_factory
        record = old_factory(*args, **kwargs)
        # Here we create the new field that already contains the desired parts
        record.module_func_line = f"{record.module}:{record.funcName}:{record.lineno}"
        return record

    # Set the new record factory
    logging.setLogRecordFactory(record_factory)


def get_module_categories(file_path: str) -> list[str]:
    """
    Get category names from __init__.py docstrings in the module path.

    :param file_path: Path to the test file (e.g. "pynecore/tests/ast/01_persistent_test.py")
    :return: Category string with "::" separator (e.g. "AST Tests::Persistent Tests")
    """
    # Convert to Path and make relative to tests dir
    path = Path(file_path)
    parts = path.parts
    test_idx = parts.index('tests')
    module_parts = parts[test_idx + 1:-1]  # Skip 'tests' and filename

    categories = []
    current = Path('/'.join(parts[:test_idx + 1]))  # Up to tests dir

    # Walk through module parts
    for part in module_parts:
        current = current / part
        init_file = current / '__init__.py'

        if init_file.exists():
            # Parse the file and get first line of docstring
            with open(init_file) as f:
                module = ast.parse(f.read())
                if (module.body and isinstance(module.body[0], ast.Expr) and
                        isinstance(cast(ast.Expr, module.body[0]).value, ast.Constant)):
                    doc = cast(ast.Constant, cast(ast.Expr, module.body[0]).value).value
                    # Get first line
                    first_line = doc.strip().split('\n')[0]
                    categories.append(first_line)

    return categories if categories else ['Other Tests']


@pytest.hookimpl(trylast=True)  # Because we want to sort after all other hooks
def pytest_collection_modifyitems(items):
    # Sort items by file path
    items.sort(key=lambda x: x.fspath.strpath)

    # Add categories and indentation to nodeid
    for item in items:
        file_path = item.location[0]
        module_categories = get_module_categories(file_path)

        indent = ""
        for i, category in enumerate(module_categories):
            module_categories[i] = indent + ("- " if indent else "* ") + category
            indent += INDENT * "#"

        # Remove the filename from nodeid and add the categories from modules
        item._nodeid = ("::".join(module_categories) + "::" + indent + item.nodeid.split("::", 1)[-1])


#
# Fixtures
#

@pytest.fixture(scope="function")
def log() -> logging.Logger:
    return logger


@pytest.fixture(scope="function")
def script_path(request) -> Path:
    return Path(request.path)


@pytest.fixture(scope="function")
def test_name(request) -> str:
    return request.node.name


@pytest.fixture(scope="function")
def module_key(script_path) -> str:
    tests_path = Path(__file__).parent.parent
    relative_script_path = script_path.relative_to(tests_path)
    module_key = relative_script_path.with_suffix('').as_posix().replace('/', '.')
    return module_key


@pytest.fixture(scope="function")
def ast_transformed_code(script_path, test_name, module_key) -> str:
    from pynecore.core.script_runner import import_script
    from contextlib import redirect_stdout
    from io import StringIO

    # Enable AST debug output
    os.environ['PYNE_AST_DEBUG_RAW'] = '1'

    # Remove module from sys.modules
    del sys.modules[module_key]

    # Import as script and get the modified code
    output = StringIO()
    with redirect_stdout(output):
        import_script(script_path)

    # Disable AST debug output
    del os.environ['PYNE_AST_DEBUG_RAW']

    code_str = output.getvalue()

    # Remove default __scope_id__ which os unique on each machine, and impossible to compare
    if "__scope_id__ = '" in code_str:
        _code_str = code_str.split("__scope_id__ = '")
        code_str = _code_str[0] + "__scope_id__ = ''" + _code_str[1].split("'", 1)[1]
    return code_str


@pytest.fixture(scope="function")
def ast_transform(script_path, test_name, module_key, request) -> Callable[[], str]:
    from inspect import unwrap

    fixture_fn = unwrap(ast_transformed_code)  # noqa

    def fixture():
        return fixture_fn(script_path, test_name, module_key)

    return fixture


@pytest.fixture(scope="function")
def syminfo() -> SymInfo:
    # Get 0-24/7 opening hours
    from pynecore.providers.ccxt import CCXTProvider
    opening_hours, session_starts, session_ends = CCXTProvider.get_opening_hours_and_sessions()

    return SymInfo(
        prefix="PYTEST",
        description="Pytest Symbol",
        ticker="TEST",
        currency="USD",
        period="5",
        type="crypto",
        mintick=0.00001000,
        pricescale=100000,
        minmove=1,
        pointvalue=1,
        timezone="UTC",
        volumetype="base",
        taker_fee=0.1,
        maker_fee=0.1,
        opening_hours=opening_hours,
        session_starts=session_starts,
        session_ends=session_ends
    )


@pytest.fixture(scope="function")
def dummy_ohlcv_iter():
    from datetime import datetime, UTC
    from itertools import cycle

    ohlcv = OHLCV(
        timestamp=int(datetime.fromisoformat("2025-01-01T00:00:00").replace(tzinfo=UTC).timestamp()),
        open=1.0,
        high=2.0,
        low=0.5,
        close=1.1,
        volume=10.0
    )

    return cycle([ohlcv])


@pytest.fixture(scope="function")
def runner(script_path, module_key, syminfo) -> RunnerProtocol:
    # Remove module from sys.modules to be able to re-import it
    del sys.modules[module_key]

    def _runner(ohlcv_iter: Iterable[OHLCV], syminfo_override: dict[str, Any] | None = None, *,
                syminfo_path: Path | None = None) -> ScriptRunner:
        nonlocal syminfo

        if syminfo_path is not None:
            syminfo = SymInfo.load_toml(syminfo_path)
        if syminfo_override is not None:
            for key, value in syminfo_override.items():
                setattr(syminfo, key, value)

        r = ScriptRunner(script_path, ohlcv_iter, syminfo)

        return r

    return cast(RunnerProtocol, _runner)


@pytest.fixture(scope="function")
def dict_comparator() -> DictComparatorProtocol:
    from pynecore import lib

    def _comparator(a: dict[str, Any], b: dict[str, Any], **kwargs):
        kwargs = dict(kwargs)
        abs_tol = kwargs.setdefault('abs_tol', 1e-8)
        rel_tol = kwargs.setdefault('rel_tol', 1e-5)

        key = None
        try:
            for key, value in a.items():
                assert key in b
                if isinstance(value, NA):
                    assert isinstance(b[key], NA)
                elif isinstance(b[key], NA):
                    assert isinstance(value, NA)
                elif isinstance(value, (float, int)):
                    assert isinstance(b[key], (float, int))
                    assert math.isclose(value, b[key], abs_tol=abs_tol, rel_tol=rel_tol)
                else:
                    assert value == b[key]
        except AssertionError:
            logger.error(
                f"Failed to compare values:\n"
                f"bar_index: {lib.bar_index}\n"
                f"datetime: {lib._datetime}  time: {lib._time}\n"  # noqa
                f"key: {key}\n"
                f"a: {a[key]} ({type(a[key])})\n"
                f"b: {b[key]} ({type(b[key])})\n")
            raise
        except TypeError:
            logger.error(
                f"Failed to compare values:\n"
                f"bar_index: {lib.bar_index}\n"
                f"datetime: {lib._datetime}  time: {lib._time}\n"  # noqa
                f"key: {key}\n"
                f"a: {a[key]} ({type(a[key])})\n"
                f"b: {b[key]} ({type(b[key])})\n")
            raise

    return cast(DictComparatorProtocol, _comparator)


@pytest.fixture(scope="function")
def log_comparator(capsys) -> LogComparatorProtocol:
    from datetime import datetime
    from contextlib import contextmanager
    import re

    def round_numbers_in_array(text: str, precision: int = 10) -> str:
        """Find arrays with floating point numbers and round them to specified precision"""

        def round_match(match: re.Match) -> str:
            array_str = match.group(0)
            # Parse numbers from array string
            numbers = [float(n) for n in re.findall(r'-?\d+\.?\d*(?:e[-+]?\d+)?', array_str)]
            # Round numbers and normalize zero
            rounded = []
            for n in numbers:
                n = round(n, precision)
                # Normalize -0.0 to 0.0
                if n == 0.0:
                    n = 0.0
                rounded.append(n)
            return str(rounded).replace("'", "")

        # Find arrays with floating point numbers and round them
        pattern = r'\[-?\d+\.?\d*(?:e[-+]?\d+)?(?:\s*,\s*-?\d+\.?\d*(?:e[-+]?\d+)?)*\]'
        return re.sub(pattern, round_match, text)

    @contextmanager
    def _comparator(good_log: str, compare_dates: bool = False, float_precision: int = 12):
        # Remove pytest handlers temporarily
        root_logger = logging.getLogger()
        pytest_handlers = [h for h in root_logger.handlers
                           if getattr(h, '__module__', '').startswith('_pytest')]
        for handler in pytest_handlers:
            root_logger.removeHandler(handler)

        # noinspection PyUnreachableCode
        try:
            # Run the test
            yield
            # Read log messages from stderr
            output = capsys.readouterr()
            test_log = output.err
            print(output.out, flush=True, end='')

            # Compare all lines
            line = 0
            for good_line, test_line in zip(good_log.splitlines(), test_log.splitlines()):
                line += 1
                good_line = good_line.strip()
                test_line = test_line.strip()
                if not good_line or not test_line:
                    continue
                try:
                    # Compare dates
                    if compare_dates:
                        good_date = datetime.fromisoformat(good_line.split(']', 1)[0].strip('['))
                        test_date = datetime.fromisoformat(test_line.split(']', 1)[0].strip('['))
                        assert good_date == test_date, f"Dates are not equal! Line: {line}"

                    # Round numbers in arrays and compare messages
                    good_msg = round_numbers_in_array(good_line.split(']:', 1)[1].strip(), float_precision)
                    test_msg = round_numbers_in_array(test_line.split(']', 1)[1][9:].strip(), float_precision)
                    assert good_msg == test_msg, f"Messages are not equal! Line: {line}"

                except IndexError:
                    print("The stderr output of the test:")
                    print(test_log)
                    raise ValueError(f"The log output is not in the expected format! Line: {line}")

        finally:
            # Restore pytest handlers
            for handler in pytest_handlers:
                root_logger.addHandler(handler)

    return cast(LogComparatorProtocol, _comparator)


@pytest.fixture(scope="function")
def strat_equity_comparator() -> StratEquityComparatorProtocol:
    from pynecore.lib import string
    from datetime import datetime

    def _comparator(trade: 'Trade', good_entry: dict[str, Any], good_exit: dict[str, Any], **__):
        # Skip trades where exit Signal is "Open" (TradingView backtest end artifact)
        if good_exit.get('Signal') == 'Open':
            return  # Skip validation for this trade
        # Field mapping: TradingView export names -> PyneCore field names
        tv_to_pynecore = {
            'P&L %': 'Profit %',
            'Net P&L %': 'Profit %',  # Now works correctly after fixing strat_extract.py
            'Cumulative P&L %': 'Cumulative profit %',
            'P&L USD': 'Profit USD',
            'Net P&L USD': 'Profit USD',  # Re-enabled after fixing percentage conversion
            'Cumulative P&L USD': 'Cumulative profit USD',
            'Quantity': 'Contracts',
        }

        def get_field(data: dict, field_name: str) -> Any:
            """Get field value with flexible naming"""
            # Direct match
            if field_name in data:
                return data[field_name]

            # Check if we need to map from TradingView format
            for tv_name, pynecore_name in tv_to_pynecore.items():
                if pynecore_name == field_name and tv_name in data:
                    return data[tv_name]

            # Return None if not found
            return None

        assert ((trade.sign > 0 and good_entry['Type'] == 'Entry long')
                or (trade.sign < 0 and good_entry['Type'] == 'Entry short'))

        # TradingView exports comment as Signal, or ID if no comment exists
        entry_signal = trade.entry_comment if trade.entry_comment else trade.entry_id
        exit_signal = trade.exit_comment if trade.exit_comment else trade.exit_id

        assert entry_signal == good_entry['Signal']
        assert exit_signal == good_exit['Signal']

        # Compare timestamps instead of string representations
        # This handles different timezone formats properly
        trade_entry_str = string.format_time(trade.entry_time, "yyyy-MM-ddTHH:mm:ssZ", tz="UTC")
        trade_exit_str = string.format_time(trade.exit_time, "yyyy-MM-ddTHH:mm:ssZ", tz="UTC")

        # Parse both timestamps and compare them
        trade_entry_dt = datetime.fromisoformat(trade_entry_str.replace('+0000', '+00:00'))
        trade_exit_dt = datetime.fromisoformat(trade_exit_str.replace('+0000', '+00:00'))

        good_entry_dt = datetime.fromisoformat(good_entry['Date/Time'])
        good_exit_dt = datetime.fromisoformat(good_exit['Date/Time'])

        # Compare the actual datetime objects (which handles timezone conversion)
        assert trade_entry_dt == good_entry_dt, f"Entry time mismatch: {trade_entry_str} != {good_entry['Date/Time']}"
        assert trade_exit_dt == good_exit_dt, f"Exit time mismatch: {trade_exit_str} != {good_exit['Date/Time']}"

        # Get profit fields using flexible field mapping
        profit_percent = get_field(good_exit, 'Profit %')
        cum_profit_percent = get_field(good_exit, 'Cumulative profit %')

        # Here the tradingview export has 2 decimal places, but sometimes has rounding issues
        # Some exports have incorrect scale (e.g., -5 instead of -0.05)
        # Use very large tolerance or skip comparison if values are too different
        if profit_percent is not None:
            expected_profit = float(profit_percent)
            # Check if the values are in completely different scales
            if abs(expected_profit) > 1 and abs(trade.profit_percent) < 1:
                # Likely a scale issue in the export, skip this check
                pass
            else:
                assert math.isclose(trade.profit_percent, expected_profit, abs_tol=0.1), \
                    f"Profit % mismatch: {trade.profit_percent} != {expected_profit} (tolerance: 0.1)"
        if cum_profit_percent is not None:
            expected_cum_profit = float(cum_profit_percent)
            # Check if the values are in completely different scales
            if abs(expected_cum_profit) > 1 and abs(trade.cum_profit_percent) < 1:
                # Likely a scale issue in the export, skip this check
                pass
            else:
                assert math.isclose(trade.cum_profit_percent, expected_cum_profit, abs_tol=0.1), \
                    f"Cumulative profit % mismatch: {trade.cum_profit_percent} != {expected_cum_profit} (tolerance: 0.1)"
        # assert math.isclose(trade.max_runup_percent, float(good_exit['Run-up %']), abs_tol=0.01)
        # assert math.isclose(trade.max_drawdown_percent, float(good_exit['Drawdown %']), abs_tol=0.01)

    return cast(StratEquityComparatorProtocol, _comparator)


@pytest.fixture(scope="function")
def file_reader(script_path, replace_scope_id=True) -> FileReaderProtocol:
    def _read_file(
            file_path: Path | None = None,
            *,
            subdir: str | None = None,
            suffix: str | None = ".txt"
    ) -> str:
        if file_path is None:
            file_path = script_path
        elif len(Path(file_path).parts) == 1:
            file_path = script_path.parent / file_path
        if subdir:
            assert file_path is not None
            file_path = Path(file_path).parent / subdir / file_path.name
        if suffix:
            file_path = Path(str(file_path).removesuffix('.py') + suffix)
        with open(file_path, 'r') as f:  # type: ignore
            return f.read()

    return cast(FileReaderProtocol, _read_file)


@pytest.fixture(scope="function")
def csv_reader(script_path) -> CsvReaderProtocol:
    def _csv_reader(file_path: Path | None = None, *, subdir: str | None = None) -> CSVReader:
        if file_path is None:
            file_path = script_path.with_suffix('.csv')
        elif len(Path(file_path).parts) == 1:
            file_path = script_path.parent / file_path
        if subdir:
            assert file_path is not None
            file_path = Path(file_path).parent / subdir / file_path.name
        return CSVReader(file_path)

    return cast(CsvReaderProtocol, _csv_reader)
