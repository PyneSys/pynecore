"""
@pyne

The per-bar memo behind the account-currency rate.
"""
# Every money expression in the strategy engine reads the symbol-to-account point value,
# a dozen times or more per bar, but the rate itself is a daily series. Sampling it once
# per bar keeps the hot loop off the rate provider without changing a single number.
#
# The memo key is the pair (script object, bar index), not the bar index alone: PyneAPI
# runs several scripts in one process and re-applies syminfo every bar, so a bar-only key
# could hand one script the rate sampled for another. A realtime bar bypasses the memo
# because the chart-pair rate source reads lib.close, which moves within the bar.
from pynecore.lib import bar_index, currency, plot, script, strategy


@script.strategy(
    "Account rate memo",
    overlay=True,
    currency=currency.USD,
    initial_capital=10000,
    default_qty_type=strategy.fixed,
    default_qty_value=1,
    margin_long=0,
    margin_short=0,
)
def main():
    # An open position makes the engine re-read the point value all through the bar
    if bar_index == 0:
        strategy.entry('L', strategy.long, qty=1)

    plot(strategy.convert_to_account(1.0), "to_acct")
    plot(strategy.convert_to_symbol(1.0), "to_sym")
    plot(strategy.openprofit, "openprofit")
    plot(strategy.equity, "equity")
    plot(strategy.opentrades.capital_held, "capital_held")


BASE_TS = 1_704_067_200_000  # 2024-01-01 00:00:00 UTC, in ms
DAY_MS = 86_400_000
BAR_COUNT = 6
RATES = [0.7, 0.8, 0.9, 1.1, 1.2, 1.3]


def _make_syminfo():
    """Chart symbol quoted in USDT, so an USD account has to convert."""
    from pynecore.core.syminfo import SymInfo
    from pynecore.providers.ccxt import CCXTProvider
    # noinspection PyProtectedMember
    opening_hours, session_starts, session_ends = CCXTProvider._create_24_7_sessions()
    return SymInfo(
        prefix="TEST", description="Test", ticker="BTCUSDT", currency="USDT",
        basecurrency="BTC", period='1D', type="crypto", mintick=0.01, pricescale=100,
        minmove=1, pointvalue=1, timezone="UTC", volumetype="base",
        mincontract=0.0001,
        opening_hours=opening_hours, session_starts=session_starts,
        session_ends=session_ends,
    )


def _write_rate_file(dir_path) -> str:
    """Daily USDT/USD series where ``RATES[i]`` is in force on chart bar ``i``."""
    from pynecore.core.ohlcv import OHLCVWriter
    from pynecore.types.ohlcv import OHLCV

    base_path = dir_path / "USDTUSD"
    with OHLCVWriter(base_path.with_suffix('.ohlcv'), "1D", truncate=True) as writer:
        for i, rate in enumerate(RATES):
            ts = BASE_TS + (i - 1) * DAY_MS
            writer.write(OHLCV(ts, rate, rate, rate, rate, 100.0))

    base_path.with_suffix('.toml').write_text(
        '[symbol]\nprefix = "TEST"\ndescription = "USDTUSD"\nticker = "USDTUSD"\n'
        'currency = "USD"\nbasecurrency = "USDT"\nperiod = "1D"\ntype = "forex"\n'
        'mintick = 0.00001\npricescale = 100000\npointvalue = 1.0\ntimezone = "UTC"\n'
        '[[opening_hours]]\nday = 1\nstart = "00:00:00"\nend = "23:59:59"\n'
        '[[session_starts]]\nday = 1\ntime = "00:00:00"\n'
        '[[session_ends]]\nday = 1\ntime = "23:59:59"\n'
    )
    return str(base_path)


def _run(script_path, module_key, rate_path: str) -> list[dict]:
    """Run the script with the rate source attached."""
    import sys
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner
    from pynecore.types.ohlcv import OHLCV

    # Both import names of this file, so every run gets a fresh strategy object: pytest
    # holds it under the dotted package path, ``import_script`` under the bare stem.
    sys.modules.pop(module_key, None)
    sys.modules.pop(Path(script_path).stem, None)

    bars = [
        OHLCV(timestamp=BASE_TS + i * DAY_MS, open=100.0, high=110.0, low=90.0,
              close=100.0 + i, volume=100.0)
        for i in range(BAR_COUNT)
    ]
    runner = ScriptRunner(Path(script_path), iter(bars), _make_syminfo(),
                          security_data={"rate_USDTUSD": rate_path})
    rows = []
    for _candle, plot_values, _closed in runner.run_iter():
        rows.append(dict(plot_values))
    return rows


# noinspection PyShadowingNames
def __test_the_rate_is_sampled_once_per_bar__(script_path, module_key):
    """One provider lookup per bar, however many money expressions read the rate."""
    import tempfile
    from pathlib import Path
    from pynecore.core.currency import CurrencyRateProvider

    calls = []
    original = CurrencyRateProvider.get_rate

    def counting_get_rate(self, from_cur, to_cur, timestamp):
        calls.append((from_cur, to_cur, timestamp))
        return original(self, from_cur, to_cur, timestamp)

    with tempfile.TemporaryDirectory() as tmpdir:
        rate_path = _write_rate_file(Path(tmpdir))
        CurrencyRateProvider.get_rate = counting_get_rate
        try:
            rows = _run(script_path, module_key, rate_path)
            first_run_calls = len(calls)
            _run(script_path, module_key, rate_path)
        finally:
            CurrencyRateProvider.get_rate = original

    assert first_run_calls == BAR_COUNT, \
        f"expected {BAR_COUNT} lookups, got {first_run_calls}"
    # The memo is dropped between runs, so the second run samples every bar again
    assert len(calls) == 2 * BAR_COUNT, f"expected {2 * BAR_COUNT} lookups, got {len(calls)}"

    # Each bar saw its own rate, so the memo is per bar and not per run
    for i, row in enumerate(rows):
        assert abs(row['to_acct'] - RATES[i]) < 1e-12, f"bar {i}: {row['to_acct']}"


# noinspection PyShadowingNames
def __test_the_memo_key_is_the_script_and_the_bar__(script_path, module_key):
    """A new bar, another script, or a realtime bar all bypass the stored value."""
    import sys
    import tempfile
    from pathlib import Path
    from pynecore import lib

    with tempfile.TemporaryDirectory() as tmpdir:
        _run(script_path, module_key, _write_rate_file(Path(tmpdir)))

    strat = sys.modules['pynecore.lib.strategy']
    saved = (lib._script, lib.bar_index, lib.barstate.isrealtime)
    # With no script the account is the symbol's own currency, so a resample returns the
    # bare point value and is unmistakable next to the planted memo
    unconverted = lib.syminfo.pointvalue
    planted = unconverted + 122.0
    try:
        lib._script = None
        lib.bar_index = 7
        lib.barstate.isrealtime = False

        strat._conv_script = None
        strat._conv_bar = 7
        strat._conv_pv = planted
        assert strat._account_point_value() == planted, "same script, same bar: memo hit"

        lib.bar_index = 8
        assert strat._account_point_value() == unconverted, "new bar: resampled"

        strat._conv_bar = 8
        strat._conv_pv = planted
        strat._conv_script = object()
        assert strat._account_point_value() == unconverted, "other script: resampled"

        strat._conv_script = None
        strat._conv_bar = 8
        strat._conv_pv = planted
        lib.barstate.isrealtime = True
        assert strat._account_point_value() == unconverted, "realtime bar: resampled"
    finally:
        lib._script, lib.bar_index, lib.barstate.isrealtime = saved
        strat._reset_currency_state()
