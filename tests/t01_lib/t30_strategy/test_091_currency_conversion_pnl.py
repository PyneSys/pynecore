"""
@pyne

Account-currency conversion of the strategy ledger.
"""
# TradingView converts a strategy's money at the cash-flow level: every amount is booked
# at the symbol-to-account rate of the bar it happens on, and nothing re-marks afterwards.
# Measured on BINANCE:BTCUSDT against currency.JPY (18.4% rate amplitude, 274/274 closed
# trades, worst 1.1e-7 relative): profit = gross * rate(exit bar).
#
# Converting the two legs separately -- entry value at the entry rate, exit value at the
# exit rate -- is refuted: on the 53 trades whose rate moved, the exit-rate model missed
# by 0.0039 in total and the per-leg model by 2768.14. Unrealized P&L follows the current
# bar's rate instead (14320/14320), and the percent metrics divide two amounts booked at
# the same rate, so the rate cancels out of them.
#
# The rate series here swings 0.7 -> 1.3 mid-trade, far wider than any real pair, so a
# wrong model cannot hide in the noise. The run-up and draw-down bars are picked so that
# the converted extreme lands on a different bar than the unconverted one.
from pynecore.lib import bar_index, currency, plot, script, strategy


@script.strategy(
    "Currency conversion P&L",
    overlay=True,
    currency=currency.USD,
    initial_capital=10000,
    default_qty_type=strategy.fixed,
    default_qty_value=1,
    margin_long=0,
    margin_short=0,
)
def main():
    if bar_index == 1:
        strategy.entry('L', strategy.long, qty=1)
    if bar_index == 3:
        strategy.close('L')

    plot(strategy.netprofit, "netprofit")
    plot(strategy.equity, "equity")
    plot(strategy.openprofit, "openprofit")
    plot(strategy.closedtrades.entry_bar_index(0), "entry_bar")
    plot(strategy.closedtrades.exit_bar_index(0), "exit_bar")
    plot(strategy.closedtrades.entry_price(0), "entry_price")
    plot(strategy.closedtrades.exit_price(0), "exit_price")
    plot(strategy.closedtrades.profit(0), "profit")
    plot(strategy.closedtrades.profit_percent(0), "profit_pct")
    plot(strategy.closedtrades.max_runup(0), "runup")
    plot(strategy.closedtrades.max_drawdown(0), "drawdown")


BASE_TS = 1_704_067_200_000  # 2024-01-01 00:00:00 UTC, in ms
DAY_MS = 86_400_000

# Chart price series. Bar 2 has the wider high/low, bar 3 the higher rate, so the
# unconverted extremes sit on bar 2 and the converted ones on bar 3.
CHART_BARS = [
    #  open,   high,    low,  close
    (100.0, 101.0, 99.0, 100.0),
    (100.0, 101.0, 99.0, 100.0),
    (100.0, 120.0, 85.0, 100.0),
    (100.0, 115.0, 90.0, 110.0),
    (120.0, 121.0, 119.0, 120.0),
    (120.0, 121.0, 119.0, 120.0),
]

# Symbol-to-account (USDT -> USD) rate in force on each chart bar
RATES = [0.7, 0.7, 0.7, 1.3, 1.3, 1.3]


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


def _write_rate_file(dir_path, rates: list[float]) -> str:
    """
    Write a daily USDT/USD rate series where ``rates[i]`` is in force on chart bar ``i``.

    Only an already-closed rate bar is readable, so the bar carrying the rate for chart
    bar ``i`` opens one day earlier.

    :param dir_path: Directory to write into.
    :param rates: Rate per chart bar.
    :return: Path stem accepted by ``security_data``.
    """
    from pynecore.core.ohlcv import OHLCVWriter
    from pynecore.types.ohlcv import OHLCV

    base_path = dir_path / "USDTUSD"
    with OHLCVWriter(base_path.with_suffix('.ohlcv'), "1D", truncate=True) as writer:
        for i, rate in enumerate(rates):
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


def _run(script_path, module_key, rate_path: str | None) -> list[dict]:
    """Run the script once, with or without a rate source attached."""
    import sys
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner
    from pynecore.types.ohlcv import OHLCV

    # Both import names of this file, so every run gets a fresh strategy object: pytest
    # holds it under the dotted package path, ``import_script`` under the bare stem.
    sys.modules.pop(module_key, None)
    sys.modules.pop(Path(script_path).stem, None)

    bars = [
        OHLCV(timestamp=BASE_TS + i * DAY_MS, open=o, high=h, low=l, close=c, volume=100.0)
        for i, (o, h, l, c) in enumerate(CHART_BARS)
    ]
    runner = ScriptRunner(
        Path(script_path), iter(bars), _make_syminfo(),
        security_data={"rate_USDTUSD": rate_path} if rate_path else None,
    )
    rows = []
    for _candle, plot_values, _closed in runner.run_iter():
        rows.append(dict(plot_values))
    return rows


# noinspection PyShadowingNames
def __test_ledger_converts_at_the_rate_of_each_booking_bar__(script_path, module_key):
    """Realized profit rides the exit rate, unrealized P&L the current bar's rate."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        rows = _run(script_path, module_key, _write_rate_file(Path(tmpdir), RATES))

    assert len(rows) == len(CHART_BARS), "the run must reach the last bar"

    # The trade fills at the open of the bar after the call, so it spans the rate change
    last = rows[-1]
    assert last['entry_bar'] == 2, f"entry filled on bar {last['entry_bar']}"
    assert last['exit_bar'] == 4, f"exit filled on bar {last['exit_bar']}"
    assert last['entry_price'] == 100.0
    assert last['exit_price'] == 120.0

    gross = 1.0 * (120.0 - 100.0)
    # Booked entirely at the exit bar's rate. Converting the legs separately would give
    # 120 * 1.3 - 100 * 0.7 = 86.0, and leaving it unconverted 20.0
    assert abs(last['profit'] - gross * RATES[4]) < 1e-9, f"profit {last['profit']} != 26.0"

    # Unrealized P&L re-marks every bar at that bar's rate
    assert abs(rows[2]['openprofit'] - 0.0) < 1e-9
    assert abs(rows[3]['openprofit'] - 10.0 * RATES[3]) < 1e-9, \
        f"openprofit {rows[3]['openprofit']} != 13.0"

    # equity = initial capital + realized + unrealized, in the account currency
    assert abs(rows[3]['equity'] - (10000.0 + rows[3]['netprofit']
                                   + rows[3]['openprofit'])) < 1e-9
    assert abs(last['equity'] - (10000.0 + gross * RATES[4])) < 1e-9
    assert abs(last['netprofit'] - gross * RATES[4]) < 1e-9


# noinspection PyShadowingNames
def __test_runup_and_drawdown_use_their_own_bar_rate__(script_path, module_key):
    """The extremes are maxima over per-bar amounts, each converted where it occurred."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        rows = _run(script_path, module_key, _write_rate_file(Path(tmpdir), RATES))

    last = rows[-1]
    # Bar 2: (120 - 100) * 0.7 = 14.0, bar 3: (115 - 100) * 1.3 = 19.5.
    # Unconverted the winner would be bar 2 with 20.0, and converting the whole extreme
    # at the exit rate would give 20 * 1.3 = 26.0
    assert abs(last['runup'] - 19.5) < 1e-9, f"runup {last['runup']} != 19.5"
    # Bar 2: (100 - 85) * 0.7 = 10.5, bar 3: (100 - 90) * 1.3 = 13.0.
    # Unconverted the winner would be bar 2 with 15.0
    assert abs(last['drawdown'] - 13.0) < 1e-9, f"drawdown {last['drawdown']} != 13.0"


# noinspection PyShadowingNames
def __test_percent_metrics_are_rate_invariant__(script_path, module_key):
    """profit_percent divides two amounts booked at the same rate, so the rate cancels."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        converted = _run(script_path, module_key, _write_rate_file(Path(tmpdir), RATES))
    plain = _run(script_path, module_key, None)

    assert converted[-1]['profit_pct'] == plain[-1]['profit_pct'], \
        "profit_percent must not depend on the account currency"
    assert abs(converted[-1]['profit_pct'] - 20.0) < 1e-9

    # The unconverted run is the reference the conversion multiplies
    assert abs(plain[-1]['profit'] - 20.0) < 1e-9
    assert abs(converted[-1]['profit'] - plain[-1]['profit'] * RATES[4]) < 1e-9


# noinspection PyShadowingNames
def __test_flat_rate_series_leaves_the_ledger_untouched__(script_path, module_key):
    """A rate of exactly 1.0 is bit-identical to running without any rate source."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        ones = _run(script_path, module_key, _write_rate_file(Path(tmpdir), [1.0] * len(RATES)))
    plain = _run(script_path, module_key, None)

    for i, (a, b) in enumerate(zip(ones, plain)):
        assert a == b, f"bar {i}: {a} != {b}"
