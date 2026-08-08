"""
@pyne

Percent commission across a rate change.
"""
# The conversion is cash-flow level, and a percent commission is a cash flow like any
# other: it is booked at the rate of the bar its leg fills on. Measured on
# BINANCE:BTCUSDT against currency.JPY, 274/274 closed trades, worst 1.1e-7 relative:
#
#     profit = gross * rate(exit) - entry_fee * rate(entry) - exit_fee * rate(exit)
#
# Every competing model -- converting the whole net at the exit rate, at the entry rate,
# or charging both fees at one of the two -- got 1 of the 188 trades whose rate moved.
# The engine gets this for free from the point value: the entry fee is computed on the
# entry bar and both exit legs on the exit bar, each with that bar's rate.
from pynecore.lib import bar_index, currency, plot, script, strategy


@script.strategy(
    "Percent commission conversion",
    overlay=True,
    currency=currency.USD,
    initial_capital=10000,
    default_qty_type=strategy.fixed,
    default_qty_value=1,
    commission_type=strategy.commission.percent,
    commission_value=1,
    margin_long=0,
    margin_short=0,
)
def main():
    if bar_index == 1:
        strategy.entry('L', strategy.long, qty=1)
    if bar_index == 3:
        strategy.close('L')

    plot(strategy.netprofit, "netprofit")
    plot(strategy.closedtrades.profit(0), "profit")
    plot(strategy.closedtrades.commission(0), "commission")


BASE_TS = 1_704_067_200_000  # 2024-01-01 00:00:00 UTC, in ms
DAY_MS = 86_400_000

CHART_BARS = [
    #  open,   high,    low,  close
    (100.0, 100.0, 100.0, 100.0),
    (100.0, 100.0, 100.0, 100.0),
    (100.0, 100.0, 100.0, 100.0),
    (100.0, 100.0, 100.0, 100.0),
    (120.0, 120.0, 120.0, 120.0),
    (120.0, 120.0, 120.0, 120.0),
]

# The rate steps between the entry fill (bar 2) and the exit fill (bar 4)
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
        OHLCV(timestamp=BASE_TS + i * DAY_MS, open=o, high=h, low=l, close=c, volume=100.0)
        for i, (o, h, l, c) in enumerate(CHART_BARS)
    ]
    runner = ScriptRunner(Path(script_path), iter(bars), _make_syminfo(),
                          security_data={"rate_USDTUSD": rate_path})
    rows = []
    for _candle, plot_values, _closed in runner.run_iter():
        rows.append(dict(plot_values))
    return rows


# noinspection PyShadowingNames
def __test_each_commission_leg_takes_its_own_bar_rate__(script_path, module_key):
    """The entry fee rides the entry rate and the exit fee the exit rate."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        rows = _run(script_path, module_key, _write_rate_file(Path(tmpdir)))

    entry_fee = 1.0 * 100.0 * RATES[2] * 0.01
    exit_fee = 1.0 * 120.0 * RATES[4] * 0.01
    expected = 1.0 * (120.0 - 100.0) * RATES[4] - entry_fee - exit_fee

    assert abs(rows[-1]['commission'] - (entry_fee + exit_fee)) < 1e-12, \
        f"commission {rows[-1]['commission']} != {entry_fee + exit_fee}"
    # 23.74. Charging both fees at the exit rate would give 23.14, both at the entry
    # rate 24.46, and converting the whole net at the exit rate 23.14 as well
    assert abs(rows[-1]['profit'] - expected) < 1e-12, f"profit {rows[-1]['profit']}"
    assert abs(rows[-1]['netprofit'] - expected) < 1e-12, f"netprofit {rows[-1]['netprofit']}"
