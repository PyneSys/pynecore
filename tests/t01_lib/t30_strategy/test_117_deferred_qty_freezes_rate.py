"""
@pyne

A resting default-sized entry converts its frozen budget at the PLACEMENT rate.

MEASURED on the wild-corpus strategy "Breakout Trend Follower" (BINANCE:BTCUSDT
30m, a USD account on a USDT-quoted symbol, so the daily COINBASE:USDTUSD rate
steps at every 00:00 bar): 12 of its 580 percent_of_equity entries came out one
lot off while the FILL bar's rate converted the per-unit cost, ten of them on a
00:00 bar. TradingView freezes the whole sizing at the bar the order was last
placed -- the money AND the quote-to-account rate -- and only the FILL PRICE
comes from the execution. With the placement rate all 580 sizes are exact.

Here the rate doubles between placement and fill, so the two models differ by a
factor of two in the resolved quantity, not by a rounding lot.
"""
from pynecore.lib import bar_index, currency, plot, script, strategy


@script.strategy(
    "Deferred qty freezes rate",
    overlay=True,
    currency=currency.USD,
    initial_capital=10000,
    default_qty_type=strategy.percent_of_equity,
    default_qty_value=100,
    margin_long=0,
    margin_short=0,
)
def main():
    # Placed on bar 1 and never re-issued, so the budget freezes at bar 1's close.
    # The stop sits above every price until bar 4, where it fills at the open.
    if bar_index == 1:
        strategy.entry('L', strategy.long, stop=120.0)

    plot(strategy.position_size, "psize")


__test_helper_BASE_TS = 1_704_067_200_000  # 2024-01-01 00:00:00 UTC, in ms
__test_helper_DAY_MS = 86_400_000

__test_helper_CHART_BARS = [
    #  open,   high,    low,  close
    (100.0, 100.0, 100.0, 100.0),
    (100.0, 100.0, 100.0, 100.0),
    (100.0, 100.0, 100.0, 100.0),
    (100.0, 100.0, 100.0, 100.0),
    (125.0, 125.0, 125.0, 125.0),   # the stop is marketable: fills at this open,
                                    # sized at its own 120 level
    (125.0, 125.0, 125.0, 125.0),
]

# The rate doubles between the placement bar (1) and the fill bar (4)
__test_helper_RATES = [0.5, 0.5, 0.5, 1.0, 1.0, 1.0]


def __test_helper_make_syminfo():
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


def __test_helper_write_rate_file(dir_path) -> str:
    """Daily USDT/USD series where ``__test_helper_RATES[i]`` is in force on chart bar ``i``."""
    from pynecore.core.ohlcv import OHLCVWriter
    from pynecore.types.ohlcv import OHLCV

    base_path = dir_path / "USDTUSD"
    with OHLCVWriter(base_path.with_suffix('.ohlcv'), "1D", truncate=True) as writer:
        for i, rate in enumerate(__test_helper_RATES):
            ts = __test_helper_BASE_TS + (i - 1) * __test_helper_DAY_MS
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


def __test_helper_run(script_path, module_key, rate_path: str) -> list[dict]:
    """Run the script with the rate source attached."""
    import sys
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner
    from pynecore.types.ohlcv import OHLCV

    sys.modules.pop(module_key, None)
    sys.modules.pop(Path(script_path).stem, None)

    bars = [
        OHLCV(timestamp=__test_helper_BASE_TS + i * __test_helper_DAY_MS,
              open=o, high=h, low=l, close=c, volume=100.0)
        for i, (o, h, l, c) in enumerate(__test_helper_CHART_BARS)
    ]
    runner = ScriptRunner(Path(script_path), iter(bars), __test_helper_make_syminfo(),
                          security_data={"rate_USDTUSD": rate_path})
    rows = []
    for _candle, plot_values, _closed in runner.run_iter():
        rows.append(dict(plot_values))
    return rows


# noinspection PyShadowingNames
def __test_resting_entry_sizes_at_the_placement_rate__(script_path, module_key):
    """The rate rides with the frozen money, not with the fill price."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        rate_path = __test_helper_write_rate_file(Path(tmpdir))
        rows = __test_helper_run(script_path, module_key, rate_path)

    # 10000 USD of equity buys 10000 / (120 * 0.5) = 166.6666 contracts at the
    # frozen placement rate (a gapped stop is sized at its level, see
    # test_143); the fill-bar rate would have bought only 83.3333.
    assert abs(rows[4]['psize'] - 166.6666) < 1e-9, (
        f"expected the placement rate {__test_helper_RATES[1]} to size the fill, "
        f"got {rows[4]['psize']}"
    )
