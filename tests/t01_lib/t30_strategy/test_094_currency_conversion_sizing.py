"""
@pyne

Account-currency conversion of percent-of-equity sizing.
"""
# strategy.percent_of_equity spends a slice of the equity, and equity is in the account
# currency while the price is in the symbol's. Measured on BINANCE:BTCUSDT with a JPY
# account: qty = floor_mc((equity / rate) / price), 581/584 entries, the residue coming
# from TradingView sizing on the placement bar's close. Dividing an account-currency
# budget by an account-currency unit cost is the same arithmetic, which is what the
# engine does by carrying the rate in the point value.
#
# The rate is a flat 0.5 here so the expected contract counts are exact integers and the
# floor to the mincontract grid cannot mask a wrong model.
from pynecore.lib import bar_index, currency, plot, script, strategy


@script.strategy(
    "Percent of equity sizing",
    overlay=True,
    currency=currency.USD,
    initial_capital=10000,
    default_qty_type=strategy.percent_of_equity,
    default_qty_value=50,
    margin_long=0,
    margin_short=0,
)
def main():
    if bar_index == 1:
        strategy.entry('L', strategy.long)

    plot(strategy.position_size, "position_size")
    plot(strategy.position_avg_price, "avg_price")
    plot(strategy.equity, "equity")


BASE_TS = 1_704_067_200_000  # 2024-01-01 00:00:00 UTC, in ms
DAY_MS = 86_400_000
BAR_COUNT = 5
RATE = 0.5


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
    """Flat daily USDT/USD series covering every chart bar."""
    from pynecore.core.ohlcv import OHLCVWriter
    from pynecore.types.ohlcv import OHLCV

    base_path = dir_path / "USDTUSD"
    with OHLCVWriter(base_path.with_suffix('.ohlcv'), "1D", truncate=True) as writer:
        for i in range(BAR_COUNT):
            ts = BASE_TS + (i - 1) * DAY_MS
            writer.write(OHLCV(ts, RATE, RATE, RATE, RATE, 100.0))

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
        OHLCV(timestamp=BASE_TS + i * DAY_MS, open=100.0, high=100.0, low=100.0,
              close=100.0, volume=100.0)
        for i in range(BAR_COUNT)
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
def __test_percent_of_equity_sizes_on_the_converted_price__(script_path, module_key):
    """The equity slice buys the contracts it can afford at the account-currency price."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        converted = _run(script_path, module_key, _write_rate_file(Path(tmpdir)))
    plain = _run(script_path, module_key, None)

    # 50% of 10000 spent at a unit cost of 100 * 0.5
    assert converted[-1]['position_size'] == 100.0, converted[-1]['position_size']
    # Without the conversion the same budget only reaches half as many contracts
    assert plain[-1]['position_size'] == 50.0, plain[-1]['position_size']
    assert converted[-1]['position_size'] == plain[-1]['position_size'] / RATE

    # The entry price stays in the symbol's currency -- it is a price, not an amount
    assert converted[-1]['avg_price'] == plain[-1]['avg_price'] == 100.0
    # Nothing is realized, so the equity is untouched on both sides
    assert converted[-1]['equity'] == plain[-1]['equity'] == 10000.0
