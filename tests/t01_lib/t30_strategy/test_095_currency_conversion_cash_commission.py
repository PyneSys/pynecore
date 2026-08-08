"""
@pyne

Cash sizing and cash commissions against an account currency.
"""
# strategy.cash spends a fixed amount of the ACCOUNT currency, so a smaller rate buys more
# contracts: qty = (cash / rate) / price, 584/584 entries measured on BINANCE:BTCUSDT with
# a JPY account.
#
# The cash commissions are the mirror image. cash_per_contract and cash_per_order are
# already denominated in the account currency and are booked verbatim -- 584/584 each,
# worst 4.8e-08 relative, against an alternative that converts them and misses by 157x.
# They are the only money inputs the engine takes without a point value, so they stay
# untouched by construction; this test is what keeps them that way.
from pynecore.lib import bar_index, currency, plot, script, strategy


@script.strategy(
    "Cash sizing and cash fees",
    overlay=True,
    currency=currency.USD,
    initial_capital=10000,
    default_qty_type=strategy.cash,
    default_qty_value=1000,
    commission_type=strategy.commission.cash_per_contract,
    commission_value=2,
    margin_long=0,
    margin_short=0,
)
def main():
    if bar_index == 1:
        strategy.entry('L', strategy.long)
    if bar_index == 3:
        strategy.close('L')

    plot(strategy.position_size, "position_size")
    plot(strategy.netprofit, "netprofit")
    plot(strategy.closedtrades.profit(0), "profit")
    plot(strategy.closedtrades.commission(0), "commission")


BASE_TS = 1_704_067_200_000  # 2024-01-01 00:00:00 UTC, in ms
DAY_MS = 86_400_000
RATE = 0.5

CHART_BARS = [
    #  open,   high,    low,  close
    (100.0, 100.0, 100.0, 100.0),
    (100.0, 100.0, 100.0, 100.0),
    (100.0, 100.0, 100.0, 100.0),
    (100.0, 100.0, 100.0, 100.0),
    (120.0, 120.0, 120.0, 120.0),
    (120.0, 120.0, 120.0, 120.0),
]


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
        for i in range(len(CHART_BARS)):
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
def __test_cash_sizing_and_cash_fees__(script_path, module_key):
    """The budget converts, the per-contract fee does not."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        converted = _run(script_path, module_key, _write_rate_file(Path(tmpdir)))
    plain = _run(script_path, module_key, None)

    # 1000 account-currency units at a unit cost of 100 * 0.5
    assert converted[-1]['position_size'] == 0.0, "the position must be closed"
    assert converted[2]['position_size'] == 20.0, converted[2]['position_size']
    assert plain[2]['position_size'] == 10.0, plain[2]['position_size']
    assert converted[2]['position_size'] == plain[2]['position_size'] / RATE

    # 20 contracts x 2 per contract on each leg, booked as-is in the account currency.
    # Converting them would charge 40 instead
    assert converted[-1]['commission'] == 80.0, converted[-1]['commission']
    # 20 contracts x 20 points x 0.5, minus the two unconverted legs
    assert converted[-1]['profit'] == 20.0 * 20.0 * RATE - 80.0
    assert converted[-1]['netprofit'] == converted[-1]['profit']

    # Half the contracts, so half the fee, and the same gross in the symbol's currency
    assert plain[-1]['commission'] == 40.0, plain[-1]['commission']
    assert plain[-1]['profit'] == 10.0 * 20.0 - 40.0
