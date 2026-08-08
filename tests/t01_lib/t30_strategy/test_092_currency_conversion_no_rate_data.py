"""
@pyne

Degradation when strategy(currency=...) has no rate data behind it.
"""
# A conversion the run cannot perform must never poison the ledger. request.currency_rate
# answers na when no rate source carries the pair, and the ledger then multiplies the
# point value by exactly 1.0 -- bit-identical to a run that never had anything to convert.
# The user is told once per run, on the engine's own logger rather than the Pine log.*
# stream, which carries script output and is compared against TradingView logs.
from pynecore.lib import bar_index, currency, plot, script, strategy


@script.strategy(
    "No rate data",
    overlay=True,
    currency=currency.USD,
    initial_capital=10000,
    default_qty_type=strategy.fixed,
    default_qty_value=2,
    margin_long=0,
    margin_short=0,
)
def main():
    if bar_index == 1:
        strategy.entry('L', strategy.long, qty=2)
    if bar_index == 3:
        strategy.close('L')
    if bar_index == 5:
        strategy.entry('S', strategy.short, qty=1)

    plot(strategy.netprofit, "netprofit")
    plot(strategy.equity, "equity")
    plot(strategy.openprofit, "openprofit")
    plot(strategy.grossprofit, "grossprofit")
    plot(strategy.grossloss, "grossloss")
    plot(strategy.max_drawdown, "max_drawdown")
    plot(strategy.max_runup, "max_runup")
    plot(strategy.position_size, "position_size")
    plot(strategy.opentrades.capital_held, "capital_held")


BASE_TS = 1_704_067_200_000  # 2024-01-01 00:00:00 UTC, in ms
DAY_MS = 86_400_000

CHART_BARS = [
    #  open,   high,    low,  close
    (100.0, 101.0, 99.0, 100.0),
    (100.0, 101.0, 99.0, 100.0),
    (100.0, 120.0, 85.0, 100.0),
    (100.0, 115.0, 90.0, 110.0),
    (120.0, 121.0, 119.0, 120.0),
    (120.0, 121.0, 119.0, 120.0),
    (90.0, 95.0, 88.0, 92.0),
]

MONEY_KEYS = ("netprofit", "equity", "openprofit", "grossprofit", "grossloss",
              "max_drawdown", "max_runup")


def _make_syminfo(quote: str):
    """Chart symbol quoted in ``quote``; only that string differs between the two runs."""
    from pynecore.core.syminfo import SymInfo
    from pynecore.providers.ccxt import CCXTProvider
    # noinspection PyProtectedMember
    opening_hours, session_starts, session_ends = CCXTProvider._create_24_7_sessions()
    return SymInfo(
        prefix="TEST", description="Test", ticker="BTCQUOTE", currency=quote,
        basecurrency="BTC", period='1D', type="crypto", mintick=0.01, pricescale=100,
        minmove=1, pointvalue=1, timezone="UTC", volumetype="base",
        mincontract=0.0001,
        opening_hours=opening_hours, session_starts=session_starts,
        session_ends=session_ends,
    )


def _run(script_path, module_key, quote: str) -> list[dict]:
    """Run the script on a symbol quoted in ``quote``, with no rate source anywhere."""
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
    runner = ScriptRunner(Path(script_path), iter(bars), _make_syminfo(quote))
    rows = []
    for _candle, plot_values, _closed in runner.run_iter():
        rows.append(dict(plot_values))
    return rows


# noinspection PyShadowingNames
def __test_missing_rate_leaves_the_ledger_unconverted__(script_path, module_key, caplog):
    """An unusable rate degrades to exactly the run that has nothing to convert."""
    import logging
    import math

    with caplog.at_level(logging.WARNING, logger='pynecore.lib.strategy'):
        degraded = _run(script_path, module_key, "USDT")
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    caplog.clear()

    # The account currency is the symbol's own here, so no rate is ever requested
    with caplog.at_level(logging.WARNING, logger='pynecore.lib.strategy'):
        native = _run(script_path, module_key, "USD")
    assert not [r for r in caplog.records if r.levelno == logging.WARNING], \
        "a run that needs no conversion must stay silent"

    assert len(degraded) == len(CHART_BARS), "the run must reach the last bar"
    for i, (a, b) in enumerate(zip(degraded, native)):
        assert a == b, f"bar {i}: {a} != {b}"

    # No na leaks in from the failed lookup
    for i, row in enumerate(degraded):
        for key in MONEY_KEYS:
            assert not math.isnan(row[key]), f"bar {i}: {key} is na"

    # Told once per run, not once per bar and not once per money expression
    assert len(warnings) == 1, f"expected one warning, got {len(warnings)}"
    assert "USDT" in warnings[0].getMessage() and "USD" in warnings[0].getMessage()
