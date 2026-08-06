"""
@pyne

strategy.opentrades.capital_held on a strategy that requires no margin at all.

Measured on TradingView (BINANCE:BTCUSDT 1D, pyramiding 3): with margin_long=0 and
margin_short=0 the value is na on every bar, flat ones included. Zeroing only one
side keeps the full entry value, so both sides have to be zero.
"""
from pynecore.lib import bar_index, na, plot, script, strategy


@script.strategy(
    "Capital held without margin",
    overlay=True,
    initial_capital=100000,
    pyramiding=3,
    margin_long=0,
    margin_short=0,
)
def main():
    if bar_index == 1:
        strategy.entry('A', strategy.long, qty=2)
    if bar_index == 5:
        strategy.close_all()
    if bar_index == 7:
        strategy.entry('S', strategy.short, qty=4)

    plot(1.0 if na(strategy.opentrades.capital_held) else 0.0, "is_na")
    plot(strategy.opentrades, "n")


def _make_syminfo():
    from pynecore.core.syminfo import SymInfo
    from pynecore.providers.ccxt import CCXTProvider
    # noinspection PyProtectedMember
    opening_hours, session_starts, session_ends = CCXTProvider._create_24_7_sessions()
    return SymInfo(
        prefix="TEST", description="Test", ticker="TEST", currency="USD",
        period='1', type="crypto", mintick=0.01, pricescale=100,
        minmove=1, pointvalue=1, timezone="UTC", volumetype="base",
        mincontract=0.0001,
        opening_hours=opening_hours, session_starts=session_starts,
        session_ends=session_ends,
    )


# noinspection PyShadowingNames
def __test_capital_held_is_na_without_margin__(script_path, module_key):
    """capital_held is na on every bar when neither side requires margin."""
    import sys
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner
    from pynecore.types.ohlcv import OHLCV

    sys.modules.pop(module_key, None)

    base_ts = 1_704_067_200_000  # 2024-01-01 00:00:00 UTC, in ms
    bars = [
        OHLCV(timestamp=base_ts + i * 60_000, open=100.0 + 10 * i, high=110.0 + 10 * i,
              low=95.0 + 10 * i, close=105.0 + 10 * i, volume=100.0)
        for i in range(10)
    ]

    runner = ScriptRunner(Path(script_path), iter(bars), _make_syminfo())
    rows = [dict(plot_values) for _candle, plot_values, _closed in runner.run_iter()]

    assert len(rows) == len(bars), "the run must reach the last bar"

    for i, row in enumerate(rows):
        assert row['is_na'] == 1.0, f"bar {i}: capital_held is not na"

    # The scenario must reach both the flat and the open state, or the loop is vacuous
    counts = [int(row['n']) for row in rows]
    assert counts[0] == 0, f"the run started in a position: {counts}"
    assert 1 in counts, f"no entry ever filled: {counts}"
    assert counts[6] == 0, f"close_all left trades open: {counts}"
