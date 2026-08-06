"""
@pyne

strategy.opentrades.capital_held — the capital the open trades hold.

Measured on TradingView (BINANCE:BTCUSDT 1D, pyramiding 3): the value is the sum
of |size| * entry price over the open trades. It stays put while the market
moves, a short trade contributes positively, and it is 0 while flat.

The margin percentages do not scale it: margin_long=50/margin_short=25 and
margin_long=50/margin_short=0 both report the full entry value. Only
margin_long=0 together with margin_short=0 is special, that reports na.
"""
from pynecore.lib import bar_index, plot, script, strategy


@script.strategy(
    "Capital held",
    overlay=True,
    initial_capital=100000,
    pyramiding=3,
    margin_long=50,
    margin_short=25,
)
def main():
    if bar_index == 1:
        strategy.entry('A', strategy.long, qty=2)
    if bar_index == 3:
        strategy.entry('B', strategy.long, qty=3)
    if bar_index == 5:
        strategy.close_all()
    if bar_index == 7:
        strategy.entry('S', strategy.short, qty=4)

    plot(strategy.opentrades.capital_held, "cap")
    plot(strategy.opentrades, "n")
    plot(strategy.opentrades.size(0), "s0")
    plot(strategy.opentrades.entry_price(0), "p0")
    plot(strategy.opentrades.size(1), "s1")
    plot(strategy.opentrades.entry_price(1), "p1")


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
def __test_capital_held_sums_the_open_entry_values__(script_path, module_key):
    """The plotted value equals the summed entry value of the open trades on every bar."""
    import sys
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner
    from pynecore.types.ohlcv import OHLCV

    sys.modules.pop(module_key, None)

    base_ts = 1_704_067_200_000  # 2024-01-01 00:00:00 UTC, in ms
    # A rising market, so a wrong mark-to-market implementation cannot hide behind
    # a flat price series
    bars = [
        OHLCV(timestamp=base_ts + i * 60_000, open=100.0 + 10 * i, high=110.0 + 10 * i,
              low=95.0 + 10 * i, close=105.0 + 10 * i, volume=100.0)
        for i in range(10)
    ]

    runner = ScriptRunner(Path(script_path), iter(bars), _make_syminfo())
    # The runner hands out the same plot dict on every bar and clears it in between,
    # so keep a copy per bar
    rows = [dict(plot_values) for _candle, plot_values, _closed in runner.run_iter()]

    assert len(rows) == len(bars), "the run must reach the last bar"

    # The strategy runs on 50% long / 25% short margins, so the full entry value here
    # also proves that the margin percentages do not scale capital_held
    for i, row in enumerate(rows):
        n = int(row['n'])
        assert n <= 2, f"bar {i}: more open trades than the test plots slots for: {n}"
        expected = sum(abs(row[f's{slot}']) * row[f'p{slot}'] for slot in range(n))
        assert abs(row['cap'] - expected) < 1e-9, f"bar {i}: {row['cap']} != {expected}"

    # The scenario must actually exercise every state, or the loop above is vacuous
    counts = [int(row['n']) for row in rows]
    assert counts[0] == 0, f"the run started in a position: {counts}"
    assert 2 in counts, f"the second entry never pyramided: {counts}"
    assert counts[6] == 0, f"close_all left trades open: {counts}"
    assert rows[6]['cap'] == 0.0, f"flat capital_held is not zero: {rows[6]['cap']}"

    # A short trade holds capital just like a long one
    assert rows[-1]['s0'] < 0.0, "the short entry did not fill"
    assert rows[-1]['cap'] > 0.0, f"the short trade held no capital: {rows[-1]['cap']}"

    # The value is entry based: it must not follow the market on a bar without a fill
    assert rows[-1]['cap'] == rows[-2]['cap'], "capital_held followed the market"
