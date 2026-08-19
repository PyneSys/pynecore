"""
@pyne

A calc_on_order_fills re-execution must not see the bar's final extremes.

A COOF pass stands part-way through the bar — that is the whole point of the
feature. TradingView nevertheless hands the body the bar's COMPLETED
open/high/low/close/volume on every pass (measured on BINANCE:BTCUSDT 60m: all
four passes of a bar report identical OHLCV). PyneCore does not reproduce
lookahead, so each pass sees the bar as built up to the node it stands on.

The entry here is placed on bar 0 with no position, so its market order fills at
bar 1's OPEN — path node 0. The first re-execution therefore stands where the
bar has only its opening price: high == low == close == open, and a quarter of
the volume. The bar's real extremes (101/99) are still in the future there.
"""
from pynecore.lib import bar_index, close, high, low, open as bar_open, plot, script, strategy, volume
from pynecore.types import IBPersistent


@script.strategy(
    "COOF No Lookahead",
    overlay=True,
    initial_capital=100000,
    default_qty_type=strategy.fixed,
    default_qty_value=1,
    calc_on_order_fills=True,
)
def main():
    # ``varip`` survives the rollback between passes, so these keep what the
    # FIRST execution of the bar saw even after the definitive run overwrites
    # the price globals with the completed bar.
    marked: IBPersistent[int] = -1
    first_open: IBPersistent[float] = 0.0
    first_high: IBPersistent[float] = 0.0
    first_low: IBPersistent[float] = 0.0
    first_close: IBPersistent[float] = 0.0
    first_volume: IBPersistent[float] = 0.0

    if marked != bar_index:
        marked = bar_index
        first_open = bar_open
        first_high = high
        first_low = low
        first_close = close
        first_volume = volume

    if strategy.position_size == 0:
        strategy.entry('Long', strategy.long)

    plot(first_open, 'first_open')
    plot(first_high, 'first_high')
    plot(first_low, 'first_low')
    plot(first_close, 'first_close')
    plot(first_volume, 'first_volume')
    plot(high, 'final_high')
    plot(low, 'final_low')


def _make_syminfo(period: str = '5'):
    """Create a minimal SymInfo for testing."""
    from pynecore.core.syminfo import SymInfo
    from pynecore.providers.ccxt import CCXTProvider
    # noinspection PyProtectedMember
    opening_hours, session_starts, session_ends = CCXTProvider._create_24_7_sessions()
    return SymInfo(
        prefix="TEST", description="Test", ticker="TEST", currency="USD",
        period=period, type="crypto", mintick=0.01, pricescale=100,
        minmove=1, pointvalue=1, timezone="UTC", volumetype="base",
        mincontract=0.0001,
        opening_hours=opening_hours, session_starts=session_starts,
        session_ends=session_ends,
    )


def _make_ohlcv(num_bars: int, base_ts: int = 1_704_067_200_000, period: int = 300_000):
    """Create simple flat OHLCV bars."""
    from pynecore.types.ohlcv import OHLCV
    return [
        OHLCV(
            timestamp=base_ts + i * period,
            open=100.0, high=101.0, low=99.0, close=100.0, volume=1000.0
        )
        for i in range(num_bars)
    ]


# noinspection PyShadowingNames
def __test_coof_pass_sees_the_bar_as_built_so_far__(script_path, module_key):
    """A COOF re-execution reads the bar up to its node, never the final bar."""
    import sys
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner

    sys.modules.pop(module_key, None)

    runner = ScriptRunner(
        Path(script_path), iter(_make_ohlcv(num_bars=5)), _make_syminfo(period='5'),
    )
    results = [dict(plot_data) for _candle, plot_data, _trades in runner.run_iter()]

    assert len(results) == 5, f"Expected 5 bars, got {len(results)}"

    # Bar 0 never re-executes (the order is only placed there), so its single
    # execution is the definitive one and legitimately sees the whole bar.
    assert results[0]['first_high'] == 101.0
    assert results[0]['first_low'] == 99.0

    # Bar 1 fills the market order at the open, so the first pass stands on
    # node 0 — the bar has not moved off its opening price yet.
    bar1 = results[1]
    assert bar1['first_open'] == 100.0, f"bar 1 first_open: {bar1['first_open']}"
    assert bar1['first_high'] == 100.0, (
        f"bar 1 first_high: {bar1['first_high']} — the pass must not see the "
        f"bar's final high of 101.0, which had not happened yet"
    )
    assert bar1['first_low'] == 100.0, (
        f"bar 1 first_low: {bar1['first_low']} — the pass must not see the "
        f"bar's final low of 99.0, which had not happened yet"
    )
    assert bar1['first_close'] == 100.0, f"bar 1 first_close: {bar1['first_close']}"
    # Volume accrues a quarter per node in TradingView's 4-ticks-per-bar model.
    assert bar1['first_volume'] == 250.0, f"bar 1 first_volume: {bar1['first_volume']}"

    # The definitive execution of the same bar still reports the complete bar —
    # truncation applies to the discarded passes only.
    assert bar1['final_high'] == 101.0, f"bar 1 final_high: {bar1['final_high']}"
    assert bar1['final_low'] == 99.0, f"bar 1 final_low: {bar1['final_low']}"

    # Bars 2+ hold a position and place no new order, so they never re-execute
    # and their single pass is definitive.
    for i in range(2, 5):
        assert results[i]['first_high'] == 101.0, f"bar {i}: {results[i]['first_high']}"
        assert results[i]['first_low'] == 99.0, f"bar {i}: {results[i]['first_low']}"
