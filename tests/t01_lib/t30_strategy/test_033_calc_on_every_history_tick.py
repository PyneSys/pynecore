"""
@pyne

calc_on_every_history_tick runs the body at every node of a historical bar.

TradingView's July 2026 release added `calc_on_every_history_tick` to
`strategy()`: the body is calculated on every available tick of a historical bar
instead of once at its close. Measured 2026-08-18 on BINANCE:BTCUSDT 60m with no
bar magnifier, the body runs FOUR times per bar — the broker emulator's assumed
path, open -> the extreme nearest the open -> the other extreme -> close.

Each pass stands mid-bar, so it sees the bar as built up to its node; only the
last one sees the completed bar. TradingView hands every pass the finished
OHLCV, which is lookahead — PyneCore does not reproduce it.

The data below rises (o=100, h=104, l=99, c=103), so the low is nearest the open
and the walk is open -> low -> high -> close: 100, 99, 104, 103.
"""
from pynecore.lib import bar_index, close, high, low, plot, script, strategy, volume
from pynecore.types import IBPersistent


@script.strategy(
    "CEHT",
    overlay=True,
    initial_capital=100000,
    default_qty_type=strategy.fixed,
    default_qty_value=1,
    calc_on_every_history_tick=True,
)
def main():
    # ``varip`` survives the rollback between passes, so the counter and the
    # recorded prices outlive the discarded runs.
    passes: IBPersistent[int] = 0
    marked: IBPersistent[int] = -1
    c0: IBPersistent[float] = 0.0
    c1: IBPersistent[float] = 0.0
    c2: IBPersistent[float] = 0.0
    h0: IBPersistent[float] = 0.0
    l0: IBPersistent[float] = 0.0
    v0: IBPersistent[float] = 0.0

    if marked != bar_index:
        marked = bar_index
        passes = 0
    passes += 1

    if passes == 1:
        c0, h0, l0, v0 = close, high, low, volume
    elif passes == 2:
        c1 = close
    elif passes == 3:
        c2 = close

    plot(passes, 'passes')
    plot(c0, 'c0')
    plot(c1, 'c1')
    plot(c2, 'c2')
    plot(h0, 'h0')
    plot(l0, 'l0')
    plot(v0, 'v0')
    plot(close, 'final_close')
    plot(high, 'final_high')


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
    """Create rising OHLCV bars whose low is the extreme nearest the open."""
    from pynecore.types.ohlcv import OHLCV
    return [
        OHLCV(
            timestamp=base_ts + i * period,
            open=100.0, high=104.0, low=99.0, close=103.0, volume=1000.0
        )
        for i in range(num_bars)
    ]


# noinspection PyShadowingNames
def __test_body_runs_on_every_node_of_the_assumed_path__(script_path, module_key):
    """Four executions per historical bar, each seeing the bar up to its node."""
    import sys
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner

    sys.modules.pop(module_key, None)

    runner = ScriptRunner(
        Path(script_path), iter(_make_ohlcv(num_bars=4)), _make_syminfo(period='5'),
    )
    results = [dict(plot_data) for _candle, plot_data, _trades in runner.run_iter()]

    assert len(results) == 4, f"Expected 4 bars, got {len(results)}"

    for i, res in enumerate(results):
        assert res['passes'] == 4, f"bar {i}: {res['passes']} passes, expected 4"
        # Node 0 — the bar has only its opening price.
        assert res['c0'] == 100.0, f"bar {i} c0: {res['c0']}"
        assert res['h0'] == 100.0, f"bar {i} h0: {res['h0']} — the final high is lookahead"
        assert res['l0'] == 100.0, f"bar {i} l0: {res['l0']} — the final low is lookahead"
        assert res['v0'] == 250.0, f"bar {i} v0: {res['v0']}"
        # Node 1 — the low, the extreme nearest this bar's open.
        assert res['c1'] == 99.0, f"bar {i} c1: {res['c1']}"
        # Node 2 — the other extreme.
        assert res['c2'] == 104.0, f"bar {i} c2: {res['c2']}"
        # Node 3 — the definitive execution sees the completed bar.
        assert res['final_close'] == 103.0, f"bar {i} final_close: {res['final_close']}"
        assert res['final_high'] == 104.0, f"bar {i} final_high: {res['final_high']}"


# noinspection PyShadowingNames
def __test_disabled_by_default__(script_path, module_key):
    """Without the flag the body still runs exactly once per bar."""
    import sys
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner

    sys.modules.pop(module_key, None)

    runner = ScriptRunner(
        Path(script_path), iter(_make_ohlcv(num_bars=4)), _make_syminfo(period='5'),
    )
    # The decorator's flag is what the runner reads, so clearing it is enough to
    # compare the two modes on one body. The ``Script`` it lives on is shared
    # with the other tests of this module, hence the restore.
    runner.script.calc_on_every_history_tick = False
    try:
        results = [dict(pd) for _c, pd, _t in runner.run_iter()]
    finally:
        runner.script.calc_on_every_history_tick = True

    for i, res in enumerate(results):
        assert res['passes'] == 1, f"bar {i}: {res['passes']} passes, expected 1"
        assert res['c0'] == 103.0, f"bar {i} c0: {res['c0']}"


def _make_sub_bars(base_ts: int, prices: list[tuple[float, float, float, float]]) -> list:
    """Create 1-minute OHLCV sub-bars from (open, high, low, close) tuples."""
    from pynecore.types.ohlcv import OHLCV
    return [
        OHLCV(timestamp=base_ts + i * 60_000, open=o, high=h, low=lo, close=c, volume=100.0)
        for i, (o, h, lo, c) in enumerate(prices)
    ]


# noinspection PyShadowingNames
def __test_magnified_run_stands_at_the_end_of_every_sub_bar__(script_path, module_key):
    """With the magnifier the sub-bars replace the assumed path's four nodes."""
    import sys
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner

    sys.modules.pop(module_key, None)

    base_ts = 1_704_067_200_000
    # Five 1-minute sub-bars per 5-minute chart bar. The chart bar aggregates to
    # o=100, h=104, l=99, c=103 — the same bar the assumed-path test above uses,
    # but here the real path inside it is known.
    prices = [
        (100.0, 100.0, 100.0, 100.0),
        (100.0, 100.0, 99.0, 99.0),
        (99.0, 102.0, 99.0, 102.0),
        (102.0, 104.0, 102.0, 104.0),
        (104.0, 104.0, 103.0, 103.0),
    ]
    sub_bars = _make_sub_bars(base_ts, prices) + _make_sub_bars(base_ts + 300_000, prices)

    runner = ScriptRunner(
        Path(script_path), iter([]), _make_syminfo(period='5'),
        magnifier_iter=iter(sub_bars),
    )
    results = [dict(plot_data) for _candle, plot_data, _trades in runner.run_iter()]

    assert len(results) == 2, f"Expected 2 chart bars, got {len(results)}"

    for i, res in enumerate(results):
        # One pass per sub-bar: four discarded ones plus the definitive execution.
        assert res['passes'] == 5, f"bar {i}: {res['passes']} passes, expected 5"
        # Sub-bar 0 — the chart bar has not left its opening price yet.
        assert res['c0'] == 100.0, f"bar {i} c0: {res['c0']}"
        assert res['h0'] == 100.0, f"bar {i} h0: {res['h0']} — the final high is lookahead"
        assert res['l0'] == 100.0, f"bar {i} l0: {res['l0']} — the final low is lookahead"
        assert res['v0'] == 100.0, f"bar {i} v0: {res['v0']}"
        # Sub-bar 1 closes at the chart bar's low.
        assert res['c1'] == 99.0, f"bar {i} c1: {res['c1']}"
        # Sub-bar 2 — still below the chart bar's high.
        assert res['c2'] == 102.0, f"bar {i} c2: {res['c2']}"
        # The definitive execution sees the completed chart bar.
        assert res['final_close'] == 103.0, f"bar {i} final_close: {res['final_close']}"
        assert res['final_high'] == 104.0, f"bar {i} final_high: {res['final_high']}"
