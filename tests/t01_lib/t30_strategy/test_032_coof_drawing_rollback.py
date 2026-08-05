"""
calc_on_order_fills regression: a re-executed bar must not leave the drawings
of its discarded runs behind.

The registries are plain module-level containers, so nothing used to remove
what a discarded run drew: a fill bar that ran its body three times ended up
with three lines instead of one. They fill the script's ``max_lines_count``
budget and evict live drawings, and ``line.all`` reports them.

Measured on TradingView (FX:EURUSD 240, calc_on_order_fills=true, one
line.new per execution plus a close issued as soon as the position is open):
``array.size(line.all)`` grows by exactly 1 on every bar, fill bars included,
while the trade series proves the body really re-ran -- the same script with
calc_on_order_fills=false closes one bar later.
"""
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent / 'data'


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


def __test_coof_drawing_rollback__():
    """ A re-executed bar leaves exactly one line behind, like TradingView """
    from pynecore.core import viz
    from pynecore.core.script_runner import ScriptRunner

    try:
        runner = ScriptRunner(
            DATA_DIR / 'coof_drawing_rollback.py', iter(_make_ohlcv(5)), _make_syminfo(),
        )
        results = [dict(plot_data) for _candle, plot_data, _trades in runner.run_iter()]
    finally:
        viz.reset_state()
        sys.modules.pop('coof_drawing_rollback', None)

    # Bar 1 fills the entry, so its body runs several times -- one line per bar
    # regardless
    assert [r['lines'] for r in results] == [1, 2, 3, 4, 5]

    # Control: the module-level counter is outside the rollback, so it reports
    # every execution. Bar 1 fills the entry and runs the body three times, so
    # the totals must outrun the bar count from there on
    assert [r['total_execs'] for r in results] == [1, 4, 5, 6, 7]
