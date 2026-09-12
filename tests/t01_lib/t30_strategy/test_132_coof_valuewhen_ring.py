"""
calc_on_order_fills regression: ``ta.valuewhen``'s occurrence ring is filled
once per EXECUTION of the body and a discarded re-execution never gives its
push back.

Measured on TradingView (CAPITALCOM:EURUSD 30, calc_on_order_fills=true,
condition ``bar_index % 10 == 0``, a market entry placed one bar earlier so it
fills at the open of the condition bar): occurrences 0 and 1 both read the
condition bar itself and occurrence 2 the previous condition bar. Adding a
one-tick profit target that fills intrabar on the same bar -- a second fill,
so a third execution -- pushes the duplicate one slot further: 0, 1 and 2 read
the current bar and 3 the previous one. The same script with
calc_on_order_fills off reads the clean current/previous/one-before ring, and
a ``var`` counter plus ``ta.cum`` stay un-doubled on the very same bars, so it
is the ring that lives outside the rollback -- which is ``varip``.
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


def __test_coof_valuewhen_ring__():
    """ A re-executed bar pushes its occurrence again, like TradingView """
    from pynecore.core import viz
    from pynecore.core.script_runner import ScriptRunner

    try:
        runner = ScriptRunner(
            DATA_DIR / 'coof_valuewhen_ring.py', iter(_make_ohlcv(4)), _make_syminfo(),
        )
        results = [dict(plot_data) for _candle, plot_data, _trades in runner.run_iter()]
    finally:
        viz.reset_state()
        sys.modules.pop('coof_valuewhen_ring', None)

    # Control: bar 1 fills the entry and runs the body three times, so the
    # execution total outruns the bar count from there on
    assert [r['total_execs'] for r in results] == [1, 4, 5, 6]

    # The condition is true on bars 0 and 1 only. Bar 1 pushes once per
    # execution, so from there on the whole ring reads bar 1
    def read(key: str) -> list[float | None]:
        """The plotted column with na spelled ``None`` instead of a bare nan."""
        return [None if value != value else value for value in (r[key] for r in results)]

    assert read('occ0') == [0.0, 1.0, 1.0, 1.0]
    assert read('occ1') == [None, 1.0, 1.0, 1.0]
    assert read('occ2') == [None, 1.0, 1.0, 1.0]
