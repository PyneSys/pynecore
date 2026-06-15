"""
@pyne

Three same-bar partial closes whose percentages sum past 100% must clamp to the
open position, never reverse it. 50% + 50% + 50% of a call-time 10-unit long each
ask to shed 5 (15 total); the fills must stop at 10 and leave the position flat,
not flip it short. Guards the stacking fix against cumulative over-close.
"""
from pynecore.lib import bar_index, script, strategy


@script.strategy(
    "Same-Bar Three-Way Over-Close",
    overlay=True,
    initial_capital=100000,
    default_qty_type=strategy.fixed,
    default_qty_value=10,
    pyramiding=0,
)
def main():
    if bar_index == 0:
        strategy.entry('Long', strategy.long)
    if bar_index == 1:
        strategy.close('Long', 'C1', qty_percent=50)
        strategy.close('Long', 'C2', qty_percent=50)
        strategy.close('Long', 'C3', qty_percent=50)


def _make_syminfo(period: str = '1'):
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


# noinspection PyShadowingNames,PyProtectedMember
def __test_three_way_overclose_clamps_to_flat__(script_path, module_key):
    """50% + 50% + 50% shed exactly the 10-unit long and leave it flat, never short."""
    import sys
    from pathlib import Path
    from pynecore import lib
    from pynecore.core.script_runner import ScriptRunner
    from pynecore.types.ohlcv import OHLCV

    sys.modules.pop(module_key, None)

    base_ts = 1704067200
    bars = [
        OHLCV(timestamp=base_ts + i * 60, open=100.0, high=100.5, low=99.5,
              close=100.0, volume=100.0)
        for i in range(4)
    ]

    runner = ScriptRunner(Path(script_path), iter(bars), _make_syminfo())
    trades = []
    min_position = 0.0
    for _candle, _plot, new_closed in runner.run_iter():
        trades.extend(new_closed)
        min_position = min(min_position, lib._script.position.size)

    closed = [t for t in trades if t.size != 0.0]
    total = sum(abs(t.size) for t in closed)
    assert abs(total - 10.0) < 1e-9, (
        f"Three 50% closes must shed exactly the 10-unit position (clamped), got {total}"
    )
    assert all(t.size > 0.0 for t in closed), \
        f"Over-close must not reverse into a short, got {[t.size for t in closed]}"
    assert min_position >= 0.0, f"Position must never flip short; saw {min_position}"
    assert abs(lib._script.position.size) < 1e-9, \
        f"Position must end flat, got {lib._script.position.size}"
