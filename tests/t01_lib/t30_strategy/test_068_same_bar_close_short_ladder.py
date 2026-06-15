"""
@pyne

Short-side mirror of the same-bar close-stacking regression (the DEMA ATR bug
that motivated the fix was ALL shorts: a sharp down bar pierced two TP levels at
once). Three same-bar strategy.close() partial closes on a SHORT entry (30% +
30% + 40%) must all stack and flatten the position, shedding -3, -3, -4.
"""
from pynecore.lib import bar_index, script, strategy


@script.strategy(
    "Same-Bar Short Close Ladder",
    overlay=True,
    initial_capital=100000,
    default_qty_type=strategy.fixed,
    default_qty_value=10,
    pyramiding=0,
)
def main():
    if bar_index == 0:
        strategy.entry('Short', strategy.short)
    if bar_index == 1:
        strategy.close('Short', 'TP1', qty_percent=30)
        strategy.close('Short', 'TP2', qty_percent=30)
        strategy.close('Short', 'TP3', qty_percent=40)


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
def __test_short_ladder_stacks_and_flattens__(script_path, module_key):
    """30% + 30% + 40% on one bar shed a 10-unit short as three closes (-3, -3, -4), ending flat."""
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
    for _candle, _plot, new_closed in runner.run_iter():
        trades.extend(new_closed)

    closed = [t for t in trades if t.size != 0.0]
    assert len(closed) == 3, (
        f"Expected 3 stacked short closes, got {len(closed)} — same-bar "
        "strategy.close() collision regressed on the short side."
    )
    # Shorts close with negative trade size.
    assert all(t.size < 0.0 for t in closed), \
        f"All short closes must be negative size, got {[t.size for t in closed]}"
    assert sorted(abs(t.size) for t in closed) == [3.0, 3.0, 4.0], \
        f"Expected slices [3, 3, 4], got {sorted(abs(t.size) for t in closed)}"
    assert abs(lib._script.position.size) < 1e-9, \
        f"The ladder must flatten the short; final position {lib._script.position.size}"
