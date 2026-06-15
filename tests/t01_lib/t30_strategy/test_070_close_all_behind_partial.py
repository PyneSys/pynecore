"""
@pyne

A same-bar strategy.close() partial followed by strategy.close_all() must flatten
the position, never reverse it. close_all captures -position.size at call time; if
the partial close already shed part of the position by the time close_all fills,
the leftover must clamp to flat instead of opening a phantom opposite trade.
A 10-unit long, 30% partial close (-3) then close_all must close 3 + 7 and end flat.
"""
from pynecore.lib import bar_index, script, strategy


@script.strategy(
    "Close-All Behind Partial",
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
        strategy.close('Long', 'TP', qty_percent=30)
        strategy.close_all()


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
def __test_close_all_behind_partial_flattens__(script_path, module_key):
    """30% partial close then close_all on one bar end flat, never reversing to a short."""
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
        f"Partial + close_all must shed exactly the 10-unit long, got {total}"
    )
    assert all(t.size > 0.0 for t in closed), \
        f"close_all behind a partial must not reverse into a short, got {[t.size for t in closed]}"
    assert min_position >= 0.0, f"Position must never flip short; saw {min_position}"
    assert abs(lib._script.position.size) < 1e-9, \
        f"Position must end flat, got {lib._script.position.size}"
