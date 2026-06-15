"""
@pyne

Short-side mirror of the close_all-behind-partial clamp. A 10-unit short, a 30%
partial strategy.close() (covers 3) then strategy.close_all() on the same bar must
cover the remaining 7 and end flat — never overshoot into a phantom long. close_all
captured +10 (to flatten the -10 short) before the partial shed 3, so the clamp
must stop it at the current position. TradingView ends flat here (verified).
"""
from pynecore.lib import bar_index, script, strategy


@script.strategy(
    "Close-All Behind Partial (Short)",
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
        strategy.close('Short', 'TP', qty_percent=30)
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
def __test_close_all_behind_partial_short_flattens__(script_path, module_key):
    """30% cover then close_all on a short end flat, never flipping to a phantom long."""
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
    max_position = 0.0
    for _candle, _plot, new_closed in runner.run_iter():
        trades.extend(new_closed)
        max_position = max(max_position, lib._script.position.size)

    closed = [t for t in trades if t.size != 0.0]
    total = sum(abs(t.size) for t in closed)
    assert abs(total - 10.0) < 1e-9, (
        f"Partial + close_all must cover exactly the 10-unit short, got {total}"
    )
    assert all(t.size < 0.0 for t in closed), \
        f"close_all behind a partial must not flip to a long, got {[t.size for t in closed]}"
    assert max_position <= 0.0, f"Position must never flip long; saw {max_position}"
    assert abs(lib._script.position.size) < 1e-9, \
        f"Position must end flat, got {lib._script.position.size}"
