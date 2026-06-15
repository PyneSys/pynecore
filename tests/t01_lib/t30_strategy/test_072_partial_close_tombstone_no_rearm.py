"""
@pyne

A fired partial strategy.close() leg becomes a consumed tombstone that stays in the
order book (its reservation must keep counting) while its entry is still open, and
is re-filled as a no-op every bar. That no-op re-fill must NOT re-arm the same-bar
partial-close marker (_partial_close_bar) — otherwise a later, unrelated close_all
overshoot (e.g. from a deferred margin call) would be wrongly clamped to flat on a
bar where no real partial close happened. This guards the marker against the
tombstone re-fill: after the partial fires once, the marker stays pinned to that
bar and is not re-stamped on subsequent close-free bars.
"""
from pynecore.lib import bar_index, script, strategy


@script.strategy(
    "Partial Close Tombstone No Re-arm",
    overlay=True,
    initial_capital=1000000,
    default_qty_type=strategy.fixed,
    default_qty_value=100,
    pyramiding=0,
)
def main():
    if bar_index == 0:
        strategy.entry('A', strategy.long)
    if bar_index == 1:
        strategy.close('A', 'P', qty=10)
    # Bars 2+: NO closes — the consumed tombstone keeps getting no-op re-filled.


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
def __test_tombstone_refill_does_not_rearm_marker__(script_path, module_key):
    """The consumed partial-close tombstone must not re-stamp _partial_close_bar each bar."""
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
        for i in range(8)
    ]

    runner = ScriptRunner(Path(script_path), iter(bars), _make_syminfo())
    marker_by_bar: dict[int, int] = {}
    for _candle, _plot, _new_closed in runner.run_iter():
        pos = lib._script.position
        marker_by_bar[int(lib.bar_index)] = pos._partial_close_bar

    # The partial close('A', qty=10) is placed on bar 1 and fills on bar 2's open,
    # so the marker must be armed exactly once, for bar 2.
    assert marker_by_bar.get(2) == 2, \
        f"Partial close should arm the marker on its fill bar (2); saw {marker_by_bar.get(2)}"
    # On every later bar the script issues no close; the tombstone re-fill must not
    # re-arm the marker, so it stays pinned at 2 and never equals the current bar.
    for b in range(3, 8):
        assert marker_by_bar[b] == 2, (
            f"Tombstone re-fill wrongly re-armed the marker on bar {b} "
            f"(got {marker_by_bar[b]}, expected it pinned at 2)"
        )
        assert marker_by_bar[b] != b, \
            f"Marker must not equal current bar {b} when no real partial close fired"
