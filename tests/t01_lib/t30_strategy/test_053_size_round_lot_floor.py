"""
@pyne

Regression test for the lot-size floor (``_size_round``) on float64 boundaries.

Order sizes are floored to the instrument's lot step (``1 / _size_round_factor``;
1e-4 for non-BTC crypto). The float64 product of an EXACT lot multiple can land a
hair below the integer -- ``173.432 * 1e4 -> 1734319.9999999998`` -- and a bare
``int()`` floor truncated such a multiple a whole lot DOWN (to ``173.4319``).
TradingView keeps ``173.432``. ``_size_round`` now snaps values within a few ULPs
of a lot boundary up before flooring, so an exact multiple survives while a
genuine sub-lot fraction is still floored down.

Two entries guard both directions:
* ``A`` qty ``173.432`` -- exact lot multiple stored just below in float64; must
  NOT be truncated to ``173.4319``.
* ``B`` qty ``50.00055`` -- a genuine half-lot fraction; must still floor to
  ``50.0005`` (the snap must not bump a real fraction up).
"""
from pynecore.lib import bar_index, script, strategy


@script.strategy(
    "Lot-Size Floor",
    overlay=True,
    initial_capital=100000,
    default_qty_type=strategy.fixed,
    default_qty_value=1,
    pyramiding=0,
)
def main():
    if bar_index == 0:
        strategy.entry('A', strategy.long, qty=173.432)
    if bar_index == 1:
        strategy.close('A')
    if bar_index == 3:
        strategy.entry('B', strategy.long, qty=50.00055)
    if bar_index == 4:
        strategy.close('B')


def _make_syminfo(period: str = '1'):
    from pynecore.core.syminfo import SymInfo
    from pynecore.providers.ccxt import CCXTProvider
    opening_hours, session_starts, session_ends = CCXTProvider._create_24_7_sessions()
    return SymInfo(
        prefix="TEST", description="Test", ticker="TEST", currency="USD",
        period=period, type="crypto", mintick=0.01, pricescale=100,
        minmove=1, pointvalue=1, timezone="UTC", volumetype="base",
        opening_hours=opening_hours, session_starts=session_starts,
        session_ends=session_ends,
    )


# noinspection PyShadowingNames
def __test_lot_floor_keeps_exact_multiple_and_floors_fraction__(script_path, module_key):
    """
    An exact lot multiple survives the floor; a genuine sub-lot fraction is floored.

    Non-BTC crypto syminfo -> lot step 1e-4. ``173.432`` is an exact multiple but
    ``173.432 * 1e4`` is ``1734319.9999999998`` in float64; the filled size must be
    ``173.432``, not the float-truncated ``173.4319``. ``50.00055`` is a real half
    lot above ``50.0005`` and must floor down to ``50.0005``.
    """
    import sys
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner
    from pynecore.types.ohlcv import OHLCV

    sys.modules.pop(module_key, None)

    syminfo = _make_syminfo(period='1')
    base_ts = 1704067200  # 2024-01-01 00:00:00 UTC

    rows = [
        # open,  high,  low,   close
        (100.0, 100.5, 99.5, 100.0),  # bar 0 - entry A signal
        (100.0, 100.5, 99.5, 100.0),  # bar 1 - A fills @100, close A placed
        (100.0, 100.5, 99.5, 100.0),  # bar 2 - A closes @100, flat
        (100.0, 100.5, 99.5, 100.0),  # bar 3 - entry B signal
        (100.0, 100.5, 99.5, 100.0),  # bar 4 - B fills @100, close B placed
        (100.0, 100.5, 99.5, 100.0),  # bar 5 - B closes @100
        (100.0, 100.5, 99.5, 100.0),  # bar 6 - tail
    ]
    bars = [
        OHLCV(timestamp=base_ts + i * 60, open=o, high=h, low=l, close=c, volume=100.0)
        for i, (o, h, l, c) in enumerate(rows)
    ]

    runner = ScriptRunner(Path(script_path), iter(bars), syminfo)
    trades = []
    for _candle, _plot, new_closed in runner.run_iter():
        trades.extend(new_closed)

    assert len(trades) == 2, f"Expected 2 closed trades, got {len(trades)}"
    by_id = {t.entry_id: t for t in trades}
    assert sorted(by_id) == ['A', 'B'], f"entry ids: {sorted(by_id)}"
    # A: exact lot multiple must NOT be truncated a lot down (173.4319).
    assert abs(abs(by_id['A'].size) - 173.432) < 1e-9, (
        f"A size {by_id['A'].size} (expected 173.432, not the float-truncated 173.4319)"
    )
    # B: genuine half-lot fraction must floor DOWN to 50.0005 (no over-bump).
    assert abs(abs(by_id['B'].size) - 50.0005) < 1e-9, (
        f"B size {by_id['B'].size} (expected the floored 50.0005)"
    )
