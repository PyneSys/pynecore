"""
@pyne

Pine v4/v5 order slots of ``strategy.close()`` (``close_by_id=True``): a close of an
entry id MODIFIES the pending close of that id placed earlier on the bar, so of
three close statements on one bar only the last one fills, while closes of two
different ids each keep their own order.

Measured on TradingView (CAPITALCOM:EURUSD 60, v4 and v5, with and without
process_orders_on_close, ``qty`` and ``qty_percent`` alike): a 10-unit short closed by
4 + 3 + 3 on one bar sheds only the last 3, the other 7 stay open; v6 sheds all 10.
Each call is sized against what the pending close leaves open before it replaces
it: qty 3 then a full close sheds 7, a full close then qty 3 sheds the full 10, and
qty 8, 5, 1 sheds 1 (the 5 is cut to the 2 the 8 left open). An ``immediately=True``
close that a later call of the same id replaced does not fill either.
"""
from pynecore.lib import bar_index, script, strategy


@script.strategy(
    "Same-Bar Close By Id",
    overlay=True,
    initial_capital=100000,
    default_qty_type=strategy.fixed,
    default_qty_value=10,
    pyramiding=2,
    close_by_id=True,
)
def main():
    if bar_index == 0:
        strategy.entry('S', strategy.short)
    if bar_index == 2:
        strategy.close('S', 'TP1', qty=4)
        strategy.close('S', 'TP2', qty=3)
        strategy.close('S', 'TP3', qty=3)
    if bar_index == 4:
        strategy.close_all('flat')
    if bar_index == 5:
        strategy.entry('A', strategy.long)
        strategy.entry('B', strategy.long)
    if bar_index == 7:
        strategy.close('A', 'TPA', qty=4)
        strategy.close('B', 'TPB', qty=3)
    if bar_index == 8:
        strategy.close_all('flat')
    if bar_index == 9:
        strategy.entry('C', strategy.short)
    if bar_index == 11:
        strategy.close('C', 'part', qty=3)
        strategy.close('C', 'full')
    if bar_index == 13:
        strategy.close_all('flat')
    if bar_index == 14:
        strategy.entry('D', strategy.short)
    if bar_index == 16:
        strategy.close('D', 'full')
        strategy.close('D', 'part', qty=3)
    if bar_index == 18:
        strategy.entry('E', strategy.short)
    if bar_index == 20:
        strategy.close('E', 'p8', qty=8)
        strategy.close('E', 'p5', qty=5)
        strategy.close('E', 'p1', qty=1)
    if bar_index == 22:
        strategy.close_all('flat')
    if bar_index == 25:
        strategy.entry('F', strategy.short)
    if bar_index == 27:
        strategy.close('F', 'i4', qty=4, immediately=True)
        strategy.close('F', 'i3a', qty=3, immediately=True)
        strategy.close('F', 'i3b', qty=3, immediately=True)
    if bar_index == 29:
        strategy.close_all('flat')
    if bar_index == 30:
        strategy.entry('G', strategy.short)
    if bar_index == 32:
        strategy.close('G', 'ipart', qty=3, immediately=True)
        strategy.close('G', 'full')
    if bar_index == 34:
        strategy.close_all('flat')


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
def __test_same_id_closes_keep_the_last__(script_path, module_key):
    """Same-id closes on one bar leave the last order; different ids both fill."""
    import sys
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner
    from pynecore.types.ohlcv import OHLCV

    sys.modules.pop(module_key, None)

    base_ts = 1_704_067_200_000  # 2024-01-01 00:00:00 UTC, in ms
    bars = [
        OHLCV(timestamp=base_ts + i * 60_000, open=100.0, high=100.5, low=99.5,
              close=100.0, volume=100.0)
        for i in range(25)
    ]

    runner = ScriptRunner(Path(script_path), iter(bars), _make_syminfo())
    trades = []
    for _candle, _plot, new_closed in runner.run_iter():
        trades.extend(new_closed)

    short = sorted((t.exit_comment, abs(t.size)) for t in trades if t.entry_id == 'S')
    assert short == [('TP3', 3.0), ('flat', 7.0)], short

    longs = sorted((t.entry_id, t.exit_comment, abs(t.size))
                   for t in trades if t.entry_id in ('A', 'B'))
    # FIFO: both closes consume entry A first, TPB takes the 3 that TPA left on A.
    assert longs[:2] == [('A', 'TPA', 4.0), ('A', 'TPB', 3.0)], longs


# noinspection PyShadowingNames,PyProtectedMember
def __test_each_close_is_sized_against_the_pending_one__(script_path, module_key):
    """A close takes only what the id's pending close leaves open, then replaces it."""
    import sys
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner
    from pynecore.types.ohlcv import OHLCV

    sys.modules.pop(module_key, None)

    base_ts = 1_704_067_200_000  # 2024-01-01 00:00:00 UTC, in ms
    bars = [
        OHLCV(timestamp=base_ts + i * 60_000, open=100.0, high=100.5, low=99.5,
              close=100.0, volume=100.0)
        for i in range(25)
    ]

    runner = ScriptRunner(Path(script_path), iter(bars), _make_syminfo())
    trades = []
    for _candle, _plot, new_closed in runner.run_iter():
        trades.extend(new_closed)

    def sheds(entry_id: str) -> list[tuple[str, float]]:
        return sorted((t.exit_comment, abs(t.size)) for t in trades if t.entry_id == entry_id)

    assert sheds('C') == [('flat', 3.0), ('full', 7.0)], sheds('C')
    assert sheds('D') == [('full', 10.0)], sheds('D')
    assert sheds('E') == [('flat', 9.0), ('p1', 1.0)], sheds('E')


# noinspection PyShadowingNames,PyProtectedMember
def __test_replaced_immediate_close_does_not_fill__(script_path, module_key):
    """Only the immediate close the slot still holds fills; a replaced one is dropped."""
    import sys
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner
    from pynecore.types.ohlcv import OHLCV

    sys.modules.pop(module_key, None)

    base_ts = 1_704_067_200_000  # 2024-01-01 00:00:00 UTC, in ms
    bars = [
        OHLCV(timestamp=base_ts + i * 60_000, open=100.0, high=100.5, low=99.5,
              close=100.0, volume=100.0)
        for i in range(36)
    ]

    runner = ScriptRunner(Path(script_path), iter(bars), _make_syminfo())
    trades = []
    for _candle, _plot, new_closed in runner.run_iter():
        trades.extend(new_closed)

    def sheds(entry_id: str) -> list[tuple[str, float, int]]:
        return sorted((t.exit_comment, abs(t.size), t.exit_bar_index)
                      for t in trades if t.entry_id == entry_id)

    assert sheds('F') == [('flat', 7.0, 30), ('i3b', 3.0, 27)], sheds('F')
    # The replacing plain close is sized against the replaced immediate one.
    assert sheds('G') == [('flat', 3.0, 35), ('full', 7.0, 33)], sheds('G')
