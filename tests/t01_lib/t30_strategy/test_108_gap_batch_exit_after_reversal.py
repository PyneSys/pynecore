"""
@pyne

A ``strategy.exit`` stop that gaps through the same bar open as the entry that
reverses the position is NOT cancelled by that reversal.

MEASURED on TradingView (BINANCE:BTCUSDT 30m, 6/6 events per variant):

* Both stops gap through the open -> the entry fills first (whatever the two
  levels are and whichever the script placed first), and the exit then sells its
  own quantity a SECOND time, opening a fresh position under its own exit id.
  Placing the exit stop BELOW the entry stop here is deliberate: an order-book
  price walk reaches the lower level first and would let the exit close the long.
* The exit level first reached INSIDE the bar (18/18) and an exit outlived by a
  MARKET reversal (6/6) are cancelled unfilled instead.
* Closing everything with one ``from_entry``-less ``strategy.exit`` reports the
  exit-opened leg's quantity off the OLDEST trade first: long 1 reversed to
  short 5 with an exit-opened 1 closes as 1, 4, 1 -- not 5, 1.
"""
from pynecore.lib import bar_index, script, strategy


@script.strategy(
    "Gap Batch Exit After Reversal",
    overlay=True,
    initial_capital=100000,
    default_qty_type=strategy.fixed,
    default_qty_value=1,
)
def main():
    if bar_index == 0:
        strategy.entry('Long', strategy.long, qty=1.0)
    if bar_index == 2:
        strategy.entry('Short', strategy.short, stop=99.5, qty=5.0)
    if strategy.position_size > 0 and bar_index >= 2:
        strategy.exit(id='L Stop', stop=99.0)
    if strategy.position_size < 0 and bar_index >= 4:
        strategy.exit(id='S Stop', stop=100.5)


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


def _bars():
    from pynecore.types.ohlcv import OHLCV
    base_ts = 1_704_067_200_000  # 2024-01-01 00:00:00 UTC, in ms
    # 0-2 flat at 100, 3 gaps down through both stops, 4 flat, 5 gaps back up
    # through the S Stop, 6 flat.
    ohlc = [
        (100.0, 100.0, 100.0, 100.0),
        (100.0, 100.0, 100.0, 100.0),
        (100.0, 100.0, 100.0, 100.0),
        (90.0, 90.5, 89.0, 90.0),
        (90.0, 90.5, 89.5, 90.0),
        (101.0, 102.0, 101.0, 101.0),
        (101.0, 101.5, 100.5, 101.0),
    ]
    return [
        OHLCV(timestamp=base_ts + i * 60_000, open=o, high=h, low=lo, close=c, volume=100.0)
        for i, (o, h, lo, c) in enumerate(ohlc)
    ]


# noinspection PyShadowingNames
def __test_gap_batch_exit_opens_a_second_leg__(script_path, module_key):
    """The exit that gapped with the reversal adds exposure instead of dying.

    Buggy code path: ``_drop_binding`` cancels the exit leg while the reversal's
    closing half runs, so the position stops at the reversal's own -5.
    """
    import sys
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner

    sys.modules.pop(module_key, None)

    runner = ScriptRunner(Path(script_path), iter(_bars()), _make_syminfo())
    sizes = []
    for _candle, _plot, _new_closed in runner.run_iter():
        sizes.append(runner.script.position.size)

    assert sizes[3] == -6.0, (
        f"Expected -6 after the gap batch (reversal -5 plus the exit's own 1), "
        f"got {sizes[3]}."
    )
    assert sizes[5] == 0.0


# noinspection PyShadowingNames
def __test_exit_opened_leg_closes_off_the_oldest_trade__(script_path, module_key):
    """The exit-opened position's leg is served FIRST, splitting the older trade."""
    import sys
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner

    sys.modules.pop(module_key, None)

    runner = ScriptRunner(Path(script_path), iter(_bars()), _make_syminfo())
    # Only what THIS run closed: the position's own deque outlives the runner.
    rows = []
    for _candle, _plot, new_closed in runner.run_iter():
        rows.extend((t.entry_id, t.exit_id, abs(t.size)) for t in new_closed)
    assert rows == [
        ('Long', 'Short', 1.0),
        ('Short', 'S Stop', 1.0),
        ('Short', 'S Stop', 4.0),
        ('L Stop', 'S Stop', 1.0),
    ], rows
