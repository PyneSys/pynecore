"""
@pyne

Two opposite entries filling at the SAME moment book the long side first.

When a reversal's closing leg and the position it annihilates land on the very
same bar, at the same point of the intrabar walk and at the same price, there is
no chronology to order them by. TradingView always makes the LONG order the
record's entry and the short one its exit, whichever of the two was placed
first.

MEASURED on BINANCE:BTCUSDT 30m across four probe shapes: ``strategy.entry``
placed short-then-long, long-then-short, three orders on one bar and unequal
sizes all report the long leg as the entry, and ``strategy.order`` behaves the
same. It is the fill MOMENT that decides, not the price -- a reversal whose legs
land on different points of the walk keeps its chronological order even when both
legs price identically (a buy-limit sitting exactly on the short's entry price
stays a short round trip), which is why the walk node is part of the test below.

Prices are equal here by construction, so only the labels and the direction move:
the profit is invariant.
"""
from pynecore.lib import bar_index, script, strategy


@script.strategy(
    "Same-instant Reversal",
    overlay=True,
    initial_capital=100000,
    default_qty_type=strategy.fixed,
    default_qty_value=1,
    pyramiding=0,
)
def main():
    # Short placed first: its trade is the one the long's closing leg consumes.
    if bar_index == 0:
        strategy.entry('S1', strategy.short, comment='s1')
        strategy.entry('L1', strategy.long, comment='l1')
    if bar_index == 3:
        strategy.close_all()

    # Long placed first: the record already comes out long, with nothing to swap.
    if bar_index == 5:
        strategy.entry('L2', strategy.long, comment='l2')
        strategy.entry('S2', strategy.short, comment='s2')
    if bar_index == 8:
        strategy.close_all()

    # A limit buy priced exactly on the short's entry: same price, LATER moment.
    if bar_index == 9:
        strategy.entry('S3', strategy.short, comment='s3')
    if bar_index == 10:
        strategy.entry('L3', strategy.long, limit=100.0, comment='l3')
    if bar_index == 12:
        strategy.close_all()


def _make_syminfo():
    from pynecore.core.syminfo import SymInfo
    from pynecore.providers.ccxt import CCXTProvider
    # noinspection PyProtectedMember
    opening_hours, session_starts, session_ends = CCXTProvider._create_24_7_sessions()
    return SymInfo(
        prefix="TEST", description="Test", ticker="TEST", currency="USD",
        period='1', type="crypto", mintick=0.01, pricescale=100,
        minmove=1, pointvalue=1, timezone="UTC", volumetype="base",
        mincontract=0.00001,
        opening_hours=opening_hours, session_starts=session_starts,
        session_ends=session_ends,
    )


def _run(script_path, module_key):
    import sys
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner
    from pynecore.types.ohlcv import OHLCV

    sys.modules.pop(module_key, None)

    base_ts = 1_704_067_200_000  # 2024-01-01 00:00:00 UTC, in ms
    bars = [
        OHLCV(timestamp=base_ts + i * 60_000, open=100.0, high=100.5, low=99.5,
              close=100.0, volume=100.0)
        for i in range(15)
    ]

    runner = ScriptRunner(Path(script_path), iter(bars), _make_syminfo())
    trades = []
    for _candle, _plot_data, new_closed in runner.run_iter():
        trades.extend(new_closed)
    return trades


# noinspection PyShadowingNames
def __test_short_first_reversal_is_reported_as_a_long_round_trip__(script_path, module_key):
    """ The short was placed first, yet the long order is the record's entry """
    trades = _run(script_path, module_key)
    assert len(trades) == 6, f"Expected 6 closed trades, got {len(trades)}"

    reversal = trades[0]
    assert reversal.entry_id == 'L1', f"entry_id {reversal.entry_id}"
    assert reversal.exit_id == 'S1', f"exit_id {reversal.exit_id}"
    assert reversal.entry_comment == 'l1', f"entry_comment {reversal.entry_comment}"
    assert reversal.exit_comment == 's1', f"exit_comment {reversal.exit_comment}"
    assert reversal.size > 0.0, f"size {reversal.size} (expected a long round trip)"
    assert reversal.profit == 0.0, f"profit {reversal.profit}"


# noinspection PyShadowingNames
def __test_long_first_reversal_reports_the_same_way__(script_path, module_key):
    """ Placement order does not move the labels: the long leg is the entry either way """
    trades = _run(script_path, module_key)

    reversal = trades[2]
    assert reversal.entry_id == 'L2', f"entry_id {reversal.entry_id}"
    assert reversal.exit_id == 'S2', f"exit_id {reversal.exit_id}"
    assert reversal.entry_comment == 'l2', f"entry_comment {reversal.entry_comment}"
    assert reversal.exit_comment == 's2', f"exit_comment {reversal.exit_comment}"
    assert reversal.size > 0.0, f"size {reversal.size} (expected a long round trip)"


# noinspection PyShadowingNames
def __test_the_positions_the_reversals_leave_behind_close_normally__(script_path, module_key):
    """ Control: the swap only relabels, the surviving leg is untouched """
    trades = _run(script_path, module_key)

    assert trades[1].entry_id == 'L1' and trades[1].exit_id == 'Close position order'
    assert trades[1].size > 0.0, f"size {trades[1].size}"
    assert trades[3].entry_id == 'S2' and trades[3].exit_id == 'Close position order'
    assert trades[3].size < 0.0, f"size {trades[3].size}"


# noinspection PyShadowingNames
def __test_a_later_fill_at_the_same_price_keeps_its_chronology__(script_path, module_key):
    """ The moment decides, not the price: a limit filling a bar later stays a short """
    trades = _run(script_path, module_key)

    reversal = trades[4]
    assert reversal.entry_id == 'S3', f"entry_id {reversal.entry_id}"
    assert reversal.exit_id == 'L3', f"exit_id {reversal.exit_id}"
    assert reversal.entry_price == reversal.exit_price, "the probe needs both legs at one price"
    assert reversal.size < 0.0, f"size {reversal.size} (expected a short round trip)"
