"""
@pyne

A reversal order the FILL ITSELF produced keeps its chronological order.

Two opposite entries that waited in the book together and fill at one moment are
booked long side first (see test_127). Under ``calc_on_order_fills`` the tie has
a second, opposite shape: the re-execution a short's own fill runs can place the
opposing long entry, which then fills at that very fill point. The long order did
NOT wait alongside the short -- it did not exist yet -- and TradingView books a
SHORT round trip there.

MEASURED on BINANCE:BTCUSDT 30m (probe "coof reversal labeling probe"), both
shapes in one script: the coof-issued reversal reports
``strategy.closedtrades.size(0) = -1`` with entry ``SA`` / exit ``LA``, while a
same-bar pair placed a few bars later reports ``+1`` with entry ``LB`` / exit
``SB``. The wild strategy "SuperTrended Moving Averages Strategy" is the shape in
the wild: ``calc_on_order_fills=true`` plus a signal that flips on the fill bar
gives it 334 such round trips, every one of them a short on TradingView.
"""
from pynecore.lib import bar_index, script, strategy
from pynecore.types import IBPersistent


@script.strategy(
    "Coof Reversal Chronology",
    overlay=True,
    initial_capital=100000,
    default_qty_type=strategy.fixed,
    default_qty_value=1,
    pyramiding=0,
    calc_on_order_fills=True,
)
def main():
    # The short fills at the next bar's open; the re-execution that fill runs is
    # what places the long, so the long is younger than the short's own trade.
    # ``varip`` survives the rollback between the bar's passes, so the long is
    # placed exactly once -- by the pass the short's fill runs.
    armed: IBPersistent[bool] = False

    if bar_index == 10:
        strategy.entry('SA', strategy.short, comment='sa')
    if bar_index == 11 and strategy.position_size < 0 and not armed:
        armed = True
        strategy.entry('LA', strategy.long, comment='la')
    if bar_index == 16:
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


# noinspection PyShadowingNames
def __test_a_fill_issued_reversal_stays_a_short_round_trip__(script_path, module_key):
    """ The long order did not wait alongside the short, so the short is the entry """
    import sys
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner
    from pynecore.types.ohlcv import OHLCV

    sys.modules.pop(module_key, None)

    base_ts = 1_704_067_200_000  # 2024-01-01 00:00:00 UTC, in ms
    bars = [
        OHLCV(timestamp=base_ts + i * 60_000, open=100.0, high=100.5, low=99.5,
              close=100.0, volume=100.0)
        for i in range(20)
    ]

    runner = ScriptRunner(Path(script_path), iter(bars), _make_syminfo())
    trades = []
    for _candle, _plot_data, new_closed in runner.run_iter():
        trades.extend(new_closed)

    assert len(trades) == 2, f"Expected 2 closed trades, got {len(trades)}"

    reversal = trades[0]
    assert reversal.entry_id == 'SA', f"entry_id {reversal.entry_id}"
    assert reversal.exit_id == 'LA', f"exit_id {reversal.exit_id}"
    assert reversal.entry_comment == 'sa', f"entry_comment {reversal.entry_comment}"
    assert reversal.size < 0.0, f"size {reversal.size} (expected a short round trip)"
    assert reversal.entry_bar_index == reversal.exit_bar_index, "the legs must share a bar"
    assert reversal.entry_price == reversal.exit_price, "the legs must share a price"

    assert trades[1].entry_id == 'LA' and trades[1].size > 0.0
