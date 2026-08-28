"""
@pyne

The pyramiding limit is judged when ``strategy.entry`` RUNS, not when its market
order is processed.

An entry placed while the limit is already spent is dropped for good — flattening
the position afterwards does not revive it. MEASURED on BINANCE:BTCUSDT 240 with
``pyramiding=1`` (probes "PYR probe 1/2"): a second entry placed on the bar the
open position is closed never fills, whether that close is a same-body
``strategy.close(immediately=true)`` or an ordinary close order filling at the
very open the entry would have used, and regardless of the entry id.

What the gate reads is the position as the market orders ALREADY PLACED on the
bar leave it: a ``close_all`` ahead of the entry hands the slot over (see
test_122), and a pending reversal entry makes the next opposite-direction entry a
reversal of its own rather than a pyramid add.
"""
from pynecore.lib import script, strategy, bar_index


@script.strategy(
    "Pyramiding Judged At Call",
    overlay=True,
    initial_capital=1000000,
    default_qty_type=strategy.fixed,
    default_qty_value=1,
    pyramiding=1,
)
def main():
    # A: entry while open, then an immediate close in the same body.
    if bar_index == 0:
        strategy.entry('A', strategy.long)
    if bar_index == 1:
        strategy.entry('A', strategy.long)
        strategy.close('A', immediately=True)

    # B: entry while open, then an ordinary close filling at the same next open.
    if bar_index == 4:
        strategy.entry('B', strategy.long)
    if bar_index == 5:
        strategy.entry('B', strategy.long)
        strategy.close('B')

    # C: a long and a short entry on one bar — the pending long reversal makes
    # the short a reversal too, so it must survive.
    if bar_index == 8:
        strategy.entry('C', strategy.short)
    if bar_index == 10:
        strategy.entry('CL', strategy.long)
        strategy.entry('CS', strategy.short)
    if bar_index == 13:
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
        mincontract=0.0001,
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
        OHLCV(timestamp=base_ts + i * 60_000,
              open=100.0, high=100.05, low=99.95, close=100.0, volume=100.0)
        for i in range(16)
    ]

    runner = ScriptRunner(Path(script_path), iter(bars), _make_syminfo())
    trades = []
    for _candle, _plot, new_closed in runner.run_iter():
        trades.extend(new_closed)
    return trades


# noinspection PyShadowingNames
def __test_spent_pyramiding_slot_drops_the_entry__(script_path, module_key):
    """
    Only the reversal-fed entries survive; the two same-direction adds are gone.

    * bars 0-1: ``A`` fills at bar 1's open, the second ``A`` is rejected at the
      call and the immediate close flattens at bar 1's close.
    * bars 4-5: same shape with an ordinary close, which fills at bar 6's open
      alongside the rejected entry — the position must stay flat.
    * bars 8-10: ``C`` short is open, the bar-10 long reverses it and the bar-10
      short reverses that back, so two more trades close on bar 11.
    """
    trades = _run(script_path, module_key)
    shape = [(t.entry_id, t.entry_bar_index, t.exit_bar_index, t.sign) for t in trades]

    assert len(trades) == 5, f"expected five closed trades, got {shape}"

    a, b, c, cl, cs = trades

    assert (a.entry_id, a.entry_bar_index, a.exit_bar_index) == ('A', 1, 1), shape
    assert (b.entry_id, b.entry_bar_index, b.exit_bar_index) == ('B', 5, 6), shape

    # The bar-10 pair reverses the short, then reverses straight back.
    assert (c.entry_id, c.entry_bar_index, c.exit_bar_index) == ('C', 9, 11), shape
    assert c.sign < 0, shape
    assert (cl.entry_id, cl.entry_bar_index, cl.exit_bar_index) == ('CL', 11, 11), shape
    assert cl.sign > 0, shape
    assert (cs.entry_id, cs.entry_bar_index, cs.exit_bar_index) == ('CS', 11, 14), shape
    assert cs.sign < 0, shape
