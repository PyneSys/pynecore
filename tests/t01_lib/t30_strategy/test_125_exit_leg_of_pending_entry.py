"""
@pyne

An exit leg issued against a still-PENDING entry order belongs to THAT entry.

``strategy.exit`` fans out one leg per open entry plus one for a pending entry
order under the same ``from_entry``. That last leg has no bound entry yet, and
two things follow from it:

* its tick offsets (``profit``/``loss``/``trail_points``) price off the fill of
  the entry it is waiting for -- never off an OLDER open trade sharing the id.
  ``ticks_resolved`` is one-shot, so a level taken from the older entry freezes a
  bar early and the bracket sits at the wrong distance for good.
* if that entry order never opens (cancelled, or margin-rejected), the leg is an
  orphan: TradingView does not let it reach the position the older entry holds.

MEASURED on the wild `How to use Leverage and Margin in PineScript` reference
(BINANCE:BTCUSDT 30m): TV fills `tp_long` at `fill + 100 ticks` of the entry that
just opened, on that entry's own bar (2025-01-27 08:00), and closes a single open
trade in ONE fill where the orphan leg used to split it in two (2025-04-02 08:30).
"""
from pynecore.lib import script, strategy, bar_index


@script.strategy(
    "Exit Leg Of Pending Entry",
    overlay=True,
    initial_capital=1000000,
    default_qty_type=strategy.fixed,
    default_qty_value=1,
    pyramiding=4,
)
def main():
    # A: a pyramid add whose bracket must price off ITS OWN fill.
    if bar_index == 0:
        strategy.entry('A', strategy.long)
    if bar_index == 2:
        strategy.entry('A', strategy.long)
        strategy.exit('XA', 'A', profit=100)

    # B: two open entries plus a third that is cancelled before it can fill:
    # the leg that third one left behind must not take a share of its own.
    if bar_index == 4 or bar_index == 6:
        strategy.entry('B', strategy.long)
    if bar_index == 5:
        strategy.exit('XB', 'B', profit=1000)
    if bar_index == 8:
        strategy.entry('B', strategy.long)
        strategy.exit('XB', 'B', profit=100)
        strategy.cancel('B')
    if bar_index == 11:
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
    rows = [
        # open,  high,   low,    close
        (100.00, 100.00, 100.00, 100.00),  # bar 0  - 'A' entry signal
        (100.00, 100.05, 99.95, 100.00),   # bar 1  - first 'A' fills at 100.00
        (100.00, 100.05, 99.95, 100.00),   # bar 2  - 'A' add + XA(profit=100 ticks)
        (90.00, 95.00, 90.00, 95.00),      # bar 3  - add fills at 90, XA -> 91.00
        (95.00, 95.05, 94.95, 95.00),      # bar 4  - first 'B' entry signal
        (90.00, 90.05, 89.95, 90.00),      # bar 5  - it fills at 90, XB -> 100.00
        (90.00, 90.05, 89.95, 90.00),      # bar 6  - second 'B' entry signal
        (95.00, 95.05, 94.95, 95.00),      # bar 7  - it fills at 95
        (95.00, 95.05, 94.95, 95.00),      # bar 8  - third 'B' + XB, third cancelled
        (90.50, 97.00, 90.00, 96.00),      # bar 9  - 91.00 and 96.00 are both reached
        (96.00, 96.05, 95.95, 96.00),      # bar 10
        (96.00, 96.05, 95.95, 96.00),      # bar 11 - close_all
        (96.00, 96.05, 95.95, 96.00),      # bar 12 - the close fills
    ]
    bars = [
        OHLCV(timestamp=base_ts + i * 60_000, open=o, high=h, low=l, close=c, volume=100.0)
        for i, (o, h, l, c) in enumerate(rows)
    ]

    runner = ScriptRunner(Path(script_path), iter(bars), _make_syminfo())
    trades = []
    for _candle, _plot, new_closed in runner.run_iter():
        trades.extend(new_closed)
    return trades


# noinspection PyShadowingNames
def __test_pending_entry_leg_prices_and_expires_with_its_entry__(script_path, module_key):
    """
    XA fills off the ADD's own fill, and XB takes exactly the two units it owns.

    * bar 3: the second 'A' fills at 90.00, so XA sits at 91.00 and fills on the
      same bar. Priced off the first entry (100.00) it would be 101.00 and the
      bar's 95.00 high could not reach it.
    * bar 9: the two open 'B' entries take their own levels, 91.00 off the 90.00
      fill and 96.00 off the 95.00 one. The leg the cancelled third entry left
      behind is an orphan; it prices off the oldest 'B' trade, so letting it fire
      would take 91.00 twice and eat the unit the second leg is owed.
    """
    trades = _run(script_path, module_key)
    shape = [(t.entry_id, t.entry_bar_index, t.exit_bar_index, round(t.exit_price, 2), t.size)
             for t in trades]

    assert len(trades) == 4, f"expected four closed trades, got {shape}"
    tp_a = trades[0]

    # The tick bracket resolved off the ADD's fill. The display book is FIFO, so
    # the fill is booked against the oldest open entry rather than the add.
    assert (tp_a.entry_bar_index, tp_a.exit_bar_index) == (1, 3), shape
    assert abs(tp_a.exit_price - 91.00) < 1e-9, shape

    # Two units -- not three -- leave on bar 9: one for each OPEN 'B' entry.
    bar9 = sorted(round(t.exit_price, 2) for t in trades if t.exit_bar_index == 9)
    assert bar9 == [91.00, 96.00], shape

    # What the cancelled entry could not close leaves on the bar-11 close_all.
    rest = trades[-1]
    assert (rest.entry_id, rest.exit_bar_index) == ('B', 12), shape
    assert abs(rest.exit_price - 96.00) < 1e-9, shape
