"""
@pyne

A ``calc_on_order_fills`` bar is walked one fill at a time.

TradingView re-runs the body after EVERY fill, so the broker emulator never
walks past a leg that filled something before the body has run again:

- a pass walks only the legs AFTER the path node it stands at, never the leg
  that led there;
- the walk stops at the end of the first leg that fills anything, and the next
  pass stands at that leg's end;
- a pass whose own market orders filled at its node walks only the next leg;
- a pass standing at the nearer extreme walks the second leg from that extreme,
  not from the open;
- the "already fillable" test of a limit or stop compares against the price of
  the node the pass stands at, not the bar open;
- two sticky ``strategy.exit`` legs bound to different adds of one entry id gap
  through together (neither evicts the other from the market book).

Reference: the Pine probe below, run on TradingView on CAPITALCOM:BTCUSD 30m
(2026-09-28). The OHLCV slice starts at a chart bar index divisible by 48, so
the probe's ``bar_index`` phases line up; the first 48 bars are warmup (``high[1]``
is ``na`` on the first one). Every trade TradingView reports inside the window
is compared: entry time, side, price, exit time, exit order and price.

    //@version=6
    strategy("COOF stepwise probe", overlay=true, calc_on_order_fills=true,
         pyramiding=0, default_qty_type=strategy.fixed, default_qty_value=1,
         initial_capital=10000000, margin_long=0, margin_short=0)
    phase = bar_index % 8
    cyc = math.floor(bar_index / 8) % 6
    isLong = cyc < 3
    k = cyc % 3 == 0 ? -0.25 : cyc % 3 == 1 ? 0.3 : 0.6
    rng = high[1] - low[1]
    if phase == 0 or phase == 1
        strategy.order("E", isLong ? strategy.long : strategy.short)
    lvl = isLong ? open + k * rng : open - k * rng
    strategy.exit("X", "E", limit = lvl)
    if phase == 4
        strategy.close_all()
"""
from pynecore.lib import bar_index, high, low, math, open, script, strategy


@script.strategy(
    "COOF stepwise probe",
    overlay=True,
    calc_on_order_fills=True,
    pyramiding=0,
    default_qty_type=strategy.fixed,
    default_qty_value=1,
    initial_capital=10000000,
    margin_long=0,
    margin_short=0,
)
def main():
    phase = bar_index % 8
    cyc = math.floor(bar_index / 8) % 6
    is_long = cyc < 3
    k = -0.25 if cyc % 3 == 0 else 0.3 if cyc % 3 == 1 else 0.6
    rng = high[1] - low[1]
    if phase == 0 or phase == 1:
        strategy.order('E', strategy.long if is_long else strategy.short)
    lvl = open + k * rng if is_long else open - k * rng
    strategy.exit('X', 'E', limit=lvl)
    if phase == 4:
        strategy.close_all()


# noinspection PyShadowingNames
def __test_coof_bar_is_walked_one_fill_at_a_time__(csv_reader, runner, script_path):
    """ Every TradingView trade of the probe window, fill by fill """
    import csv
    import math as pymath
    from datetime import datetime

    syminfo_override = dict(
        prefix="CAPITALCOM",
        ticker="BTCUSD",
        currency="USD",
        period="30",
        type="crypto",
        mintick=0.05,
        pricescale=100,
        minmove=5,
        pointvalue=1,
        mincontract=0.000001,
    )

    def ts(text: str) -> int:
        return int(datetime.fromisoformat(text).timestamp() * 1000)

    data_dir = script_path.parent / "data"
    with (data_dir / "coof_stepwise_trades.csv").open(newline="") as f:
        expected = [(ts(r['entry_time']), r['entry_type'], float(r['entry_price']),
                     ts(r['exit_time']), r['exit_name'], float(r['exit_price']))
                    for r in csv.DictReader(f)]
    # Scored from the 49th bar (the first 48 are warmup) to the last one
    with (data_dir / "coof_stepwise_ohlcv.csv").open(newline="") as f:
        bar_times = [ts(r['time']) for r in csv.DictReader(f)]
    window_from, window_to = bar_times[48], bar_times[-1]

    closed = []
    with csv_reader('coof_stepwise_ohlcv.csv', subdir="data") as cr:
        r = runner(cr, syminfo_override=syminfo_override)
        for _candle, _plot, new_closed in r.run_iter():
            closed.extend(new_closed)

    actual = [(t.entry_time, 'long' if t.size > 0 else 'short', t.entry_price,
               t.exit_time, t.exit_comment or t.exit_id, t.exit_price)
              for t in closed
              if t.entry_time >= window_from and t.exit_time <= window_to]

    assert len(actual) == len(expected), f"{len(actual)} trades, TradingView has {len(expected)}"
    for i, (got, want) in enumerate(zip(actual, expected)):
        same = (got[0] == want[0] and got[1] == want[1] and got[3] == want[3]
                and got[4] == want[4]
                and pymath.isclose(got[2], want[2], abs_tol=1e-6)
                and pymath.isclose(got[5], want[5], abs_tol=1e-6))
        assert same, f"trade {i}: got {got}, TradingView {want}"
