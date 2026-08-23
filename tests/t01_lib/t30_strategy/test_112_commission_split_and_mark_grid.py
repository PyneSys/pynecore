"""
@pyne

Regression test: a percent commission is split over the ORDER's own quantity and over
the legs' exact LOT COUNTS, and an open position is marked on the fill tick grid.

Measured on TradingView with the wild "Built-in Kelly ratio for dynamic position
sizing" (currency=USD on BINANCE:BTCUSDT 30m, 28777 bars, 284 closed trades). Three
laws come out of it, each worth a few 1e-10 to 1e-12 on the plotted ledger:

* the split base is the order's whole quantity, not the running sum of the legs
  booked so far -- otherwise the first leg is priced as if it were the whole order
  and the difference is dumped on the last one;
* the split ratio is a ratio of lot counts, not of the materialized contract
  doubles, so a reversal that closes and opens the same size splits its fee on an
  exact half whose tie rounds UP;
* ``strategy.openprofit`` marks at ``_tick_snap(close)`` -- the grid every booked
  price rides -- not at the OHLC quantization ``SimPosition.c`` carries.

With all three, ``grossprofit``, ``grossloss``, ``netprofit``, ``openprofit`` and
``equity`` reproduce the Kelly script's plots on all 28777 bars bit-exact (they were
310, 236, 177, 27399 and 2055 of 28777 before).

The numbers below follow from those laws rather than from the engine. The commission
rate is 96611.33 * 0.075% = 72.4584975 per contract, ``r10`` is the ten-significant-
digit half-up cash rounding, and every quantity is an exact count of 1e-8 lots:

* bar 0 opens 15000 lots on an order of its own: ``r10(72.4584975 * 0.00015)``
  = 0.01086877462.
* bar 1 reverses 15000 against 15000. The order total is
  ``r10(72.4584975 * 0.0003)`` = 0.02173754925, whose exact half 0.010868774625
  rounds up to 0.01086877463 per leg, so the trade bar 0 opened reports
  0.01086877462 + 0.01086877463. Pricing that closing leg alone -- or taking the
  ratio over the contract doubles 0.00015000000000000001 and
  0.00030000000000000003, which falls a hair short of one half -- gives
  0.01086877462 and one grid step less on the trade.
* bar 2 reverses 15000 against 1000, bar 3 reverses that 1000 back against 14000.
  The 1000-lot trade in between pays a sixteenth of bar 2's
  ``r10(72.4584975 * 0.00016)`` and a fifteenth of bar 3's
  ``r10(72.4584975 * 0.00015)``: 0.000724584975 + 0.0007245849747. Priced on its own
  quantity both legs would round to 0.000724584975.
* bar 4 marks 14000 short lots entered at 96611.33 against a 94208.43 close.
  ``floor(94208.43 / 0.01 + 0.5) * 0.01`` is 94208.43000000001, one ULP above the
  ``int(94208.43 / 0.01 + 0.5) * 1 / 100`` the bar itself carries, and the position
  size scales that ULP into the last two digits of the unrealized P&L.
"""
from pynecore.lib import script, strategy, plot, bar_index


@script.strategy(
    "Commission Split And Mark Grid",
    initial_capital=1_000_000,
    commission_type=strategy.commission.percent,
    commission_value=0.075,
    margin_long=0,
    margin_short=0,
    process_orders_on_close=True,
)
def main():
    if bar_index == 0:
        strategy.entry('A', strategy.long, qty=0.00015)
    if bar_index == 1:
        strategy.entry('B', strategy.short, qty=0.00015)
    if bar_index == 2:
        strategy.entry('C', strategy.long, qty=0.00001)
    if bar_index == 3:
        strategy.entry('D', strategy.short, qty=0.00014)

    plot(strategy.closedtrades.commission(0), "comm0")
    plot(strategy.closedtrades.commission(2), "comm2")
    plot(strategy.opentrades.commission(0), "opencomm")
    plot(strategy.openprofit, "openprofit")
    plot(strategy.grossprofit, "gp")
    plot(strategy.grossloss, "gl")
    plot(strategy.netprofit, "np")


def _make_syminfo():
    from pynecore.core.syminfo import SymInfo
    from pynecore.providers.ccxt import CCXTProvider
    # noinspection PyProtectedMember
    opening_hours, session_starts, session_ends = CCXTProvider._create_24_7_sessions()
    return SymInfo(
        prefix="TEST", description="Test", ticker="TEST", currency="USD",
        period='30', type="crypto", mintick=0.01, pricescale=100,
        minmove=1, pointvalue=1, timezone="UTC", volumetype="base",
        mincontract=1e-8,
        opening_hours=opening_hours, session_starts=session_starts,
        session_ends=session_ends,
    )


def _run(script_path, module_key):
    """Run the script over the five bars the laws above are stated on."""
    import sys
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner
    from pynecore.types.ohlcv import OHLCV

    sys.modules.pop(module_key, None)

    base_ts = 1_704_067_200_000  # 2024-01-01 00:00:00 UTC, in ms
    closes = [96611.33, 96611.33, 96611.33, 96611.33, 94208.43]
    bars = [
        OHLCV(timestamp=base_ts + i * 1_800_000, open=c, high=c, low=c, close=c, volume=1.0)
        for i, c in enumerate(closes)
    ]
    runner = ScriptRunner(Path(script_path), iter(bars), _make_syminfo())
    return [dict(p) for _candle, p, _closed in runner.run_iter()]


# noinspection PyShadowingNames
def __test_half_half_reversal_splits_on_the_lot_grid__(script_path, module_key):
    """The 15000-against-15000 tie rounds up on both legs."""
    plots = _run(script_path, module_key)

    # The order of bar 0 is a single leg, so it rounds on its own quantity.
    assert plots[1]["opencomm"] == 0.01086877462, plots[1]["opencomm"]
    # Entry plus the reversal's closing leg, the latter on the rounded-up half.
    assert plots[2]["comm0"] == 0.02173754925, plots[2]["comm0"]


# noinspection PyShadowingNames
def __test_closing_leg_is_priced_over_the_whole_order__(script_path, module_key):
    """A 1000-lot close inside a 15000-lot order pays its fifteenth, not its own rounding."""
    plots = _run(script_path, module_key)

    assert plots[4]["comm2"] == 0.0014491699496999993, plots[4]["comm2"]


# noinspection PyShadowingNames
def __test_open_position_marks_on_the_fill_tick_grid__(script_path, module_key):
    """14000 short lots against a close whose two tick roundings differ by an ULP."""
    plots = _run(script_path, module_key)

    last = plots[4]
    assert last["openprofit"] == 0.3364059999999992, last["openprofit"]
    # The same subtraction off the unsnapped close lands one step away.
    assert last["openprofit"] != -0.00014000000000000001 * (94208.43 - 96611.33)
    # netprofit peels the published grossloss off in one step.
    assert last["np"] == last["gp"] - last["gl"], (last["np"], last["gp"], last["gl"])
