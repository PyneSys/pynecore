"""
@pyne

A pyramid add filling AT THE CLOSE is closed by its own leg on that same close.

With ``process_orders_on_close``, a bar can both open a pyramid add and reach the
level of the ``strategy.exit`` standing on its ``from_entry`` id. When the script
issues that exit BEFORE ordering the add, the call leaves no leg behind for it --
the leg only comes into being when the add fills, which on this bar happens inside
the close pass, after it has picked its candidates. Such a leg is a current-bar
exit order whose level the same close already meets, so the close pass has to
re-scan the exit book once the entries are in.

MEASURED on the wild ``Double Supertrend`` reference (BINANCE:BTCUSDT 240m,
``process_orders_on_close=true``, ``pyramiding=10``): the ``Add Short`` filling at
2022-01-21 00:00 for 38465.65 is closed by ``Exit Add Short`` on that same bar at
the same price, not on the next one.
"""
from pynecore.lib import script, strategy, bar_index


@script.strategy(
    "Close Pass Leg Of Same Bar Add",
    overlay=True,
    initial_capital=1000000,
    default_qty_type=strategy.fixed,
    default_qty_value=1,
    pyramiding=2,
    process_orders_on_close=True,
)
def main():
    if bar_index == 0:
        strategy.entry('E', strategy.long)
    if bar_index == 2:
        # The exit call comes FIRST, so it leaves no leg for an add that is not
        # ordered yet -- the add's leg can only be spawned by its own fill.
        strategy.exit('X', 'E', limit=110.0)
        strategy.entry('E', strategy.long)


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
        (100.00, 100.00, 100.00, 100.00),  # bar 0 - 'E' entry, fills at this close
        (100.00, 100.00, 100.00, 100.00),  # bar 1
        (100.00, 110.00, 100.00, 110.00),  # bar 2 - 'E' add + X(limit=110) at the close
        (110.00, 110.00, 110.00, 110.00),  # bar 3 - where a missed leg would fill
        (110.00, 110.00, 110.00, 110.00),  # bar 4
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
def __test_close_pass_fills_the_leg_of_a_same_bar_add__(script_path, module_key):
    """Both units leave on bar 2 at 110.00 -- the add's leg does not wait a bar."""
    trades = _run(script_path, module_key)
    shape = [(t.entry_bar_index, t.exit_bar_index, round(t.exit_price, 2), t.size)
             for t in trades]

    assert len(trades) == 2, f"expected both units closed, got {shape}"
    assert [t.exit_bar_index for t in trades] == [2, 2], shape
    assert all(abs(t.exit_price - 110.00) < 1e-9 for t in trades), shape
