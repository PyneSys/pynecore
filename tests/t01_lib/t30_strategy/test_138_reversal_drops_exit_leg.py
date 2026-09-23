"""
@pyne

A ``strategy.exit`` leg does not carry over the reversal that spent its binding.

``strategy.exit`` legs of an entry id survive a ``close_all`` + same-id LONG
re-entry (test 122): the leg is handed to the order still waiting to fill. A
reversal looks alike -- the short binding is spent while a long order of the same
id is pending -- but that pending order is the reversing entry itself, and its
opening half gets no leg from the position it just closed.

MEASURED on the wild script `3 EMA + Stochastic RSI + ATR` (BINANCE:BTCUSDT 30m,
``process_orders_on_close``): short "3ESRA" reversed by long "3ESRA", with the
reversal bar's ``strategy.exit`` bracket reached on the very next bar --
TradingView fills nothing there in 5/5 events; the long is first covered by the
next bar's re-issued call.
"""
from pynecore.lib import script, strategy, bar_index


@script.strategy(
    "Reversal Drops Exit Leg",
    overlay=True,
    initial_capital=1000000,
    default_qty_type=strategy.fixed,
    default_qty_value=1,
    pyramiding=1,
    process_orders_on_close=True,
)
def main():
    if bar_index == 0:
        strategy.entry('R', strategy.short)
    if bar_index == 2:
        strategy.entry('R', strategy.long)
    strategy.exit('X', 'R', limit=100.50 if bar_index >= 2 else 50.0)


def __test_helper_make_syminfo():
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


# noinspection PyShadowingNames
def __test_reversal_does_not_carry_exit_leg__(script_path, module_key):
    """
    The reversed long is first exited by the leg issued on the bar after the flip.

    * bar 0: the short fills at the close (100.00); ``X`` is far out of reach.
    * bar 2: the long entry reverses the short at the close (100.00) and ``X`` is
      re-issued at 100.50.
    * bar 3: the bar reaches 101.00, but the short's leg died with the reversal,
      so the fresh long stays open.
    * bar 4: the leg re-issued on bar 3 fills the long at 100.50.
    """
    import sys
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner
    from pynecore.types.ohlcv import OHLCV

    sys.modules.pop(module_key, None)

    base_ts = 1_704_067_200_000  # 2024-01-01 00:00:00 UTC, in ms
    rows = [
        # open,   high,   low,    close
        (100.00, 100.05, 99.95, 100.00),   # bar 0 - short fills at the close
        (100.00, 100.05, 99.95, 100.00),   # bar 1 - short open
        (100.00, 100.05, 99.95, 100.00),   # bar 2 - reversal at the close, X=100.50
        (100.00, 101.00, 99.90, 100.30),   # bar 3 - X reached, must not fill
        (100.30, 101.00, 100.20, 100.30),  # bar 4 - the re-issued X fills
        (100.30, 100.35, 100.25, 100.30),  # bar 5 - flat
    ]
    bars = [
        OHLCV(timestamp=base_ts + i * 60_000, open=o, high=h, low=l, close=c, volume=100.0)
        for i, (o, h, l, c) in enumerate(rows)
    ]

    runner = ScriptRunner(Path(script_path), iter(bars), __test_helper_make_syminfo())
    trades = []
    for _candle, _plot, new_closed in runner.run_iter():
        trades.extend(new_closed)

    shape = [(t.entry_bar_index, t.exit_bar_index, round(t.exit_price, 2)) for t in trades]
    assert len(trades) == 2, f"expected two closed trades, got {shape}"

    reversed_short, reversing_long = trades

    assert (reversed_short.entry_bar_index, reversed_short.exit_bar_index) == (0, 2), shape
    assert abs(reversed_short.exit_price - 100.00) < 1e-9, shape

    assert reversing_long.entry_bar_index == 2, shape
    assert reversing_long.exit_bar_index == 4, (
        f"the short's leg must not carry over the reversal, got {shape}"
    )
    assert abs(reversing_long.exit_price - 100.50) < 1e-9, shape
