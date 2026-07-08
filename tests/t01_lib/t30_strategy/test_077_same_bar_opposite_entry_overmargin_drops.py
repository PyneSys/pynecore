"""
@pyne

Regression test: same-bar opposite-direction entries — over-margin flip drops.

When two ``strategy.entry`` orders with different IDs fire on the same bar in
opposite directions, TradingView margins BOTH legs of the flip at once (the
closing leg's margin is not freed before the opening leg is gated). At
``percent_of_equity=100`` each leg needs ~100% of equity, so the pair cannot
coexist: TV keeps the first entry and drops the second. Verified against a
live TradingView probe on BINANCE:BTCUSDT (a same-bar 0.9 BTC pair, ~55% of
equity per leg, is rejected). This is the root of the ThinkTech AI Signals
extra-round-trip divergence.
"""
from pynecore.lib import bar_index, script, strategy


@script.strategy(
    "Same-Bar Opposite Entry Over-Margin",
    overlay=True,
    initial_capital=100000,
    default_qty_type=strategy.percent_of_equity,
    default_qty_value=100,
)
def main():
    if bar_index == 0:
        strategy.entry('A', strategy.long)
        strategy.entry('B', strategy.short)
    if bar_index == 3:
        strategy.close_all(comment='flat')


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


# noinspection PyShadowingNames
def __test_same_bar_opposite_over_margin_keeps_first__(script_path, module_key):
    """
    The first-created entry (long ``A``) survives; the opposite second entry
    (short ``B``) cannot be margined alongside it and is dropped, not reversed.
    Exactly one long round-trip results.
    """
    import sys
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner
    from pynecore.types.ohlcv import OHLCV

    sys.modules.pop(module_key, None)

    base_ts = 1704067200
    rows = [(100.0, 100.0, 100.0, 100.0) for _ in range(6)]
    bars = [
        OHLCV(timestamp=base_ts + i * 60, open=o, high=h, low=l, close=c, volume=100.0)
        for i, (o, h, l, c) in enumerate(rows)
    ]

    runner = ScriptRunner(Path(script_path), iter(bars), _make_syminfo())
    closed_trades = []
    for _candle, _plot, new_closed in runner.run_iter():
        closed_trades.extend(new_closed)

    # Over-margin flip rejected: a single long round-trip for ``A``; ``B`` never opens.
    assert len(closed_trades) == 1, [t.entry_id for t in closed_trades]
    trade = closed_trades[0]
    assert trade.entry_id == 'A'
    assert trade.size > 0.0  # long
    assert trade.entry_bar_index == 1  # placed bar 0, filled next open
    assert all(t.entry_id != 'B' for t in closed_trades)
