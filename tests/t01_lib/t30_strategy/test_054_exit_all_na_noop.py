"""
@pyne

Regression test for ``strategy.exit`` with ALL triggers na -> must be a no-op.

A global-scope ``strategy.exit`` re-issued every bar with stop/limit derived from
``strategy.position_avg_price`` resolves to ``stop=na, limit=na`` on every bar
BEFORE the entry fills (position_avg_price is na while flat). TradingView treats
such a trigger-less call as a no-op; the position opens and is protected only once
the brackets become real.

Before the fix PyneCore created a LEVEL-LESS exit order for the all-na call and
filled it as a market close on the entry bar -- every trade became a degenerate
same-bar round trip at the entry price (profit 0). The fix returns early from
``exit()`` when no trigger (limit/stop/profit/loss/trail) is set, so the entry is
held until a real bracket level is hit.
"""
from pynecore.lib import bar_index, script, strategy


# noinspection PyTypeChecker
@script.strategy(
    "Exit All-NA No-Op",
    overlay=True,
    initial_capital=100000,
    default_qty_type=strategy.fixed,
    default_qty_value=1,
    pyramiding=0,
)
def main():
    if bar_index == 0:
        strategy.entry('E', strategy.long)
    # Brackets from position_avg_price (na while flat -> stop/limit na on every
    # pre-fill bar), re-issued every bar like a global-scope strategy.exit.
    stop_lvl = strategy.position_avg_price - 5.0
    limit_lvl = strategy.position_avg_price + 10.0
    strategy.exit('X', from_entry='E', stop=stop_lvl, limit=limit_lvl)


def _make_syminfo(period: str = '1'):
    from pynecore.core.syminfo import SymInfo
    from pynecore.providers.ccxt import CCXTProvider
    # noinspection PyProtectedMember
    opening_hours, session_starts, session_ends = CCXTProvider._create_24_7_sessions()
    return SymInfo(
        prefix="TEST", description="Test", ticker="TEST", currency="USD",
        period=period, type="crypto", mintick=0.01, pricescale=100,
        minmove=1, pointvalue=1, timezone="UTC", volumetype="base",
        opening_hours=opening_hours, session_starts=session_starts,
        session_ends=session_ends,
    )


# noinspection PyShadowingNames
def __test_all_na_exit_is_noop_not_market_close__(script_path, module_key):
    """
    The all-na exit on the pre-fill bar must NOT close the position same-bar.

    * bar 0: entry signal; position flat -> exit X resolves to stop=na, limit=na.
    * bar 1: E fills @100; the real bracket (stop 95, limit 110) is armed here.
    * bars 2-3: price rises; the limit 110 fills on bar 3.

    Before the fix the bar-0 all-na exit became a level-less market close that
    fired on bar 1 at the entry price 100 (profit 0). After the fix the trade is
    held and closes at the limit 110 on bar 3.
    """
    import sys
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner
    from pynecore.types.ohlcv import OHLCV

    sys.modules.pop(module_key, None)

    syminfo = _make_syminfo(period='1')
    base_ts = 1704067200  # 2024-01-01 00:00:00 UTC

    rows = [
        # open,  high,   low,   close
        (100.0, 100.5, 99.5, 100.0),  # bar 0 - entry signal, exit X all-na
        (100.0, 100.5, 99.5, 100.0),  # bar 1 - E fills @100; bug would exit @100 here
        (100.0, 105.0, 99.0, 104.0),  # bar 2 - no level hit (limit 110, stop 95)
        (108.0, 112.0, 107.0, 111.0),  # bar 3 - high 112 >= limit 110 -> exit @110
        (110.0, 110.5, 109.5, 110.0),  # bar 4 - tail
    ]
    bars = [
        OHLCV(timestamp=base_ts + i * 60, open=o, high=h, low=l, close=c, volume=100.0)
        for i, (o, h, l, c) in enumerate(rows)
    ]

    runner = ScriptRunner(Path(script_path), iter(bars), syminfo)
    trades = []
    for _candle, _plot, new_closed in runner.run_iter():
        trades.extend(new_closed)

    assert len(trades) == 1, f"Expected 1 closed trade, got {len(trades)}"
    t = trades[0]
    assert t.entry_id == 'E', f"entry id {t.entry_id}"
    assert t.entry_bar_index == 1, f"entry bar {t.entry_bar_index}"
    # The trade must close at the real limit (110) on bar 3, NOT be force-closed
    # same-bar at the entry price 100 by the all-na exit.
    assert t.exit_bar_index == 3, (
        f"exit bar {t.exit_bar_index} (the all-na exit must not close on the entry bar 1)"
    )
    assert abs(t.exit_price - 110.0) < 1e-9, (
        f"exit price {t.exit_price} (expected the limit 110.0, not the entry price 100.0)"
    )
