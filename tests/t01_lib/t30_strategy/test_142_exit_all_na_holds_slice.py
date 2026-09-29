"""
@pyne

Regression test: a ``strategy.exit`` call whose price levels are ALL na leaves a
leg that never fills but still HOLDS its slice of the entry.

TradingView replaces the leg with an order that cannot fill (test 139 locks the
cancel side of that), yet the order keeps its reservation, so a sibling leg
issued later cannot take that part of the entry. Measured on CAPITALCOM:EURUSD
30m (probes xb/xc, ~135 cycles per variant):

- Filled entry: a 10-lot long with legs A (80%) and B (50%) on one limit, A
  issued first every bar. B issued once with ``limit = na`` beforehand makes the
  level fill B 5 + A 5; without it A takes 8 and B the remaining 2.
- Pending entry: the legs issued AFTER the entry call on the entry bar -- all-na,
  as a flat ``position_avg_price`` makes them -- take the entry before the legs
  issued above the call, which find nothing left.

PyneCore used to drop the all-na leg entirely (wild corpus "Kahlman HullMA / WT
Cross Strategy": the short's 10% LONG leg filled where TradingView fills the 50%
SHORT leg, 205/210 trades).
"""
from pynecore.lib import bar_index, na, plot, script, strategy


@script.strategy(
    "Exit All-na Holds Slice",
    overlay=True,
    initial_capital=100000,
    default_qty_type=strategy.fixed,
    default_qty_value=10,
    pyramiding=1,
)
def main():
    avg = strategy.position_avg_price
    tp = avg * 1.01
    tp2 = avg * 1.1

    # Phase 1: filled entry, B issued all-na before A and B get levels
    if bar_index == 0:
        strategy.entry('E1', strategy.long)
    if bar_index == 2:
        strategy.exit('B', limit=na, qty_percent=50, comment='B')
    if 3 <= bar_index <= 6:
        strategy.exit('A', limit=tp, qty_percent=80, comment='A')
        strategy.exit('B', limit=tp, qty_percent=50, comment='B')

    # Phase 2: pending entry, D/D1 issued after the entry call while flat (all-na)
    if bar_index >= 10:
        strategy.exit('C', limit=tp, qty_percent=10, comment='C')
        strategy.exit('C1', limit=tp2, qty_percent=100, comment='C1')
        if bar_index == 10:
            strategy.entry('E2', strategy.long)
        strategy.exit('D', limit=tp, qty_percent=50, comment='D')
        strategy.exit('D1', limit=tp2, qty_percent=100, comment='D1')
    if bar_index == 15:
        strategy.close_all(comment='CA')
    plot(strategy.position_size, 'psize')


def _make_syminfo():
    from pynecore.core.syminfo import SymInfo
    from pynecore.providers.ccxt import CCXTProvider
    # noinspection PyProtectedMember
    opening_hours, session_starts, session_ends = CCXTProvider._create_24_7_sessions()
    return SymInfo(
        prefix="TEST", description="Test", ticker="TEST", currency="USD",
        period='1', type="crypto", mintick=0.01, pricescale=100,
        minmove=1, pointvalue=1, timezone="UTC", volumetype="base",
        mincontract=1.0,
        opening_hours=opening_hours, session_starts=session_starts,
        session_ends=session_ends,
    )


# noinspection PyShadowingNames
def __test_all_na_exit_holds_its_slice__(script_path, module_key):
    """
    The all-na leg keeps its reservation on a filled and on a pending entry.
    """
    import sys
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner
    from pynecore.types.ohlcv import OHLCV

    sys.modules.pop(module_key, None)

    syminfo = _make_syminfo()
    base_ts = 1_704_067_200_000  # 2024-01-01 00:00:00 UTC, in ms

    # Every entry fills at 100.0, so the 1% target is 101.0; bars 5 and 13 reach it.
    bars = [
        OHLCV(timestamp=base_ts + i * 60_000, open=100.0,
              high=101.5 if i in (5, 13) else 100.5, low=99.5, close=100.0, volume=100.0)
        for i in range(18)
    ]

    runner = ScriptRunner(Path(script_path), iter(bars), syminfo)
    trades = []
    for _candle, _plot_data, new_closed in runner.run_iter():
        trades.extend(new_closed)

    fills = [(t.exit_comment, abs(t.size)) for t in trades]
    assert fills == [('A', 5.0), ('B', 5.0), ('D', 5.0), ('CA', 5.0)], fills
