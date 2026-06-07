"""
@pyne

Regression test for ``strategy.risk.max_drawdown`` enforcement.

Until the rule was wired into ``SimPosition._enforce_post_bar_risk``, the
setter merely stored the threshold and the simulator never compared it
against the running drawdown — a strategy that breached the limit kept
trading.

The script enters long on bar 0 at price 100 with $1000 initial capital
and a fixed 10-unit position. When price drops to $80 the open drawdown
is $200 (20% of equity). With ``max_drawdown(100, cash)`` the rule must
fire on that bar: position closes, ``risk_halt_trading`` flips to ``True``,
and the entry attempt on the recovery bar is suppressed.
"""
from pynecore.lib import bar_index, script, strategy


@script.strategy(
    "Max Drawdown Halt",
    overlay=True,
    initial_capital=1000,
    default_qty_type=strategy.fixed,
    default_qty_value=10,
    pyramiding=0,
)
def main():
    strategy.risk.max_drawdown(100, strategy.cash)
    if bar_index == 0:
        strategy.entry('Long', strategy.long)
    # Re-attempt entry after the halt — must be suppressed.
    if bar_index == 4:
        strategy.entry('Long2', strategy.long)


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
def __test_max_drawdown_halts_and_blocks_re_entry__(script_path, module_key):
    """
    A breached ``max_drawdown`` closes the long, sets ``risk_halt_trading``, and blocks re-entry.

    Drawdown of $200 (20% of $1000 equity) on bar 2 must trigger
    ``strategy.risk.max_drawdown(100, cash)``: the open long is closed at the
    bar's close price, ``risk_halt_trading`` becomes True, and the bar 4
    re-entry attempt does NOT open a new position.
    """
    import sys
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner
    from pynecore.types.ohlcv import OHLCV

    sys.modules.pop(module_key, None)

    syminfo = _make_syminfo(period='1')
    base_ts = 1704067200  # 2024-01-01 00:00:00 UTC

    # Bar 0: signal       — entry queued
    # Bar 1: entry fills at open=100; price wobbles around 100
    # Bar 2: price plunges to 80 → unrealized P&L = -$200, drawdown rule fires
    # Bar 3: price recovers, but position already closed, halt set
    # Bar 4: re-entry attempt — must be suppressed
    bars = [
        OHLCV(timestamp=base_ts + 0 * 60, open=100.0, high=100.5, low=99.5, close=100.0, volume=100.0),
        OHLCV(timestamp=base_ts + 1 * 60, open=100.0, high=101.0, low=99.0, close=100.0, volume=100.0),
        OHLCV(timestamp=base_ts + 2 * 60, open=100.0, high=100.5, low=80.0, close=80.0, volume=100.0),
        OHLCV(timestamp=base_ts + 3 * 60, open=80.0, high=85.0, low=80.0, close=85.0, volume=100.0),
        OHLCV(timestamp=base_ts + 4 * 60, open=85.0, high=90.0, low=84.0, close=88.0, volume=100.0),
        OHLCV(timestamp=base_ts + 5 * 60, open=88.0, high=92.0, low=87.0, close=90.0, volume=100.0),
    ]

    runner = ScriptRunner(Path(script_path), iter(bars), syminfo)

    closed_trades: list = []
    for _candle, _plot, new_closed in runner.run_iter():
        closed_trades.extend(new_closed)

    position = runner.script.position

    assert position.risk_halt_trading is True, (
        "Drawdown limit breached but risk_halt_trading still False"
    )
    assert position.size == 0.0, (
        f"Drawdown halt should have closed the position, got size={position.size}"
    )
    assert len(closed_trades) == 1, (
        f"Expected exactly one closed trade from the halt, got {len(closed_trades)}"
    )
    halt_trade = closed_trades[0]
    assert halt_trade.exit_id == 'Risk management close', (
        f"Expected halt close exit_id 'Risk management close', got {halt_trade.exit_id!r}"
    )
    assert 'Max drawdown' in (halt_trade.exit_comment or ''), (
        f"Halt close comment should mention drawdown, got {halt_trade.exit_comment!r}"
    )
