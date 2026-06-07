"""
@pyne

Regression test for ``strategy.risk.max_intraday_loss`` enforcement.

Until the rule was wired into ``SimPosition._enforce_post_bar_risk``, the
setter merely stored the threshold and the simulator never compared it
against the equity drop since the start of the trading day. The script
below opens a long, takes the position to a -$120 unrealized loss within
the same day with ``max_intraday_loss(100, cash)``, and verifies the rule
fires: position closes, ``risk_halt_trading`` is set, the comment
identifies the rule.
"""
from pynecore.lib import bar_index, script, strategy


@script.strategy(
    "Max Intraday Loss Halt",
    overlay=True,
    initial_capital=1000,
    default_qty_type=strategy.fixed,
    default_qty_value=10,
    pyramiding=0,
)
def main():
    strategy.risk.max_intraday_loss(100, strategy.cash)
    if bar_index == 0:
        strategy.entry('Long', strategy.long)


def _make_syminfo(period: str = '60'):
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
def __test_max_intraday_loss_halts_within_day__(script_path, module_key):
    """
    Intraday loss exceeding the limit within the same trading day fires the halt on that bar.

    Long entry filled at $100, price drops to $88 within the same trading day:
    unrealized intraday loss = $120 > limit $100 → halt must fire on that bar.
    """
    import sys
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner
    from pynecore.types.ohlcv import OHLCV

    sys.modules.pop(module_key, None)

    syminfo = _make_syminfo(period='60')
    base_ts = 1704067200  # 2024-01-01 00:00:00 UTC — start of day, hourly bars

    # Day 1, hourly bars. Position opens at bar 1, breaches loss limit on bar 3.
    bars = [
        OHLCV(timestamp=base_ts + 0 * 3600, open=100.0, high=100.0, low=100.0, close=100.0, volume=100.0),
        OHLCV(timestamp=base_ts + 1 * 3600, open=100.0, high=100.5, low=99.5, close=100.0, volume=100.0),
        OHLCV(timestamp=base_ts + 2 * 3600, open=100.0, high=100.0, low=95.0, close=95.0, volume=100.0),
        OHLCV(timestamp=base_ts + 3 * 3600, open=95.0, high=95.0, low=88.0, close=88.0, volume=100.0),
        OHLCV(timestamp=base_ts + 4 * 3600, open=88.0, high=92.0, low=88.0, close=92.0, volume=100.0),
    ]

    runner = ScriptRunner(Path(script_path), iter(bars), syminfo)

    closed_trades: list = []
    for _candle, _plot, new_closed in runner.run_iter():
        closed_trades.extend(new_closed)

    position = runner.script.position

    assert position.risk_halt_trading is True, "Intraday loss limit not enforced"
    assert position.size == 0.0, f"Position not closed by halt; size={position.size}"
    assert len(closed_trades) == 1, f"Expected one halt-close, got {len(closed_trades)}"
    assert 'Max intraday loss' in (closed_trades[0].exit_comment or ''), (
        f"Halt close comment should mention intraday loss, got "
        f"{closed_trades[0].exit_comment!r}"
    )
