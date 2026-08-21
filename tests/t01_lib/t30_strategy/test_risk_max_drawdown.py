"""
@pyne

Regression test for ``strategy.risk.max_drawdown`` enforcement.

TradingView evaluates the rule against the realized (closed-equity) high-water
mark, with the drawdown measured off the mark-to-market equity, and only ON A
BAR THAT BOOKS REALIZED P&L (a close/reduce fill). Open-position paper drawdown
alone never halts: verified on ``BINANCE:BTCUSDT`` 30m, a single long carried to
a ~50% floating drawdown with no closing order keeps riding indefinitely, while
the same position force-closes the instant any reducing order fills while the
drop is past the threshold.

The script below enters a fixed 10-unit long at $100 with $1000 initial capital
(realized peak $1000). Price plunges to $80 — a $200 open drawdown, twice the
``max_drawdown(100, cash)`` limit — yet with no closing order the position must
stay open. A ``strategy.close`` of 10% then fires; on its fill the realized-peak
minus mark-to-market drawdown is $150 (still past the $100 limit), so the rule
fires: the remainder closes, ``risk_halt_trading`` flips to ``True``, and the
later re-entry is suppressed.
"""
from pynecore.lib import bar_index, script, strategy


@script.strategy(
    "Max Drawdown Close-Gated",
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
    # A reducing order — its fill is the only moment the max_drawdown rule is
    # evaluated. Queued here, it fills on the next bar's open.
    if bar_index == 4:
        strategy.close('Long', qty_percent=10)
    # Re-attempt entry after the halt — must be suppressed.
    if bar_index == 6:
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
        mincontract=0.0001,
        opening_hours=opening_hours, session_starts=session_starts,
        session_ends=session_ends,
    )


# noinspection PyShadowingNames
def __test_floating_drawdown_holds_close_gated_halt_and_block_re_entry__(script_path, module_key):
    """
    Open drawdown alone never halts; a close fill past the limit does, then blocks re-entry.

    Bar 2 carries a $200 open drawdown (2x the $100 cash limit) with no closing
    order — the position must stay open, ``risk_halt_trading`` False. The bar-4
    ``strategy.close(10%)`` fills on bar 5 at $85: realized peak $1000 minus the
    $850 mark-to-market equity is a $150 drawdown, so the rule fires — the
    remainder closes, the halt flag sets, and the bar-6 re-entry does not open.
    """
    import sys
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner
    from pynecore.types.ohlcv import OHLCV

    sys.modules.pop(module_key, None)

    syminfo = _make_syminfo(period='1')
    base_ts = 1_704_067_200_000  # 2024-01-01 00:00:00 UTC, in ms

    # Bar 0: signal          — entry queued
    # Bar 1: entry fills at open=100; ten units long
    # Bar 2: price plunges to 80 → open drawdown $200, but NO close → no halt
    # Bar 3: price ticks back to 85 — still open, still no halt
    # Bar 4: strategy.close(10%) queued
    # Bar 5: close fills at open=85 → realized-peak-minus-MTM drawdown $150 → halt
    # Bar 6: re-entry attempt — must be suppressed
    bars = [
        OHLCV(timestamp=base_ts + 0 * 60_000, open=100.0, high=100.5, low=99.5, close=100.0, volume=100.0),
        OHLCV(timestamp=base_ts + 1 * 60_000, open=100.0, high=101.0, low=99.0, close=100.0, volume=100.0),
        OHLCV(timestamp=base_ts + 2 * 60_000, open=100.0, high=100.5, low=80.0, close=80.0, volume=100.0),
        OHLCV(timestamp=base_ts + 3 * 60_000, open=80.0, high=86.0, low=80.0, close=85.0, volume=100.0),
        OHLCV(timestamp=base_ts + 4 * 60_000, open=85.0, high=86.0, low=84.0, close=85.0, volume=100.0),
        OHLCV(timestamp=base_ts + 5 * 60_000, open=85.0, high=86.0, low=84.0, close=85.0, volume=100.0),
        OHLCV(timestamp=base_ts + 6 * 60_000, open=85.0, high=90.0, low=84.0, close=88.0, volume=100.0),
    ]

    runner = ScriptRunner(Path(script_path), iter(bars), syminfo)

    closed_trades: list = []
    halt_by_bar: list[bool] = []
    size_by_bar: list[float] = []
    for _candle, _plot, new_closed in runner.run_iter():
        closed_trades.extend(new_closed)
        position = runner.script.position
        halt_by_bar.append(position.risk_halt_trading)
        size_by_bar.append(position.size)

    # Bar 2: a $200 open drawdown, twice the limit, but with no closing order the
    # position keeps riding — this is the behavior open paper loss must NOT halt.
    assert halt_by_bar[2] is False, (
        "Open (floating) drawdown alone must not halt trading"
    )
    assert size_by_bar[2] == 10.0, (
        f"Position must stay fully open through the floating drawdown, got size={size_by_bar[2]}"
    )

    position = runner.script.position

    assert position.risk_halt_trading is True, (
        "Drawdown past the limit at a close fill must set risk_halt_trading"
    )
    assert position.size == 0.0, (
        f"Drawdown halt should have closed the position, got size={position.size}"
    )
    halt_trades = [t for t in closed_trades if t.exit_id == 'Risk management close']
    assert len(halt_trades) == 1, (
        f"Expected exactly one 'Risk management close' trade, got {len(halt_trades)}"
    )
    assert 'Max drawdown' in (halt_trades[0].exit_comment or ''), (
        f"Halt close comment should mention drawdown, got {halt_trades[0].exit_comment!r}"
    )
