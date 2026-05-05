"""
@pyne

Regression test for ``strategy.risk.max_cons_loss_days`` enforcement.

Until the rule was wired into ``SimPosition._enforce_post_bar_risk``, the
setter stored the limit but neither incremented the consecutive-loss-day
counter nor halted on breach.

The script below uses a **fixed-quantity short** so price-up days produce
a loss while the strategy is exposed; we open the short on bar 0, hold
through three rising daily bars, then expect the halt on the third
consecutive losing day.
"""
from pynecore.lib import bar_index, script, strategy


@script.strategy(
    "Max Cons Loss Days Halt",
    overlay=True,
    initial_capital=10000,
    default_qty_type=strategy.fixed,
    default_qty_value=10,
    pyramiding=0,
)
def main():
    strategy.risk.max_cons_loss_days(3)
    if bar_index == 0:
        strategy.entry('Short', strategy.short)


def _make_syminfo(period: str = '1D'):
    from pynecore.core.syminfo import SymInfo
    from pynecore.providers.ccxt import CCXTProvider
    opening_hours, session_starts, session_ends = CCXTProvider._create_24_7_sessions()
    return SymInfo(
        prefix="TEST", description="Test", ticker="TEST", currency="USD",
        period=period, type="crypto", mintick=0.01, pricescale=100,
        minmove=1, pointvalue=1, timezone="UTC", volumetype="base",
        opening_hours=opening_hours, session_starts=session_starts,
        session_ends=session_ends,
    )


# noinspection PyShadowingNames
def __test_max_cons_loss_days_halts_after_three_losing_days__(script_path, module_key):
    """
    Short opened bar 0; price rises every day → equity drops every day. After
    3 consecutive losing days the halt must fire and close the short.

    Equity comparison happens at the start of each new day, so the halt
    triggers on the bar that *opens* the next day after the third losing day.
    """
    import sys
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner
    from pynecore.types.ohlcv import OHLCV

    sys.modules.pop(module_key, None)

    syminfo = _make_syminfo(period='1D')
    day_seconds = 24 * 3600
    base_ts = 1704067200  # 2024-01-01 UTC

    # Day 0 — entry placed, fills at day-1 open at price 100.
    # Day 1..3 — price rises 100 → 110 → 120 → 130; each day's equity is
    # below the previous day's. After day 3 close we have 3 consecutive
    # losing days; the halt fires the moment day 4 opens and the rollover
    # observes the third loss day. The test stops there — once halted, the
    # strategy stays flat, so a further day rollover would (correctly) reset
    # the counter and obscure the breach assertion.
    bars = [
        OHLCV(timestamp=base_ts + 0 * day_seconds,
              open=100.0, high=100.5, low=99.5, close=100.0, volume=100.0),
        OHLCV(timestamp=base_ts + 1 * day_seconds,
              open=100.0, high=110.0, low=99.0, close=110.0, volume=100.0),
        OHLCV(timestamp=base_ts + 2 * day_seconds,
              open=110.0, high=120.0, low=109.0, close=120.0, volume=100.0),
        OHLCV(timestamp=base_ts + 3 * day_seconds,
              open=120.0, high=130.0, low=119.0, close=130.0, volume=100.0),
        OHLCV(timestamp=base_ts + 4 * day_seconds,
              open=130.0, high=140.0, low=129.0, close=140.0, volume=100.0),
    ]

    runner = ScriptRunner(Path(script_path), iter(bars), syminfo)

    closed_trades: list = []
    for _candle, _plot, new_closed in runner.run_iter():
        closed_trades.extend(new_closed)

    position = runner.script.position

    assert position.risk_cons_loss_days >= 3, (
        f"Cons-loss-day counter should reach 3, got {position.risk_cons_loss_days}"
    )
    assert position.risk_halt_trading is True, (
        "max_cons_loss_days breached but risk_halt_trading still False"
    )
    assert position.size == 0.0, f"Halt should close the short; size={position.size}"
    halt_close = next(
        (t for t in closed_trades if t.exit_id == 'Risk management close'), None,
    )
    assert halt_close is not None, "No halt-close trade emitted"
    assert 'Max consecutive loss days' in (halt_close.exit_comment or ''), (
        f"Halt close comment should mention consecutive loss days, got "
        f"{halt_close.exit_comment!r}"
    )
