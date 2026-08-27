"""
@pyne

Regression tests for ``strategy.risk.max_intraday_loss`` enforcement.

The script below opens a long and takes the position to a -$120 unrealized
loss within the same trading day with ``max_intraday_loss(100, cash)``.
Three measured properties of the rule are covered here:

* it fires INTRA-BAR at the position's unfavorable extreme (the bar's low
  for a long), so the forced close prints there rather than at the close;
* the halt it raises expires with the trading day — a permanent one would
  strand the strategy for the rest of the run;
* the comment identifies which ``strategy.risk.*`` rule fired.

The third measured property — the rule stays dormant while the day OPENS
with non-positive equity — needs a position that can outrun its capital
and lives in ``test_risk_max_intraday_loss_negative_equity``.
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
        mincontract=0.0001,
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

    # Two keys: pytest imports the file under its dotted package path, while
    # ``import_script`` imports the bare stem — only dropping both hands the next
    # test in this file a fresh ``@script.strategy`` object (and with it a fresh
    # position, ledger and bar counter).
    sys.modules.pop(module_key, None)
    sys.modules.pop(Path(script_path).stem, None)

    syminfo = _make_syminfo(period='60')
    base_ts = 1_704_067_200_000  # 2024-01-01 00:00:00 UTC (ms) — start of day, hourly bars

    # Day 1, hourly bars. Position opens at bar 1, breaches loss limit on bar 3.
    bars = [
        OHLCV(timestamp=base_ts + 0 * 3_600_000, open=100.0, high=100.0, low=100.0, close=100.0, volume=100.0),
        OHLCV(timestamp=base_ts + 1 * 3_600_000, open=100.0, high=100.5, low=99.5, close=100.0, volume=100.0),
        OHLCV(timestamp=base_ts + 2 * 3_600_000, open=100.0, high=100.0, low=95.0, close=95.0, volume=100.0),
        OHLCV(timestamp=base_ts + 3 * 3_600_000, open=95.0, high=95.0, low=88.0, close=88.0, volume=100.0),
        OHLCV(timestamp=base_ts + 4 * 3_600_000, open=88.0, high=92.0, low=88.0, close=92.0, volume=100.0),
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


# noinspection PyShadowingNames
def __test_max_intraday_loss_closes_at_the_bar_extreme__(script_path, module_key):
    """
    The forced close prints at the bar's unfavorable extreme, not at its close.

    MEASURED on TradingView: the rule is an intra-bar check, so a long is
    flattened at the LOW the emulator walks to, even when the bar recovers
    before closing. Bar 3 dips to $88 (a $120 loss on 10 contracts, past the
    $100 limit) and closes back at $93 — a bar-end check would print $93.
    """
    import sys
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner
    from pynecore.types.ohlcv import OHLCV

    # Two keys: pytest imports the file under its dotted package path, while
    # ``import_script`` imports the bare stem — only dropping both hands the next
    # test in this file a fresh ``@script.strategy`` object (and with it a fresh
    # position, ledger and bar counter).
    sys.modules.pop(module_key, None)
    sys.modules.pop(Path(script_path).stem, None)

    syminfo = _make_syminfo(period='60')
    base_ts = 1_704_067_200_000  # 2024-01-01 00:00:00 UTC (ms) — start of day, hourly bars

    bars = [
        OHLCV(timestamp=base_ts + 0 * 3_600_000, open=100.0, high=100.0, low=100.0, close=100.0, volume=100.0),
        OHLCV(timestamp=base_ts + 1 * 3_600_000, open=100.0, high=100.5, low=99.5, close=100.0, volume=100.0),
        OHLCV(timestamp=base_ts + 2 * 3_600_000, open=100.0, high=100.0, low=95.0, close=95.0, volume=100.0),
        # Dips past the limit intra-bar, then recovers before the close.
        OHLCV(timestamp=base_ts + 3 * 3_600_000, open=95.0, high=95.0, low=88.0, close=93.0, volume=100.0),
    ]

    runner = ScriptRunner(Path(script_path), iter(bars), syminfo)

    closed_trades: list = []
    for _candle, _plot, new_closed in runner.run_iter():
        closed_trades.extend(new_closed)

    assert len(closed_trades) == 1, f"Expected one halt-close, got {len(closed_trades)}"
    assert closed_trades[0].exit_price == 88.0, (
        f"Halt close should print at the bar low, got {closed_trades[0].exit_price}"
    )


# noinspection PyShadowingNames
def __test_max_intraday_loss_halt_expires_with_the_day__(script_path, module_key):
    """
    ``max_intraday_loss`` blocks the rest of the DAY, not the rest of the run.

    TradingView re-arms the rule at the next trading day (measured: one wild
    strategy fires it on 98 separate days of a single run), so the halt flag
    must be cleared once the day rolls over.
    """
    import sys
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner
    from pynecore.types.ohlcv import OHLCV

    # Two keys: pytest imports the file under its dotted package path, while
    # ``import_script`` imports the bare stem — only dropping both hands the next
    # test in this file a fresh ``@script.strategy`` object (and with it a fresh
    # position, ledger and bar counter).
    sys.modules.pop(module_key, None)
    sys.modules.pop(Path(script_path).stem, None)

    syminfo = _make_syminfo(period='60')
    base_ts = 1_704_067_200_000  # 2024-01-01 00:00:00 UTC (ms) — start of day, hourly bars
    day2_ts = base_ts + 24 * 3_600_000

    bars = [
        OHLCV(timestamp=base_ts + 0 * 3_600_000, open=100.0, high=100.0, low=100.0, close=100.0, volume=100.0),
        OHLCV(timestamp=base_ts + 1 * 3_600_000, open=100.0, high=100.5, low=99.5, close=100.0, volume=100.0),
        OHLCV(timestamp=base_ts + 2 * 3_600_000, open=100.0, high=100.0, low=88.0, close=88.0, volume=100.0),
        # Next trading day — the halt must be gone by the time this bar is done.
        OHLCV(timestamp=day2_ts, open=88.0, high=89.0, low=88.0, close=89.0, volume=100.0),
    ]

    runner = ScriptRunner(Path(script_path), iter(bars), syminfo)

    halt_flags: list[bool] = []
    for _candle, _plot, _new_closed in runner.run_iter():
        halt_flags.append(runner.script.position.risk_halt_trading)

    position = runner.script.position
    assert halt_flags[2] is True, "Intraday loss limit not enforced on the breaching bar"
    assert position.risk_halt_trading is False, "Halt survived the trading-day rollover"
    assert position.risk_halt_day == -1, "Day-scoped halt marker not cleared"
