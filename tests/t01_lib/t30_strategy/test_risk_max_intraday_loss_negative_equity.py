"""
@pyne

Regression test for the dormancy half of ``strategy.risk.max_intraday_loss``.

MEASURED on TradingView: the rule is enforced only while the trading day
OPENS with positive equity. A strategy whose equity has gone negative rides
arbitrarily large intraday losses untouched, and the rule re-arms on the
first day that opens back above zero. The script below shorts a position
that outruns its capital, so day one halts on the rule and day two — which
opens at -$500 — must not.
"""
from pynecore.lib import script, strategy


@script.strategy(
    "Max Intraday Loss Dormancy",
    overlay=True,
    initial_capital=1000,
    default_qty_type=strategy.fixed,
    default_qty_value=10,
    pyramiding=0,
    # No margin requirement: a margin call would flatten the short long before
    # the intraday-loss rule gets a chance to be measured.
    margin_long=0,
    margin_short=0,
)
def main():
    strategy.risk.max_intraday_loss(100, strategy.cash)
    if strategy.position_size == 0:
        strategy.entry('Short', strategy.short)


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
def __test_max_intraday_loss_dormant_while_equity_is_negative__(script_path, module_key):
    """
    A day that opens with non-positive equity does not enforce the rule.

    Day one: the short fills at $100 and the bar spikes to $250, a $1500 loss
    on 10 contracts — the rule fires and leaves the account at -$500. Day two
    re-enters and takes the same $1500 hit; with the day's opening equity
    negative the position must survive it.
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

    def bar(ts: int, o: float, h: float, lo: float, c: float) -> OHLCV:
        return OHLCV(timestamp=ts, open=o, high=h, low=lo, close=c, volume=100.0)

    bars = [
        bar(base_ts + 0 * 3_600_000, 100.0, 100.0, 100.0, 100.0),
        bar(base_ts + 1 * 3_600_000, 100.0, 100.0, 100.0, 100.0),   # short fills at 100
        bar(base_ts + 2 * 3_600_000, 100.0, 250.0, 100.0, 250.0),   # -1500 -> rule fires
        bar(base_ts + 3 * 3_600_000, 250.0, 250.0, 250.0, 250.0),   # halted for the rest of day 1
        bar(day2_ts + 0 * 3_600_000, 250.0, 250.0, 250.0, 250.0),   # halt lifts, anchor = -500
        bar(day2_ts + 1 * 3_600_000, 250.0, 250.0, 250.0, 250.0),
        bar(day2_ts + 2 * 3_600_000, 250.0, 250.0, 250.0, 250.0),
        bar(day2_ts + 3 * 3_600_000, 250.0, 400.0, 250.0, 400.0),   # another -1500, no halt
        bar(day2_ts + 4 * 3_600_000, 400.0, 400.0, 400.0, 400.0),
    ]

    runner = ScriptRunner(Path(script_path), iter(bars), syminfo)

    closed_trades: list = []
    for _candle, _plot, new_closed in runner.run_iter():
        closed_trades.extend(new_closed)

    position = runner.script.position

    assert len(closed_trades) == 1, (
        f"Only day one may close on the rule, got {len(closed_trades)} closes"
    )
    assert closed_trades[0].exit_price == 250.0, (
        f"Day-one close should print at the bar high, got {closed_trades[0].exit_price}"
    )
    assert position.risk_halt_trading is False, "Halt survived the trading-day rollover"
    assert position.size == -10.0, (
        f"Day two opened at negative equity, the rule must stay dormant; size={position.size}"
    )
