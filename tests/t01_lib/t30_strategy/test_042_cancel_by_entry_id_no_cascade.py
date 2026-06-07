"""
@pyne

TV-verified: ``strategy.cancel(entry_id)`` after the entry has filled is a no-op.
It MUST NOT cascade-cancel exits that were placed with ``from_entry=<that id>``.

Verified on FX:EURUSD 60min via ``pyne tradingview run`` on 2026-05-04 — both
TP1 and TP2 fired in TV despite a bar-2 ``cancel("Long")`` call after the entry
had already filled at bar 1 open.
"""
from pynecore.lib import bar_index, script, strategy


@script.strategy(
    "Cancel Entry Id No Cascade",
    overlay=True,
    initial_capital=100000,
    default_qty_type=strategy.fixed,
    default_qty_value=2,
    pyramiding=0,
)
def main():
    if bar_index == 0:
        strategy.entry('Long', strategy.long)

    if bar_index == 1:
        strategy.exit('TP1', from_entry='Long', qty=1, limit=110.0)
        strategy.exit('TP2', from_entry='Long', qty=1, limit=120.0)

    # Cancel by ENTRY id after entry has filled — must be no-op for the exits.
    if bar_index == 2:
        strategy.cancel('Long')


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
def __test_cancel_by_entry_id_does_not_cascade_to_exits__(script_path, module_key):
    """
    Both TP1 and TP2 fire because cancel("Long") after the entry filled is a no-op for exits.

    Both TP1 and TP2 must fire — cancel("Long") on bar 2 is a no-op (entry already
    filled, no pending entry to cancel; exits NOT cascade-cancelled per TV).
    """
    import sys
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner
    from pynecore.types.ohlcv import OHLCV

    sys.modules.pop(module_key, None)

    syminfo = _make_syminfo(period='1')
    base_ts = 1704067200

    bars = [
        OHLCV(timestamp=base_ts + 0 * 60, open=100.0, high=100.5, low=99.5,  close=100.0, volume=100.0),
        OHLCV(timestamp=base_ts + 1 * 60, open=100.0, high=100.5, low=99.5,  close=100.0, volume=100.0),
        OHLCV(timestamp=base_ts + 2 * 60, open=100.0, high=100.5, low=99.5,  close=100.0, volume=100.0),
        OHLCV(timestamp=base_ts + 3 * 60, open=100.0, high=115.0, low=100.0, close=115.0, volume=100.0),
        OHLCV(timestamp=base_ts + 4 * 60, open=115.0, high=125.0, low=115.0, close=125.0, volume=100.0),
        OHLCV(timestamp=base_ts + 5 * 60, open=125.0, high=125.5, low=124.5, close=125.0, volume=100.0),
    ]

    runner = ScriptRunner(Path(script_path), iter(bars), syminfo)

    trades = []
    for _candle, _plot, new_closed in runner.run_iter():
        trades.extend(new_closed)

    assert len(trades) == 2, (
        f"Expected 2 closed trades (TP1+TP2 — entry-id cancel must NOT cascade), "
        f"got {len(trades)}."
    )
    exit_ids = sorted(t.exit_id for t in trades)
    assert exit_ids == ['TP1', 'TP2'], f"Expected [TP1, TP2], got {exit_ids}"
