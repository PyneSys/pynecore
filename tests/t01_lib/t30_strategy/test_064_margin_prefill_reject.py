"""
@pyne

Regression test for pre-fill margin rejection of market entries.
"""
from pynecore.lib import bar_index, script, strategy


@script.strategy(
    "Margin Prefill Reject",
    overlay=True,
    initial_capital=100000,
    default_qty_type=strategy.percent_of_equity,
    default_qty_value=100,
    margin_long=100,
    margin_short=100,
)
def main():
    if bar_index == 0:
        strategy.entry('L', strategy.long)


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
def __test_market_entry_rejected_when_fill_open_exceeds_margin__(script_path, module_key):
    """
    A 100% equity market entry sized at the signal close must be rejected when
    the next open makes the resulting position unaffordable.
    """
    import sys
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner
    from pynecore.types.ohlcv import OHLCV

    sys.modules.pop(module_key, None)

    base_ts = 1704067200
    rows = [
        (100.00, 100.00, 100.00, 100.00),
        (100.01, 100.01, 99.50, 100.00),
        (100.00, 100.00, 99.50, 100.00),
    ]
    bars = [
        OHLCV(timestamp=base_ts + i * 60, open=o, high=h, low=l, close=c, volume=100.0)
        for i, (o, h, l, c) in enumerate(rows)
    ]

    runner = ScriptRunner(Path(script_path), iter(bars), _make_syminfo())
    closed_trades = []
    for _candle, _plot, new_closed in runner.run_iter():
        closed_trades.extend(new_closed)

    position = runner.script.position
    assert closed_trades == []
    assert position.size == 0.0
