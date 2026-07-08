"""
@pyne

Guard test: a prior-bar position still reverses at full margin.

The same-bar both-legs margin gate must NOT leak into prior-bar reversals. A
position opened on an earlier bar frees its margin when the reversing entry
closes it, so only the new leg is margined (the net check). A live
TradingView probe on BINANCE:BTCUSDT confirmed a prior-bar 0.9 BTC reversal
fills even though both legs together exceed equity — unlike the same-bar
pair, which is rejected. Here ``percent_of_equity=100`` makes each leg ~100%
of equity: the same-bar case would drop the flip, but the prior-bar case must
still reverse.
"""
from pynecore.lib import bar_index, script, strategy


@script.strategy(
    "Prior-Bar Opposite Entry Reverses",
    overlay=True,
    initial_capital=100000,
    default_qty_type=strategy.percent_of_equity,
    default_qty_value=100,
)
def main():
    if bar_index == 0:
        strategy.entry('L', strategy.long)
    if bar_index == 2:
        strategy.entry('S', strategy.short)
    if bar_index == 4:
        strategy.close_all(comment='flat')


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
def __test_prior_bar_opposite_entry_reverses__(script_path, module_key):
    """
    Long ``L`` opened on bar 0 (filled bar 1); opposite short ``S`` placed on
    bar 2 (filled bar 3) reverses it via the net check — ``L`` closes and the
    position flips short. This must NOT be swallowed by the same-bar gate.
    """
    import sys
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner
    from pynecore.types.ohlcv import OHLCV

    sys.modules.pop(module_key, None)

    base_ts = 1704067200
    rows = [(100.0, 100.0, 100.0, 100.0) for _ in range(6)]
    bars = [
        OHLCV(timestamp=base_ts + i * 60, open=o, high=h, low=l, close=c, volume=100.0)
        for i, (o, h, l, c) in enumerate(rows)
    ]

    runner = ScriptRunner(Path(script_path), iter(bars), _make_syminfo())
    closed_trades = []
    for _candle, _plot, new_closed in runner.run_iter():
        closed_trades.extend(new_closed)

    # Reversal: ``L`` (long) closes at bar-3 open when ``S`` flips short, then
    # ``S`` (short) closes at the final close_all. Two round-trips confirm the
    # prior-bar reversal was NOT suppressed by the same-bar both-legs gate.
    assert len(closed_trades) == 2, [t.entry_id for t in closed_trades]
    assert closed_trades[0].entry_id == 'L'
    assert closed_trades[0].size > 0.0  # long leg
    assert closed_trades[1].entry_id == 'S'
    assert closed_trades[1].size < 0.0  # short leg (reversal opened it)
