"""
@pyne

Regression test for ``strategy.opentrades.size()`` with a negative trade index.

The common "do I have an open position?" guard
``strategy.opentrades.size(strategy.opentrades - 1) <= 0`` evaluates ``size(-1)``
while flat (``opentrades == 0`` -> index ``-1``). PyneCore returned NA for a
negative index, so ``NA <= 0`` was False on every bar and the strategy never
opened a single trade. ``closedtrades.size()`` (and ``opentrades.size()``'s own
out-of-range branch) already return 0.0 for an invalid index; the negative-index
branch now matches, so the guard passes and the first entry fires.
"""
from pynecore.lib import bar_index, script, strategy


# noinspection PyTypeChecker
@script.strategy(
    "Opentrades Size Negative Index",
    overlay=True,
    initial_capital=100000,
    default_qty_type=strategy.fixed,
    default_qty_value=1,
    pyramiding=0,
)
def main():
    # Flat-position guard: the size of the last open trade must be <= 0 when no
    # position is held. With opentrades == 0 the index is -1, i.e. size(-1).
    flat = strategy.opentrades.size(strategy.opentrades - 1) <= 0
    if bar_index == 0 and flat:
        strategy.entry('E', strategy.long)
    if bar_index == 1:
        strategy.close('E')


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
def __test_opentrades_size_negative_index_allows_entry__(script_path, module_key):
    """
    The flat-guard idiom must not block the first entry.

    * bar 0: flat -> ``size(-1) <= 0`` must be True -> entry signal E.
    * bar 1: E fills @100; close signal E.
    * bar 2: E closes @100.

    Before the fix ``size(-1)`` returned NA, the guard ``NA <= 0`` was False, and
    no trade ever opened. After the fix it returns 0.0 (like ``closedtrades.size()``)
    so exactly one trade opens and closes.
    """
    import sys
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner
    from pynecore.types.ohlcv import OHLCV

    sys.modules.pop(module_key, None)

    syminfo = _make_syminfo(period='1')
    base_ts = 1704067200  # 2024-01-01 00:00:00 UTC

    rows = [
        # open,  high,   low,   close
        (100.0, 100.5, 99.5, 100.0),  # bar 0 - flat guard passes -> entry E
        (100.0, 101.0, 99.5, 100.5),  # bar 1 - E fills; close signal E
        (100.5, 101.0, 100.0, 100.5),  # bar 2 - E closes
        (100.5, 101.0, 100.0, 100.5),  # bar 3 - tail
    ]
    bars = [
        OHLCV(timestamp=base_ts + i * 60, open=o, high=h, low=l, close=c, volume=100.0)
        for i, (o, h, l, c) in enumerate(rows)
    ]

    runner = ScriptRunner(Path(script_path), iter(bars), syminfo)
    trades = []
    for _candle, _plot, new_closed in runner.run_iter():
        trades.extend(new_closed)

    assert len(trades) == 1, (
        f"Expected exactly 1 trade (flat guard size(-1) <= 0 must pass), got {len(trades)}"
    )
