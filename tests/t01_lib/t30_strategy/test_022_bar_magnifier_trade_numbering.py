"""
@pyne

Bar magnifier trade numbering.

The magnifier runs its chart bars in ``ScriptRunner._run_iter_magnified`` while the
end-of-data export of still-open positions stays in ``run_iter``. Both write the
same trades CSV, so they must share one trade counter — otherwise the open trade
is numbered 1 again and collides with the first closed trade.

The strategy closes its first long inside the window and enters a second one that
is never closed, so the export has to emit exactly one closed and one open trade.
"""
from pynecore.lib import script, strategy, bar_index


@script.strategy(
    "Magnifier Trade Numbering",
    overlay=True,
    initial_capital=100000,
    default_qty_type=strategy.fixed,
    default_qty_value=1,
    use_bar_magnifier=True,
)
def main():
    if bar_index == 1:
        strategy.entry('A', strategy.long)
    if bar_index == 3:
        strategy.close('A')
    if bar_index == 5:
        strategy.entry('B', strategy.long)


def _make_syminfo(period: str = '5'):
    """Minimal 24/7 crypto SymInfo for the magnified run."""
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
def __test_magnifier_open_trade_continues_closed_numbering__(script_path, module_key, tmp_path):
    """The open trade must follow the closed ones, not restart the numbering."""
    import csv
    import sys
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner
    from pynecore.types.ohlcv import OHLCV

    sys.modules.pop(module_key, None)

    base_ts = 1_704_067_200_000  # 2024-01-01 00:00:00 UTC (ms)
    sub_bars = [
        OHLCV(timestamp=base_ts + i * 60_000, open=100.0, high=101.0, low=99.0,
              close=100.0, volume=1.0)
        for i in range(7 * 5)  # 7 chart bars of 5 one-minute sub-bars
    ]

    trade_path = tmp_path / "trades.csv"
    runner = ScriptRunner(
        Path(script_path), iter([]), _make_syminfo(),
        trade_path=trade_path,
        magnifier_iter=iter(sub_bars),
    )
    for _ in runner.run_iter():
        pass
    del runner  # the open-trade export runs in run_iter's finally

    with trade_path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    closed = [r for r in rows if r["Signal"] != "Open" and "Exit" in r["Type"]]
    opened = [r for r in rows if r["Signal"] == "Open"]
    assert len(closed) == 1, f"Expected 1 closed trade, got {len(closed)}"
    assert len(opened) == 1, f"Expected 1 open trade, got {len(opened)}"
    assert closed[0]["Trade #"] == "1"
    assert opened[0]["Trade #"] == "2", (
        f"Open trade restarted the numbering: {opened[0]['Trade #']}")

    numbers = [r["Trade #"] for r in rows]
    assert len(set(numbers)) == 2, f"Trade numbers collide: {numbers}"
