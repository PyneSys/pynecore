"""
@pyne

Issue #51 — strategy.commission.cash_per_contract must charge BOTH legs of
a round-trip, not just the entry leg.

Pine v6 semantics (TradingView strategy tester, verified against the same
script on CAPITALCOM:EURUSD 60m): a closed trade with qty=1 and
commission_value=1.70 is charged 1.70 on entry AND 1.70 on exit, for a
total of 3.40 per round-trip. PyneCore's strategy fill path currently
charges only the entry leg — the exit-fill branch in
``strategy/__init__.py`` defers cash_per_contract into ``closed_trade_size``
(lines 703-705) but the ``delete``-block deferred-realization (lines
791-799) only handles ``cash_per_order``, so cash_per_contract is never
realized on the exit side.

Test data: real TradingView OHLCV slice exported via
``pyne data download tradingview -s CAPITALCOM:EURUSD -tf 60 -tr`` and
copied into ``data/`` next to this file. Trade count is ~300+, so every
divergent commission is reflected in the assertion failure listing.
"""
from pynecore.lib import close, script, strategy, ta


@script.strategy(
    "cash_per_contract_repro",
    overlay=True,
    initial_capital=10000,
    default_qty_type=strategy.fixed,
    default_qty_value=1,
    pyramiding=0,
    commission_type=strategy.commission.cash_per_contract,
    commission_value=1.70,
    slippage=0,
)
def main():
    fast = ta.sma(close, 10)
    slow = ta.sma(close, 30)

    if ta.crossover(fast, slow):
        strategy.entry("L", strategy.long)
    if ta.crossunder(fast, slow):
        strategy.entry("S", strategy.short)


# noinspection PyShadowingNames
def __test_cash_per_contract_charges_both_legs__(script_path, module_key):
    """Every closed trade must reflect commission = 2 * qty * commission_value."""
    import math
    import sys
    from pathlib import Path

    from pynecore.core.ohlcv_file import OHLCVReader
    from pynecore.core.script_runner import ScriptRunner
    from pynecore.core.syminfo import SymInfo

    sys.modules.pop(module_key, None)

    data_dir = Path(script_path).parent / "data"
    ohlcv_path = data_dir / "cash_per_contract_repro.ohlcv"
    toml_path = data_dir / "cash_per_contract_repro.toml"

    syminfo = SymInfo.load_toml(toml_path)

    expected_commission = 2 * 1.70  # entry leg + exit leg, qty=1

    with OHLCVReader(ohlcv_path) as reader:
        ohlcv_iter = reader.read_from(reader.start_timestamp, reader.end_timestamp)
        runner = ScriptRunner(Path(script_path), ohlcv_iter, syminfo)

        closed = []
        for _candle, _plot, new_closed in runner.run_iter():
            closed.extend(new_closed)

    assert len(closed) > 0, "no closed trades — data window too short?"

    bad = [
        (i, t.commission)
        for i, t in enumerate(closed)
        if not math.isclose(t.commission, expected_commission, rel_tol=1e-9, abs_tol=1e-9)
    ]
    assert not bad, (
        f"{len(bad)}/{len(closed)} trades have wrong commission "
        f"(expected {expected_commission}, first three: {bad[:3]})"
    )
