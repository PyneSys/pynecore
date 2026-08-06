"""
@pyne

strategy.account_currency and the convert functions with the default currency=NONE.

Measured on TradingView (FX:EURUSD 1D, no currency argument): the account
currency falls back to the symbol's quote currency ("USD" on EURUSD) and both
convert functions are the identity (rate 1).
"""
from pynecore.lib import plot, script, strategy


@script.strategy("Account currency default", overlay=True)
def main():
    plot(strategy.convert_to_account(2.5), "to_acct")
    plot(strategy.convert_to_symbol(2.5), "to_sym")
    plot(1.0 if strategy.account_currency == "USD" else 0.0, "acct_is_usd")


def _make_syminfo():
    from pynecore.core.syminfo import SymInfo
    from pynecore.providers.ccxt import CCXTProvider
    # noinspection PyProtectedMember
    opening_hours, session_starts, session_ends = CCXTProvider._create_24_7_sessions()
    return SymInfo(
        prefix="TEST", description="Test", ticker="EURUSD", currency="USD",
        basecurrency="EUR", period='1', type="forex", mintick=0.00001,
        pricescale=100000, minmove=1, pointvalue=1, timezone="UTC",
        volumetype="base", mincontract=0.0001,
        opening_hours=opening_hours, session_starts=session_starts,
        session_ends=session_ends,
    )


# noinspection PyShadowingNames
def __test_account_currency_defaults_to_the_symbol_currency__(script_path, module_key):
    """currency=NONE: account_currency is the quote currency and conversion is identity."""
    import sys
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner
    from pynecore.types.ohlcv import OHLCV

    sys.modules.pop(module_key, None)

    base_ts = 1_704_067_200_000  # 2024-01-01 00:00:00 UTC, in ms
    bars = [
        OHLCV(timestamp=base_ts + i * 60_000, open=1.10, high=1.11, low=1.09,
              close=1.10, volume=100.0)
        for i in range(3)
    ]

    runner = ScriptRunner(Path(script_path), iter(bars), _make_syminfo())
    rows = [dict(plot_values) for _candle, plot_values, _closed in runner.run_iter()]

    for i, row in enumerate(rows):
        assert row['acct_is_usd'] == 1.0, f"bar {i}: account_currency is not USD"
        assert row['to_acct'] == 2.5, f"bar {i}: to_account is not identity: {row['to_acct']}"
        assert row['to_sym'] == 2.5, f"bar {i}: to_symbol is not identity: {row['to_sym']}"
