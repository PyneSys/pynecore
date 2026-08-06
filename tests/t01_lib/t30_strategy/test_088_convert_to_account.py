"""
@pyne

strategy.convert_to_account / convert_to_symbol / account_currency with currency=EUR
on a EURUSD chart (quote currency USD, account currency EUR).

Measured on TradingView (FX:EURUSD 1D, currency=currency.EUR): TV keeps one
symbol->account rate; convert_to_account multiplies by it and convert_to_symbol
divides by it, so the two directions are exactly reciprocal
(0.8608 and 1/0.8608 = 1.161710037175). convert_to_account(na) is na and
convert_to_account(0) is 0. account_currency is "EUR".

Here the chart itself is the currency pair, so the rate source is the chart's
own close (the CurrencyRateProvider chart-pair fast path): the USD->EUR rate
is 1/close.
"""
from pynecore.lib import close, currency, na, plot, script, strategy


@script.strategy("Convert to account", overlay=True, currency=currency.EUR)
def main():
    plot(strategy.convert_to_account(1.0), "to_acct")
    plot(strategy.convert_to_symbol(1.0), "to_sym")
    plot(strategy.convert_to_account(na(float)), "to_acct_na")
    plot(strategy.convert_to_account(na), "to_acct_bare_na")
    plot(strategy.convert_to_account(0.0), "to_acct_zero")
    plot(1.0 if strategy.account_currency == "EUR" else 0.0, "acct_is_eur")
    plot(close, "close")


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
def __test_convert_uses_the_chart_pair_rate__(script_path, module_key):
    """to_account = 1/close, to_symbol its exact reciprocal, na passes through, EUR account."""
    import sys
    import math
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner
    from pynecore.types.ohlcv import OHLCV

    sys.modules.pop(module_key, None)

    base_ts = 1_704_067_200_000  # 2024-01-01 00:00:00 UTC, in ms
    closes = [1.10, 1.12, 1.08, 1.15]
    bars = [
        OHLCV(timestamp=base_ts + i * 60_000, open=c, high=c + 0.01, low=c - 0.01,
              close=c, volume=100.0)
        for i, c in enumerate(closes)
    ]

    runner = ScriptRunner(Path(script_path), iter(bars), _make_syminfo())
    rows = [dict(plot_values) for _candle, plot_values, _closed in runner.run_iter()]

    assert len(rows) == len(bars), "the run must reach the last bar"

    for i, row in enumerate(rows):
        assert row['acct_is_eur'] == 1.0, f"bar {i}: account_currency is not EUR"
        rate = 1.0 / row['close']
        assert abs(row['to_acct'] - rate) < 1e-12, f"bar {i}: {row['to_acct']} != {rate}"
        # convert_to_symbol divides by the same rate — exactly reciprocal, like TV
        assert row['to_sym'] == 1.0 / row['to_acct'], f"bar {i}: not reciprocal"
        assert math.isnan(row['to_acct_na']), f"bar {i}: na did not pass through"
        # Bare value-position na is rewritten to the lib._na_none sentinel by the
        # transform pipeline; the function must answer na for it too
        assert math.isnan(row['to_acct_bare_na']), f"bar {i}: bare na did not pass through"
        assert row['to_acct_zero'] == 0.0, f"bar {i}: zero is not zero"
