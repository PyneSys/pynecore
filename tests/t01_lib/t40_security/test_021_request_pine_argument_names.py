"""
Regression: the parameter NAMES of the request.* functions must be Pine's.

A compiled Pine script emits Pine's own keyword for every named argument, so a
parameter spelled differently from TradingView raises ``TypeError`` and halts the
script. The names below were verified against the TradingView compiler.

``request.dividends`` / ``request.earnings`` are not implemented yet, but with
``ignore_invalid_symbol=True`` they return ``na`` instead of raising, so the call
itself -- which is where a wrong keyword would fail -- can be exercised directly.
"""
from math import isnan

import pytest

from pynecore.lib import barmerge, dividends, earnings, request, splits


def __test_request_dividends_has_currency__():
    """request.dividends takes Pine's 6 parameters, ``currency`` last"""
    assert isnan(request.dividends(ticker="AAPL", field=dividends.gross,
                                   gaps=barmerge.gaps_off,
                                   lookahead=barmerge.lookahead_off,
                                   ignore_invalid_symbol=True, currency="USD"))
    # The same arguments fully positional -- guards the parameter ORDER
    assert isnan(request.dividends("AAPL", dividends.gross, barmerge.gaps_off,
                                   barmerge.lookahead_off, True, "USD"))


def __test_request_earnings_has_currency__():
    """request.earnings takes Pine's 6 parameters, ``currency`` last"""
    assert isnan(request.earnings(ticker="AAPL", field=earnings.actual,
                                  gaps=barmerge.gaps_off,
                                  lookahead=barmerge.lookahead_off,
                                  ignore_invalid_symbol=True, currency="USD"))
    # The same arguments fully positional -- guards the parameter ORDER
    assert isnan(request.earnings("AAPL", earnings.actual, barmerge.gaps_off,
                                  barmerge.lookahead_off, True, "USD"))


def __test_request_splits_has_no_currency__():
    """request.splits stops at Pine's 5 parameters -- no ``currency``"""
    with pytest.raises(TypeError):
        # noinspection PyArgumentList
        request.splits("AAPL", splits.numerator, barmerge.gaps_off,  # type: ignore[call-arg]
                       barmerge.lookahead_off, True, "USD")
