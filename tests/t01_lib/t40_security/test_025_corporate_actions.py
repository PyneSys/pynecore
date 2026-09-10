from collections.abc import Callable
from math import isnan

import pytest

from pynecore.lib import barmerge, dividends, earnings, request, splits, syminfo


@pytest.mark.parametrize(('operation', 'field'), [
    (request.earnings, earnings.actual),
    (request.dividends, dividends.gross),
    (request.splits, splits.denominator),
])
def __test_crypto_chart_corporate_actions_are_empty__(
        monkeypatch: pytest.MonkeyPatch, operation: Callable[..., float], field: str,
):
    monkeypatch.setattr(syminfo, 'type', 'crypto')
    monkeypatch.setattr(syminfo, 'tickerid', 'BINANCE:BTCUSDT')

    assert isnan(operation(syminfo.tickerid, field, barmerge.gaps_on, barmerge.lookahead_on))


@pytest.mark.parametrize(('operation', 'field'), [
    (request.earnings, earnings.actual),
    (request.dividends, dividends.gross),
    (request.splits, splits.denominator),
])
@pytest.mark.parametrize(('chart_type', 'chart_ticker', 'requested_ticker'), [
    ('stock', 'NASDAQ:AAPL', 'NASDAQ:AAPL'),
    ('crypto', 'BINANCE:BTCUSDT', 'NASDAQ:AAPL'),
    ('crypto', 'BINANCE:BTCUSDT', 'OTHER:BTCUSDT'),
    ('crypto', 'BINANCE:BTCUSDT', 'BINANCE:UNKNOWN'),
    ('crypto', 'BINANCE:BTCUSDT', ''),
    ('crypto', 'BINANCE:BTCUSDT', None),
    ('crypto', '', ''),
    ('', 'BINANCE:BTCUSDT', 'BINANCE:BTCUSDT'),
])
def __test_unsupported_corporate_action_feed_stays_explicit__(
        monkeypatch: pytest.MonkeyPatch, operation: Callable[..., float], field: str,
        chart_type: str, chart_ticker: str, requested_ticker: str | None,
):
    monkeypatch.setattr(syminfo, 'type', chart_type)
    monkeypatch.setattr(syminfo, 'tickerid', chart_ticker)

    with pytest.raises(NotImplementedError):
        operation(requested_ticker, field, barmerge.gaps_on, barmerge.lookahead_on)


@pytest.mark.parametrize(('operation', 'field'), [
    (request.earnings, earnings.actual),
    (request.dividends, dividends.gross),
    (request.splits, splits.denominator),
])
def __test_corporate_actions_preserve_ignore_invalid_symbol__(
        monkeypatch: pytest.MonkeyPatch, operation: Callable[..., float], field: str,
):
    monkeypatch.setattr(syminfo, 'type', 'stock')
    monkeypatch.setattr(syminfo, 'tickerid', 'NASDAQ:AAPL')

    assert isnan(operation('UNKNOWN:UNKNOWN', field, ignore_invalid_symbol=True))
