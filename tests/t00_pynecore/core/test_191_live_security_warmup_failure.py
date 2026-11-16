"""
A live ``request.security`` context whose warmup download the provider turns down
stops the run with a coded provider error naming the provider, symbol and
timeframe, not with an internal error.
"""
from datetime import datetime, timedelta, UTC

import pytest

from pynecore.core.errors import PyneError
from pynecore.core.plugin.live_provider import PluginSymbol
# noinspection PyProtectedMember
from pynecore.core.security_process import _download_warmup, _failure_report
from pynecore.errorcodes import PROVIDER_UNAVAILABLE, ProviderUnavailableError


class _RejectingProvider:
    """A provider whose venue rejects the requested candle interval."""

    @staticmethod
    def save_ohlcv_data(data):
        raise AssertionError(f"nothing is downloaded: {data!r}")

    @staticmethod
    def download_ohlcv(time_from, time_to):
        raise RuntimeError('binance: binance {"code":-1120,"msg":"Invalid interval."}')


class _CodedFailureProvider:
    """A provider that already reports its own coded error."""

    @staticmethod
    def save_ohlcv_data(data):
        raise AssertionError(f"nothing is downloaded: {data!r}")

    @staticmethod
    def download_ohlcv(time_from, time_to):
        raise ProviderUnavailableError(provider="ccxt", why="rate limited")


_SOURCE = PluginSymbol(provider_name="ccxt", symbol="BINANCE:BTC/USDT:USDT", timeframe="180")
_TO = datetime(2026, 11, 16, 15, 0, tzinfo=UTC)
_FROM = _TO - timedelta(days=10)


def __test_rejected_warmup_is_a_coded_provider_error__():
    """The rejection names the provider, the symbol and the timeframe as R0049"""
    # noinspection PyTypeChecker
    with pytest.raises(ProviderUnavailableError) as info:
        _download_warmup(_RejectingProvider(), _SOURCE, _FROM, _TO)
    error = info.value
    assert error.code is PROVIDER_UNAVAILABLE
    message = str(error)
    assert "`ccxt`" in message
    assert "BINANCE:BTC/USDT:USDT" in message
    assert "timeframe 180" in message
    assert "Invalid interval" in message
    assert isinstance(error.__cause__, RuntimeError)


def __test_child_hands_the_coded_error_to_the_chart__():
    """The child's failure report is the coded error itself, not a text report"""
    try:
        # noinspection PyTypeChecker
        _download_warmup(_RejectingProvider(), _SOURCE, _FROM, _TO)
    except ProviderUnavailableError as exc:
        report = _failure_report(exc)
    else:
        raise AssertionError("the warmup did not fail")
    assert isinstance(report, PyneError)
    assert report.code is PROVIDER_UNAVAILABLE


def __test_coded_provider_error_passes_through__():
    """A provider's own coded error is not wrapped again"""
    # noinspection PyTypeChecker
    with pytest.raises(ProviderUnavailableError) as info:
        _download_warmup(_CodedFailureProvider(), _SOURCE, _FROM, _TO)
    assert "rate limited" in str(info.value)
    assert "timeframe" not in str(info.value)
