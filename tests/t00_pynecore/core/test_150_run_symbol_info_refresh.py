"""Provider-mode runs refresh the symbol metadata from the venue on every start."""
from typing import cast

import pytest

from pynecore.cli.commands.run import _fetch_provider_symbol_info
from pynecore.core.plugin.provider import ProviderPlugin
from pynecore.core.syminfo import SymInfo


def _make_syminfo(ticker: str) -> SymInfo:
    return SymInfo(prefix='TEST', description=ticker, ticker=ticker, currency='USD',
                   basecurrency=None, period='1', type='crypto', mintick=0.01,
                   pricescale=100, minmove=1, pointvalue=1.0, mincontract=0.001,
                   timezone='UTC', opening_hours=[], session_starts=[], session_ends=[])


class _FakeProvider:
    """A provider whose venue metadata differs from its cached sidecar."""

    def __init__(self, *, cached: bool, fail: Exception | None = None) -> None:
        self.cached = cached
        self.fail = fail
        self.calls: list[bool] = []

    def is_symbol_info_exists(self) -> bool:
        return self.cached

    def get_symbol_info(self, force_update: bool = False) -> SymInfo:
        self.calls.append(force_update)
        if force_update:
            if self.fail is not None:
                raise self.fail
            return _make_syminfo('VENUE')
        assert self.cached
        return _make_syminfo('CACHED')


def __test_provider_run_refreshes_symbol_info_despite_a_cached_sidecar__():
    """A cached sidecar does not short-circuit the venue refresh.

    Measured live (ctrader lane, 2026-10-02): the lane ran on a sidecar from
    four months earlier, so the venue's dated trading break (published days
    ahead) never reached the live session calendar and the bot churned
    through reconnects against a closed market.
    """
    provider = _FakeProvider(cached=True)

    syminfo = _fetch_provider_symbol_info(cast(ProviderPlugin, provider))

    assert syminfo.ticker == 'VENUE'
    assert provider.calls == [True]


def __test_provider_run_falls_back_to_the_cached_symbol_info_when_the_venue_fails__(capsys):
    provider = _FakeProvider(cached=True, fail=ConnectionError("venue down"))

    syminfo = _fetch_provider_symbol_info(cast(ProviderPlugin, provider))

    assert syminfo.ticker == 'CACHED'
    assert provider.calls == [True, False]
    assert "symbol info refresh failed" in capsys.readouterr().err


def __test_provider_run_without_a_cache_surfaces_the_venue_failure__():
    provider = _FakeProvider(cached=False, fail=ConnectionError("venue down"))

    with pytest.raises(ConnectionError, match="venue down"):
        _fetch_provider_symbol_info(cast(ProviderPlugin, provider))
    assert provider.calls == [True]
