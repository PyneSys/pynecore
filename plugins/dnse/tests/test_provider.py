"""Tests for :mod:`pynecore_dnse.provider` — the data-only plugin whose
``resolve_contract`` / ``market_type`` feed ``broker._place``'s order symbol
and book selection. A wrong resolution here trades the wrong instrument or
mis-quantizes every stock order (tick-size bands), so this file leans hard on
the money-adjacent branches per ``docs/test_plan.md``.

Seam: inject a :class:`_FakeClient` (see ``conftest.py``) as ``provider._client``
— the ``client`` property returns it directly, so no network/FS ever happens.
"""
from datetime import datetime, timezone

import pytest

from pynecore_dnse import provider as provider_module
from pynecore_dnse.provider import DNSEProvider, DNSEConfig


def _provider(symbol="VN30F1M", timeframe="5", **config_kwargs):
    config = DNSEConfig(api_key="k", api_secret="s", **config_kwargs)
    return DNSEProvider(symbol=symbol, timeframe=timeframe, config=config)


def _wired(fake_client, symbol="VN30F1M", timeframe="5", **config_kwargs):
    """A provider instance with ``fake_client`` already injected as ``_client``."""
    p = _provider(symbol=symbol, timeframe=timeframe, **config_kwargs)
    p._client = fake_client
    return p


# --- resolve_contract --------------------------------------------------------

def __test_resolve_contract_alias_maps_to_dated_contract__(fake_client):
    fake = fake_client(get_instruments=(200, {"data": [
        {"symbolType": "VN30F2M", "symbol": "41I1G9000"},
        {"symbolType": "VN30F1M", "symbol": "41I1G8000"},
    ]}))
    p = _wired(fake, symbol="VN30F1M")

    resolved = p.resolve_contract()

    assert resolved == "41I1G8000", "must match the row whose symbolType equals the alias"
    assert fake.count("get_instruments") == 1, "one lookup for the first resolution"


def __test_resolve_contract_non_alias_symbol_returned_unchanged__(fake_client):
    fake = fake_client(get_instruments=(200, {"data": [
        {"symbolType": "VN30F1M", "symbol": "41I1G8000"},
    ]}))
    p = _wired(fake, symbol="HPG")

    resolved = p.resolve_contract("HPG")

    assert resolved == "HPG", "stocks are already their own tradable code"
    assert fake.count("get_instruments") == 1


@pytest.mark.parametrize("status, body", [
    (500, {}),                       # API failure
    (200, {"data": []}),             # no matching row
    (200, {}),                       # malformed body (no "data" key)
])
def __test_resolve_contract_silently_falls_back_to_input_on_failure__(fake_client, status, body):
    """SAFETY: an unresolved alias silently degrades to the input symbol rather
    than raising — if a caller (broker._place) doesn't notice, it would place
    an order using the untradeable alias instead of the dated contract."""
    fake = fake_client(get_instruments=(status, body))
    p = _wired(fake, symbol="VN30F1M")

    resolved = p.resolve_contract()

    assert resolved == "VN30F1M", "silent-degrade: falls back to the unresolved alias"
    assert fake.count("get_instruments") == 1, "still only one attempt, no retry storm"


def __test_resolve_contract_caches_per_instance__(fake_client):
    fake = fake_client(get_instruments=(200, {"data": [
        {"symbolType": "VN30F1M", "symbol": "41I1G8000"},
    ]}))
    p = _wired(fake, symbol="VN30F1M")

    first = p.resolve_contract()
    second = p.resolve_contract()

    assert first == second == "41I1G8000"
    assert fake.count("get_instruments") == 1, "second call must be served from the per-instance cache"


def __test_resolve_contract_caches_the_fallback_too__(fake_client):
    """Even the silent-degrade result is cached — repeated calls don't retry."""
    fake = fake_client(get_instruments=(500, {}))
    p = _wired(fake, symbol="VN30F1M")

    p.resolve_contract()
    p.resolve_contract()

    assert fake.count("get_instruments") == 1, "fallback result is cached, not retried every call"


def __test_normalize_symbol_delegates_to_resolve_contract__(fake_client):
    fake = fake_client(get_instruments=(200, {"data": [
        {"symbolType": "VN30F1M", "symbol": "41I1G8000"},
    ]}))
    p = _wired(fake, symbol="VN30F1M")

    assert p.normalize_symbol("VN30F1M") == "41I1G8000"
    assert fake.count("get_instruments") == 1


# --- market_type --------------------------------------------------------------

def __test_market_type_derivative_from_security_group_fu__(fake_client):
    fake = fake_client(get_security_definition=(200, [{"securityGroupId": "FU"}]))
    p = _wired(fake, symbol="41I1G8000")  # already a dated code, no VN30F prefix

    result = p.market_type

    assert result == "DERIVATIVE", "securityGroupId=='FU' must classify as DERIVATIVE"
    assert fake.count("get_security_definition") == 1


def __test_market_type_stock_from_non_fu_security_group__(fake_client):
    fake = fake_client(get_security_definition=(200, {"securityGroupId": "ST"}))
    p = _wired(fake, symbol="HPG")

    result = p.market_type

    assert result == "STOCK", "a non-empty, non-FU group must classify as STOCK"
    assert fake.count("get_security_definition") == 1


def __test_market_type_falls_back_to_vn30f_prefix_heuristic_when_secdef_silent__(fake_client):
    fake = fake_client(
        get_instruments=(500, {}),           # resolve_contract fallback (alias unresolved)
        get_security_definition=(200, []),   # empty list -> row == {}
    )
    p = _wired(fake, symbol="VN30F1M")

    result = p.market_type

    assert result == "DERIVATIVE", "empty secdef must fall back to the VN30F prefix heuristic"
    assert fake.count("get_security_definition") == 1


def __test_market_type_heuristic_stock_when_no_vn30f_prefix_and_secdef_silent__(fake_client):
    fake = fake_client(get_security_definition=(404, {}))
    p = _wired(fake, symbol="HPG")

    result = p.market_type

    assert result == "STOCK", "non-VN30F symbol with silent secdef must default to STOCK"
    assert fake.count("get_security_definition") == 1


# --- _secdef -------------------------------------------------------------------

def __test_secdef_caches_per_symbol__(fake_client):
    fake = fake_client(get_security_definition=(200, {"securityGroupId": "ST"}))
    p = _wired(fake, symbol="HPG")

    first = p._secdef("HPG")
    second = p._secdef("HPG")

    assert first == second == {"securityGroupId": "ST"}
    assert fake.count("get_security_definition") == 1, "second lookup must be served from cache"


def __test_secdef_derivative_alias_pre_resolved_before_lookup__(fake_client):
    fake = fake_client(
        get_instruments=(200, {"data": [{"symbolType": "VN30F1M", "symbol": "41I1G8000"}]}),
        get_security_definition=(200, {"securityGroupId": "FU"}),
    )
    p = _wired(fake, symbol="VN30F1M")

    p._secdef("VN30F1M")

    secdef_calls = [c for c in fake.calls if c[0] == "get_security_definition"]
    assert len(secdef_calls) == 1
    assert secdef_calls[0][1][0] == "41I1G8000", (
        "the resolved dated contract, not the raw alias, must be sent to secdef"
    )


def __test_secdef_list_response_uses_first_row__(fake_client):
    fake = fake_client(get_security_definition=(200, [
        {"securityGroupId": "FU", "basicPrice": 1234.0},
        {"securityGroupId": "FU", "basicPrice": 9999.0},
    ]))
    p = _wired(fake, symbol="41I1G8000")

    row = p._secdef("41I1G8000")

    assert row["basicPrice"] == 1234.0, "list body must use the first row"
    assert row["securityGroupId"] == "FU"


def __test_secdef_dict_response_used_directly__(fake_client):
    fake = fake_client(get_security_definition=(200, {"securityGroupId": "ST", "basicPrice": 22.15}))
    p = _wired(fake, symbol="HPG")

    row = p._secdef("HPG")

    assert row["basicPrice"] == 22.15
    assert row["securityGroupId"] == "ST"


@pytest.mark.parametrize("status, body", [
    (404, {"code": "RESOURCE_NOT_FOUND"}),
    (200, []),      # empty list
    (200, None),    # neither list nor dict
])
def __test_secdef_non_200_or_empty_caches_empty_dict__(fake_client, status, body):
    fake = fake_client(get_security_definition=(status, body))
    p = _wired(fake, symbol="HPG")

    first = p._secdef("HPG")
    second = p._secdef("HPG")

    assert first == {} and second == {}, "non-200/empty body must resolve to {}"
    assert fake.count("get_security_definition") == 1, "the empty result must still be cached"


# --- update_symbol_info: derivative branch -------------------------------------

def __test_update_symbol_info_derivative_fixed_mintick__(fake_client):
    fake = fake_client(get_security_definition=(200, {"securityGroupId": "FU"}))
    p = _wired(fake, symbol="41I1G8000", timeframe="15")

    info = p.update_symbol_info()

    assert info.mintick == 0.1, "derivative mintick is fixed regardless of price"
    assert info.minmove == 1 and info.pricescale == 10
    assert info.type == "futures" and info.pointvalue == 100_000


# --- update_symbol_info: stock tick-size bands (highest-value case) -----------

@pytest.mark.parametrize("reference_price, expected_mintick", [
    (5.0, 0.01),      # well inside low band
    (9.99, 0.01),     # just below the 10 boundary
    (10.0, 0.05),     # exactly at the 10 boundary -> mid band
    (49.99, 0.05),    # just below the 50 boundary
    (50.0, 0.10),     # exactly at the 50 boundary -> high band
    (150.0, 0.10),    # well inside high band
])
def __test_update_symbol_info_stock_tick_bands__(fake_client, reference_price, expected_mintick):
    fake = fake_client(get_security_definition=(200, {
        "securityGroupId": "ST", "basicPrice": reference_price,
    }))
    p = _wired(fake, symbol="HPG", timeframe="15")

    info = p.update_symbol_info()

    assert info.mintick == expected_mintick, (
        f"reference={reference_price} must select the {expected_mintick} tick band "
        f"(wrong band mis-quantizes every stock order at this price)"
    )
    assert info.minmove == round(expected_mintick * 100)
    assert info.pricescale == 100
    assert info.type == "stock" and info.pointvalue == 1_000


def __test_update_symbol_info_stock_uses_ceiling_then_floor_price_fallback__(fake_client):
    fake = fake_client(get_security_definition=(200, {
        "securityGroupId": "ST", "basicPrice": 0, "ceilingPrice": 55.0, "floorPrice": 45.0,
    }))
    p = _wired(fake, symbol="HPG", timeframe="15")

    info = p.update_symbol_info()

    assert info.mintick == 0.10, "falsy basicPrice must fall through to ceilingPrice (55 -> high band)"
    assert info.minmove == 10


def __test_update_symbol_info_stock_defaults_to_mid_band_when_secdef_empty__(fake_client):
    """Edge: no basicPrice/ceilingPrice/floorPrice at all -> the 20.0 mid-band default."""
    fake = fake_client(get_security_definition=(404, {}))
    p = _wired(fake, symbol="HPG", timeframe="15")

    info = p.update_symbol_info()

    assert info.mintick == 0.05, "no price data must fall back to the documented 20.0 mid-band default"
    assert info.minmove == 5


def __test_update_symbol_info_sessions_populated_mon_through_fri__(fake_client):
    fake = fake_client(get_security_definition=(200, {"securityGroupId": "FU"}))
    p = _wired(fake, symbol="41I1G8000", timeframe="15")

    info = p.update_symbol_info()

    assert [s.day for s in info.session_starts] == [1, 2, 3, 4, 5], "one session-start per weekday"
    assert [s.day for s in info.session_ends] == [1, 2, 3, 4, 5], "one session-end per weekday"
    assert all(s.time.hour == 9 and s.time.minute == 0 for s in info.session_starts), (
        "morning session must start at 09:00"
    )
    assert all(s.time.hour == 14 and s.time.minute == 45 for s in info.session_ends), (
        "afternoon session must end at 14:45"
    )
    assert len(info.opening_hours) == 10, "two trading blocks (morning+afternoon) x 5 weekdays"


# --- download_ohlcv -------------------------------------------------------------

def _ohlcv_body(count, base_ts=1_700_000_000):
    times = [base_ts + i * 300 for i in range(count)]
    return {
        "t": times,
        "o": [float(100 + i) for i in range(count)],
        "h": [float(101 + i) for i in range(count)],
        "l": [float(99 + i) for i in range(count)],
        "c": [float(100.5 + i) for i in range(count)],
        "v": [float(10 + i) for i in range(count)],
    }


def __test_download_ohlcv_saves_each_bar_with_ms_timestamp__(fake_client):
    body = _ohlcv_body(3)
    fake = fake_client(get_ohlc=(200, body))
    p = _wired(fake, symbol="41I1G8000", timeframe="5")
    saved = []
    p.save_ohlcv_data = lambda data: saved.append(data)

    p.download_ohlcv(datetime(2026, 1, 1, tzinfo=timezone.utc), datetime(2026, 1, 2, tzinfo=timezone.utc))

    assert len(saved) == 3, "one save_ohlcv_data call per returned bar"
    assert saved[0].timestamp == body["t"][0] * 1000, "seconds from the API must become milliseconds"
    assert saved[1].open == 101.0 and saved[1].close == 101.5 and saved[1].volume == 11.0


def __test_download_ohlcv_progress_cadence_every_200_bars_plus_final__(fake_client):
    body = _ohlcv_body(401)  # indices 0, 200, 400 -> per-loop progress fires 3x
    fake = fake_client(get_ohlc=(200, body))
    p = _wired(fake, symbol="41I1G8000", timeframe="5")
    p.save_ohlcv_data = lambda data: None
    calls = []
    time_to = datetime(2026, 1, 2, tzinfo=timezone.utc)

    p.download_ohlcv(datetime(2026, 1, 1, tzinfo=timezone.utc), time_to, on_progress=calls.append)

    assert len(calls) == 4, "3 per-loop cadence calls (idx 0/200/400) + 1 final call"
    assert calls[-1] == time_to, "the final callback must receive time_to exactly"


def __test_download_ohlcv_progress_datetime_naive_when_time_to_naive__(fake_client):
    body = _ohlcv_body(1)
    fake = fake_client(get_ohlc=(200, body))
    p = _wired(fake, symbol="41I1G8000", timeframe="5")
    p.save_ohlcv_data = lambda data: None
    calls = []
    naive_time_to = datetime(2026, 1, 2)  # no tzinfo

    p.download_ohlcv(datetime(2026, 1, 1), naive_time_to, on_progress=calls.append)

    assert calls[0].tzinfo is None, "per-loop progress must match time_to's naive-ness"
    assert calls[-1] is naive_time_to


def __test_download_ohlcv_progress_datetime_aware_when_time_to_aware__(fake_client):
    body = _ohlcv_body(1)
    fake = fake_client(get_ohlc=(200, body))
    p = _wired(fake, symbol="41I1G8000", timeframe="5")
    p.save_ohlcv_data = lambda data: None
    calls = []
    aware_time_to = datetime(2026, 1, 2, tzinfo=timezone.utc)

    p.download_ohlcv(datetime(2026, 1, 1, tzinfo=timezone.utc), aware_time_to, on_progress=calls.append)

    assert calls[0].tzinfo is timezone.utc, "per-loop progress must gain UTC tzinfo to match aware time_to"
    assert calls[-1] is aware_time_to


@pytest.mark.parametrize("status, body", [
    (500, {"code": "REMOTE_SERVER_ERROR"}),   # non-200
    (200, "not-a-dict"),                      # malformed body
    (200, None),                              # empty body
])
def __test_download_ohlcv_raises_runtime_error_on_bad_response__(fake_client, status, body):
    fake = fake_client(get_ohlc=(status, body))
    p = _wired(fake, symbol="41I1G8000", timeframe="5")
    p.save_ohlcv_data = lambda data: (_ for _ in ()).throw(AssertionError("must not save on failure"))

    with pytest.raises(RuntimeError):
        p.download_ohlcv(datetime(2026, 1, 1, tzinfo=timezone.utc), datetime(2026, 1, 2, tzinfo=timezone.utc))


def __test_download_ohlcv_empty_bars_still_calls_final_progress__(fake_client):
    """Edge: zero bars returned — the loop body never runs, but the final callback still must."""
    fake = fake_client(get_ohlc=(200, _ohlcv_body(0)))
    p = _wired(fake, symbol="41I1G8000", timeframe="5")
    p.save_ohlcv_data = lambda data: (_ for _ in ()).throw(AssertionError("nothing to save"))
    calls = []
    time_to = datetime(2026, 1, 2, tzinfo=timezone.utc)

    p.download_ohlcv(datetime(2026, 1, 1, tzinfo=timezone.utc), time_to, on_progress=calls.append)

    assert calls == [time_to], "final progress callback must fire even with no bars"


# --- client property: fail-fast on missing credentials -------------------------

@pytest.mark.parametrize("api_key, api_secret", [
    ("", ""),
    ("", "s"),
    ("k", ""),
])
def __test_client_property_missing_credentials_raises_before_building_client__(monkeypatch, api_key, api_secret):
    config = DNSEConfig(api_key=api_key, api_secret=api_secret)
    p = DNSEProvider(symbol="HPG", timeframe="5", config=config)
    assert p._client is None, "precondition: no client injected"

    built = []
    monkeypatch.setattr(provider_module, "DNSEClient",
                        lambda *a, **k: built.append((a, k)) or object())

    with pytest.raises(ValueError):
        _ = p.client

    assert built == [], "the real client must never be constructed when credentials are missing"


def __test_client_property_builds_and_caches_real_client_when_credentials_present__(monkeypatch):
    """Happy path: valid credentials build (and cache) the wrapper exactly once,
    pinned to config's base_url — the DNSEClient constructor is stubbed so no
    real TLS/socket setup happens."""
    built = []

    class _FakeDNSEClient:
        def __init__(self, api_key, api_secret, base_url):
            built.append((api_key, api_secret, base_url))

    monkeypatch.setattr(provider_module, "DNSEClient", _FakeDNSEClient)
    p = _provider(symbol="HPG", timeframe="5", base_url="http://localhost:1234")

    first = p.client
    second = p.client

    assert first is second, "the client must be built once and cached on the instance"
    assert built == [("k", "s", "http://localhost:1234")], (
        "constructor must receive config's api_key/api_secret/base_url exactly once"
    )


def __test_get_list_of_symbols__():
    with_symbol = _provider(symbol="HPG", timeframe="5")
    without_symbol = _provider(symbol=None, timeframe=None)

    assert with_symbol.get_list_of_symbols() == ["HPG"]
    assert without_symbol.get_list_of_symbols() == [], "no configured symbol must yield an empty list"


def __test_client_property_returns_injected_client_without_network__(fake_client):
    fake = fake_client()
    p = _provider(symbol="HPG", timeframe="5")
    p._client = fake

    assert p.client is fake, "the property must short-circuit to the injected client"
    assert fake.calls == [], "merely accessing .client must not issue any request"


# --- is_production / announce_endpoints -----------------------------------------

@pytest.mark.parametrize("base_url, ws_url, expect_production, expect_tag", [
    ("https://openapi.dnse.com.vn", "wss://ws-openapi.dnse.com.vn", True, "LIVE DNSE"),
    ("http://localhost:8080", "ws://localhost:8081", False, "TEST VENUE (not DNSE)"),
    ("https://openapi.dnse.com.vn", "ws://localhost:8081", True, "MIXED"),
    ("http://localhost:8080", "wss://ws-openapi.dnse.com.vn", True, "MIXED"),
])
def __test_is_production_and_announce_endpoints__(base_url, ws_url, expect_production, expect_tag):
    p = _provider(symbol="HPG", timeframe="5", base_url=base_url, ws_url=ws_url)

    assert p.is_production is expect_production, (
        f"is_production must be {expect_production} for base={base_url!r} ws={ws_url!r}"
    )
    banner = p.announce_endpoints()
    assert expect_tag in banner, f"banner {banner!r} must contain {expect_tag!r}"


def __test_is_production_mixed_endpoint_never_silently_test_venue__(fake_client):
    """SAFETY: production REST + local WS must be flagged, never mistaken for a
    test venue while real orders go out over the production REST leg."""
    p = _provider(symbol="HPG", timeframe="5",
                  base_url="https://openapi.dnse.com.vn", ws_url="ws://localhost:9999")

    assert p.is_production is True, "any production endpoint means production"
    assert "TEST VENUE" not in p.announce_endpoints(), "must never read as a safe test venue"


# --- timeframe conversion --------------------------------------------------------

@pytest.mark.parametrize("tv_timeframe, dnse_resolution", [
    ("1", "1"),
    ("5", "5"),
    ("15", "15"),
    ("60", "1H"),
    ("1D", "1D"),
    ("D", "1D"),
    ("1W", "1W"),
])
def __test_to_exchange_timeframe_supported__(tv_timeframe, dnse_resolution):
    assert DNSEProvider.to_exchange_timeframe(tv_timeframe) == dnse_resolution


def __test_to_exchange_timeframe_unsupported_raises_with_supported_list__():
    with pytest.raises(ValueError) as excinfo:
        DNSEProvider.to_exchange_timeframe("2")

    message = str(excinfo.value)
    assert "2" in message, "the offending timeframe must be named in the error"
    assert "supported" in message.lower(), "the error must list the supported timeframes"


@pytest.mark.parametrize("dnse_resolution, tv_timeframe", [
    ("1", "1"),
    ("15", "15"),
    ("1H", "60"),
    ("1D", "1D"),
    ("1W", "1W"),
])
def __test_to_tradingview_timeframe_supported__(dnse_resolution, tv_timeframe):
    assert DNSEProvider.to_tradingview_timeframe(dnse_resolution) == tv_timeframe


def __test_to_tradingview_timeframe_unknown_falls_back_to_identity__():
    """Unlike to_exchange_timeframe, this direction never raises — an unrecognized
    exchange resolution is passed through unchanged rather than erroring."""
    result = DNSEProvider.to_tradingview_timeframe("bogus-resolution")

    assert result == "bogus-resolution"
    assert isinstance(result, str)


# === get_expected_price (Giá dự khớp — DNSE 2026-08-06 endpoint) ===============

def __test_get_expected_price_resolves_alias_and_returns_body__(fake_client):
    fake = fake_client(
        get_instruments=(200, {"data": [{"symbolType": "VN30F1M", "symbol": "41I1G8000"}]}),
        get_expected_price=(200, {"expectedPrices": [{"symbol": "41I1G8000", "price": 1900.0}]}))
    p = _wired(fake, symbol="VN30F1M")

    body = p.get_expected_price()

    assert body["expectedPrices"][0]["price"] == 1900.0, "returns the parsed venue body"
    call = next(c for c in fake.calls if c[0] == "get_expected_price")
    assert call[1][0] == "41I1G8000", \
        "the derivative alias must be resolved to its contract before the price call"


@pytest.mark.parametrize("status, body", [
    (500, {}),                        # API failure
    (404, {"code": "RESOURCE_NOT_FOUND"}),
    (200, "not-a-dict"),              # 200 but malformed body
])
def __test_get_expected_price_returns_empty_dict_on_non_200_or_non_dict__(fake_client, status, body):
    fake = fake_client(
        get_instruments=(200, {"data": [{"symbolType": "VN30F1M", "symbol": "41I1G8000"}]}),
        get_expected_price=(status, body))
    p = _wired(fake, symbol="VN30F1M")

    assert p.get_expected_price() == {}, "non-200 or non-dict must degrade to an empty dict"
