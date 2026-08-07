"""Tests for the version-pinned, JSON-parsing, TLS-verifying client wrapper
(``pynecore_dnse.client.DNSEClient``) over the vendored DNSE SDK.

No network: constructing ``DNSEClient`` only builds an urllib3 ``PoolManager``
object (no I/O) -- the vendored SDK's own ``__init__`` does the same. External
behavior is intercepted by monkeypatching ``pynecore_dnse.client._SdkClient``
(the constructor) or attributes on the already-built ``client._sdk`` instance,
per ``docs/test_plan.md``.
"""
import json

import certifi
import pytest
import urllib3

from pynecore_dnse import client as client_module
from pynecore_dnse.client import API_VERSION, DNSEClient


# --- version pin (highest value: the live-proven cancel landmine) ----------

@pytest.mark.parametrize("base_url", [
    pytest.param(None, id="default-base-url"),
    pytest.param("https://staging.dnse.example.com", id="custom-base-url"),
    pytest.param("https://openapi.dnse.com.vn/", id="trailing-slash-base-url"),
])
def __test_version_always_pinned_regardless_of_base_url__(monkeypatch, base_url):
    """A floating date silently no-ops conditional cancels (proven live
    2026-08-07: DNSE reads ``orderId`` as an int and returns 200 while
    cancelling nothing). The wrapper must ALWAYS pass the pinned version,
    no matter what else is passed to the constructor."""
    captured = {}

    class _FakeSdk:
        def __init__(self, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

    monkeypatch.setattr(client_module, "_SdkClient", _FakeSdk)

    if base_url is None:
        DNSEClient("key", "secret")
    else:
        DNSEClient("key", "secret", base_url=base_url)

    assert captured["kwargs"].get("api_version") == "2026-07-23", (
        f"api_version must always be pinned to 2026-07-23, "
        f"got {captured['kwargs'].get('api_version')!r}"
    )
    assert captured["kwargs"]["api_version"] == API_VERSION, \
        "the wrapper's pin and the exported module constant must agree"


def __test_version_pin_reaches_the_real_vendored_sdk_instance__():
    """No mocking: confirm the pin actually lands on the real SDK object that
    will sign and send requests (not just on a test double)."""
    c = DNSEClient("key", "secret")
    assert c._sdk._api_version == "2026-07-23", \
        "real vendored SDK instance must carry the pinned version"
    assert c._sdk._api_version == API_VERSION


# --- _parse (staticmethod, called directly) ---------------------------------

@pytest.mark.parametrize("value", [
    pytest.param(None, id="none"),
    pytest.param("just a string", id="bare-string"),
    pytest.param(["a", "b"], id="list-not-tuple"),
    pytest.param((1, 2, 3), id="3-tuple"),
    pytest.param((1,), id="1-tuple"),
])
def __test_parse_passes_through_non_status_body_values_unchanged__(value):
    result = DNSEClient._parse(value)
    assert result == value, "anything that isn't a 2-tuple must pass through untouched"
    assert result is value, "must be the identical object, not a rebuilt copy"


@pytest.mark.parametrize("raw, expected", [
    pytest.param(b'{"a": 1}', {"a": 1}, id="bytes-valid-json"),
    pytest.param(bytearray(b'{"a": 1}'), {"a": 1}, id="bytearray-valid-json"),
])
def __test_parse_decodes_bytes_body_then_parses_json__(raw, expected):
    status, body = DNSEClient._parse((200, raw))
    assert status == 200, "status must pass through unchanged"
    assert body == expected, f"bytes/bytearray body must decode utf-8 then json.loads, got {body!r}"


@pytest.mark.parametrize("raw, expected", [
    pytest.param('{"a": 1, "b": [1, 2]}', {"a": 1, "b": [1, 2]}, id="json-object"),
    pytest.param('[1, 2, 3]', [1, 2, 3], id="json-array"),
    pytest.param('{"unicode": "ti\\u1ebfng Vi\\u1ec7t", "quote": "a\\"b"}',
                 {"unicode": "tiếng Việt", "quote": 'a"b'}, id="unicode-and-escaped-quote"),
])
def __test_parse_valid_json_string_becomes_dict_or_list__(raw, expected):
    status, body = DNSEClient._parse((200, raw))
    assert status == 200
    assert body == expected, f"valid JSON string must parse into dict/list, got {body!r}"
    assert isinstance(body, (dict, list))


@pytest.mark.parametrize("raw", [
    pytest.param("not json at all", id="plain-text"),
    pytest.param("{unterminated", id="truncated-json"),
    pytest.param("{'single': 'quotes'}", id="single-quoted-not-valid-json"),
])
def __test_parse_invalid_json_left_as_raw_text_not_raised__(raw):
    status, body = DNSEClient._parse((200, raw))
    assert status == 200
    assert body == raw, "malformed JSON must be left as the original text, not raised or dropped"
    assert isinstance(body, str), "must stay a string, never coerced to another type"


@pytest.mark.parametrize("raw", [
    pytest.param("", id="empty-string"),
    pytest.param("   ", id="spaces-only"),
    pytest.param("\t\n", id="tabs-and-newline"),
])
def __test_parse_empty_or_whitespace_string_left_as_is_not_none__(raw):
    status, body = DNSEClient._parse((200, raw))
    assert status == 200
    assert body == raw, "empty/whitespace body must be preserved verbatim"
    assert body is not None, "must NOT be coerced to None"


def __test_parse_json_null_literal_parses_to_none__():
    """Distinct from the empty-string case above: the 4-character string
    "null" is valid JSON and legitimately parses to Python None."""
    status, body = DNSEClient._parse((200, "null"))
    assert status == 200
    assert body is None


@pytest.mark.parametrize("body", [
    pytest.param({"already": "parsed"}, id="already-dict"),
    pytest.param(42, id="int-body"),
    pytest.param(None, id="none-body"),
])
def __test_parse_non_string_non_bytes_body_left_untouched__(body):
    status, parsed = DNSEClient._parse((200, body))
    assert status == 200
    assert parsed == body
    assert parsed is body, "non-str/bytes bodies must not be rebuilt or mutated"


def __test_parse_handles_large_json_body__():
    """Boundary: a large payload must round-trip fully, not get truncated."""
    raw = json.dumps({"items": list(range(10_000))})
    status, body = DNSEClient._parse((200, raw))
    assert status == 200
    assert len(body["items"]) == 10_000, "large body must parse completely, not be truncated"
    assert body["items"][-1] == 9999


# --- __getattr__ delegation ---------------------------------------------------

def __test_getattr_delegates_sdk_method_and_parses_result__(monkeypatch):
    c = DNSEClient("key", "secret")
    monkeypatch.setattr(c._sdk, "get_accounts", lambda *a, **k: (200, '{"x": 1}'))

    status, body = c.get_accounts()

    assert status == 200, "delegated call must return the SDK's status"
    assert body == {"x": 1}, "delegated call's raw JSON string must come back parsed"


def __test_getattr_forwards_args_and_kwargs_to_sdk_method__(monkeypatch):
    c = DNSEClient("key", "secret")
    received = {}

    def _fake_cancel_order(account_no, order_id, market_type, trading_token,
                            order_category=None, dry_run=False):
        received["positional"] = (account_no, order_id, market_type, trading_token)
        received["order_category"] = order_category
        return 200, '{"orderStatus": "Canceled"}'

    monkeypatch.setattr(c._sdk, "cancel_order", _fake_cancel_order)

    status, body = c.cancel_order("ACC1", "123", "DERIVATIVE", "tok", order_category="STOP")

    assert received["positional"] == ("ACC1", "123", "DERIVATIVE", "tok"), \
        "positional args must reach the SDK method unchanged"
    assert received["order_category"] == "STOP", "kwargs must reach the SDK method unchanged"
    assert (status, body) == (200, {"orderStatus": "Canceled"})


def __test_getattr_underscore_name_raises_immediately_without_reaching_sdk__(monkeypatch):
    c = DNSEClient("key", "secret")
    # If delegation happened anyway, this would be what comes back -- proving
    # the early-return guard, not a lucky AttributeError from the SDK itself.
    monkeypatch.setattr(c._sdk, "_secret_field", "leaked-if-delegated", raising=False)

    with pytest.raises(AttributeError):
        c._secret_field


def __test_getattr_noncallable_sdk_attribute_returned_unwrapped__(monkeypatch):
    c = DNSEClient("key", "secret")
    sentinel = {"already": "a value, not a (status, body) tuple"}
    monkeypatch.setattr(c._sdk, "custom_field", sentinel, raising=False)

    result = c.custom_field

    assert result is sentinel, "non-callable SDK attributes must be returned as-is, untouched by _parse"
    assert not callable(result)


# --- TLS-verify swap (security) ---------------------------------------------

def __test_tls_verify_swap_enforces_cert_verification__():
    c = DNSEClient("key", "secret")
    http = c._sdk._http

    assert isinstance(http, urllib3.PoolManager), "must swap in a urllib3 PoolManager"
    pool_kwargs = http.connection_pool_kw
    assert pool_kwargs.get("cert_reqs") == "CERT_REQUIRED", (
        f"TLS verification must be enforced, got cert_reqs={pool_kwargs.get('cert_reqs')!r}"
    )
    assert pool_kwargs.get("ca_certs") == certifi.where(), "ca_certs must point at certifi's bundle"


def __test_tls_verify_swap_overrides_sdk_insecure_default__():
    """Before/after: the vendored SDK's own default is unverified
    (``cert_reqs=CERT_NONE``, ``assert_hostname=False``); the wrapper must
    replace it, not merely coexist with it."""
    from pynecore_dnse._sdk import DNSEClient as RawSdkClient

    raw = RawSdkClient("key", "secret")
    assert raw._http.connection_pool_kw.get("cert_reqs") == "CERT_NONE", \
        "sanity check: confirms the SDK's own default really is insecure"

    wrapped = DNSEClient("key", "secret")
    assert wrapped._sdk._http.connection_pool_kw.get("cert_reqs") == "CERT_REQUIRED", \
        "wrapper must override the SDK's insecure default"
