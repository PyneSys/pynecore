"""Phase A1 repro (#67) — transport failures must reach the engine CLASSIFIED,
never as raw urllib3 exceptions.

The vendored SDK's ``_request`` catches only ``HTTPError``-with-response;
timeouts and connection failures RAISE RAW through the wrapper into the
broker (verified: client.py's ``wrapped()`` has no try, and
``errors.classify``'s ``status==0`` "no response" sentinel is DEAD code —
nothing ever produces it). Consequences, red-proven here:

- a read-timeout on a POST is a LOST REPLY — the order may exist at the
  venue — and instead of ``OrderDispositionUnknownError`` (the engine's
  park-and-verify contract for exactly this), the raw urllib3 error
  propagates and kills the run (round-2 item 1: "a client timeout KILLS
  the run");
- a connection failure on a read escapes ``get_open_orders`` unclassified,
  missing the engine's ``except ExchangeConnectionError`` reconnect path;
- the same raw class crashes the cancel-disposition core instead of
  resolving to UNKNOWN.

The GREEN control pins that classified HTTP replies are untouched.
"""
import asyncio

import pytest
import urllib3
import pynecore.lib as lib

lib.bar_index = 0

from pynecore_dnse import broker
from pynecore.core.broker.models import (
    CancelDispositionOutcome, CancelIntent, DispatchEnvelope, EntryIntent, OrderType,
)
from pynecore.core.broker.exceptions import (
    ExchangeConnectionError, OrderDispositionUnknownError,
)

_SECDEF_ROW = [{"ceilingPrice": "1550", "floorPrice": "1450", "securityGroupId": "FU"}]
_LOAN_OK = (200, {"loanPackages": [{"id": 42}]})


def _broker(fake_client, tmp_path, **client_responses):
    responses = {"get_security_definition": (200, _SECDEF_ROW), "get_loan_packages": _LOAN_OK}
    responses.update(client_responses)
    config = broker.DNSEBrokerConfig(
        api_key="k", api_secret="s", account_no="ACC001", trading_token="tok-A",
        token_file=str(tmp_path / "missing_token.json"))
    instance = broker.DNSEBroker(symbol="VN30F1M", timeframe="15", config=config)
    instance._client = fake_client(**responses)
    instance._cancel_verify_attempts = 1
    instance._cancel_verify_delay = 0.0
    return instance


def _entry_envelope():
    return DispatchEnvelope(
        intent=EntryIntent(pine_id="L", symbol="VN30F1M", side="buy", qty=1,
                           order_type=OrderType.LIMIT, limit=1500.0),
        run_tag="abcd", bar_ts_ms=1_700_000_000_000, retry_seq=0, coid_max_len=30)


def _read_timeout(*_args, **_kwargs):
    # POST is not retried by urllib3 (POST not in Retry.DEFAULT_ALLOWED_METHODS),
    # so post_order raises the BARE leaf.
    raise urllib3.exceptions.ReadTimeoutError(None, "https://x", "read timed out")


def _connect_fail_wrapped(*_args, **_kwargs):
    # GET/PUT/DELETE are retried, so they raise MaxRetryError wrapping the leaf
    # in .reason (the shape the #67 panel proved the old baseline missed).
    raise urllib3.exceptions.MaxRetryError(
        None, "https://x",
        reason=urllib3.exceptions.NewConnectionError(None, "connection refused"))


def _read_timeout_wrapped(*_args, **_kwargs):
    raise urllib3.exceptions.MaxRetryError(
        None, "https://x",
        reason=urllib3.exceptions.ReadTimeoutError(None, "https://x", "read timed out"))


# --- RED 1: a lost POST reply must park-and-verify, never kill the run -------

def __test_post_timeout_is_a_lost_reply_not_a_crash__(fake_client, tmp_path):
    """The POST may have REACHED the venue before the reply was lost — the
    order can exist with no local record. The engine's contract for exactly
    this is OrderDispositionUnknownError (park the dispatch + verify);
    a raw urllib3 exception kills the run instead."""
    b = _broker(fake_client, tmp_path, post_order=_read_timeout)

    with pytest.raises(OrderDispositionUnknownError):
        asyncio.run(b.execute_entry(_entry_envelope()))


# --- RED 2: a read connection failure must hit the reconnect path ------------

def __test_read_connection_failure_is_classified_for_reconnect__(fake_client, tmp_path):
    """`get_open_orders` raising raw urllib3 misses every engine
    `except ExchangeConnectionError` (reconcile skip, reconnect) — the run
    crashes on a network blip instead of retrying."""
    b = _broker(fake_client, tmp_path, get_orders=_connect_fail_wrapped)

    with pytest.raises(ExchangeConnectionError):
        asyncio.run(b.get_open_orders("VN30F1M"))


def __test_positions_connection_failure_is_classified_for_reconnect__(fake_client, tmp_path):
    b = _broker(fake_client, tmp_path, get_positions=_connect_fail_wrapped)

    with pytest.raises(ExchangeConnectionError):
        asyncio.run(b.get_position("VN30F1M"))


# --- RED 3: the cancel-disposition core must resolve, not crash --------------

def __test_cancel_transport_failure_resolves_unknown__(fake_client, tmp_path):
    """A cancel write that dies in transport has an unknown disposition —
    the #55 core must answer UNKNOWN (engine retries), not propagate raw."""
    b = _broker(fake_client, tmp_path, cancel_order=_read_timeout_wrapped)
    b._order_ids["K"] = ["ID1"]
    b._order_category["ID1"] = "NORMAL"

    envelope = DispatchEnvelope(
        intent=CancelIntent(pine_id="K", symbol="VN30F1M"),
        run_tag="abcd", bar_ts_ms=1_700_000_000_000, retry_seq=0, coid_max_len=30)
    outcome = asyncio.run(b.execute_cancel_with_outcome(envelope))

    assert outcome is CancelDispositionOutcome.UNKNOWN


# --- GREEN control: classified HTTP replies are untouched --------------------

def __test_http_replies_unaffected__(fake_client, tmp_path):
    b = _broker(fake_client, tmp_path,
                post_order=(201, {"id": "1", "symbol": "VN30F1M", "side": "NB",
                                  "quantity": 1, "orderStatus": "New"}))

    orders = asyncio.run(b.execute_entry(_entry_envelope()))

    assert orders[0].id == "1"


# --- amend (PUT, MaxRetryError-wrapped) — panel P1's uncovered channel -------

def __test_amend_transport_failure_parks__(fake_client, tmp_path):
    b = _broker(fake_client, tmp_path, put_order=_connect_fail_wrapped)
    b._order_ids["L"] = ["ID1"]
    b._order_category["ID1"] = "NORMAL"
    with pytest.raises(OrderDispositionUnknownError):
        asyncio.run(b.modify_entry(_entry_envelope(), _entry_envelope()))


# --- bar feed (get_ohlc, GET) — engine-hard-called; a blip must not crash ----

def __test_bar_feed_transport_failure_is_classified__(fake_client, tmp_path):
    """provider/broker get_ohlc has no try today (broker.py:316/380/505); a raw
    blip in the engine-hard-called bar feed kills the whole feed. The guarded
    call returns the sentinel, so the loop sees a non-200 and retries."""
    from pynecore_dnse.transport_errors import guard, SENTINEL_STATUS
    status, body = guard(_connect_fail_wrapped)
    assert status == SENTINEL_STATUS and body["code"] == "NO_RESPONSE"
    # and a REAL guarded bar-feed read (the sync tick-close path at broker.py:380)
    # returns None on the sentinel instead of raising the raw urllib3 error:
    b = _broker(fake_client, tmp_path, get_ohlc=_connect_fail_wrapped)
    b._tick_slot = 1_700_000_000
    assert b._tick_fetch_official_close(900) is None


# --- pure taxonomy: both shapes, phase is metadata ---------------------------

def __test_sentinel_maps_leaf_and_wrapped__():
    from pynecore_dnse.transport_errors import to_sentinel
    leaf = urllib3.exceptions.ReadTimeoutError(None, "https://x", "t")
    wrapped = urllib3.exceptions.MaxRetryError(
        None, "https://x",
        reason=urllib3.exceptions.NewConnectionError(None, "refused"))
    s_leaf = to_sentinel(leaf)
    s_wrapped = to_sentinel(wrapped)
    assert s_leaf[0] == 0 and s_leaf[1]["phase"] == "sent"
    assert s_wrapped[0] == 0 and s_wrapped[1]["phase"] == "connect"
    assert s_wrapped[1]["transport"] == "NewConnectionError"   # unwrapped .reason


def __test_guard_does_not_swallow_non_transport_errors__():
    """The guard converts ONLY urllib3 transport exceptions — a real bug
    (e.g. a KeyError) must propagate, never masquerade as 'no response'."""
    from pynecore_dnse.transport_errors import guard
    with pytest.raises(KeyError):
        guard(lambda: (_ for _ in ()).throw(KeyError("real bug")))
