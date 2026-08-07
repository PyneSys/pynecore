"""Tests for :class:`DNSEBroker`'s OCO-resolution, amend, cancel wrappers,
and account/token plumbing (``broker.py``).

Covers ``_resolve_oco_lo`` (the synchronous poll loop that finds an OCO's
spawned working LO), ``_amend``/``modify_entry``/``modify_exit`` (price
precedence, tracked-id-missing fallback, non-dict body, error-path wiring),
``execute_cancel``/``execute_cancel_with_outcome`` (no-id / multi-id / the
first-id-only asymmetry), ``account_id`` (config vs. resolved, caching,
malformed-body handling), and ``_token`` (state-file precedence, malformed
JSON, OSError, and the final ``RuntimeError``).

Same fake-client seam as ``test_errors.py`` / ``test_broker_orders.py``: a
real ``DNSEBroker`` instance is built with a tiny in-memory config and
``broker._client`` is swapped for a canned :class:`_FakeClient`. No live
network, no real filesystem (the trading-token state file always lives
under ``tmp_path`` or is deliberately absent). ``time.sleep`` is
monkeypatched wherever ``_resolve_oco_lo``'s poll loop runs, so tests never
actually wait. Test functions use the repo convention ``__test_*__`` (see
``pytest.ini``).
"""
import asyncio
import json

import pytest
import pynecore.lib as lib

lib.bar_index = 0  # let the [BROKER] log formatter render during broker._emit()

from pynecore_dnse import broker
from pynecore.core.broker.models import (
    EntryIntent, ExitIntent, CancelIntent, DispatchEnvelope, LegType, OrderType,
    OrderStatus, CancelDispositionOutcome,
)
from pynecore.core.broker.exceptions import (
    ExchangeOrderRejectedError, OrderDispositionUnknownError,
)

#: A derivatives secdef row so ``market_type`` resolves to DERIVATIVE without
#: an extra round-trip, and a resolvable loan package for every ``_place`` call.
_SECDEF_ROW = [{"ceilingPrice": "1550", "floorPrice": "1450", "securityGroupId": "FU"}]
_LOAN_OK = (200, {"loanPackages": [{"id": 42}]})


def _broker(fake_client, tmp_path, *, account_no="ACC001", trading_token="tok-A",
           token_file=None, **client_responses):
    """A ``DNSEBroker`` wired to a fake client — no network, no real files."""
    responses = {"get_security_definition": (200, _SECDEF_ROW), "get_loan_packages": _LOAN_OK}
    responses.update(client_responses)
    config = broker.DNSEBrokerConfig(
        api_key="k", api_secret="s", account_no=account_no, trading_token=trading_token,
        token_file=token_file if token_file is not None else str(tmp_path / "missing_token.json"))
    instance = broker.DNSEBroker(symbol="VN30F1M", timeframe="15", config=config)
    instance._client = fake_client(**responses)
    return instance


def _bare_broker(tmp_path, *, account_no="ACC001", trading_token="tok-A", token_file=None):
    """A ``DNSEBroker`` with no client attached — for config/token-only tests."""
    config = broker.DNSEBrokerConfig(
        api_key="k", api_secret="s", account_no=account_no, trading_token=trading_token,
        token_file=token_file if token_file is not None else str(tmp_path / "missing_token.json"))
    return broker.DNSEBroker(symbol="VN30F1M", timeframe="15", config=config)


def _envelope(intent, *, run_tag="abcd", bar_ts_ms=1_700_000_000_000, retry_seq=0,
             coid_max_len=30):
    return DispatchEnvelope(intent=intent, run_tag=run_tag, bar_ts_ms=bar_ts_ms,
                            retry_seq=retry_seq, coid_max_len=coid_max_len)


def _put_calls(client):
    return [c for c in client.calls if c[0] == "put_order"]


# === _resolve_oco_lo =========================================================

def _immediate_oco_detail(account, order_id, market, order_category=None):
    """externalOrderId is present on the very first OCO poll."""
    if order_category == "OCO":
        return (200, {"externalOrderId": "LO-IMMEDIATE"})
    return (200, {"id": "LO-IMMEDIATE", "orderStatus": "New", "quantity": 4})


def _late_oco_detail(appears_at, lo_id="LO-LATE"):
    """externalOrderId appears only on the ``appears_at``-th OCO poll (1-indexed)."""
    counter = {"n": 0}

    def _detail(account, order_id, market, order_category=None):
        if order_category == "OCO":
            counter["n"] += 1
            if counter["n"] >= appears_at:
                return (200, {"externalOrderId": lo_id})
            return (200, {"id": order_id})
        return (200, {"id": lo_id, "orderStatus": "New", "quantity": 3})

    return _detail


def _never_oco_detail(account, order_id, market, order_category=None):
    """externalOrderId never appears; the LO-detail book must never be touched."""
    if order_category == "OCO":
        return (200, {"id": order_id})
    raise AssertionError("LO detail must never be fetched when externalOrderId never appears")


def _nondict_lo_detail(account, order_id, market, order_category=None):
    """externalOrderId resolves immediately, but the LO's own detail is malformed."""
    if order_category == "OCO":
        return (200, {"externalOrderId": "LO-BAD"})
    return (200, "not-a-dict-body")


def __test_resolve_oco_lo_external_id_on_first_attempt__(fake_client, tmp_path, monkeypatch):
    sleeps = []
    monkeypatch.setattr(broker.time, "sleep", lambda seconds: sleeps.append(seconds))
    b = _broker(fake_client, tmp_path, get_order_detail=_immediate_oco_detail)

    result = b._resolve_oco_lo("OCO-1")

    assert result is not None, "externalOrderId present on the first poll must resolve"
    assert result.id == "LO-IMMEDIATE", "resolved order must carry the LO's own id, not the OCO id"
    assert b._client.count("get_order_detail") == 2, \
        "exactly one OCO poll + one LO-detail fetch, no extra polling"
    assert sleeps == [], "a first-attempt resolution must not sleep at all"


def __test_resolve_oco_lo_external_id_appears_late__(fake_client, tmp_path, monkeypatch):
    sleeps = []
    monkeypatch.setattr(broker.time, "sleep", lambda seconds: sleeps.append(seconds))
    b = _broker(fake_client, tmp_path, get_order_detail=_late_oco_detail(appears_at=5))

    result = b._resolve_oco_lo("OCO-2", attempts=6, delay=0.15)

    assert result is not None, "externalOrderId appearing on attempt 5 (within attempts=6) must resolve"
    assert result.id == "LO-LATE"
    oco_polls = sum(1 for c in b._client.calls
                    if c[0] == "get_order_detail" and c[2].get("order_category") == "OCO")
    assert oco_polls == 5, "must stop polling exactly at the attempt where externalOrderId appears"
    assert len(sleeps) == 4, "sleeps between the 4 unresolved attempts only, none after resolving"


def __test_resolve_oco_lo_never_appears_returns_none__(fake_client, tmp_path, monkeypatch):
    sleeps = []
    monkeypatch.setattr(broker.time, "sleep", lambda seconds: sleeps.append(seconds))
    b = _broker(fake_client, tmp_path, get_order_detail=_never_oco_detail)

    result = b._resolve_oco_lo("OCO-3", attempts=6, delay=0.15)

    assert result is None, "externalOrderId never appearing must give up and return None, not raise"
    assert b._client.count("get_order_detail") == 6, "must exhaust exactly the 6-attempt budget"
    assert len(sleeps) == 6, "every exhausted attempt sleeps once before the next poll"


def __test_resolve_oco_lo_nondict_lo_detail_falls_back_to_bare_order__(fake_client, tmp_path,
                                                                        monkeypatch):
    monkeypatch.setattr(broker.time, "sleep", lambda seconds: None)
    b = _broker(fake_client, tmp_path, get_order_detail=_nondict_lo_detail)

    result = b._resolve_oco_lo("OCO-4")

    assert result is not None, "a resolved externalOrderId must still produce an order"
    assert result.id == "LO-BAD", \
        "falls back to _to_exchange_order({'id': lo_id}) when the LO detail body is not a dict"
    assert result.status is OrderStatus.PENDING, "the bare fallback carries no status -> default PENDING"


def __test_place_oco_keeps_umbrella_when_resolve_never_completes__(fake_client, tmp_path,
                                                                    monkeypatch):
    """When ``_resolve_oco_lo`` gives up, ``_place`` must keep tracking the OCO
    umbrella order itself (id + category) rather than crashing or losing it."""
    monkeypatch.setattr(broker.time, "sleep", lambda seconds: None)
    b = _broker(fake_client, tmp_path,
               post_order=(201, {"id": "OCO-UMBRELLA", "symbol": "VN30F1M", "side": "NS",
                                  "quantity": 2, "orderStatus": "New"}),
               get_order_detail=_never_oco_detail)
    envelope = _envelope(ExitIntent(pine_id="TP", from_entry="L", symbol="VN30F1M", side="sell",
                                    qty=2, tp_price=110.0, sl_price=90.0))

    result = b._place(envelope, "sell", 2, price=110.0, category="OCO",
                      stop_price=90.0, stop_order_price=90.0, leg_type=LegType.TAKE_PROFIT)

    assert result[0].id == "OCO-UMBRELLA", "keeps the umbrella OCO id when the LO never resolves"
    assert b._order_category["OCO-UMBRELLA"] == "OCO", \
        "tracked_category must stay OCO (never swapped to NORMAL) when re-tracking failed"


# === _amend / modify_entry / modify_exit =====================================

@pytest.mark.parametrize("intent, expected_price, order_id", [
    pytest.param(EntryIntent(pine_id="P1", symbol="VN30F1M", side="buy", qty=2,
                             order_type=OrderType.STOP, limit=101.5, stop=99.0),
                 101.5, "ORD-E1", id="entry-limit-wins-over-stop"),
    pytest.param(EntryIntent(pine_id="P1", symbol="VN30F1M", side="buy", qty=2,
                             order_type=OrderType.STOP, limit=None, stop=99.0),
                 99.0, "ORD-E2", id="entry-stop-wins-when-no-limit"),
    pytest.param(ExitIntent(pine_id="X1", from_entry="P1", symbol="VN30F1M", side="sell",
                            qty=2, tp_price=120.0, sl_price=90.0),
                 120.0, "ORD-X1", id="exit-tp-wins-over-sl"),
    pytest.param(ExitIntent(pine_id="X1", from_entry="P1", symbol="VN30F1M", side="sell",
                            qty=2, tp_price=None, sl_price=90.0),
                 90.0, "ORD-X2", id="exit-sl-wins-when-no-tp"),
])
def __test_amend_price_precedence_limit_or_stop_or_tp_or_sl__(
        fake_client, tmp_path, intent, expected_price, order_id):
    b = _broker(fake_client, tmp_path,
               put_order=(200, {"id": order_id, "orderStatus": "New", "quantity": 2}))
    key = intent.intent_key
    b._order_ids[key] = [order_id]
    b._order_category[order_id] = "NORMAL"
    is_exit = isinstance(intent, ExitIntent)

    result = asyncio.run(b._amend(_envelope(intent), _envelope(intent), is_exit=is_exit))

    assert result[0].id == order_id, "amend must return the amended order"
    calls = _put_calls(b._client)
    assert calls[-1][1][3]["price"] == expected_price, (
        f"price precedence (limit or stop or tp_price or sl_price) must pick "
        f"{expected_price}, payload was {calls[-1][1][3]}")


def __test_amend_price_falls_back_to_zero_when_no_price_field_set__(fake_client, tmp_path):
    """Documents current behavior when NONE of limit/stop/tp_price/sl_price is
    set on the new intent (e.g. a bare-market entry amend): ``_amend`` sends
    ``price: 0.0`` to the venue instead of omitting the field or refusing.
    Flagging as a money-path risk for review — see the writer's final report."""
    intent = EntryIntent(pine_id="P1", symbol="VN30F1M", side="buy", qty=2,
                         order_type=OrderType.MARKET)  # no limit, no stop
    b = _broker(fake_client, tmp_path,
               put_order=(200, {"id": "ORD-E3", "orderStatus": "New", "quantity": 2}))
    b._order_ids["P1"] = ["ORD-E3"]
    b._order_category["ORD-E3"] = "NORMAL"

    result = asyncio.run(b._amend(_envelope(intent), _envelope(intent), is_exit=False))

    assert result[0].id == "ORD-E3"
    calls = _put_calls(b._client)
    assert calls[-1][1][3]["price"] == 0.0, \
        "current behavior: no price field present -> payload price falls back to 0.0"


def __test_amend_no_tracked_id_falls_back_without_crashing__(fake_client, tmp_path):
    b = _broker(fake_client, tmp_path,
               post_order=(201, {"id": "NEW1", "symbol": "VN30F1M", "side": "NB",
                                  "quantity": 2, "orderStatus": "New"}))
    old = EntryIntent(pine_id="P1", symbol="VN30F1M", side="buy", qty=2,
                      order_type=OrderType.LIMIT, limit=100.0)
    new = EntryIntent(pine_id="P1", symbol="VN30F1M", side="buy", qty=2,
                      order_type=OrderType.LIMIT, limit=105.0)

    result = asyncio.run(b._amend(_envelope(old), _envelope(new), is_exit=False))

    assert result[0].id == "NEW1", "fallback must still return the freshly placed order"
    assert b._client.count("put_order") == 0, "no tracked id -> must not take the amend path"


def __test_modify_entry_no_tracked_id_falls_back_to_super_cancel_and_execute__(
        fake_client, tmp_path):
    b = _broker(fake_client, tmp_path,
               post_order=(201, {"id": "NEW1", "symbol": "VN30F1M", "side": "NB",
                                  "quantity": 2, "orderStatus": "New"}))
    old = _envelope(EntryIntent(pine_id="P1", symbol="VN30F1M", side="buy", qty=2,
                                order_type=OrderType.LIMIT, limit=100.0))
    new = _envelope(EntryIntent(pine_id="P1", symbol="VN30F1M", side="buy", qty=2,
                                order_type=OrderType.LIMIT, limit=105.0))

    result = asyncio.run(b.modify_entry(old, new))

    assert result[0].id == "NEW1", "the base cancel+execute fallback must still return an order"
    assert b._client.count("post_order") == 1, "fallback must dispatch a fresh entry via execute_entry"
    assert b._client.count("put_order") == 0, "fallback must never call the amend endpoint"


def __test_modify_exit_no_tracked_id_falls_back_to_super_cancel_and_execute__(
        fake_client, tmp_path):
    b = _broker(fake_client, tmp_path,
               post_order=(201, {"id": "NEW2", "symbol": "VN30F1M", "side": "NB",
                                  "quantity": 2, "orderStatus": "New"}))
    old = _envelope(ExitIntent(pine_id="X1", from_entry="P1", symbol="VN30F1M", side="sell",
                               qty=2, sl_price=90.0))
    new = _envelope(ExitIntent(pine_id="X1", from_entry="P1", symbol="VN30F1M", side="sell",
                               qty=2, sl_price=88.0))

    result = asyncio.run(b.modify_exit(old, new))

    assert result[0].id == "NEW2"
    assert b._client.count("post_order") == 1, "fallback must dispatch a fresh exit via execute_exit"
    assert b._client.count("put_order") == 0


def __test_modify_entry_with_tracked_id_amends_in_place__(fake_client, tmp_path):
    b = _broker(fake_client, tmp_path,
               put_order=(200, {"id": "ORD1", "orderStatus": "New", "quantity": 3}))
    b._order_ids["P1"] = ["ORD1"]
    b._order_category["ORD1"] = "NORMAL"
    old = _envelope(EntryIntent(pine_id="P1", symbol="VN30F1M", side="buy", qty=2,
                                order_type=OrderType.LIMIT, limit=100.0))
    new = _envelope(EntryIntent(pine_id="P1", symbol="VN30F1M", side="buy", qty=3,
                                order_type=OrderType.LIMIT, limit=103.0))

    result = asyncio.run(b.modify_entry(old, new))

    assert result[0].id == "ORD1"
    assert b._client.count("put_order") == 1, "a tracked id must take the atomic amend path"
    assert b._client.count("post_order") == 0, "must not cancel+replace when an amend id is tracked"


def __test_modify_exit_with_tracked_id_amends_in_place__(fake_client, tmp_path):
    b = _broker(fake_client, tmp_path,
               put_order=(200, {"id": "ORD2", "orderStatus": "New", "quantity": 2}))
    old_intent = ExitIntent(pine_id="X1", from_entry="P1", symbol="VN30F1M", side="sell",
                            qty=2, sl_price=90.0)
    new_intent = ExitIntent(pine_id="X1", from_entry="P1", symbol="VN30F1M", side="sell",
                            qty=2, sl_price=88.0)
    b._order_ids[old_intent.intent_key] = ["ORD2"]
    b._order_category["ORD2"] = "STOP"
    old = _envelope(old_intent)
    new = _envelope(new_intent)

    result = asyncio.run(b.modify_exit(old, new))

    assert result[0].id == "ORD2"
    assert b._client.count("put_order") == 1, "a tracked id must take the atomic amend path"
    assert b._client.count("post_order") == 0


def __test_amend_non_dict_success_body_raises_rejected__(fake_client, tmp_path):
    b = _broker(fake_client, tmp_path, put_order=(200, "oops-not-a-dict"))
    b._order_ids["P1"] = ["ORD1"]
    old = EntryIntent(pine_id="P1", symbol="VN30F1M", side="buy", qty=2,
                      order_type=OrderType.LIMIT, limit=100.0)
    new = EntryIntent(pine_id="P1", symbol="VN30F1M", side="buy", qty=2,
                      order_type=OrderType.LIMIT, limit=105.0)

    with pytest.raises(ExchangeOrderRejectedError) as exc_info:
        asyncio.run(b._amend(_envelope(old), _envelope(new), is_exit=False))

    assert "non-dict" in str(exc_info.value), "message must call out the malformed success body"
    assert b._client.count("put_order") == 1, "the write must actually have been attempted"


@pytest.mark.parametrize("status, body, exc_type", [
    (400, {"code": "INVALID_PRICE"}, ExchangeOrderRejectedError),
    (0, {}, OrderDispositionUnknownError),
])
def __test_amend_error_path_raises_via_raise_write_error__(fake_client, tmp_path, status, body,
                                                             exc_type):
    b = _broker(fake_client, tmp_path, put_order=(status, body))
    b._order_ids["P1"] = ["ORD1"]
    old = EntryIntent(pine_id="P1", symbol="VN30F1M", side="buy", qty=2,
                      order_type=OrderType.LIMIT, limit=100.0)
    new = EntryIntent(pine_id="P1", symbol="VN30F1M", side="buy", qty=2,
                      order_type=OrderType.LIMIT, limit=105.0)

    with pytest.raises(exc_type) as exc_info:
        asyncio.run(b._amend(_envelope(old), _envelope(new), is_exit=False))

    assert "amend" in str(exc_info.value), "the classified error must be tagged with the amend action"
    if exc_type is OrderDispositionUnknownError:
        assert exc_info.value.client_order_id == "ORD1", (
            "coid wiring: _amend must pass the venue order_id as the disposition-unknown coid")


# === execute_cancel / execute_cancel_with_outcome ============================

def _cancel_envelope(pine_id="K", from_entry=None):
    return _envelope(CancelIntent(pine_id=pine_id, symbol="VN30F1M", from_entry=from_entry))


def __test_execute_cancel_no_tracked_ids_returns_false_not_exception__(fake_client, tmp_path):
    b = _broker(fake_client, tmp_path)

    result = asyncio.run(b.execute_cancel(_cancel_envelope("NOPE")))

    assert result is False, "no tracked ids must return False, never raise"
    assert b._client.calls == [], "no client call should be attempted when nothing is tracked"


def __test_execute_cancel_with_outcome_no_tracked_ids_returns_unknown__(fake_client, tmp_path):
    b = _broker(fake_client, tmp_path)

    outcome = asyncio.run(b.execute_cancel_with_outcome(_cancel_envelope("NOPE")))

    assert outcome is CancelDispositionOutcome.UNKNOWN, \
        "no tracked ids must map to UNKNOWN, never raise"
    assert b._client.calls == [], "no client call should be attempted when nothing is tracked"


def __test_execute_cancel_multiple_ids_attempts_both_and_ands_results__(fake_client, tmp_path):
    def _cancel(account, order_id, market, token, order_category=None):
        if order_id == "ID1":
            return (200, {"orderStatus": "Canceled"})
        return (400, {"code": "CANNOT_CANCEL_THE_ORDER_IN_THE_ATO_SESSION"})  # session-refused

    b = _broker(fake_client, tmp_path, cancel_order=_cancel)
    b._order_ids["K"] = ["ID1", "ID2"]
    b._order_category["ID1"] = "STOP"
    b._order_category["ID2"] = "NORMAL"

    result = asyncio.run(b.execute_cancel(_cancel_envelope("K")))

    assert result is False, "AND of True (ID1 cancelled) and False (ID2 session-refused) must be False"
    cancelled_ids = [c[1][1] for c in b._client.calls if c[0] == "cancel_order"]
    assert cancelled_ids == ["ID1", "ID2"], "both legs of the bracket must be attempted"


def __test_execute_cancel_with_outcome_uses_only_first_id__(fake_client, tmp_path):
    def _cancel(account, order_id, market, token, order_category=None):
        assert order_id == "ID1", \
            f"execute_cancel_with_outcome must not touch id {order_id!r} — only the first id"
        return (200, {"orderStatus": "Canceled"})

    b = _broker(fake_client, tmp_path, cancel_order=_cancel)
    b._order_ids["K"] = ["ID1", "ID2"]
    b._order_category["ID1"] = "STOP"

    outcome = asyncio.run(b.execute_cancel_with_outcome(_cancel_envelope("K")))

    assert outcome is CancelDispositionOutcome.CANCEL_CONFIRMED
    assert b._client.count("cancel_order") == 1, "only the first id's cancel should be attempted"


# === account_id ===============================================================

def __test_account_id_uses_config_value_and_caches__(fake_client, tmp_path):
    def _boom(*a, **k):
        raise AssertionError("get_accounts must not be called when config.account_no is set")

    b = _bare_broker(tmp_path, account_no="ACC1")
    b._client = fake_client(get_accounts=_boom)

    first = b.account_id
    second = b.account_id

    assert first == "ACC1" and second == "ACC1", "config.account_no must be used verbatim"
    assert b._client.calls == [], "get_accounts must never be called when account_no is preset"


def __test_account_id_resolves_via_get_accounts_and_caches__(fake_client, tmp_path):
    b = _bare_broker(tmp_path, account_no="")
    b._client = fake_client(get_accounts=(200, {"accounts": [{"id": "RESOLVED1"}]}))

    first = b.account_id
    second = b.account_id

    assert first == "RESOLVED1" and second == "RESOLVED1", \
        "must resolve to body['accounts'][0]['id']"
    assert b._client.count("get_accounts") == 1, "second access must be served from the cache"


@pytest.mark.parametrize("status, body", [
    (500, {"code": "REMOTE_SERVER_ERROR"}),
    (200, "not-a-dict"),
    (0, {}),
])
def __test_account_id_raises_runtime_error_on_non200_or_non_dict__(fake_client, tmp_path,
                                                                     status, body):
    b = _bare_broker(tmp_path, account_no="")
    b._client = fake_client(get_accounts=(status, body))

    with pytest.raises(RuntimeError) as exc_info:
        _ = b.account_id

    assert "cannot resolve account" in str(exc_info.value), \
        "the RuntimeError must name the failing operation for operator diagnosis"
    assert str(status) in str(exc_info.value)


@pytest.mark.parametrize("body", [{}, {"accounts": []}])
def __test_account_id_missing_or_empty_accounts_raises_runtimeerror__(fake_client, tmp_path, body):
    """A 200 whose body lacks (or empties) "accounts" must raise the documented
    RuntimeError, not a raw KeyError/IndexError — the guard now covers the
    missing/empty-accounts shape, not just non-200/non-dict."""
    b = _bare_broker(tmp_path, account_no="")
    b._client = fake_client(get_accounts=(200, body))

    with pytest.raises(RuntimeError):
        _ = b.account_id


# === _token ====================================================================

def __test_token_state_file_wins_over_config__(tmp_path):
    token_file = tmp_path / "token.json"
    token_file.write_text(json.dumps({"trading_token": "FILE_TOKEN"}))
    b = _bare_broker(tmp_path, trading_token="CONFIG_TOKEN", token_file=str(token_file))

    result = b._token()

    assert result == "FILE_TOKEN", "a present, valid state file must win over the config fallback"
    assert result != "CONFIG_TOKEN"


def __test_token_missing_file_falls_back_to_config__(tmp_path):
    b = _bare_broker(tmp_path, trading_token="CONFIG_TOKEN",
                     token_file=str(tmp_path / "does_not_exist.json"))

    result = b._token()

    assert result == "CONFIG_TOKEN", "an absent state file must fall back to config.trading_token"


def __test_token_malformed_json_is_caught_and_falls_back_to_config__(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json")
    b = _bare_broker(tmp_path, trading_token="CONFIG_TOKEN", token_file=str(bad))

    result = b._token()

    assert result == "CONFIG_TOKEN", "malformed JSON must be caught (ValueError), not propagate"


def __test_token_oserror_reading_file_is_caught_and_falls_back_to_config__(tmp_path):
    """A directory at the token_file path makes ``read_text()`` raise
    ``IsADirectoryError`` (an ``OSError`` subclass) -- the exact class of
    failure the ``except (ValueError, OSError)`` guard exists for."""
    as_directory = tmp_path / "token_dir"
    as_directory.mkdir()
    b = _bare_broker(tmp_path, trading_token="CONFIG_TOKEN", token_file=str(as_directory))

    result = b._token()

    assert result == "CONFIG_TOKEN", "an OSError while reading the state file must be caught"


@pytest.mark.parametrize("file_contents", [
    {},                        # no "trading_token" key at all
    {"trading_token": ""},     # key present but falsy
])
def __test_token_valid_json_without_usable_token_falls_back_to_config__(tmp_path, file_contents):
    present_but_empty = tmp_path / "empty_token.json"
    present_but_empty.write_text(json.dumps(file_contents))
    b = _bare_broker(tmp_path, trading_token="CONFIG_TOKEN", token_file=str(present_but_empty))

    result = b._token()

    assert result == "CONFIG_TOKEN", \
        "a falsy/missing trading_token in an otherwise-valid file must still fall back"


def __test_token_neither_file_nor_config_raises_with_guidance__(tmp_path):
    b = _bare_broker(tmp_path, trading_token="",
                     token_file=str(tmp_path / "does_not_exist.json"))

    with pytest.raises(RuntimeError, match="no trading_token"):
        b._token()
