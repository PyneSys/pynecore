"""Tests for the DNSE broker's ORDER-CONSTRUCTION path (``broker.py``).

Covers ``execute_entry`` / ``execute_exit`` / ``execute_close`` routing,
``_place`` payload construction (rounding, qty truncation, lazy+cached
``loanPackageId``, OCO re-tracking, non-dict success body, bookkeeping),
``_write`` token-retry, ``_to_exchange_order`` status-map + numeric
coercions, and the small standalone helpers (``_marketable_price``,
``_gtd``, ``_loan_package_id``, ``get_capabilities``, connection lifecycle).

Same fake-client seam as ``test_errors.py``: a per-instance
``broker._client`` intercepts the whole REST surface, so no live network and
no real filesystem I/O (the trading-token state file path always points at a
non-existent file under ``tmp_path``).
"""
import asyncio
from datetime import datetime, timedelta, timezone

import pytest

import pynecore.lib as lib

lib.bar_index = 0  # let the [BROKER] log formatter render during broker._emit()

from pynecore_dnse import broker
from pynecore.core.broker.models import (
    EntryIntent, ExitIntent, CloseIntent, DispatchEnvelope, LegType, OrderType,
    OrderStatus, CapabilityLevel,
)
from pynecore.core.broker.exceptions import (
    AuthenticationError, ExchangeOrderRejectedError, InsufficientMarginError,
    OrderSkippedByPlugin,
)

#: A derivatives secdef row with both bands set — the default every broker in
#: this file gets, so ``market_type`` resolves to DERIVATIVE and
#: ``_marketable_price`` never has to raise unless a test deliberately wants it to.
_SECDEF_ROW = [{"ceilingPrice": "1550", "floorPrice": "1450", "securityGroupId": "FU"}]
_LOAN_OK = (200, {"loanPackages": [{"id": 42}]})


def _broker(fake_client, tmp_path, **client_responses):
    """A ``DNSEBroker`` wired to a fake client — no network, no real files.

    ``token_file`` always points at a path that does not exist, so ``_token``
    falls through to the config's ``trading_token`` deterministically.
    """
    responses = {"get_security_definition": (200, _SECDEF_ROW), "get_loan_packages": _LOAN_OK}
    responses.update(client_responses)
    config = broker.DNSEBrokerConfig(
        api_key="k", api_secret="s", account_no="ACC001", trading_token="tok-A",
        token_file=str(tmp_path / "missing_token.json"))
    b = broker.DNSEBroker(symbol="VN30F1M", timeframe="15", config=config)
    b._client = fake_client(**responses)
    return b


def _envelope(intent, *, run_tag="abcd", bar_ts_ms=1_700_000_000_000):
    return DispatchEnvelope(intent=intent, run_tag=run_tag, bar_ts_ms=bar_ts_ms)


def _last_call(client, name):
    """The most recent recorded call to ``name`` — ``(args, kwargs)``."""
    matches = [c for c in client.calls if c[0] == name]
    assert matches, f"{name} was never called"
    return matches[-1][1], matches[-1][2]


# --- execute_entry routing --------------------------------------------------

@pytest.mark.parametrize("side, expected_operator, dnse_side", [
    ("buy", ">=", "NB"), ("sell", "<=", "NS"),
])
def __test_execute_entry_stop_condition_operator__(fake_client, tmp_path, side,
                                                     expected_operator, dnse_side):
    b = _broker(fake_client, tmp_path, post_order=(
        201, {"id": "1", "symbol": "VN30F1M", "side": dnse_side, "quantity": 1,
              "orderStatus": "New"}))
    envelope = _envelope(EntryIntent(pine_id="L", symbol="VN30F1M", side=side, qty=1,
                                     order_type=OrderType.STOP, stop=1500.0))
    orders = asyncio.run(b.execute_entry(envelope))
    args, kwargs = _last_call(b._client, "post_order")
    payload = args[2]
    assert payload["conditionOperator"] == expected_operator, \
        f"{side} stop must use {expected_operator}"
    assert kwargs["order_category"] == "STOP"
    assert payload["durationType"] == "GTD"
    assert payload["durationDateTime"].endswith("Z"), "GTD expiry must be RFC3339 Zulu"
    assert orders[0].id == "1"


def __test_execute_entry_stop_with_limit_prices_at_limit__(fake_client, tmp_path):
    """A stop-limit entry (both ``limit`` and ``stop`` set) prices the STOP
    order at the limit — not the trigger — per ``execute_entry``."""
    b = _broker(fake_client, tmp_path, post_order=(
        201, {"id": "1", "symbol": "VN30F1M", "side": "NB", "quantity": 1,
              "orderStatus": "New"}))
    envelope = _envelope(EntryIntent(pine_id="L", symbol="VN30F1M", side="buy", qty=1,
                                     order_type=OrderType.STOP, limit=1495.0, stop=1500.0))
    asyncio.run(b.execute_entry(envelope))
    args, kwargs = _last_call(b._client, "post_order")
    payload = args[2]
    assert payload["price"] == 1495.0, "stop-limit prices the order at the LIMIT, not the trigger"
    assert payload["stopPrice"] == 1500.0
    assert kwargs["order_category"] == "STOP"


def __test_execute_entry_limit_only_is_normal_lo__(fake_client, tmp_path):
    b = _broker(fake_client, tmp_path, post_order=(
        201, {"id": "2", "symbol": "VN30F1M", "side": "NB", "quantity": 4,
              "orderStatus": "New"}))
    envelope = _envelope(EntryIntent(pine_id="L", symbol="VN30F1M", side="buy", qty=4,
                                     order_type=OrderType.LIMIT, limit=1490.0))
    orders = asyncio.run(b.execute_entry(envelope))
    args, kwargs = _last_call(b._client, "post_order")
    payload = args[2]
    assert payload["price"] == 1490.0
    assert kwargs["order_category"] == "NORMAL"
    assert "stopPrice" not in payload, "a plain LIMIT entry must not carry a stopPrice"
    assert orders[0].id == "2"


@pytest.mark.parametrize("side, expected_price, dnse_side", [
    ("buy", 1550.0, "NB"), ("sell", 1450.0, "NS"),
])
def __test_execute_entry_bare_market_uses_marketable_band__(fake_client, tmp_path, side,
                                                              expected_price, dnse_side):
    """A bare (no limit/stop) entry prices at the band edge — ceiling to buy, floor to sell."""
    b = _broker(fake_client, tmp_path, post_order=(
        201, {"id": "3", "symbol": "VN30F1M", "side": dnse_side, "quantity": 1,
              "orderStatus": "New"}))
    envelope = _envelope(EntryIntent(pine_id="L", symbol="VN30F1M", side=side, qty=1,
                                     order_type=OrderType.MARKET))
    asyncio.run(b.execute_entry(envelope))
    args, kwargs = _last_call(b._client, "post_order")
    payload = args[2]
    assert payload["price"] == expected_price
    assert kwargs["order_category"] == "NORMAL"


# --- execute_exit routing ---------------------------------------------------

def __test_execute_exit_tp_and_sl_is_native_oco__(fake_client, tmp_path):
    def get_order_detail(*args, order_category=None, **kwargs):
        if order_category == "OCO":
            return (200, {"externalOrderId": "9001"})
        return (200, {"id": "9001", "symbol": "VN30F1M", "side": "NS", "quantity": 3,
                      "orderStatus": "New"})

    b = _broker(fake_client, tmp_path,
               post_order=(201, {"id": "8000", "symbol": "VN30F1M", "side": "NS",
                                  "quantity": 3, "orderStatus": "New"}),
               get_order_detail=get_order_detail)
    envelope = _envelope(ExitIntent(pine_id="TP", from_entry="L", symbol="VN30F1M", side="sell",
                                    qty=3, tp_price=1520.5, sl_price=1480.5))
    orders = asyncio.run(b.execute_exit(envelope))
    args, kwargs = _last_call(b._client, "post_order")
    payload = args[2]
    assert kwargs["order_category"] == "OCO"
    assert payload["price"] == 1520.5 and payload["stopPrice"] == 1480.5, \
        "OCO payload must carry the TP limit and the SL trigger"
    assert payload["stopOrderPrice"] == 1480.5
    assert payload["durationType"] == "DAY"
    # Re-tracking: the OCO's spawned NORMAL LO replaces the umbrella order.
    assert orders[0].id == "9001", "must re-track to the spawned NORMAL LO id"
    assert b._order_category["9001"] == "NORMAL", "tracked_category swaps to NORMAL"
    assert b._identity["9001"] == ("TP", "L", LegType.TAKE_PROFIT)


def __test_execute_exit_sl_only_is_native_stop__(fake_client, tmp_path):
    b = _broker(fake_client, tmp_path, post_order=(
        201, {"id": "5", "symbol": "VN30F1M", "side": "NB", "quantity": 2, "orderStatus": "New"}))
    envelope = _envelope(ExitIntent(pine_id="X", from_entry="L", symbol="VN30F1M", side="sell",
                                    qty=2, sl_price=1440.0))
    orders = asyncio.run(b.execute_exit(envelope))
    args, kwargs = _last_call(b._client, "post_order")
    payload = args[2]
    assert kwargs["order_category"] == "STOP"
    assert payload["price"] == 1440.0 and payload["stopPrice"] == 1440.0
    assert orders[0].id == "5"


def __test_execute_exit_tp_only_is_normal_lo__(fake_client, tmp_path):
    b = _broker(fake_client, tmp_path, post_order=(
        201, {"id": "6", "symbol": "VN30F1M", "side": "NB", "quantity": 2, "orderStatus": "New"}))
    envelope = _envelope(ExitIntent(pine_id="X", from_entry="L", symbol="VN30F1M", side="sell",
                                    qty=2, tp_price=1560.0))
    orders = asyncio.run(b.execute_exit(envelope))
    args, kwargs = _last_call(b._client, "post_order")
    payload = args[2]
    assert kwargs["order_category"] == "NORMAL"
    assert payload["price"] == 1560.0
    assert "stopPrice" not in payload
    assert orders[0].id == "6"


@pytest.mark.xfail(
    strict=True, raises=TypeError,
    reason="PRODUCT BUG: broker.py execute_exit raises "
           "OrderSkippedByPlugin('...') with only the positional message, but "
           "OrderSkippedByPlugin.__init__ (pynecore.core.broker.exceptions) now "
           "requires a keyword-only `intent_key` argument -> TypeError instead "
           "of the intended graceful skip for a no-tp/no-sl exit (e.g. a "
           "trailing-only strategy.exit()). Do not weaken this assertion; fix "
           "is to pass intent_key=envelope.intent.intent_key at the call site.")
def __test_execute_exit_neither_tp_nor_sl_is_skipped__(fake_client, tmp_path):
    b = _broker(fake_client, tmp_path)
    envelope = _envelope(ExitIntent(pine_id="X", from_entry="L", symbol="VN30F1M", side="sell",
                                    qty=2))
    with pytest.raises(OrderSkippedByPlugin):
        asyncio.run(b.execute_exit(envelope))
    assert b._client.count("post_order") == 0, "no order may be sent when tp/sl are both absent"


# --- execute_close routing --------------------------------------------------

@pytest.mark.parametrize("side, expected_price, dnse_side", [
    ("buy", 1550.0, "NB"), ("sell", 1450.0, "NS"),
])
def __test_execute_close_uses_marketable_band_edge__(fake_client, tmp_path, side,
                                                       expected_price, dnse_side):
    b = _broker(fake_client, tmp_path, post_order=(
        201, {"id": "9", "symbol": "VN30F1M", "side": dnse_side, "quantity": 3,
              "orderStatus": "New"}))
    envelope = _envelope(CloseIntent(pine_id="L", symbol="VN30F1M", side=side, qty=3))
    order = asyncio.run(b.execute_close(envelope))
    args, kwargs = _last_call(b._client, "post_order")
    payload = args[2]
    assert payload["price"] == expected_price, "close must price at the band edge on the closing side"
    assert kwargs["order_category"] == "NORMAL"
    assert order.id == "9"


# --- _place mechanics --------------------------------------------------------

def __test_place_rounds_price_and_truncates_qty__(fake_client, tmp_path):
    b = _broker(fake_client, tmp_path, post_order=(
        201, {"id": "1", "symbol": "VN30F1M", "side": "NB", "quantity": 5, "orderStatus": "New"}))
    envelope = _envelope(EntryIntent(pine_id="L", symbol="VN30F1M", side="buy", qty=5.9,
                                     order_type=OrderType.LIMIT, limit=1500.34))
    asyncio.run(b.execute_entry(envelope))
    args, _ = _last_call(b._client, "post_order")
    payload = args[2]
    assert payload["price"] == 1500.3, f"price must round to 1dp, got {payload['price']}"
    assert payload["quantity"] == 5 and isinstance(payload["quantity"], int), \
        "quantity must TRUNCATE to int, not round"
    assert payload["loanPackageId"] == 42


def __test_place_caches_loan_package_id_across_calls__(fake_client, tmp_path):
    b = _broker(fake_client, tmp_path, post_order=(
        201, {"id": "1", "symbol": "VN30F1M", "side": "NB", "quantity": 1, "orderStatus": "New"}))
    envelope1 = _envelope(EntryIntent(pine_id="L1", symbol="VN30F1M", side="buy", qty=1,
                                      order_type=OrderType.LIMIT, limit=1500.0))
    envelope2 = _envelope(EntryIntent(pine_id="L2", symbol="VN30F1M", side="buy", qty=1,
                                      order_type=OrderType.LIMIT, limit=1500.0))
    asyncio.run(b.execute_entry(envelope1))
    asyncio.run(b.execute_entry(envelope2))
    assert b._client.count("get_loan_packages") == 1, \
        "loanPackageId must be resolved once and cached, not re-fetched per order"
    assert b._client.count("post_order") == 2


def __test_place_bookkeeping_recorded_on_success__(fake_client, tmp_path):
    b = _broker(fake_client, tmp_path, post_order=(
        201, {"id": "77", "symbol": "VN30F1M", "side": "NB", "quantity": 2,
              "orderStatus": "New"}))
    envelope = _envelope(EntryIntent(pine_id="Long", symbol="VN30F1M", side="buy", qty=2,
                                     order_type=OrderType.LIMIT, limit=1500.0))
    asyncio.run(b.execute_entry(envelope))
    assert b._order_ids["Long"] == ["77"], "the venue order id must be tracked under the intent key"
    assert b._identity["77"] == ("Long", None, LegType.ENTRY)
    assert b._order_category["77"] == "NORMAL"


@pytest.mark.parametrize("code, exc", [
    ("INVALID_PRICE", ExchangeOrderRejectedError),
    ("PURCHASING_POWER_NOT_ENOUGH", InsufficientMarginError),
])
def __test_place_propagates_classified_reject_without_bookkeeping__(fake_client, tmp_path,
                                                                      code, exc):
    b = _broker(fake_client, tmp_path, post_order=(400, {"code": code}))
    envelope = _envelope(EntryIntent(pine_id="L", symbol="VN30F1M", side="buy", qty=1,
                                     order_type=OrderType.LIMIT, limit=1500.0))
    with pytest.raises(exc):
        asyncio.run(b.execute_entry(envelope))
    assert b._order_ids == {}, "a rejected place must leave no order-id bookkeeping behind"
    assert b._identity == {}


def __test_place_rejects_non_dict_success_body__(fake_client, tmp_path):
    b = _broker(fake_client, tmp_path, post_order=(201, "OK"))
    envelope = _envelope(EntryIntent(pine_id="L", symbol="VN30F1M", side="buy", qty=1,
                                     order_type=OrderType.LIMIT, limit=1500.0))
    with pytest.raises(ExchangeOrderRejectedError):
        asyncio.run(b.execute_entry(envelope))
    assert b._order_ids == {}, "a non-dict success body must not be treated as a placed order"


def __test_place_oco_falls_back_to_umbrella_when_lo_never_appears__(fake_client, tmp_path,
                                                                      monkeypatch):
    """When the OCO's spawned LO never resolves, the umbrella OCO record is
    kept as-is (``tracked_category`` stays ``"OCO"``) rather than crashing."""
    monkeypatch.setattr(broker.time, "sleep", lambda *_: None)
    b = _broker(fake_client, tmp_path,
               post_order=(201, {"id": "8000", "symbol": "VN30F1M", "side": "NS",
                                  "quantity": 3, "orderStatus": "New"}),
               get_order_detail=(200, {}))  # no externalOrderId, ever
    envelope = _envelope(ExitIntent(pine_id="TP", from_entry="L", symbol="VN30F1M", side="sell",
                                    qty=3, tp_price=1550.0, sl_price=1450.0))
    orders = asyncio.run(b.execute_exit(envelope))
    assert orders[0].id == "8000", "keeps the umbrella OCO id when the LO never resolves"
    assert b._order_category["8000"] == "OCO"
    assert b._client.count("get_order_detail") == 6, "must poll the fixed attempt budget, then give up"


# --- _write token-retry ------------------------------------------------------

def __test_write_retries_once_on_invalid_token__(fake_client, tmp_path):
    seen_tokens = []

    def post_order(*args, **kwargs):
        seen_tokens.append(args[3])
        if len(seen_tokens) == 1:
            return (400, {"code": "INVALID_TRADING_TOKEN"})
        return (201, {"id": "1", "symbol": "VN30F1M", "side": "NB", "quantity": 1,
                      "orderStatus": "New"})

    b = _broker(fake_client, tmp_path, post_order=post_order)
    envelope = _envelope(EntryIntent(pine_id="L", symbol="VN30F1M", side="buy", qty=1,
                                     order_type=OrderType.LIMIT, limit=1500.0))
    orders = asyncio.run(b.execute_entry(envelope))
    assert b._client.count("post_order") == 2, "must retry EXACTLY once on token failure"
    assert orders[0].id == "1"


def __test_write_propagates_when_still_invalid_after_retry__(fake_client, tmp_path):
    b = _broker(fake_client, tmp_path,
               post_order=(400, {"code": "INVALID_TRADING_TOKEN"}))
    envelope = _envelope(EntryIntent(pine_id="L", symbol="VN30F1M", side="buy", qty=1,
                                     order_type=OrderType.LIMIT, limit=1500.0))
    with pytest.raises(AuthenticationError):
        asyncio.run(b.execute_entry(envelope))
    assert b._client.count("post_order") == 2, "must NOT loop forever retrying the same failure"


# --- _to_exchange_order ------------------------------------------------------

@pytest.mark.parametrize(
    "raw_status, expected",
    list(broker._STATUS_MAP.items()) + [("SOME_UNKNOWN_STATUS", OrderStatus.PENDING)],
)
def __test_to_exchange_order_status_map__(fake_client, tmp_path, raw_status, expected):
    b = _broker(fake_client, tmp_path)
    order = b._to_exchange_order({"id": "1", "symbol": "VN30F1M", "side": "NB", "quantity": 1,
                                  "orderStatus": raw_status})
    assert order.status is expected, f"{raw_status!r} must map to {expected}"
    assert order.symbol == "VN30F1M"


def __test_to_exchange_order_normalizes_case_and_separators__(fake_client, tmp_path):
    b = _broker(fake_client, tmp_path)
    order = b._to_exchange_order({"id": "1", "orderStatus": "Partially_Filled"})
    assert order.status is OrderStatus.PARTIALLY_FILLED, \
        "status match must be case- and separator-insensitive"


def __test_to_exchange_order_defaults_when_fields_missing__(fake_client, tmp_path):
    b = _broker(fake_client, tmp_path)
    order = b._to_exchange_order({"id": "42"})
    assert order.qty == 0.0 and order.filled_qty == 0.0
    assert order.remaining_qty == 0.0
    assert order.price is None and order.stop_price is None
    assert order.average_fill_price is None
    assert order.status is OrderStatus.PENDING
    assert order.side == "buy", "an unrecognised/missing side must default to buy, never crash"
    assert order.symbol == b.symbol, "falls back to the broker's own symbol when raw lacks one"


def __test_to_exchange_order_remaining_qty_from_qty_minus_filled__(fake_client, tmp_path):
    b = _broker(fake_client, tmp_path)
    order = b._to_exchange_order({"id": "1", "quantity": 10, "fillQuantity": 4})
    assert order.qty == 10.0 and order.filled_qty == 4.0
    assert order.remaining_qty == 6.0, "remaining defaults to qty - filled when leaveQuantity is absent"


def __test_to_exchange_order_prefers_explicit_leave_quantity__(fake_client, tmp_path):
    b = _broker(fake_client, tmp_path)
    order = b._to_exchange_order({"id": "1", "quantity": 10, "fillQuantity": 4, "leaveQuantity": 1})
    assert order.remaining_qty == 1.0, "explicit leaveQuantity must win over the qty-filled fallback"
    assert order.filled_qty == 4.0


# --- quick wins --------------------------------------------------------------

def __test_marketable_price_raises_without_ceiling_or_floor__(fake_client, tmp_path):
    b = _broker(fake_client, tmp_path, get_security_definition=(200, {}))
    with pytest.raises(RuntimeError, match="ceiling/floor"):
        b._marketable_price("buy")


def __test_gtd_is_rfc3339_about_a_week_out__():
    before = datetime.now(timezone.utc)
    result = broker.DNSEBroker._gtd(days=7)
    after = datetime.now(timezone.utc)
    assert result.endswith("Z"), "GTD expiry must be RFC3339 Zulu-suffixed"
    parsed = datetime.strptime(result, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    assert before + timedelta(days=6, hours=23) <= parsed <= after + timedelta(days=7, minutes=1), \
        f"expected ~7 days out, got {parsed} (bounds around {before}..{after})"


def __test_loan_package_id_caches_after_first_resolve__(fake_client, tmp_path):
    b = _broker(fake_client, tmp_path)
    first = b._loan_package_id()
    second = b._loan_package_id()
    assert first == 42 and second == 42
    assert b._client.count("get_loan_packages") == 1, "must resolve once and cache thereafter"


def __test_loan_package_id_raises_when_no_packages__(fake_client, tmp_path):
    b = _broker(fake_client, tmp_path, get_loan_packages=(200, {"loanPackages": []}))
    with pytest.raises(RuntimeError):
        b._loan_package_id()


def __test_get_capabilities_snapshot__(fake_client, tmp_path):
    b = _broker(fake_client, tmp_path)
    caps = b.get_capabilities()
    assert caps.stop_order is CapabilityLevel.NATIVE
    assert caps.tp_sl_bracket is CapabilityLevel.NATIVE
    assert caps.oca_cancel is CapabilityLevel.NATIVE
    assert caps.watch_orders is CapabilityLevel.SOFTWARE
    assert caps.short_selling is CapabilityLevel.NATIVE
    assert caps.trailing_stop is CapabilityLevel.SOFTWARE
    assert caps.idempotency is CapabilityLevel.SOFTWARE


def __test_connect_disconnect_lifecycle__(fake_client, tmp_path):
    b = _broker(fake_client, tmp_path)
    assert b.is_connected is False, "must start disconnected"
    asyncio.run(b.connect())
    assert b.is_connected is True
    asyncio.run(b.disconnect())
    assert b.is_connected is False
