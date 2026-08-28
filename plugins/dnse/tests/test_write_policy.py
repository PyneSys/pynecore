"""#58 repro — `_write` must not auto-retry an INVALID_TRADING_TOKEN refusal.

The retry's docstring rationale ("the cron may have refreshed the state
file") is void: ``_token()`` reads the state file fresh on EVERY call
(broker.py), so the first attempt already carried the freshest token and the
retry is a second IDENTICAL write. The measured #51/#46 venue windows make it
actively harmful: a 30-second-old token was refused, re-minting was measured
not to reclaim (three-for-three), and the live rules forbid retrying into the
lockout — yet every refused write fires twice (a refused cancel loop = ~2
extra venue writes per engine retry, flagged on #65 too).

RED-1/RED-2 pin the new policy (one write per refusal, loud named reason) on
the entry and cancel paths; the GREEN control pins that healthy writes are
untouched. Five existing tests PIN the old behavior and are rewritten with
the venue-fact rationale in the fix commit (the #47 characterization-pin
lesson, at suite scale).
"""
import asyncio

import pytest
import pynecore.lib as lib

lib.bar_index = 0  # let the [BROKER] log formatter render during broker._emit()

from pynecore_dnse import broker
from pynecore.core.broker.models import (
    CancelDispositionOutcome, EntryIntent, DispatchEnvelope, OrderType,
)
from pynecore.core.broker.exceptions import AuthenticationError

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


# --- RED 1: a refused PLACE writes exactly once ------------------------------

def __test_invalid_token_place_writes_exactly_once__(fake_client, tmp_path):
    """INVALID_TRADING_TOKEN on a place must surface (AuthenticationError with
    the named reason) after ONE venue write — the auto-retry was a second
    identical write into the measured #51 lockout."""
    b = _broker(fake_client, tmp_path,
                post_order=(400, {"code": "INVALID_TRADING_TOKEN",
                                  "message": "Invalid trading token"}))

    with pytest.raises(AuthenticationError):
        asyncio.run(b.execute_entry(_entry_envelope()))

    assert b._client.count("post_order") == 1, (
        f"{b._client.count('post_order')} writes for one refusal — the "
        f"token-reread retry fires a second identical write (#58)")


# --- RED 2: a refused CANCEL writes exactly once per book --------------------

def __test_invalid_token_cancel_writes_exactly_once__(fake_client, tmp_path):
    """The cancel path routes through the same `_write`; during a #51 window
    the engine already retries per tick, so the plugin-level double makes it
    ~2 venue writes per tick, unbounded (#65)."""
    b = _broker(fake_client, tmp_path,
                cancel_order=(400, {"code": "INVALID_TRADING_TOKEN",
                                    "message": "Invalid trading token"}))
    b._order_ids["K"] = ["ID1"]
    b._order_category["ID1"] = "STOP"

    from pynecore.core.broker.models import CancelIntent
    envelope = DispatchEnvelope(
        intent=CancelIntent(pine_id="K", symbol="VN30F1M"),
        run_tag="abcd", bar_ts_ms=1_700_000_000_000, retry_seq=0, coid_max_len=30)
    outcome = asyncio.run(b.execute_cancel_with_outcome(envelope))

    assert outcome is CancelDispositionOutcome.UNKNOWN
    assert b._client.count("cancel_order") == 1, (
        f"{b._client.count('cancel_order')} cancel writes for one refusal — "
        f"the retry doubles every write into the lockout")


# --- GREEN control: healthy writes are untouched -----------------------------

def __test_healthy_place_single_write_unchanged__(fake_client, tmp_path):
    b = _broker(fake_client, tmp_path,
                post_order=(201, {"id": "1", "symbol": "VN30F1M", "side": "NB",
                                  "quantity": 1, "orderStatus": "New"}))

    orders = asyncio.run(b.execute_entry(_entry_envelope()))

    assert orders[0].id == "1"
    assert b._client.count("post_order") == 1


# --- the AMEND path was unpinned (panel P1) ----------------------------------

def __test_invalid_token_amend_writes_exactly_once__(fake_client, tmp_path):
    """`_amend` routes through the same `_write` (put_order) — one refusal,
    one venue write, surfaced; never a second identical attempt."""
    from pynecore.core.broker.exceptions import AuthenticationError as _AuthErr
    b = _broker(fake_client, tmp_path,
                put_order=(400, {"code": "INVALID_TRADING_TOKEN"}))
    b._order_ids["L"] = ["ID1"]
    b._order_category["ID1"] = "NORMAL"

    envelope = _entry_envelope()
    with pytest.raises(_AuthErr):
        asyncio.run(b.modify_entry(envelope, envelope))

    assert b._client.count("put_order") == 1, (
        f"{b._client.count('put_order')} amend writes for one refusal")
