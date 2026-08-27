"""#55 repro — a cancel that loses the race to a FILL must never read as
``CANCEL_CONFIRMED``.

The engine's entry-stop resolver fires a MARKET on ``CANCEL_CONFIRMED``
(``sync_engine.py`` — "provably gone with no fill"), and its ``ALREADY_FILLED``
branch marks the limit won and withholds the market. DNSE never produces
``ALREADY_FILLED``: every terminal read-back (``_TERMINAL_STATUSES`` includes
``FILLED``), every ``TERMINAL_CODES`` reject (``ORDER_IS_DONE`` covers filled
orders) and even absence-from-every-book all collapse to ``_cancel_one() ->
True`` -> ``CANCEL_CONFIRMED`` -> a second MARKET on a filled entry = DOUBLE
OPEN (round-1 #53 item 7, red-proven; blind-rediscovered round-2 #59 item 7).

Same fake-client seam as ``test_broker_lifecycle.py``. Tests A / B / B2 are RED
on the unmodified tree; C is the green control proving a genuine cancel still
confirms.
"""
import asyncio

import pynecore.lib as lib

lib.bar_index = 0  # let the [BROKER] log formatter render during broker._emit()

from pynecore_dnse import broker
from pynecore.core.broker.models import (
    CancelIntent, DispatchEnvelope, CancelDispositionOutcome,
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
    instance._cancel_verify_attempts = 2
    instance._cancel_verify_delay = 0.0
    return instance


def _cancel_envelope(pine_id="K"):
    return DispatchEnvelope(intent=CancelIntent(pine_id=pine_id, symbol="VN30F1M"),
                            run_tag="abcd", bar_ts_ms=1_700_000_000_000,
                            retry_seq=0, coid_max_len=30)


def _detail(status, *, filled=0.0, qty=1):
    return (200, {"id": "ID1", "symbol": "VN30F1M", "side": "NB", "quantity": qty,
                  "orderStatus": status, "fillQuantity": filled})


def _track(b, order_id="ID1", category="NORMAL"):
    b._order_ids["K"] = [order_id]
    b._order_category[order_id] = category


# --- A (RED): cancel ACKed, read-back says Filled -> must be ALREADY_FILLED --

def __test_cancel_raced_by_fill_reports_already_filled__(fake_client, tmp_path):
    """Cancel 200-ACKs, but the venue read-back shows the order FILLED (the
    race was lost). Today ``_cancel_took_effect`` counts FILLED as "the cancel
    took effect" and the outcome surfaces as CANCEL_CONFIRMED — the engine
    then fires the entry-stop MARKET into an already-filled LIMIT."""
    b = _broker(fake_client, tmp_path,
                cancel_order=(200, {"orderStatus": "New"}),
                get_order_detail=_detail("Filled", filled=1.0))
    _track(b)

    outcome = asyncio.run(b.execute_cancel_with_outcome(_cancel_envelope()))

    assert outcome is CancelDispositionOutcome.ALREADY_FILLED, (
        f"fill-raced cancel must surface ALREADY_FILLED (engine restores legs, "
        f"withholds the market); got {outcome!r} — CANCEL_CONFIRMED here is the "
        f"#55 double-open")


# --- B (RED): ORDER_IS_DONE on a filled order -> must be ALREADY_FILLED -----

def __test_order_is_done_reject_on_filled_order_reports_already_filled__(fake_client, tmp_path):
    """The venue rejects the cancel with ``ORDER_IS_DONE`` and the detail
    read-back shows FILLED. Today TERMINAL_CODES short-circuits to
    "treated-gone" without ever asking WHY the order is done."""
    b = _broker(fake_client, tmp_path,
                cancel_order=(400, {"code": "ORDER_IS_DONE"}),
                get_order_detail=_detail("Filled", filled=1.0))
    _track(b)

    outcome = asyncio.run(b.execute_cancel_with_outcome(_cancel_envelope()))

    assert outcome is CancelDispositionOutcome.ALREADY_FILLED, (
        f"ORDER_IS_DONE with a FILLED read-back must surface ALREADY_FILLED; "
        f"got {outcome!r}")


# --- B2 (RED): absent from every book -> must be UNKNOWN, never CONFIRMED ---

def __test_absence_from_every_book_never_confirms_cancel__(fake_client, tmp_path):
    """The id 404s on every book. Absence proves nothing about disposition
    (fill vs cancel vs wrong id) — today ``all_not_found`` returns True and
    the outcome surfaces as CANCEL_CONFIRMED, concluding "gone with no fill"
    from silence."""
    b = _broker(fake_client, tmp_path,
                cancel_order=(400, {"code": "RESOURCE_NOT_FOUND"}),
                get_order_detail=(400, {"code": "RESOURCE_NOT_FOUND"}))
    b._order_ids["K"] = ["ID1"]          # no category hint -> probes every book

    outcome = asyncio.run(b.execute_cancel_with_outcome(_cancel_envelope()))

    assert outcome is not CancelDispositionOutcome.CANCEL_CONFIRMED, (
        "absence from every book is not a positive observation of 'cancelled "
        "with no fill' — CANCEL_CONFIRMED from silence is the #55 anti-pattern")
    assert outcome is CancelDispositionOutcome.UNKNOWN, (
        f"unresolvable disposition must stay UNKNOWN so the engine retries; "
        f"got {outcome!r}")


# --- C (GREEN control): genuine cancel still confirms ------------------------

def __test_genuine_cancel_with_zero_fill_confirms__(fake_client, tmp_path):
    """Red-first discipline: the control. Cancel ACKs and the read-back shows
    ``Canceled`` with ``fillQuantity=0`` — a positive observation of
    cancelled-with-no-fill. This MUST stay CANCEL_CONFIRMED before and after
    the fix (a fix that degrades genuine cancels to UNKNOWN would stall every
    legitimate cancel into the retry loop)."""
    b = _broker(fake_client, tmp_path,
                cancel_order=(200, {"orderStatus": "New"}),
                get_order_detail=_detail("Canceled", filled=0.0))
    _track(b)

    outcome = asyncio.run(b.execute_cancel_with_outcome(_cancel_envelope()))

    assert outcome is CancelDispositionOutcome.CANCEL_CONFIRMED, (
        f"a read-back-verified cancel with zero fill must confirm; got {outcome!r}")


# --- G6: fill outranks status on every read path -----------------------------

def __test_fill_beats_status_precedence_in_pure_classifier__():
    """Rejected/Expired/Canceled rows carrying ANY fill classify ALREADY_FILLED —
    the engine fires the market on TOO_LATE_TO_CANCEL exactly as it does on
    CANCEL_CONFIRMED, so a partial-fill-then-terminal row mapped by status
    alone recreates the double-open through the fixed code (#55 panel, P3)."""
    from pynecore.core.broker.models import OrderStatus
    from pynecore_dnse.cancel_disposition import classify_readback

    for status in (OrderStatus.REJECTED, OrderStatus.EXPIRED,
                   OrderStatus.CANCELLED, OrderStatus.FILLED):
        assert classify_readback(status, 1.0) \
            is CancelDispositionOutcome.ALREADY_FILLED, status


def __test_order_is_done_reject_with_partial_fill_reports_already_filled__(fake_client, tmp_path):
    """G6 through the reject path: the venue says DONE, the detail says
    Expired with a partial fill — the position exists, the market must not fire."""
    b = _broker(fake_client, tmp_path,
                cancel_order=(400, {"code": "ORDER_CANCEL_STATUS_REJECTED"}),
                get_order_detail=_detail("Expired", filled=1.0, qty=2))
    _track(b)

    outcome = asyncio.run(b.execute_cancel_with_outcome(_cancel_envelope()))

    assert outcome is CancelDispositionOutcome.ALREADY_FILLED


# --- G2: Activated conditional classifies on the CHILD -----------------------

def __test_activated_conditional_classifies_on_the_normal_book_child__(fake_client, tmp_path):
    """#41: the shell stays Activated forever; the NORMAL-book child did the
    work. CO-ORD-013 -> resolve externalOrderId -> classify the CHILD (G2)."""
    def _detail_by_id(account, order_id, market, order_category=None):
        if order_id == "SHELL":
            return (200, {"id": "SHELL", "externalOrderId": "CH1"})
        return (200, {"id": "CH1", "symbol": "VN30F1M", "side": "NB", "quantity": 1,
                      "orderStatus": "Filled", "fillQuantity": 1.0})

    b = _broker(fake_client, tmp_path,
                cancel_order=(400, {"code": "CO-ORD-013"}),
                get_order_detail=_detail_by_id)
    _track(b, "SHELL", "STOP")

    outcome = asyncio.run(b.execute_cancel_with_outcome(_cancel_envelope()))

    assert outcome is CancelDispositionOutcome.ALREADY_FILLED, (
        "an Activated shell whose child FILLED must surface ALREADY_FILLED — "
        "confirming on the shell is how a stop-entry double-opens")


# --- history: a positive row classifies; absence already pinned UNKNOWN ------

def __test_history_row_with_fill_classifies_already_filled__(fake_client, tmp_path):
    """Absent from every book, but /orders/history carries the terminal row
    (date-prefixed id, under 'data') — the POSITIVE observation decides."""
    b = _broker(fake_client, tmp_path,
                cancel_order=(400, {"code": "RESOURCE_NOT_FOUND"}),
                get_order_detail=(400, {"code": "RESOURCE_NOT_FOUND"}),
                get_order_history=(200, {"data": [
                    {"id": "20260826_ID1", "symbol": "VN30F1M", "side": "NB",
                     "quantity": 1, "orderStatus": "Filled", "fillQuantity": 1.0}]}))
    b._order_ids["K"] = ["ID1"]

    outcome = asyncio.run(b.execute_cancel_with_outcome(_cancel_envelope()))

    assert outcome is CancelDispositionOutcome.ALREADY_FILLED


# --- declared bool change: execute_cancel over the same disposition core -----

def __test_execute_cancel_bool_absence_is_false_declared_change__(fake_client, tmp_path):
    """#55 DECLARED CHANGE: absence-from-every-book was True ("gone == done");
    it is now False — the same no-conclusion-from-silence rule as the outcome
    contract (core collapses both bools to UNKNOWN; False = retry, safe)."""
    b = _broker(fake_client, tmp_path,
                cancel_order=(400, {"code": "RESOURCE_NOT_FOUND"}),
                get_order_detail=(400, {"code": "RESOURCE_NOT_FOUND"}))
    b._order_ids["K"] = ["ID1"]

    assert asyncio.run(b.execute_cancel(_cancel_envelope())) is False


# --- the pure module's terminal set mirrors the broker's ---------------------

def __test_terminal_status_sets_stay_in_sync__():
    from pynecore_dnse import broker as broker_module
    from pynecore_dnse import cancel_disposition

    assert broker_module._TERMINAL_STATUSES == cancel_disposition.TERMINAL_STATUSES, (
        "cancel_disposition duplicates the terminal set to stay import-pure; "
        "this anchor is the drift alarm")


# --- G9/liveness: client reads run off-loop, the loop stays live -------------

def __test_event_loop_stays_live_during_cancel_verify__(fake_client, tmp_path):
    """The old verify loop blocked the SHARED event loop 8.4 s (measured) —
    starving the watch_orders fill feed. Reads now run in a worker thread and
    pacing is await-based, so a concurrent ticker must keep ticking while a
    SLOW read (0.15 s each) is in flight."""
    import time as _time

    def _slow_detail(*a, **k):
        _time.sleep(0.15)
        return (200, {"id": "ID1", "symbol": "VN30F1M", "side": "NB",
                      "quantity": 1, "orderStatus": "New"})

    b = _broker(fake_client, tmp_path,
                cancel_order=(200, {"orderStatus": "New"}),
                get_order_detail=_slow_detail)
    _track(b)
    b._cancel_verify_attempts = 2

    async def _run():
        ticks = 0

        async def _ticker():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.02)
                ticks += 1

        task = asyncio.get_running_loop().create_task(_ticker())
        try:
            outcome = await b._cancel_one_disposition("ID1")
        finally:
            task.cancel()
        return outcome, ticks

    outcome, ticks = asyncio.run(_run())

    assert outcome is CancelDispositionOutcome.UNKNOWN      # G5: still New
    assert ticks >= 3, (
        f"only {ticks} ticks during ~0.3 s of venue reads — the loop is being "
        f"blocked; client calls must run via asyncio.to_thread")
