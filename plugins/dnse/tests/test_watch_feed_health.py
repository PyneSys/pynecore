"""#54 repro — a persistently failing poll must not leave the fill feed
PERMANENTLY SILENT.

Post-#62 shape of the hazard: a permanent 401 (broken API key, venue refusal)
makes ``_read_book_rows_sync`` return None every cycle, ``_iter_orders``
yields nothing, and ``watch_orders`` loops forever emitting only DEBUG lines —
strategies keep trading, bars keep arriving, fills never do. The bare
``except Exception: continue`` at the poll site is even quieter: an
every-cycle crash (parser bug, thread error) produces ZERO log evidence.
Round-1 measurement: 18,936 silent polls under a permanent 401.

The baseline is MECHANISM-NEUTRAL: red tests demand *any* loud signal —
a WARNING+ log line or a raised escalation — within ~60 poll cycles.
Today there is neither. The control pins that a healthy feed stays quiet.
"""
import asyncio
import logging

import pynecore.lib as lib

lib.bar_index = 0  # let the [BROKER] log formatter render during broker._emit()

from pynecore_dnse import broker

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
    instance._poll_interval = 0.001
    return instance


def _pump_watch(instance, cycles, *, wall_clock=2.0):
    """Drive ``watch_orders`` until the client has served ``cycles`` polls (or
    the generator raises). Returns (raised_exception_or_None, poll_count)."""
    async def _run():
        stream = instance.watch_orders()
        raised = None
        task = asyncio.ensure_future(stream.__anext__())
        deadline = asyncio.get_running_loop().time() + wall_clock
        try:
            while (instance._client.count("get_orders") < cycles
                   and asyncio.get_running_loop().time() < deadline):
                if task.done():
                    exc = task.exception()
                    if exc is not None and not isinstance(exc, StopAsyncIteration):
                        raised = exc
                    break
                await asyncio.sleep(0.005)
        finally:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            await stream.aclose()
        return raised
    return asyncio.run(_run()), instance._client.count("get_orders")


def _loud_records(caplog):
    return [r for r in caplog.records if r.levelno >= logging.WARNING]


# --- RED 1: permanent 401 must produce a loud signal -------------------------

def __test_permanent_auth_failure_is_not_silent__(fake_client, tmp_path, caplog):
    """Every poll answers 401 UNAUTHORIZED. Today: DEBUG-only, forever
    (measured round 1: 18,936 silent polls). The feed must get LOUD — a
    WARNING+ line or a raised escalation — within ~60 cycles."""
    b = _broker(fake_client, tmp_path,
                get_orders=(401, {"code": "UNAUTHORIZED", "message": "bad key"}))

    with caplog.at_level(logging.DEBUG):
        raised, polls = _pump_watch(b, cycles=60)

    assert polls >= 40, f"driver starved: only {polls} polls ran"
    assert raised is not None or _loud_records(caplog), (
        f"{polls} consecutive 401 polls produced no WARNING+ log and no "
        f"raise — the fill feed is PERMANENTLY SILENT under a dead credential")


# --- RED 2: an every-cycle crash must not vanish into the bare except --------

def __test_repeated_poll_crash_is_not_silent__(fake_client, tmp_path, caplog):
    """``get_orders`` raises every cycle (parser/thread-class failure). The
    bare ``except Exception: continue`` swallows it with ZERO log evidence
    today — strictly quieter than the 401 case."""
    def _boom(*_args, **_kwargs):
        raise RuntimeError("synthetic every-cycle failure")

    b = _broker(fake_client, tmp_path, get_orders=_boom)

    with caplog.at_level(logging.DEBUG):
        raised, polls = _pump_watch(b, cycles=60)

    assert polls >= 40, f"driver starved: only {polls} polls ran"
    assert raised is not None or _loud_records(caplog), (
        f"{polls} consecutive poll crashes produced no WARNING+ log and no "
        f"raise — zero evidence the feed is dead")


# --- GREEN control: a healthy feed stays quiet -------------------------------

def __test_healthy_feed_stays_quiet__(fake_client, tmp_path, caplog):
    """Empty books, all reads 200 — no warnings, no raise. The escalation
    must fire on PERSISTENT failure, never on a healthy idle feed."""
    b = _broker(fake_client, tmp_path,
                get_orders=(200, {"orders": [], "totalPages": 1}))

    with caplog.at_level(logging.DEBUG):
        raised, polls = _pump_watch(b, cycles=60)

    assert polls >= 40, f"driver starved: only {polls} polls ran"
    assert raised is None, f"healthy feed raised {raised!r}"
    assert not _loud_records(caplog), (
        f"healthy feed logged WARNING+: {[r.message for r in _loud_records(caplog)]}")


# --- per-book: a healthy NORMAL book must not mask a dead STOP book ----------

def __test_single_dead_book_escalates_despite_healthy_sibling__(fake_client, tmp_path, caplog):
    """Panel P1+P3 (independent): 'any successful book resets' would re-create
    the card's bug for one book — and a conditional-book-specific outage is
    the measured-plausible live shape. Counters are per book."""
    def _orders(account, market, order_category=None, page_index=0, page_size=100):
        if order_category == "NORMAL":
            return (200, {"orders": [], "totalPages": 1})
        return (401, {"code": "UNAUTHORIZED"})

    b = _broker(fake_client, tmp_path, get_orders=_orders)
    b._feed_warn_after = 5

    with caplog.at_level(logging.DEBUG):
        raised, polls = _pump_watch(b, cycles=30)

    assert raised is None, "a single-book failure must warn, never halt/raise"
    stop_warnings = [r for r in _loud_records(caplog) if "STOP" in r.getMessage()]
    assert stop_warnings, (
        "the dead STOP book never escalated — the healthy NORMAL book masked it")


# --- halt shape: ALL books AUTH-persistent halts; single book never ----------

def __test_all_books_auth_persistent_raises_designed_halt__(fake_client, tmp_path):
    from pynecore.core.broker.exceptions import BrokerManualInterventionError

    b = _broker(fake_client, tmp_path,
                get_orders=(401, {"code": "UNAUTHORIZED", "message": "dead key"}))
    b._feed_warn_after = 3
    b._feed_halt_after = 6

    raised, polls = _pump_watch(b, cycles=40)

    assert isinstance(raised, BrokerManualInterventionError), (
        f"a persistent all-books AUTH refusal must raise the DESIGNED halt "
        f"(engine latches it); got {raised!r} after {polls} polls")


def __test_transient_failures_warn_but_never_halt__(fake_client, tmp_path, caplog):
    """A dead network (500s) warns forever but must not latch the
    irreversible halt — only a provably dead CREDENTIAL does."""
    b = _broker(fake_client, tmp_path,
                get_orders=(500, {"code": "REMOTE_SERVER_ERROR"}))
    b._feed_warn_after = 3
    b._feed_halt_after = 6

    with caplog.at_level(logging.DEBUG):
        raised, polls = _pump_watch(b, cycles=40)

    assert raised is None, f"transients must never halt; got {raised!r}"
    assert _loud_records(caplog), "persistent transients must still warn"


# --- G7: a poisoned row / drain crash must not kill the stream ---------------

def __test_poisoned_row_does_not_kill_the_stream__(fake_client, tmp_path, caplog):
    """One malformed row crashes the scan every cycle — the generator must
    survive (the engine's supervisor kills the task permanently on ANY
    raise, sync_engine.py:2054) and eventually get loud."""
    poisoned = {"id": "BAD", "quantity": "not-a-number", "side": "NB"}   # float() raises in _to_exchange_order
    b = _broker(fake_client, tmp_path,
                get_orders=(200, {"orders": [poisoned], "totalPages": 1}))
    b._feed_warn_after = 5

    with caplog.at_level(logging.DEBUG):
        raised, polls = _pump_watch(b, cycles=30)

    assert polls >= 20, f"stream died early: only {polls} polls"
    assert raised is None, f"a poisoned row must never kill the stream; got {raised!r}"
    scan_warnings = [r for r in _loud_records(caplog) if "scan" in r.getMessage()]
    assert scan_warnings, "a persistent scan crash must escalate to WARNING"


def __test_drain_crash_does_not_kill_the_stream__(fake_client, tmp_path, caplog):
    """The pre-poll OCO drain sat OUTSIDE every guard (verified) — one
    exception there killed the stream permanently. It must survive + warn."""
    b = _broker(fake_client, tmp_path,
                get_orders=(200, {"orders": [], "totalPages": 1}))
    b._feed_warn_after = 5
    b._pending_oco.add("da203umbrella")

    async def _boom_drain():
        raise RuntimeError("drain path failure")
        yield  # pragma: no cover — makes this an async generator

    b._drain_pending_oco = _boom_drain

    with caplog.at_level(logging.DEBUG):
        raised, polls = _pump_watch(b, cycles=30)

    assert polls >= 20, f"stream died early: only {polls} polls"
    assert raised is None, f"a drain crash must never kill the stream; got {raised!r}"
    drain_warnings = [r for r in _loud_records(caplog) if "drain" in r.getMessage()]
    assert drain_warnings, "a persistent drain crash must escalate to WARNING"


# --- single-flight: a hung read never stacks worker threads ------------------

def __test_hung_read_is_single_flight__(fake_client, tmp_path):
    """Panel P2+P3: abandoning a hung read per cycle exhausts the SHARED
    default executor. With single-flight, a read hung far past the wait
    deadline results in at most ONE in-flight get_orders call."""
    import threading
    started = []
    release = threading.Event()

    def _hang(*_args, order_category=None, **_kwargs):
        if order_category == "NORMAL":      # first book of a poll = one poll START
            started.append(1)
        release.wait(0.8)
        return (200, {"orders": [], "totalPages": 1})

    b = _broker(fake_client, tmp_path, get_orders=_hang)
    b._watch_read_deadline_s = 0.02

    try:
        raised, _ = _pump_watch(b, cycles=999, wall_clock=0.5)
    finally:
        release.set()

    assert raised is None
    assert len(started) == 1, (
        f"{len(started)} reads started against a hung venue — single-flight "
        f"must re-await the SAME read, never spawn another")


# --- pure ladder: throttle + reset (deterministic, no timing) ----------------

def __test_feed_health_ladder_throttle_and_reset__():
    from pynecore_dnse.feed_health import FeedHealth

    ladder = FeedHealth(warn_after=3, rewarn_every=5, halt_after=4,
                        books=("NORMAL", "STOP"))
    # below threshold: quiet
    for _ in range(2):
        ladder.record_failure("NORMAL", "x")
    assert ladder.warnings_due() == []
    # success resets ONLY that book
    ladder.record_failure("STOP", "x")
    ladder.record_success("NORMAL")
    ladder.record_failure("NORMAL", "x")
    assert ladder.warnings_due() == [], "reset-on-success must clear the streak"
    # crossing the threshold warns once, then throttles until rewarn_every
    for _ in range(2):
        ladder.record_failure("NORMAL", "x")
    assert len(ladder.warnings_due()) == 1
    for _ in range(4):
        ladder.record_failure("NORMAL", "x")
        assert ladder.warnings_due() == []
    ladder.record_failure("NORMAL", "x")
    assert len(ladder.warnings_due()) == 1, "re-warn due after rewarn_every"
    # halt: ALL books AUTH-persistent, never a single book
    for _ in range(6):
        ladder.record_failure("NORMAL", "401", is_auth=True)
    assert ladder.halt_due() is None, "one AUTH book must not halt"
    for _ in range(6):
        ladder.record_failure("STOP", "401", is_auth=True)
    assert ladder.halt_due() is not None
    # a transient interleaved into one book breaks ITS auth streak
    ladder.record_failure("STOP", "500", is_auth=False)
    assert ladder.halt_due() is None, "an interleaved transient breaks the AUTH streak"


def __test_bad_secret_shape_reaches_the_designed_halt__(fake_client, tmp_path):
    """#68 (panel P3's missing anchor): the MEASURED bad-secret reply
    (OA-400 + Authorization message, live 2026-08-31) must drive the
    all-books-AUTH streak to the designed halt — before #68 it classified
    REJECT and could never halt."""
    from pynecore.core.broker.exceptions import BrokerManualInterventionError

    b = _broker(fake_client, tmp_path,
                get_orders=(400, {"code": "OA-400",
                                  "message": "Authorization field missing, "
                                             "malformed or invalid"}))
    b._feed_warn_after = 3
    b._feed_halt_after = 6

    raised, polls = _pump_watch(b, cycles=40)

    assert isinstance(raised, BrokerManualInterventionError), (
        f"the measured dead-secret shape never reached the designed halt; "
        f"got {raised!r} after {polls} polls")
